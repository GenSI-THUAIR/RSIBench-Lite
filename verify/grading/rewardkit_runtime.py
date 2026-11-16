"""本仓 Reward Kit 入口：仅为 CLI judge 保留历史非零退出重试。

函数参数沿用 rewardkit.run；调用示例：run(rubric_dir, workspace=workdir, output=output)。
LLM、判分、格式校验、隔离与产物格式沿用官方 0.1.7，不修改其安装文件或全局函数。
CLI 调用方法迁自 npu-fly 1da796b2（Apache-2.0）。
"""
import asyncio
from contextlib import nullcontext
import json
import logging
from pathlib import Path

from rewardkit import discover
from rewardkit.models import AgentJudge, Criterion, Score
from rewardkit.reward import Reward, _guarded
from rewardkit.isolation import aisolate
from rewardkit.judges import build_prompt, parse_judge_response, _build_response_schema, _timeout_scores, _MAX_JUDGE_RETRIES
from rewardkit.runner import _load_reward_specs, _validate_reward_specs, _run_all, _group_scores, _apply_specs, _write_outputs

logger = logging.getLogger(__name__)


class RetryingAgentReward(Reward):
    async def _arun_inner(self, sem=None):
        if not isinstance(self.judge, AgentJudge):
            return await super()._arun_inner(sem)

        async def invoke():
            context = aisolate(self.workspace) if (self.workspace and self.judge.isolated) else nullcontext(self.workspace)
            async with context as workspace:
                return await arun_agent(self.judge, self.criteria, self.weights, workspace, self.system_prompt)

        self.scores, self.judge_output, self.warnings = await _guarded(invoke(), sem)
        return self.scores


async def arun_agent(
    judge: AgentJudge,
    criteria: list[Criterion],
    weights: list[float] | None = None,
    workspace: str | Path | None = None,
    system_prompt: str | None = None,
) -> tuple[list[Score], str, list[str]]:
    if judge.mcp_servers:
        from rewardkit.agents import get_agent

        backend = get_agent(judge.agent)
        backend.ensure_installed()
        cwd = judge.cwd or (
            str(workspace) if workspace and Path(workspace).is_dir() else None
        )
        backend.add_mcp_servers(judge.mcp_servers, cwd=cwd)
    if judge.mode == "individual":
        return await _arun_agent_individual(
            judge, criteria, weights, workspace, system_prompt
        )
    return await _arun_agent_call(judge, criteria, weights, workspace, system_prompt)


async def _arun_agent_individual(
    judge: AgentJudge,
    criteria: list[Criterion],
    weights: list[float] | None,
    workspace: str | Path | None,
    system_prompt: str | None,
) -> tuple[list[Score], str, list[str]]:
    """Grade each criterion in its own agent call, sequentially.

    Unlike the LLM individual path (which fans criteria out concurrently),
    agent calls are heavy local CLI subprocesses, so they run one at a time;
    N parallel agent processes would exhaust a small verifier container.
    """
    scores: list[Score] = []
    outputs: list[str] = []
    warns: list[str] = []
    for i, c in enumerate(criteria):
        s, raw, w = await _arun_agent_call(
            judge,
            [c],
            [weights[i]] if weights else None,
            workspace,
            system_prompt,
        )
        scores.extend(s)
        outputs.append(f"--- {c.name} ---\n{raw}")
        warns.extend(w)
    return scores, "\n\n".join(outputs), warns


async def _arun_agent_call(
    judge: AgentJudge,
    criteria: list[Criterion],
    weights: list[float] | None,
    workspace: str | Path | None,
    system_prompt: str | None,
) -> tuple[list[Score], str, list[str]]:
    from rewardkit.agents import get_agent

    warn_list: list[str] = []
    if system_prompt:
        prompt = build_prompt(criteria, template=system_prompt)
    else:
        prompt = build_prompt(criteria, kind="agent")
    if judge.atif_trajectory:
        prompt += f"\n\nThe agent's trajectory is stored at: {judge.atif_trajectory}"

    schema = _build_response_schema(criteria)
    backend = get_agent(judge.agent)
    cwd = judge.cwd or (
        str(workspace) if workspace and Path(workspace).is_dir() else None
    )

    backend.ensure_installed()

    allowed_tools = tuple(
        name for server in judge.mcp_servers for name in server.allowed_tool_names()
    )
    cmd = backend.build_command(prompt, schema, allowed_tools=allowed_tools)
    if judge.model:
        cmd.extend(backend.model_args(judge.model))

    try:
        for attempt in range(_MAX_JUDGE_RETRIES):
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            try:
                stdout, _stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=judge.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                warn_list.append(
                    f"judge timed out after {judge.timeout}s; recording affected "
                    "criteria as 0.0"
                )
                return _timeout_scores(criteria, weights, judge.timeout), "", warn_list
            raw_output = stdout.decode()
            if proc.returncode != 0:
                stderr_text = _stderr.decode().strip() if _stderr else ""
                if attempt == _MAX_JUDGE_RETRIES - 1:
                    raise ValueError(
                        f"Agent CLI '{backend.cli_name}' exited with code "
                        f"{proc.returncode}: {stderr_text or raw_output[:200]}"
                    )
                logger.debug(
                    "Agent judge CLI exited with code %d, retrying (%d/%d)",
                    proc.returncode,
                    attempt + 1,
                    _MAX_JUDGE_RETRIES,
                )
                continue
            raw_output = backend.parse_output(raw_output)
            try:
                scores = parse_judge_response(raw_output, criteria, weights)
                return scores, raw_output, warn_list
            except ValueError:
                if attempt == _MAX_JUDGE_RETRIES - 1:
                    raise
                logger.debug(
                    "Agent judge response did not match schema, retrying (%d/%d)",
                    attempt + 1,
                    _MAX_JUDGE_RETRIES,
                )
        raise RuntimeError("Unreachable")
    finally:
        backend.cleanup()


def run(
    tests_dir: str | Path,
    *,
    workspace: str | Path = "/app",
    output: str | Path = "/logs/verifier/reward.json",
    max_concurrent_programmatic: int = 8,
    max_concurrent_llm: int = 8,
    max_concurrent_agent: int = 2,
) -> dict[str, float]:
    rewards = discover(tests_dir, workspace=workspace)
    fields = ("criteria", "weights", "judge", "workspace", "name", "reward_weight", "system_prompt", "aggregation", "threshold")
    rewards = [RetryingAgentReward(**{key: getattr(r, key) for key in fields}) if isinstance(r.judge, AgentJudge) else r for r in rewards]

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rewards:
        out_path.write_text(json.dumps({}, indent=2))
        return {}

    # Validate names before running judges so conflicts fail fast.
    specs = _load_reward_specs(tests_dir)
    if specs:
        _validate_reward_specs(specs, {r.name for r in rewards})

    asyncio.run(
        _run_all(
            rewards,
            max_concurrent_programmatic=max_concurrent_programmatic,
            max_concurrent_llm=max_concurrent_llm,
            max_concurrent_agent=max_concurrent_agent,
        )
    )

    by_name, flat = _group_scores(rewards)
    main = _apply_specs(flat, by_name, specs)
    _write_outputs(out_path, main, by_name, flat)
    return main

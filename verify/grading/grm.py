"""GRM(Generative Reward Model):用 LLM 对 agent 的过程与产物做质量判定。

现有 reward 的全部自由度是一个吞吐比值,正确性只是 0/1 闸门,回答不了两个
决定"这个 patch 值不值得合入"的问题:

1. 成功时:性能是不是来自**工程 infra 层、可迁移到生产**的优化,而非拿
   质量换速度 / 对着评测 workload 特化这类不可泛化的手段?(0902 用户
   裁决:原"实质优化"与"非过拟合"两问本质是这一条的两面,已合并单条;
   patch 卫生短期不关键,暂不判)
2. 失败时:该归咎于谁(agent 过错 / 平台故障 / 灰区)?

实现走 RewardKit 的官方声明式形态(docs/content/docs/rewardkit/):rubric 是
`grm_rubric/` 下的 TOML,由 `rewardkit.run` 统一发现、并发执行、聚合,并写出
`reward-details.json`(逐条 score + judge reasoning,harbor view 原生渲染,
viewer/server.py:2517)。本模块只保留 RewardKit 没有的三样:

1. **patch 摘要** —— RewardKit 对 >1MB 的文件整个跳过(judges.py:173),
   15MB 的脏 patch 走原生 files 字段 judge 会什么都看不到,必须先压摘要;
2. **rubric 的每-trial 实例化** —— RewardKit 按字面路径读 files /
   atif-trajectory、不拼 workspace(judges.py:206),而 verifier 跑在宿主机、
   每个 trial 路径不同,须先把 {grm_dir} 占位符绑定到本 trial;
3. **硬门策略**(`apply_gate`)—— 否决项不过则主 reward 置零是 npu-fly 的
   harness 决策,不是打分逻辑。否决集的单一事实来源是 succ_quality rubric 里
   非 optional 的 criterion(`[scoring] required_pass` 是其原生语义)。

**判定结果只作为独立 reward 键旁挂,不改动主 reward 的计算** —— GRM 是质量
维度的增量,不替换既有的可信性层;接线收口见 grm_verifier.GrmGradedVerifier。
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import shutil
import tomllib
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

_RUBRIC_ROOT = Path(__file__).resolve().parent / "grm_rubric"

# judge 的 prompt 里放不下 15MB 的 patch(实测 qwen3p8 那份 1325 个 .pyc),
# 二进制段也没有阅读价值。只喂文件清单 + 文本 diff 头部。
_MAX_DIFF_BYTES = 60_000
_MAX_LISTED_FILES = 80

def _export_judge_env() -> None:
    """Use only caller-provided judge credentials; fail before an unconfigured request."""
    base = os.environ.get("GRM_JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("GRM_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not base or not key:
        raise ValueError("Set GRM_JUDGE_BASE_URL and GRM_JUDGE_API_KEY (or OPENAI_BASE_URL and OPENAI_API_KEY)")
    os.environ["OPENAI_BASE_URL"] = base
    os.environ["OPENAI_API_KEY"] = key


def _summarize_layer(layer_path: Path, metadata_path: Path) -> str:
    """OCI layer 交付物(默认协议)的 judge 摘要:成员清单 + 元数据;文本文件正文只取头部。

    layer 没有 diff 正文,judge 看到的是"改了哪些路径、删了什么(whiteout)、新增了
    什么二进制",足以判卫生与可迁移性;源码级审阅靠轨迹。
    """
    import tarfile

    metadata = json.loads(metadata_path.read_text())
    listed: list[str] = []
    body = b""
    with tarfile.open(layer_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in members[:_MAX_LISTED_FILES]:
            kind = "dir" if member.isdir() else "symlink" if member.issym() else "file"
            listed.append(f"  {member.name} ({kind}, {member.size} bytes)")
        for member in members:
            if member.isfile() and member.name.endswith((".py", ".sh")) and len(body) < _MAX_DIFF_BYTES:
                source = tar.extractfile(member)
                if source is not None:
                    body += f"\n### {member.name}\n".encode() + source.read(_MAX_DIFF_BYTES - len(body))
    omitted = (
        f"\n  ... and {len(members) - _MAX_LISTED_FILES} more entries\n"
        if len(members) > _MAX_LISTED_FILES
        else "\n"
    )
    return (
        f"# OCI image layer produced by the agent\n"
        f"# size: {layer_path.stat().st_size:,} bytes (diff_id {metadata.get('diff_id', '?')})\n"
        f"# base_commit: {metadata.get('base_commit', '?')}\n"
        f"# entries ({len(members)}); '.wh.' prefixed names delete base files:\n"
        + "\n".join(listed) + omitted
        + f"\n# text sources in layer (first {_MAX_DIFF_BYTES} bytes):\n"
        f"{body[:_MAX_DIFF_BYTES].decode('utf-8', 'replace')}\n"
    )


def _summarize_patch(patch_path: Path, metadata_path: Path) -> str:
    """把 patch 压成 judge 读得懂的摘要;输入两个产物路径,输出纯文本。

    保留判定卫生所需的关键证据(完整文件清单、总大小),diff 正文只取头部。
    """
    raw = patch_path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        return _summarize_layer(patch_path, metadata_path)
    names = [
        line.split(b" b/")[-1].decode("utf-8", "replace")
        for line in raw.splitlines()
        if line.startswith(b"diff --git")
    ]
    listed = "\n".join(f"  {n}" for n in names[:_MAX_LISTED_FILES])
    omitted = (
        f"\n  ... and {len(names) - _MAX_LISTED_FILES} more files\n"
        if len(names) > _MAX_LISTED_FILES
        else "\n"
    )
    metadata = json.loads(metadata_path.read_text())
    return (
        f"# Git patch produced by the agent\n"
        f"# size: {len(raw):,} bytes\n"
        f"# base_commit: {metadata.get('base_commit', '?')}\n"
        f"# changed files ({len(names)}):\n"
        f"{listed}{omitted}"
        f"\n# diff (first {_MAX_DIFF_BYTES} bytes):\n"
        f"{raw[:_MAX_DIFF_BYTES].decode('utf-8', 'replace')}\n"
    )


def _materialize_rubric(workdir: Path, *, dim: str, has_trajectory: bool) -> Path:
    """把静态 rubric 模板绑定到本 trial;返回可交给 rewardkit 发现的目录。

    每个 episode 只判一个维度(成功 → succ_quality,失败 → fail_blame,
    由 grade() 分派)。没有轨迹时剥掉 atif-trajectory 行,否则 judge 会拿到
    [not found] 噪声。
    """
    rubric_dir = workdir / "rubric"
    if rubric_dir.exists():
        shutil.rmtree(rubric_dir)  # 残留的另一维度会让重试 episode 判错方向
    (rubric_dir / dim).mkdir(parents=True)
    for template in sorted((_RUBRIC_ROOT / dim).glob("*.toml")):
        text = template.read_text().replace("{grm_dir}", str(workdir))
        if not has_trajectory:
            text = (
                "\n".join(
                    line
                    for line in text.splitlines()
                    if not line.startswith("atif-trajectory")
                )
                + "\n"
            )
        (rubric_dir / dim / template.name).write_text(text)
    return rubric_dir


def _collect_rewards(
    dim_scores: dict[str, float], details: dict[str, Any]
) -> dict[str, Any]:
    """把 RewardKit 的维度分与逐条明细折成 grm_* reward 键。

    judge 超时时 RewardKit 把受影响 criterion 记 0.0 并附 error
    (judges.py:427)—— 这种 0 不是判定,进了 reward 就等于端点抖动触发否决,
    正是本项目吃过亏的"判定口径与被判定对象不匹配"。带 error 的条目跳过,
    其所在维度的聚合分一并弃用。
    """
    values: dict[str, float] = {}
    tainted_dims: set[str] = set()
    for dim, detail in details.items():
        entries = detail if isinstance(detail, list) else [detail]
        for entry in entries:
            for score in entry.get("criteria", []):
                if score.get("error"):
                    tainted_dims.add(dim)
                    _LOGGER.warning(
                        "GRM criterion %s 出错,弃用该分:%s",
                        score.get("name"),
                        score["error"],
                    )
                    continue
                values[str(score["name"]).replace("-", "_")] = float(score["value"])
    if not values:
        raise RuntimeError("GRM judge 没有产出任何有效判定")
    rewards: dict[str, Any] = {f"grm_{name}": value for name, value in values.items()}
    rewards["grm_score"] = sum(values.values()) / len(values)
    if "succ_quality" in dim_scores and "succ_quality" not in tainted_dims:
        # required_pass 裁决:1.0 = 全部否决项通过。硬门以逐条键判定,
        # 这个键留作下游单键过滤用。
        rewards["grm_succ_quality"] = float(dim_scores["succ_quality"])
    return rewards


def _configure_judge_model_metadata(rubric_dir: Path) -> None:
    """为轨迹裁剪核实模型输入上限;输入实例化 rubric 目录,不改变 judge 身份。

    GRM_JUDGE_MODEL_METADATA 是以完整 LiteLLM 模型名为键的 JSON 对象,
    值沿用 register_model 的 metadata 格式,必须含正整数 max_input_tokens
    和匹配的 litellm_provider。只应配置 provider 契约或已核实部署上限;
    短请求/controls 成功不证明上下文上限,也不能借用别名模型的数值。
    显式配置优先于 LiteLLM catalog;未配置则保留 catalog 原值。缺少有效
    上限直接报错,不让 RewardKit 落到猜测的 200k 默认值。注册与凭据一样
    是进程级配置,不适合在同一进程并发切换不同 judge 部署。
    """
    import litellm

    configured = json.loads(os.environ.get("GRM_JUDGE_MODEL_METADATA", "{}"))
    if not isinstance(configured, dict):
        raise ValueError("GRM_JUDGE_MODEL_METADATA 必须是以完整模型名为键的 JSON 对象")
    models = set()
    for path in sorted(rubric_dir.rglob("*.toml")):
        judge = tomllib.loads(path.read_text()).get("judge", {})
        if judge.get("atif-trajectory"):
            models.add(os.environ.get("REWARDKIT_JUDGE") or judge["judge"])
    for model in sorted(models):
        if model in configured:
            metadata = configured[model]
            if (
                not isinstance(metadata, dict)
                or type(metadata.get("max_input_tokens")) is not int
                or metadata["max_input_tokens"] <= 0
            ):
                raise ValueError(
                    f"GRM_JUDGE_MODEL_METADATA[{model!r}].max_input_tokens "
                    "必须是已核实的正整数输入上限"
                )
            provider = litellm.get_llm_provider(model=model)[1]
            if metadata.get("litellm_provider") != provider:
                raise ValueError(
                    f"GRM_JUDGE_MODEL_METADATA[{model!r}].litellm_provider "
                    f"必须与模型路由一致: {provider!r}"
                )
            # 官方 API 注册精确模型名;不改 model、不伪造价格或其他能力。
            litellm.register_model({model: metadata})
        try:
            info = litellm.get_model_info(model)
        except Exception as exc:  # LiteLLM 的未映射模型异常就是基类 Exception。
            raise RuntimeError(
                f"GRM judge {model!r} metadata 不可用;请通过 "
                "GRM_JUDGE_MODEL_METADATA 配置可信的 max_input_tokens 和 "
                "litellm_provider;不能从 controls 成功推断上下文上限"
            ) from exc
        if type(info.get("max_input_tokens")) is not int or info["max_input_tokens"] <= 0:
            raise RuntimeError(
                f"GRM judge {model!r} 缺少有效 max_input_tokens;请通过 "
                "GRM_JUDGE_MODEL_METADATA 配置可信输入上限,禁止猜测默认值"
            )


async def grade(
    patch_path: Path,
    metadata_path: Path,
    trajectory_path: Path | None = None,
    *,
    failed: bool = False,
    output_dir: Path | None = None,
) -> dict[str, float]:
    """对一次 episode 的产物与过程打 GRM 分;返回可并入 reward.json 的键值。

    维度按 episode 结局分派、互斥:成功 → succ_quality(产出物质量),
    失败 → fail_blame(失败归因);失败且无轨迹则整体跳过。返回逐条
    `grm_<criterion>`、均值 `grm_score`;成功 episode 另有否决裁决
    `grm_succ_quality`。`output_dir`(通常是 trial 的 verifier 目录)下落
    三样:`grm/` 工作目录(摘要 + 实例化的 rubric)、`grm-reward.json`
    (维度分)与 `reward-details.json`(逐条 reasoning,harbor view 原生渲染)。
    """
    has_trajectory = trajectory_path is not None and trajectory_path.is_file()
    if failed and not has_trajectory:
        # 没有轨迹就无法归因;宁可不出分,也不要凭 patch 猜。
        _LOGGER.warning("GRM: 失败 episode 无轨迹,跳过判定")
        return {}

    import litellm
    from verify.grading.rewardkit_runtime import run as rewardkit_run

    # RewardKit 的 LLMJudge 无条件传 reasoning_effort(judges.py:421),litellm
    # 对不支持该参数的模型会抛 UnsupportedParamsError;drop_params 静默丢弃,
    # 避免侵入式改 harbor。
    litellm.drop_params = True
    _export_judge_env()

    out_dir = output_dir if output_dir is not None else patch_path.parent
    workdir = out_dir / "grm"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "patch-summary.txt").write_text(
        _summarize_patch(patch_path, metadata_path)
    )
    if has_trajectory:
        assert trajectory_path is not None
        shutil.copyfile(trajectory_path, workdir / "trajectory.json")
    rubric_dir = _materialize_rubric(
        workdir,
        dim="fail_blame" if failed else "succ_quality",
        has_trajectory=has_trajectory,
    )
    if has_trajectory:
        _configure_judge_model_metadata(rubric_dir)

    # rewardkit.run 内部 asyncio.run,不能在事件循环里直接调,丢线程执行。
    dim_scores = await asyncio.to_thread(
        rewardkit_run,
        rubric_dir,
        workspace=workdir,
        output=out_dir / "grm-reward.json",
    )
    details = json.loads((out_dir / "reward-details.json").read_text())
    return _collect_rewards(dim_scores, details)


async def grade_best_effort(
    patch_path: Path,
    metadata_path: Path,
    trajectory_path: Path | None = None,
    *,
    failed: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """`grade()` 的宽容包装:GRM 失败不得影响主评测结论。

    GRM 依赖外部 LLM 端点,而主 reward 不依赖。judge 挂了就返回 `grm_error`,
    让主流程照常出分 —— 这是**旁挂信号**,不是评测的必经门。
    置 `GRM_DISABLED=1` 可整体跳过(例如离线环境没有 rewardkit)。
    """
    if os.environ.get("GRM_DISABLED") == "1":
        return {}
    try:
        return await grade(
            patch_path,
            metadata_path,
            trajectory_path,
            failed=failed,
            output_dir=output_dir,
        )
    except Exception as exc:  # noqa: BLE001 旁挂信号:判定失败不阻断主评测
        _LOGGER.warning("GRM 判定失败(不影响主 reward): %s", exc, exc_info=True)
        return {"grm_error": str(exc)[:500]}


@functools.cache
def veto_criteria() -> tuple[str, ...]:
    """否决项的 reward 键;单一事实来源是 succ_quality rubric 里非 optional 的项。"""
    spec = tomllib.loads(
        (_RUBRIC_ROOT / "succ_quality" / "succ_quality.toml").read_text()
    )
    return tuple(
        "grm_" + str(criterion["name"]).replace("-", "_")
        for criterion in spec["criterion"]
        if not criterion.get("optional", False)
    )


def apply_gate(rewards: dict[str, Any]) -> dict[str, Any]:
    """GRM 硬门:否决项不过则 reward 一票否决置零;输入输出同为 rewards 字典。

    hacking(夹带构建垃圾 / 对着评测 workload 特化调参)默认一票否决,
    没有开关(2026-09-02 用户裁决)。置零前的原值记为
    `reward_before_grm_gate` 留审计 —— 0902 那次 negate 误判(oracle 标准
    答案 pi05 的 1.0474 被打成 0,见 succ_quality.toml 头注)正是靠原值定位的。

    否决集来自 rubric(veto_criteria = 非 optional 的 criterion),当前仅
    `transferable-infra-optimization` 一条。门只在 GRM 真出了分时生效:
    judge 挂了(`grm_error`)或该键缺失时不置零(_collect_rewards 也已丢弃
    超时污染的 0 分),否则外部端点抖动会把健康 patch 判成 0。
    """
    failed_keys = [k for k in veto_criteria() if rewards.get(k) == 0.0]
    if not failed_keys:
        return rewards
    return {
        **rewards,
        "reward": 0.0,
        "reward_before_grm_gate": rewards.get("reward", 0.0),
        "grm_gate_failed": ",".join(failed_keys),
    }

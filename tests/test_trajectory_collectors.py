"""OpenCodeTrajectoryCollector:事件流到 ATIF 的实时转换与半写容忍。

参数：无。夹具是 v19 真实 trace 的最小化事件(step_start/text/tool_use/
step_finish),外加一条被截断的尾行。

示例：
    python3 -m unittest tests.test_trajectory_collectors
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from harbor.agents.installed.opencode import OpenCode
from harbor.models.agent.context import AgentContext
from harbor_envs.abstract_mlplat.trajectory import (
    OpenCodeTrajectoryCollector,
    reconcile_trial_trajectory,
)
from tests.test_agent_trajectory_collectors import LocalSyncEnvironment, _jsonl

_EVENTS = [
    {
        "type": "step_start",
        "timestamp": 1787673716274,
        "sessionID": "ses_x",
        "part": {"id": "prt_1", "messageID": "msg_1", "type": "step-start"},
    },
    {
        "type": "text",
        "timestamp": 1787673716300,
        "sessionID": "ses_x",
        "part": {"id": "prt_2", "type": "text", "text": "I'll check the environment."},
    },
    {
        "type": "tool_use",
        "timestamp": 1787673717000,
        "sessionID": "ses_x",
        "part": {
            "id": "prt_3",
            "type": "tool",
            "tool": "bash",
            "callID": "call_1",
            "state": {
                "status": "completed",
                "input": {"command": "npu-smi info"},
                "output": "8 devices",
                "time": {"start": 1787673716500, "end": 1787673716900},
            },
            "title": "npu-smi info",
        },
    },
    {
        "type": "step_finish",
        "timestamp": 1787673718000,
        "sessionID": "ses_x",
        "part": {
            "id": "prt_4",
            "type": "step-finish",
            "reason": "tool-calls",
            "tokens": {
                "total": 120,
                "input": 100,
                "output": 20,
                "reasoning": 0,
                "cache": {"write": 0, "read": 40},
            },
            "cost": 0,
        },
    },
]


def _stream(*, truncate_tail: bool) -> bytes:
    """构造事件流字节;输入是否附截断尾行,输出 opencode.txt 内容。"""
    lines = [json.dumps(event) for event in _EVENTS]
    if truncate_tail:
        lines.append(json.dumps(_EVENTS[1])[:40])
    return ("\n".join(lines) + "\n").encode()


class OpenCodeTrajectoryCollectorTest(unittest.TestCase):
    def test_converts_events_to_atif(self) -> None:
        artifacts = OpenCodeTrajectoryCollector().artifacts(
            _stream(truncate_tail=False), session_id=""
        )
        self.assertIn("opencode.txt", artifacts)
        trajectory = json.loads(artifacts["trajectory.json"])
        self.assertEqual(trajectory["session_id"], "ses_x")
        self.assertTrue(trajectory["schema_version"].startswith("ATIF"))
        steps = trajectory["steps"]
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["source"], "agent")
        self.assertEqual(steps[0]["tool_calls"][0]["function_name"], "bash")

    def test_tolerates_truncated_tail_line(self) -> None:
        # 远端文件正在被 opencode 追加,拉取时机可能落在半行上。
        artifacts = OpenCodeTrajectoryCollector().artifacts(
            _stream(truncate_tail=True), session_id=""
        )
        trajectory = json.loads(artifacts["trajectory.json"])
        self.assertEqual(len(trajectory["steps"]), 1)

    def test_stream_without_steps_still_mirrors_raw(self) -> None:
        artifacts = OpenCodeTrajectoryCollector().artifacts(
            b"plain text banner\n", session_id=""
        )
        self.assertEqual(artifacts, {"opencode.txt": b"plain text banner\n"})


class OpenCodeHistoryRegressionTest(unittest.IsolatedAsyncioTestCase):
    """全源账本不得改变 OpenCode completed-step/stdout/真实 instruction 契约。"""

    async def test_final_postconvert_keeps_real_instruction_and_native_events(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = LocalSyncEnvironment(Path(directory))
            raw = _stream(truncate_tail=False)
            environment.remote_file("/logs/agent/opencode.txt", raw)
            await environment.finalize_trajectory_sync(strict=True)
            agent_dir = environment.trial_paths.agent_dir
            self.assertEqual((agent_dir / "opencode.txt").read_bytes(), raw)
            live = environment.trajectory()
            self.assertEqual(len(live.steps), 1)
            self.assertEqual(live.final_metrics.total_prompt_tokens, 140)
            self.assertEqual(live.final_metrics.total_cached_tokens, 40)
            self.assertEqual(live.final_metrics.total_cost_usd, 0)
            upstream = OpenCode(agent_dir, model_name="test-model")
            upstream._instruction = "The real instruction supplied by runtime.\n"
            upstream.populate_context_post_run(AgentContext())
            self.assertEqual(len(environment.trajectory().steps), 2)
            reconciled = reconcile_trial_trajectory(agent_dir, model_name="test-model")
            self.assertEqual(len(reconciled.steps), 2)
            self.assertEqual(reconciled.steps[0].source, "user")
            self.assertEqual(reconciled.steps[0].message, upstream._instruction)
            self.assertEqual(
                reconciled.steps[0].extra["input_provenance"]["source_key"],
                "atif:trajectory.json",
            )
            self.assertEqual(reconciled.steps[1].tool_calls[0].function_name, "bash")
            self.assertEqual(
                reconciled.steps[1].observation.results[0].content, "8 devices"
            )
            self.assertEqual(reconciled.steps[1].model_name, "test-model")
            self.assertEqual(reconciled.final_metrics.total_steps, 2)
            self.assertEqual(
                reconcile_trial_trajectory(agent_dir, model_name="test-model"),
                reconciled,
            )

    async def test_unfinished_step_keeps_raw_without_fake_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = LocalSyncEnvironment(Path(directory))
            raw = _jsonl(_EVENTS[:-1])
            environment.remote_file("/logs/agent/opencode.txt", raw)
            await environment.finalize_trajectory_sync(strict=True)
            self.assertEqual(
                (environment.trial_paths.agent_dir / "opencode.txt").read_bytes(), raw
            )
            self.assertIsNone(
                reconcile_trial_trajectory(environment.trial_paths.agent_dir)
            )
            self.assertTrue(environment.manifest()["pending_sources"])

    async def test_mixed_known_missing_and_zero_usage_not_partial_sum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = LocalSyncEnvironment(Path(directory))
            first = copy.deepcopy(_EVENTS)
            first[-1]["part"]["tokens"].update(input=0, output=0)
            first[-1]["part"]["tokens"]["cache"]["read"] = 0
            environment.remote_file("/logs/agent/opencode.txt", _jsonl(first))
            await environment.finalize_trajectory_sync(strict=True)
            metrics = environment.trajectory().final_metrics
            self.assertEqual(
                (
                    metrics.total_prompt_tokens,
                    metrics.total_completion_tokens,
                    metrics.total_cached_tokens,
                    metrics.total_cost_usd,
                ),
                (0, 0, 0, 0),
            )
            later = copy.deepcopy(_EVENTS)
            later[-1]["part"]["tokens"].pop("cache")
            later[-1]["part"].pop("cost")
            environment.remote_file("/logs/agent/opencode.txt", _jsonl(first + later))
            await environment.finalize_trajectory_sync(strict=True)
            metrics = environment.trajectory().final_metrics
            self.assertEqual(metrics.total_completion_tokens, 20)
            self.assertIsNone(metrics.total_prompt_tokens)
            self.assertIsNone(metrics.total_cached_tokens)
            self.assertIsNone(metrics.total_cost_usd)


if __name__ == "__main__":
    unittest.main()

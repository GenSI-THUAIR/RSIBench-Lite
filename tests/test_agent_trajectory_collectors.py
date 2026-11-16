"""Codex / ClaudeCode 的实时轨迹 collector:session JSONL 到 ATIF 的转换。

参数:无。夹具是两家真实 session 的最小化片段,外加退化输入(空文件、
半写尾行、纯文本)——远端文件正在被 agent 追加,拉取时机可能落在任意位置。

示例:
    python3 -m unittest tests.test_agent_trajectory_collectors
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import shlex
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from harbor.agents.installed.codex import Codex
from harbor.environments.base import ExecResult
from harbor.models.trajectories.trajectory import Trajectory
from harbor_envs.abstract_mlplat.trajectory import (
    ClaudeCodeTrajectoryCollector,
    CodexTrajectoryCollector,
    TrajectorySyncMixin,
    reconcile_trial_trajectory,
)

# Codex rollout:session_meta + 一次 function_call/输出(codex.py 的 ATIF 源)。
_CODEX_LINES = [
    {
        "timestamp": "2026-09-01T14:43:11.161Z",
        "type": "session_meta",
        "payload": {
            "session_id": "01a05d6c-908b-77d0-9708-0dbb47066987",
            "id": "01a05d6c-908b-77d0-9708-0dbb47066987",
            "timestamp": "2026-09-01T14:43:11.161Z",
            "cwd": "/sgl-workspace/sglang",
            "originator": "codex_cli_rs",
            "cli_version": "0.147.0",
            "instructions": None,
        },
    },
    {
        "timestamp": "2026-09-01T14:43:20.000Z",
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": "shell",
            "arguments": '{"command":["bash","-lc","ls"]}',
            "call_id": "call_1",
        },
    },
    {
        "timestamp": "2026-09-01T14:43:21.000Z",
        "type": "response_item",
        "payload": {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "launch.sh\n",
        },
    },
]

# Claude transcript:user + assistant(带 tool_use)+ tool_result。
_CLAUDE_LINES = [
    {
        "type": "user",
        "uuid": "u-1",
        "parentUuid": None,
        "sessionId": "20c7c049-c32b-4f1b-89ea-0c5dfdb88cad",
        "timestamp": "2026-09-02T08:30:00.000Z",
        "cwd": "/workspace/deploy",
        "message": {"role": "user", "content": "Optimize the server."},
    },
    {
        "type": "assistant",
        "uuid": "a-1",
        "parentUuid": "u-1",
        "sessionId": "20c7c049-c32b-4f1b-89ea-0c5dfdb88cad",
        "timestamp": "2026-09-02T08:30:05.000Z",
        "message": {
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [
                {"type": "text", "text": "Checking the launch script."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Bash",
                    "input": {"command": "cat launch.sh"},
                },
            ],
        },
    },
    {
        "type": "user",
        "uuid": "u-2",
        "parentUuid": "a-1",
        "sessionId": "20c7c049-c32b-4f1b-89ea-0c5dfdb88cad",
        "timestamp": "2026-09-02T08:30:06.000Z",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "#!/usr/bin/env bash\n",
                }
            ],
        },
    },
]


def _jsonl(lines: list[dict], truncate_tail: bool = False) -> bytes:
    """拼 JSONL 字节;truncate_tail 追加一条半写行模拟 agent 正在落盘。"""
    text = "\n".join(json.dumps(line) for line in lines) + "\n"
    if truncate_tail:
        text += json.dumps(lines[-1])[:40]
    return text.encode()


class CodexTrajectoryCollectorTest(unittest.TestCase):
    def test_converts_rollout_to_atif(self) -> None:
        artifacts = CodexTrajectoryCollector().artifacts(
            _jsonl(_CODEX_LINES), session_id=""
        )
        # 原生 rollout 必须保留:ATIF 是派生物,原文才是可回溯的事实。
        self.assertIn("codex-session.jsonl", artifacts)
        trajectory = json.loads(artifacts["trajectory.json"])
        self.assertTrue(trajectory["schema_version"].startswith("ATIF"))
        self.assertGreaterEqual(len(trajectory["steps"]), 1)

    def test_reads_codex_home_not_logs_agent(self) -> None:
        # 上游只在 run 收尾的 finally 里才把 sessions 拷进 /logs/agent,随后
        # rm -rf "$CODEX_HOME"(codex.py:1955-1982)。跑到一半时只有 CODEX_HOME
        # 里有活的 rollout,读 /logs/agent 会永远同步不到东西。
        pattern = CodexTrajectoryCollector.remote_path
        self.assertTrue(pattern.startswith("/tmp/codex-home/sessions/"))
        self.assertNotIn("/logs/agent", pattern)
        self.assertIn("rollout-", pattern)


class ClaudeCodeTrajectoryCollectorTest(unittest.TestCase):
    def test_converts_transcript_to_atif(self) -> None:
        artifacts = ClaudeCodeTrajectoryCollector().artifacts(
            _jsonl(_CLAUDE_LINES), session_id=""
        )
        self.assertIn("claude-code-session.jsonl", artifacts)
        trajectory = json.loads(artifacts["trajectory.json"])
        self.assertTrue(trajectory["schema_version"].startswith("ATIF"))
        self.assertGreaterEqual(len(trajectory["steps"]), 1)

    def test_reads_claude_config_dir_sessions(self) -> None:
        # 上游把 CLAUDE_CONFIG_DIR 设为 /logs/agent/sessions
        # (claude_code.py:1696),CLI 在其下按 cwd slug 建 projects/<slug>/。
        pattern = ClaudeCodeTrajectoryCollector.remote_path
        self.assertTrue(pattern.startswith("/logs/agent/sessions/projects/"))
        self.assertTrue(pattern.endswith(".jsonl"))


class DegradedInputTest(unittest.TestCase):
    """半写/空/垃圾输入只留原文,绝不抛异常也绝不产出假 ATIF。"""

    def test_partial_and_garbage_inputs_keep_native_only(self) -> None:
        cases = (
            (CodexTrajectoryCollector(), "codex-session.jsonl", _CODEX_LINES),
            (
                ClaudeCodeTrajectoryCollector(),
                "claude-code-session.jsonl",
                _CLAUDE_LINES,
            ),
        )
        for collector, native, lines in cases:
            for label, raw in (
                ("empty", b""),
                ("half-written", json.dumps(lines[0]).encode()[:30]),
                ("not-json", b"plain text banner\n"),
            ):
                with self.subTest(collector=type(collector).__name__, case=label):
                    artifacts = collector.artifacts(raw, session_id="")
                    # 转不出轨迹时只镜像原文,等下一轮签名变化后重试。
                    self.assertEqual(artifacts, {native: raw})

    def test_truncated_tail_still_converts(self) -> None:
        # 完整事件已够转出轨迹,尾部半行不应让整次同步作废。
        artifacts = CodexTrajectoryCollector().artifacts(
            _jsonl(_CODEX_LINES, truncate_tail=True), session_id=""
        )
        self.assertIn("trajectory.json", artifacts)


class CollectorRegistrationTest(unittest.TestCase):
    def test_both_collectors_registered_and_atif_last(self) -> None:
        collectors = TrajectorySyncMixin()._trajectory_collectors()
        names = [type(c).__name__ for c in collectors]
        self.assertIn("CodexTrajectoryCollector", names)
        self.assertIn("ClaudeCodeTrajectoryCollector", names)
        # ATIF 仍注册，但 canonical 由原生全历史 reconcile，不按注册顺序覆盖。
        self.assertIn("AtifTrajectoryCollector", names)

    def test_remote_paths_are_unique(self) -> None:
        # 签名缓存以 remote_path 为键,重复会互相顶掉。
        paths = [c.remote_path for c in TrajectorySyncMixin()._trajectory_collectors()]
        self.assertEqual(len(paths), len(set(paths)))


class RemotePathResolutionTest(unittest.IsolatedAsyncioTestCase):
    """发现接口返回全部路径，不能再按 mtime 截出最新一个。"""

    @staticmethod
    def _mixin(stdout: str) -> TrajectorySyncMixin:
        """装返回给定 NUL 路径列表的假 exec。"""
        mixin = TrajectorySyncMixin()
        mixin._exec_cmd_platform = AsyncMock(
            return_value=ExecResult(stdout=stdout, return_code=0)
        )
        return mixin

    async def test_glob_resolves_every_match(self) -> None:
        mixin = self._mixin("/sessions/B.jsonl\0/sessions/A.jsonl\0")
        self.assertEqual(
            await mixin._resolve_remote_paths("/sessions/*.jsonl"),
            ["/sessions/A.jsonl", "/sessions/B.jsonl"],
        )
        command = mixin._exec_cmd_platform.await_args.args[0]
        self.assertNotIn("head", command)
        self.assertNotIn("ls -1t", command)

    async def test_no_match_returns_empty(self) -> None:
        for stdout in ("", "\n", "\0", "\0\n"):
            with self.subTest(stdout=repr(stdout)):
                self.assertEqual(
                    await self._mixin(stdout)._resolve_remote_paths("/nope/*.jsonl"), []
                )

    async def test_transport_newline_does_not_change_native_paths(self) -> None:
        paths = [
            "/sessions/space name.jsonl ",
            "/sessions/tab\t.jsonl",
            "/sessions/newline.jsonl\n",
        ]
        for suffix in ("", "\n"):
            with self.subTest(suffix=repr(suffix)):
                stdout = "\0".join(paths) + "\0" + suffix
                self.assertEqual(
                    await self._mixin(stdout)._resolve_remote_paths("/sessions/*"),
                    sorted(paths),
                )

    async def test_malformed_discovery_response_is_not_silently_stripped(self) -> None:
        for stdout in (
            "\n\n",
            "transport chatter\n",
            "/sessions/no-terminator.jsonl\n",
            "/sessions/a.jsonl\0junk\n",
            "/sessions/a.jsonl\0\0\n",
            "\0/sessions/a.jsonl\0\n",
        ):
            with (
                self.subTest(stdout=repr(stdout)),
                self.assertRaisesRegex(RuntimeError, "Invalid .*trajectory"),
            ):
                await self._mixin(stdout)._resolve_remote_paths("/sessions/*")

    async def test_fixed_path_still_works(self) -> None:
        self.assertEqual(
            await self._mixin("/logs/agent/opencode.txt\0")._resolve_remote_paths(
                "/logs/agent/opencode.txt"
            ),
            ["/logs/agent/opencode.txt"],
        )




def _codex_lines(index: int, *, usage: dict | None = None) -> list[dict]:
    """不同原生 session 使用不同真实 call ID；文本相似不应被猜测去重。"""
    lines = copy.deepcopy(_CODEX_LINES)
    sid = f"native-session-{index}"
    lines[0]["payload"].update(id=sid, session_id=sid)
    for event in lines:
        event["timestamp"] = event["timestamp"].replace("09-01", f"09-{index + 1:02d}")
    lines[0]["payload"]["timestamp"] = lines[0]["timestamp"]
    lines[1]["payload"]["call_id"] = f"call-{index}"
    lines[2]["payload"]["call_id"] = f"call-{index}"
    lines[2]["payload"]["output"] = f"evidence-session-{index}"
    lines.insert(
        1,
        {
            "timestamp": lines[0]["timestamp"],
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"instruction-session-{index}"},
                ],
            },
        },
    )
    if usage is not None:
        lines.append(
            {
                "timestamp": lines[-1]["timestamp"],
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": usage,
                        "last_token_usage": usage,
                        "total_cost": 0,
                    },
                },
            }
        )
    return lines


class LocalSyncEnvironment(TrajectorySyncMixin):
    """只在临时目录执行真实 shell discovery/hash/base64，不连接远端或 live artifact。"""

    def __init__(self, root: Path) -> None:
        """root 划分伪 pod 和本地 trial，便于验证实际 shell 命令和离线下载。"""
        self.root = root
        self.pod = root / "pod"
        self.trial_paths = SimpleNamespace(agent_dir=root / "trial" / "agent")
        self.session_id = "stable-trial-id"
        self.logger = MagicMock(spec=logging.Logger)
        self._trajectory_sync_task = None
        self._trajectory_remote_signatures = {}
        self.commands: list[str] = []
        self.fail_transfers: set[str] = set()

    def remote_file(self, remote: str, content: bytes) -> Path:
        """写测试自己的临时文件，返回其宿主路径。"""
        path = self.pod / remote.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    async def _exec_cmd_platform(self, command: str, **kwargs: object) -> ExecResult:
        """映射绝对路径后运行本地短命令；产物绝不离开临时目录。"""
        self.commands.append(command)
        if (
            command.startswith("base64 -w0 ")
            and shlex.split(command)[2] in self.fail_transfers
        ):
            return ExecResult(return_code=1, stderr="injected transfer failure")
        mapped = command.replace("/tmp/codex-home", str(self.pod / "tmp/codex-home"))
        mapped = mapped.replace("/logs/agent", str(self.pod / "logs/agent"))
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            mapped,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return ExecResult(
            # 与实际 Merlin sentinel 回归测试保持一致，不能再用干净 stdout
            # 掩盖空 glob 被误判成 "\\n" 路径的协议缺陷。
            stdout=stdout.decode().replace(str(self.pod), "") + "\n",
            stderr=stderr.decode(),
            return_code=process.returncode,
        )

    def manifest(self) -> dict:
        """读取受管源账本用于断言，非实现捷径。"""
        return json.loads(
            (self.trial_paths.agent_dir / "trajectory-raw/manifest.json").read_bytes()
        )

    def trajectory(self) -> Trajectory:
        """用实际 Harbor 模型校验磁盘 canonical。"""
        return Trajectory.model_validate_json(
            (self.trial_paths.agent_dir / "trajectory.json").read_bytes()
        )


def _codex_remote(index: int) -> str:
    """构造含原生日期的测试 rollout 路径。"""
    return f"/tmp/codex-home/sessions/2026/09/{index + 1:02d}/rollout-{index}.jsonl"


class FullTrajectoryHistoryTest(unittest.IsolatedAsyncioTestCase):
    """通过真实上游 converter 验证全源归档、累计投影和 final 覆盖修复。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.environment = LocalSyncEnvironment(Path(self.temporary.name))

    def _put(self, index: int, *, usage: dict | None = None) -> bytes:
        """放入一个独立 Codex 原生会话并返回期望字节。"""
        raw = _jsonl(_codex_lines(index, usage=usage))
        self.environment.remote_file(_codex_remote(index), raw)
        return raw

    async def test_five_sources_raw_before_conversion_and_idempotent_history(
        self,
    ) -> None:
        raws = [
            self._put(
                index,
                usage={
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cached_input_tokens": 40,
                },
            )
            for index in range(5)
        ]
        converter = Codex._convert_events_to_trajectory

        def convert(agent: Codex, directory: Path) -> Trajectory | None:
            # 连第一个 converter 被调用前，所有五源都必须已经原子留存。
            sources = self.environment.manifest()["sources"]
            self.assertEqual(len(sources), 5)
            for entry in sources.values():
                path = (
                    self.environment.trial_paths.agent_dir
                    / "trajectory-raw"
                    / entry["raw_path"]
                )
                self.assertIn(path.read_bytes(), raws)
            return converter(agent, directory)

        with patch.object(Codex, "_convert_events_to_trajectory", convert):
            await self.environment.finalize_trajectory_sync(strict=True)
        first = self.environment.trajectory()
        self.assertEqual(first.session_id, "stable-trial-id")
        self.assertEqual([step.step_id for step in first.steps], list(range(1, 11)))
        self.assertEqual(
            [step.message for step in first.steps if step.source == "user"],
            [f"instruction-session-{index}" for index in range(5)],
        )
        self.assertIsNone(first.subagent_trajectories)
        self.assertEqual(
            first.final_metrics.total_prompt_tokens, 500
        )  # cache 已含在 input
        self.assertEqual(first.final_metrics.total_cached_tokens, 200)
        self.assertEqual(first.final_metrics.total_completion_tokens, 100)
        self.assertEqual(first.final_metrics.total_cost_usd, 0)
        self.assertEqual(first.final_metrics.total_steps, 10)
        for step in first.steps:
            provenance = step.extra["trajectory_provenance"]
            self.assertEqual(len(provenance["native_session_ids"]), 1)
            self.assertEqual(len(provenance["source_keys"]), 1)
            self.assertIn(provenance["native_step_id"], [1, 2])
        downloads = sum(
            command.startswith("base64 -w0") for command in self.environment.commands
        )
        await self.environment.finalize_trajectory_sync(strict=True)
        self.assertEqual(self.environment.trajectory(), first)
        self.assertEqual(len(self.environment.manifest()["sources"]), 5)
        self.assertEqual(
            sum(
                command.startswith("base64 -w0")
                for command in self.environment.commands
            ),
            downloads,
        )

    async def test_same_size_mtime_new_source_and_in_place_rewrite(self) -> None:
        first = self._put(0)
        file0 = self.environment.pod / _codex_remote(0).lstrip("/")
        os.utime(file0, (100, 100))
        await self.environment.finalize_trajectory_sync(strict=True)
        second = self._put(1)
        file1 = self.environment.pod / _codex_remote(1).lstrip("/")
        os.utime(file1, (100, 100))
        self.assertEqual(len(first), len(second))
        await self.environment.finalize_trajectory_sync(strict=True)
        self.assertEqual(len(self.environment.trajectory().steps), 4)
        self.assertIn(_codex_remote(0), self.environment._trajectory_remote_signatures)
        self.assertIn(_codex_remote(1), self.environment._trajectory_remote_signatures)
        replacement = first.replace(b"evidence-session-0", b"replaced-session-0")
        self.assertEqual(len(replacement), len(first))
        file0.write_bytes(replacement)
        os.utime(file0, (100, 100))
        with self.assertRaisesRegex(ValueError, "non_append_rewrite"):
            await self.environment.finalize_trajectory_sync(strict=True)
        entry = self.environment.manifest()["sources"][
            "codex:2026/09/01/rollout-0.jsonl"
        ]
        raw_root = self.environment.trial_paths.agent_dir / "trajectory-raw"
        self.assertEqual((raw_root / entry["raw_path"]).read_bytes(), replacement)
        self.assertEqual(
            (raw_root / entry["issues"][0]["previous_raw"]).read_bytes(), first
        )
        self.assertIsNone(
            self.environment.trajectory().final_metrics.total_prompt_tokens
        )

    async def test_live_final_roots_and_downloads_share_one_identity(self) -> None:
        raw = self._put(0)
        await self.environment.finalize_trajectory_sync(strict=True)
        first = self.environment.trajectory()
        final = _codex_remote(0).replace(
            "/tmp/codex-home/sessions/", "/logs/agent/sessions/"
        )
        self.environment.remote_file(final, raw)
        await self.environment.finalize_trajectory_sync(strict=True)
        self.assertEqual(self.environment.trajectory(), first)
        self.assertEqual(len(self.environment.manifest()["sources"]), 1)
        shutil.copytree(
            self.environment.pod / "logs/agent/sessions",
            self.environment.trial_paths.agent_dir / "sessions",
        )
        self.assertEqual(
            reconcile_trial_trajectory(self.environment.trial_paths.agent_dir), first
        )
        self.assertEqual(len(self.environment.manifest()["sources"]), 1)
        entry = next(iter(self.environment.manifest()["sources"].values()))
        self.assertEqual(len(entry["origins"]), 3)

    async def test_same_session_prefix_snapshots_replace_not_append(self) -> None:
        lines = _codex_lines(0)
        self.environment.remote_file(_codex_remote(0), _jsonl(lines[:2]))
        self.environment.remote_file(_codex_remote(1), _jsonl(lines))
        await self.environment.finalize_trajectory_sync(strict=True)
        trajectory = self.environment.trajectory()
        self.assertEqual(len(trajectory.steps), 2)
        self.assertEqual(len(trajectory.extra["trajectory_contributions"]), 1)
        self.assertEqual(
            len(trajectory.extra["trajectory_contributions"][0]["source_keys"]), 2
        )
        await self.environment.finalize_trajectory_sync(strict=True)
        self.assertEqual(self.environment.trajectory(), trajectory)

    async def test_converter_error_retains_all_raw_and_publishes_explicit_partial(
        self,
    ) -> None:
        for index in range(5):
            self._put(index)
        converter = Codex._convert_events_to_trajectory

        def convert(agent: Codex, directory: Path) -> Trajectory | None:
            if b"native-session-2" in (directory / "session.jsonl").read_bytes():
                raise RuntimeError("injected converter failure")
            return converter(agent, directory)

        with (
            patch.object(Codex, "_convert_events_to_trajectory", convert),
            self.assertRaisesRegex(ValueError, "injected converter failure"),
        ):
            await self.environment.finalize_trajectory_sync(strict=True)
        self.assertEqual(len(self.environment.manifest()["sources"]), 5)
        partial = self.environment.trajectory()
        self.assertEqual(len(partial.steps), 8)
        self.assertTrue(partial.extra["collection_incomplete"])
        self.assertIsNone(partial.final_metrics.total_prompt_tokens)
        repaired = reconcile_trial_trajectory(self.environment.trial_paths.agent_dir)
        self.assertFalse(repaired.extra["collection_incomplete"])
        self.assertEqual(len(repaired.steps), 10)

    async def test_transfer_failure_strict_and_retry_keep_other_sources(self) -> None:
        self._put(0)
        self._put(1)
        self.environment.fail_transfers.add(_codex_remote(0))
        with self.assertRaisesRegex(RuntimeError, "transfer failed"):
            await self.environment.finalize_trajectory_sync(strict=True)
        partial = self.environment.trajectory()
        self.assertTrue(partial.extra["collection_incomplete"])
        self.assertEqual(len(partial.steps), 2)
        self.environment.fail_transfers.clear()
        await self.environment.finalize_trajectory_sync(strict=True)
        self.assertFalse(self.environment.trajectory().extra["collection_incomplete"])
        self.assertEqual(len(self.environment.trajectory().steps), 4)

    async def test_pending_and_partial_jsonl_are_not_converter_failures(self) -> None:
        self.environment.remote_file(_codex_remote(0), _jsonl(_CODEX_LINES[:2]))
        await self.environment.finalize_trajectory_sync(strict=True)
        self.assertIsNone(
            reconcile_trial_trajectory(self.environment.trial_paths.agent_dir)
        )
        self.assertTrue(self.environment.manifest()["pending_sources"])
        self._put(1)
        await self.environment.finalize_trajectory_sync(strict=True)
        self.assertEqual(len(self.environment.trajectory().steps), 2)
        self.assertFalse(self.environment.trajectory().extra["collection_incomplete"])
        self.environment.remote_file(
            _codex_remote(2), _jsonl(_codex_lines(2), truncate_tail=True)
        )
        await self.environment.finalize_trajectory_sync(strict=True)
        self.assertEqual(len(self.environment.trajectory().steps), 4)
        self.assertFalse(self.environment.trajectory().extra["collection_incomplete"])
        self.assertTrue(self.environment.trajectory().extra["collection_pending"])

    async def test_unknown_and_explicit_zero_metrics_not_fabricated(self) -> None:
        self._put(
            0, usage={"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
        )
        await self.environment.finalize_trajectory_sync(strict=True)
        metrics = self.environment.trajectory().final_metrics
        self.assertEqual(
            (
                metrics.total_prompt_tokens,
                metrics.total_completion_tokens,
                metrics.total_cached_tokens,
                metrics.total_cost_usd,
            ),
            (0, 0, 0, 0),
        )
        self._put(1, usage={"input_tokens": 100, "output_tokens": 20})
        await self.environment.finalize_trajectory_sync(strict=True)
        metrics = self.environment.trajectory().final_metrics
        self.assertEqual(metrics.total_prompt_tokens, 100)
        self.assertEqual(metrics.total_completion_tokens, 20)
        self.assertIsNone(metrics.total_cached_tokens)
        self._put(2)
        await self.environment.finalize_trajectory_sync(strict=True)
        metrics = self.environment.trajectory().final_metrics
        self.assertIsNone(metrics.total_prompt_tokens)
        self.assertIsNone(metrics.total_completion_tokens)
        self.assertIsNone(metrics.total_cost_usd)

    async def test_final_upstream_overwrite_repair_and_grm_sees_all_sessions(
        self,
    ) -> None:
        from harbor.models.agent.context import AgentContext
        from rewardkit.trajectory import format_trajectory

        for index in range(5):
            self._put(index)
        await self.environment.finalize_trajectory_sync(strict=True)
        agent_dir = self.environment.trial_paths.agent_dir
        shutil.copytree(
            self.environment.pod / "tmp/codex-home/sessions", agent_dir / "sessions"
        )
        last = agent_dir / "sessions/2026/09/06/rollout-5.jsonl"
        last.parent.mkdir(parents=True)
        last.write_bytes(_jsonl(_codex_lines(5)))
        # 实际 upstream post-convert 会 latest-only 覆盖，不通过手写假 ATIF 绕过。
        Codex(agent_dir).populate_context_post_run(AgentContext())
        self.assertEqual(len(self.environment.trajectory().steps), 2)
        repaired = reconcile_trial_trajectory(agent_dir, model_name="test-model")
        self.assertEqual(repaired.session_id, "stable-trial-id")
        self.assertEqual(len(repaired.steps), 12)
        self.assertEqual(len(self.environment.manifest()["sources"]), 6)
        rendered = format_trajectory(agent_dir / "trajectory.json")
        self.assertIn("instruction-session-0", rendered)
        self.assertIn("evidence-session-5", rendered)
        self.assertEqual(
            reconcile_trial_trajectory(agent_dir, model_name="test-model"), repaired
        )

    async def test_remote_latest_atif_cannot_overwrite_native_history(self) -> None:
        self._put(0)
        raw = self._put(1)
        latest = CodexTrajectoryCollector().artifacts(raw, "")["trajectory.json"]
        self.environment.remote_file("/logs/agent/trajectory.json", latest)
        await self.environment.finalize_trajectory_sync(strict=True)
        self.assertEqual(len(self.environment.trajectory().steps), 4)
        self.assertEqual(len(self.environment.manifest()["sources"]), 3)

    async def test_copied_history_and_truncation_are_explicit_not_guessed(self) -> None:
        self._put(0)
        copied = _codex_lines(1)
        copied[0]["payload"]["forked_from_id"] = "native-session-0"
        self.environment.remote_file(_codex_remote(1), _jsonl(copied))
        with self.assertRaisesRegex(ValueError, "unsupported_copied_history"):
            await self.environment.finalize_trajectory_sync(strict=True)
        self.assertEqual(len(self.environment.trajectory().steps), 4)
        # 真正 source 截断必须留存旧字节并报错；不是一次普通 append 快照。
        file0 = self.environment.pod / _codex_remote(0).lstrip("/")
        file0.write_bytes(_jsonl(_codex_lines(0)[:2]))
        with self.assertRaisesRegex(ValueError, "non_append_rewrite"):
            await self.environment.finalize_trajectory_sync(strict=True)
        self.assertTrue(self.environment.trajectory().extra["collection_incomplete"])

    async def test_stale_download_cannot_truncate_newer_live_snapshot(self) -> None:
        original = self._put(0)
        await self.environment.finalize_trajectory_sync(strict=True)
        grown = original + _jsonl(
            [
                {
                    "timestamp": "2026-09-01T14:43:25.000Z",
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Later live evidence."},
                        ],
                    },
                }
            ]
        )
        self.environment.remote_file(_codex_remote(0), grown)
        await self.environment.finalize_trajectory_sync(strict=True)
        expected = self.environment.trajectory()
        downloaded = (
            self.environment.trial_paths.agent_dir
            / "sessions/2026/09/01/rollout-0.jsonl"
        )
        downloaded.parent.mkdir(parents=True)
        downloaded.write_bytes(original)
        for _ in range(2):
            self.assertEqual(
                reconcile_trial_trajectory(self.environment.trial_paths.agent_dir),
                expected,
            )
        entry = next(iter(self.environment.manifest()["sources"].values()))
        self.assertFalse(entry["issues"])
        self.assertEqual(
            (
                self.environment.trial_paths.agent_dir
                / "trajectory-raw"
                / entry["raw_path"]
            ).read_bytes(),
            grown,
        )

    async def test_live_unchanged_does_not_reconvert_but_final_repairs_overwrite(
        self,
    ) -> None:
        raw = self._put(0)
        self._put(1)
        converter = Codex._convert_events_to_trajectory
        with patch.object(
            Codex, "_convert_events_to_trajectory", autospec=True, side_effect=converter
        ) as converted:
            await self.environment._sync_trajectory_once()
            self.assertEqual(converted.call_count, 2)
            await self.environment._sync_trajectory_once()
            self.assertEqual(converted.call_count, 2)
        path = self.environment.trial_paths.agent_dir / "trajectory.json"
        path.write_bytes(
            CodexTrajectoryCollector().artifacts(raw, "")["trajectory.json"]
        )
        await (
            self.environment.finalize_trajectory_sync()
        )  # 默认非 strict 也必须末次投影
        self.assertEqual(len(self.environment.trajectory().steps), 4)

    async def test_unstable_transfer_retained_without_poisoning_source_history(
        self,
    ) -> None:
        import base64

        raw = self._put(0)
        await self.environment.finalize_trajectory_sync(strict=True)
        grown = raw + _jsonl(
            [{"type": "event_msg", "payload": {"type": "task_complete"}}]
        )
        self.environment.remote_file(_codex_remote(0), grown)
        original = self.environment._exec_cmd_platform

        async def truncated(command: str, **kwargs: object) -> ExecResult:
            if command.startswith("base64 -w0 "):
                return ExecResult(
                    stdout=base64.b64encode(raw[:80]).decode(), return_code=0
                )
            return await original(command, **kwargs)

        with (
            patch.object(self.environment, "_exec_cmd_platform", truncated),
            self.assertRaisesRegex(RuntimeError, "Unstable trajectory transfer"),
        ):
            await self.environment.finalize_trajectory_sync(strict=True)
        entry = next(iter(self.environment.manifest()["sources"].values()))
        self.assertFalse(entry["issues"])
        root = self.environment.trial_paths.agent_dir / "trajectory-raw"
        self.assertEqual((root / entry["raw_path"]).read_bytes(), raw)
        self.assertEqual(
            [path.read_bytes() for path in (root / "unstable").glob("*/*")], [raw[:80]]
        )
        await self.environment.finalize_trajectory_sync(strict=True)
        self.assertFalse(self.environment.trajectory().extra["collection_incomplete"])

    async def test_utf8_partial_tail_keeps_complete_history_and_original_bytes(
        self,
    ) -> None:
        raw = self._put(0) + b'{"type":"response_item","text":"' + "中".encode()[:2]
        self.environment.remote_file(_codex_remote(0), raw)
        await self.environment.finalize_trajectory_sync(strict=True)
        self.assertEqual(len(self.environment.trajectory().steps), 2)
        self.assertTrue(self.environment.trajectory().extra["collection_pending"])
        entry = next(iter(self.environment.manifest()["sources"].values()))
        self.assertEqual(
            (
                self.environment.trial_paths.agent_dir
                / "trajectory-raw"
                / entry["raw_path"]
            ).read_bytes(),
            raw,
        )

    async def test_no_archive_and_relocated_archive_have_stable_semantics(self) -> None:
        self.assertIsNone(
            reconcile_trial_trajectory(self.environment.trial_paths.agent_dir)
        )
        self._put(0)
        await self.environment.finalize_trajectory_sync(strict=True)
        destination = self.environment.root / "relocated-agent"
        shutil.copytree(self.environment.trial_paths.agent_dir, destination)
        self.assertEqual(
            reconcile_trial_trajectory(destination), self.environment.trajectory()
        )

    async def test_invalid_middle_line_and_usage_reset_fail_strict(self) -> None:
        lines = _codex_lines(0, usage={"input_tokens": 100, "output_tokens": 20})
        raw = _jsonl(lines[:2]) + b'{"broken":\n' + _jsonl(lines[2:])
        self.environment.remote_file(_codex_remote(0), raw)
        with self.assertRaisesRegex(ValueError, "malformed_jsonl"):
            await self.environment.finalize_trajectory_sync(strict=True)
        self.assertEqual(len(self.environment.trajectory().steps), 2)
        reset = copy.deepcopy(lines[-1])
        reset["payload"]["info"]["total_token_usage"]["input_tokens"] = 10
        self.environment.remote_file(_codex_remote(0), raw + _jsonl([reset]))
        with self.assertRaisesRegex(ValueError, "unsupported_usage_reset"):
            await self.environment.finalize_trajectory_sync(strict=True)
        self.assertIsNone(
            self.environment.trajectory().final_metrics.total_prompt_tokens
        )

    async def test_native_opaque_id_overlap_is_flagged_without_text_guessing(
        self,
    ) -> None:
        self._put(0)
        other = _codex_lines(1)
        other[2]["payload"]["call_id"] = "call-0"
        other[3]["payload"]["call_id"] = "call-0"
        self.environment.remote_file(_codex_remote(1), _jsonl(other))
        with self.assertRaisesRegex(ValueError, "unsupported_copied_history"):
            await self.environment.finalize_trajectory_sync(strict=True)
        self.assertEqual(len(self.environment.trajectory().steps), 4)

    async def test_archive_write_failure_propagates_after_other_sources_saved(
        self,
    ) -> None:
        import harbor_envs.abstract_mlplat.trajectory as module

        self._put(0)
        self._put(1)
        original = module._atomic_write

        def write(path: Path, raw: bytes) -> None:
            if path.name == "rollout-0.jsonl":
                raise OSError("injected raw archive failure")
            original(path, raw)

        with (
            patch.object(module, "_atomic_write", write),
            self.assertRaisesRegex(OSError, "injected raw archive failure"),
        ):
            await self.environment.finalize_trajectory_sync(strict=True)
        self.assertEqual(len(self.environment.manifest()["sources"]), 1)
        self.assertTrue(self.environment.trajectory().extra["collection_incomplete"])
        await self.environment.finalize_trajectory_sync(strict=True)
        self.assertEqual(len(self.environment.trajectory().steps), 4)

    async def test_failed_all_conversions_do_not_leave_latest_only_success(
        self,
    ) -> None:
        self._put(0)
        await self.environment.finalize_trajectory_sync(strict=True)
        with (
            patch.object(
                Codex,
                "_convert_events_to_trajectory",
                side_effect=RuntimeError("schema drift"),
            ),
            self.assertRaisesRegex(ValueError, "schema drift"),
        ):
            await self.environment.finalize_trajectory_sync(strict=True)
        self.assertFalse(
            (self.environment.trial_paths.agent_dir / "trajectory.json").exists()
        )
        self.assertEqual(len(self.environment.manifest()["sources"]), 1)
        self.assertTrue(self.environment.manifest()["projection_issues"])


    async def test_finalize_awaits_poller_before_collecting_and_is_reentrant(
        self,
    ) -> None:
        self._put(0)
        stopped = asyncio.Event()
        started = asyncio.Event()

        async def poller() -> None:
            """取消期间延迟退出，证明 finalize 真正 await 后才开始采集。"""
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                stopped.set()

        self.environment._trajectory_sync_task = asyncio.create_task(poller())
        await started.wait()
        original = self.environment._sync_trajectory_once

        async def sync(*, strict: bool = False) -> None:
            self.assertTrue(stopped.is_set())
            await original(strict=strict)

        with patch.object(self.environment, "_sync_trajectory_once", sync):
            await asyncio.gather(
                self.environment.finalize_trajectory_sync(strict=True),
                self.environment.finalize_trajectory_sync(strict=True),
            )
        self.assertIsNone(self.environment._trajectory_sync_task)
        self.assertEqual(len(self.environment.trajectory().steps), 2)


class ClaudeDirectoryHistoryTest(unittest.IsolatedAsyncioTestCase):
    """一次 whole-directory 转换复用上游 UUID、message 与 subagent 去重。"""

    async def test_whole_directory_uuid_message_subagent_dedup_and_metrics(
        self,
    ) -> None:
        from harbor.agents.installed.claude_code import ClaudeCode

        with tempfile.TemporaryDirectory() as directory:
            environment = LocalSyncEnvironment(Path(directory))
            main = copy.deepcopy(_CLAUDE_LINES)
            main[1]["message"].update(
                id="main-message",
                usage={
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 40,
                    "cache_creation_input_tokens": 10,
                },
            )
            update = copy.deepcopy(main[1])
            update.update(
                uuid="assistant-stream-update", timestamp="2026-09-02T08:30:07.000Z"
            )
            update["message"]["usage"].update(input_tokens=120, output_tokens=30)
            update["message"]["content"] = [
                {"type": "text", "text": "Completed main analysis."}
            ]
            subagent = copy.deepcopy(main[1])
            subagent.update(
                uuid="subagent-event",
                timestamp="2026-09-02T08:30:08.000Z",
                isSidechain=True,
            )
            subagent["message"].update(
                id="subagent-message",
                content=[
                    {"type": "text", "text": "Subagent real result."},
                ],
                usage={
                    "input_tokens": 20,
                    "output_tokens": 7,
                    "cache_read_input_tokens": 5,
                    "cache_creation_input_tokens": 0,
                },
            )
            environment.remote_file(
                "/logs/agent/sessions/projects/-app/main.jsonl", _jsonl(main)
            )
            environment.remote_file(
                "/logs/agent/sessions/projects/-other/copy.jsonl",
                _jsonl(main + [update]),
            )
            environment.remote_file(
                "/logs/agent/sessions/projects/-app/main/subagents/agent-x.jsonl",
                _jsonl([subagent]),
            )
            original = ClaudeCode._convert_events_to_trajectory
            calls = []

            def convert(agent: ClaudeCode, path: Path) -> Trajectory | None:
                calls.append(sorted(path.rglob("*.jsonl")))
                return original(agent, path)

            with patch.object(ClaudeCode, "_convert_events_to_trajectory", convert):
                await environment.finalize_trajectory_sync(strict=True)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(calls[0]), 3)
            trajectory = environment.trajectory()
            self.assertEqual(len(trajectory.steps), 3)
            self.assertEqual(len(trajectory.extra["trajectory_contributions"]), 1)
            self.assertEqual(
                len(trajectory.extra["trajectory_contributions"][0]["source_keys"]), 3
            )
            self.assertEqual(trajectory.steps[0].message, "Optimize the server.")
            self.assertIn("Completed main analysis.", trajectory.steps[1].message)
            self.assertEqual(len(trajectory.steps[1].tool_calls), 1)
            self.assertEqual(
                trajectory.steps[1].observation.results[0].source_call_id, "toolu_1"
            )
            self.assertTrue(trajectory.steps[2].extra["is_sidechain"])
            self.assertIsNone(trajectory.subagent_trajectories)
            metrics = trajectory.final_metrics
            self.assertEqual(
                (
                    metrics.total_prompt_tokens,
                    metrics.total_completion_tokens,
                    metrics.total_cached_tokens,
                    metrics.total_steps,
                ),
                (195, 37, 45, 3),
            )
            native_metrics = trajectory.extra["trajectory_contributions"][0][
                "native_final_metrics"
            ]
            self.assertEqual(metrics.total_cost_usd, native_metrics["total_cost_usd"])
            await environment.finalize_trajectory_sync(strict=True)
            self.assertEqual(environment.trajectory(), trajectory)

    async def test_missing_usage_stays_unknown_and_native_zero_cost_survives(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = LocalSyncEnvironment(Path(directory))
            events = copy.deepcopy(_CLAUDE_LINES)
            events[1]["message"]["usage"] = {"input_tokens": 100, "output_tokens": None}
            environment.remote_file(
                "/logs/agent/sessions/projects/-app/main.jsonl", _jsonl(events)
            )
            agent_dir = environment.trial_paths.agent_dir
            agent_dir.mkdir(parents=True)
            (agent_dir / "claude-code.txt").write_text(
                json.dumps({"type": "result", "total_cost_usd": 0})
            )
            await environment.finalize_trajectory_sync(strict=True)
            metrics = environment.trajectory().final_metrics
            self.assertIsNone(
                metrics.total_prompt_tokens
            )  # cache 两项未提供，不可默认 0
            self.assertIsNone(metrics.total_completion_tokens)
            self.assertIsNone(metrics.total_cached_tokens)
            self.assertEqual(metrics.total_cost_usd, 0)
            step_metrics = environment.trajectory().steps[1].metrics
            self.assertIsNone(step_metrics.prompt_tokens)
            self.assertIsNone(step_metrics.completion_tokens)
            self.assertIsNone(step_metrics.cached_tokens)

    async def test_conflicting_uuid_is_not_silently_deduped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = LocalSyncEnvironment(Path(directory))
            environment.remote_file(
                "/logs/agent/sessions/projects/-app/main.jsonl", _jsonl(_CLAUDE_LINES)
            )
            conflicting = copy.deepcopy(_CLAUDE_LINES)
            conflicting[1]["message"]["content"][0]["text"] = (
                "Changed same-UUID history"
            )
            environment.remote_file(
                "/logs/agent/sessions/projects/-app/copy.jsonl", _jsonl(conflicting)
            )
            with self.assertRaisesRegex(ValueError, "conflicting_event_uuid"):
                await environment.finalize_trajectory_sync(strict=True)
            self.assertEqual(len(environment.manifest()["sources"]), 2)
            self.assertTrue(environment.trajectory().extra["collection_incomplete"])


if __name__ == "__main__":
    unittest.main()

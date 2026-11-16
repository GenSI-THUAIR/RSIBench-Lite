"""同步全部原生轨迹，先保全 raw，再复用上游转换器生成 trial 级 ATIF。

remote source 按原生相对路径记账，逻辑 session 只用于转换去重；轮询不追加历史。
collector-owned ``trajectory-raw/manifest.json`` 保存 trial 身份和采集诊断，最终
日志下载/上游 post-convert 后可离线重建。示例：
    trajectory = reconcile_trial_trajectory(Path("trial/agent"), model_name="model")
    await environment.finalize_trajectory_sync(strict=True)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import shlex
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from harbor.models.trajectories.final_metrics import FinalMetrics
from harbor.models.trajectories.trajectory import Trajectory
from harbor.utils.trajectory_utils import format_trajectory_json

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from harbor.environments.base import ExecResult
    from harbor.models.trial.paths import TrialPaths

_TRAJECTORY_SYNC_INTERVAL_SEC = 10
_RAW_DIR = "trajectory-raw"
_LOGGER = logging.getLogger(__name__)
_USAGE_FIELDS = (
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_cached_tokens",
    "total_cost_usd",
)


def _atomic_write(path: Path, content: bytes) -> None:
    """在目标目录原子替换文件；失败清理临时文件，不碰旧产物。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _RawArchive:
    """单个 trial 的原生源账本；只有非追加改写才保留冲突版本，不存轮询快照。"""

    def __init__(self, agent_dir: Path, session_id: str | None = None) -> None:
        """加载已有账本；新账本必须由环境提供稳定 trial/session 身份。"""
        self.root = agent_dir / _RAW_DIR
        self.path = self.root / "manifest.json"
        if self.path.exists():
            self._refresh()
        else:
            self.data = {
                "version": 1,
                "trial_session_id": session_id,
                "sources": {},
                "collection_errors": {},
            }

    def _refresh(self) -> None:
        """采集器可跨远端await持有旧实例；同步写入前重读同一事件循环的最新账本。

        refresh到save之间不能await；这不提供跨进程写锁，也不修复已有哈希损坏。
        """
        if self.path.exists():
            data = json.loads(self.path.read_bytes())
            if data["version"] != 1:
                raise ValueError("Unsupported trajectory raw archive version")
            self.data = data

    def save(self) -> None:
        """原子提交账本；raw 必须在调用前已落盘。"""
        if not self.data["trial_session_id"]:
            raise ValueError("Trajectory archive requires a stable trial session id")
        _atomic_write(
            self.path, json.dumps(self.data, indent=2, sort_keys=True).encode()
        )

    def store(self, family: str, relative: str, raw: bytes, origin: str) -> None:
        """先存 raw 再记来源；镜像的旧前缀不覆盖新数据，原源截断明确报诊断。"""
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe trajectory source path: {relative}")
        self._refresh()
        key = f"{family}:{relative}"
        digest = hashlib.sha256(raw).hexdigest()
        entry = self.data["sources"].get(key)
        path = self.root / family / relative
        if entry is None:
            entry = {
                "family": family,
                "relative_path": relative,
                "raw_path": f"{family}/{relative}",
                "primary_origin": origin,
                "origins": {},
                "issues": [],
            }
        elif entry["origins"].get(origin) == digest:
            return
        elif entry["sha256"] != digest and family in {
            "codex",
            "claude-code",
            "opencode",
        }:
            previous = path.read_bytes()
            if not raw.startswith(previous):
                # final 的 cp/download 可以比最后一次 live snapshot 旧，但同一
                # origin 从长变短是截断；两者不能根据 mtime/长度混为一谈。
                stale_mirror = previous.startswith(raw) and (
                    origin != entry["primary_origin"] or origin.startswith("download:")
                )
                if not stale_mirror:
                    revision = f"revisions/{hashlib.sha256(key.encode()).hexdigest()}/{entry['sha256']}"
                    _atomic_write(self.root / revision, previous)
                    issue = {"kind": "non_append_rewrite", "previous_raw": revision}
                    if issue not in entry["issues"]:
                        entry["issues"].append(issue)
                else:
                    entry["origins"][origin] = digest
                    self.save()
                    return
        if entry.get("sha256") != digest:
            _atomic_write(path, raw)
        entry["sha256"] = digest
        entry["origins"][origin] = digest
        self.data["sources"][key] = entry
        self.save()

    def collection_error(self, key: str, error: Exception | None) -> None:
        """持久化未解决的传输错误；成功重采后才清除，离线 final 不假装完整。"""
        self._refresh()
        errors = self.data["collection_errors"]
        if error is not None:
            errors[key] = f"{type(error).__name__}: {error}"
        elif key in errors:
            del errors[key]
        else:
            return
        self.save()


class TrajectoryCollector(ABC):
    """声明原生路径及单文件转换；多 session 的 canonical 投影由 reconcile 负责。"""

    remote_path: str
    family: str

    @property
    def remote_paths(self) -> tuple[str, ...]:
        """返回所有发现模式，固定文件是单一模式的特例。"""
        return (self.remote_path,)

    def relative_path(self, remote: str) -> str:
        """固定文件使用原生文件名作为稳定 source identity。"""
        return PurePosixPath(remote).name

    @abstractmethod
    def artifacts(self, raw: bytes, session_id: str) -> dict[str, bytes]:
        """把原生字节转换为兼容快照；调用者须先独立归档 raw。"""


def _session_bytes_to_atif(
    raw: bytes, agent_factory: Callable[..., Any], native_name: str
) -> dict[str, bytes]:
    """单文件兼容接口：临时目录喂给真实上游转换器，不复制事件映射。"""
    with tempfile.TemporaryDirectory() as staging:
        session_dir = Path(staging)
        (session_dir / "session.jsonl").write_bytes(_conversion_jsonl_bytes(raw))
        trajectory = agent_factory(session_dir)._convert_events_to_trajectory(
            session_dir
        )
    artifacts = {native_name: raw}
    if trajectory is not None:
        artifacts["trajectory.json"] = format_trajectory_json(
            trajectory.to_json_dict()
        ).encode()
    return artifacts


class AtifTrajectoryCollector(TrajectoryCollector):
    """直接输出 ATIF 的 agent；存在原生历史时它不是覆盖历史的事实源。"""

    family = "atif"

    def __init__(self, remote_path: str = "/logs/agent/trajectory.json") -> None:
        """设置远端 ATIF 文件，默认使用上游标准日志路径。"""
        self.remote_path = remote_path

    def artifacts(self, raw: bytes, session_id: str) -> dict[str, bytes]:
        """保留直接 ATIF 的兼容输出约定。"""
        return {"trajectory.json": raw}


class MiniSweTrajectoryCollector(TrajectoryCollector):
    """mini-swe-agent 的原生轨迹及上游 ATIF 转换。"""

    family = "mini-swe"
    remote_path = "/logs/agent/mini-swe-agent.trajectory.json"

    def artifacts(self, raw: bytes, session_id: str) -> dict[str, bytes]:
        """保留原文件并调用上游 mini-swe 转换。"""
        from harbor.agents.installed.mini_swe_agent import (
            convert_mini_swe_agent_to_atif,
        )

        atif = convert_mini_swe_agent_to_atif(json.loads(raw), session_id)
        return {
            "mini-swe-agent.trajectory.json": raw,
            "trajectory.json": format_trajectory_json(atif.to_json_dict()).encode(),
        }


class CodexTrajectoryCollector(TrajectoryCollector):
    """Codex 全部 live/final rollout，两个根下相同相对路径是同一物理源。"""

    family = "codex"
    remote_path = "/tmp/codex-home/sessions/*/*/*/rollout-*.jsonl"
    roots = ("/tmp/codex-home/sessions/", "/logs/agent/sessions/")

    @property
    def remote_paths(self) -> tuple[str, ...]:
        """同时发现 live home 和上游 finally 的 sessions 归档。"""
        return tuple(root + "**/rollout-*.jsonl" for root in self.roots)

    def relative_path(self, remote: str) -> str:
        """去掉 live/final 根，保留原生日期和 rollout 文件名。"""
        return next(
            remote.removeprefix(root) for root in self.roots if remote.startswith(root)
        )

    def artifacts(self, raw: bytes, session_id: str) -> dict[str, bytes]:
        """兼容单文件调用；正常同步不会靠该快照累计 session。"""
        from harbor.agents.installed.codex import Codex

        return _session_bytes_to_atif(raw, Codex, "codex-session.jsonl")


class ClaudeCodeTrajectoryCollector(TrajectoryCollector):
    """Claude 的主 transcript 和新版本单独落盘的 subagent transcript。"""

    family = "claude-code"
    root = "/logs/agent/sessions/projects/"
    remote_path = root + "*/*.jsonl"

    @property
    def remote_paths(self) -> tuple[str, ...]:
        """发现全部 project 主会话及 subagent，转换时仍整目录去重。"""
        return (self.root + "**/*.jsonl",)

    def relative_path(self, remote: str) -> str:
        """保留 project/session/subagent 原生相对路径。"""
        return remote.removeprefix(self.root)

    def artifacts(self, raw: bytes, session_id: str) -> dict[str, bytes]:
        """兼容单文件调用；reconcile 使用上游整目录 UUID/message 去重。"""
        from harbor.agents.installed.claude_code import ClaudeCode

        return _session_bytes_to_atif(raw, ClaudeCode, "claude-code-session.jsonl")


class OpenCodeTrajectoryCollector(TrajectoryCollector):
    """保留 OpenCode completed-step 转换与 opencode.txt 输出约定。"""

    family = "opencode"
    remote_path = "/logs/agent/opencode.txt"

    def artifacts(self, raw: bytes, session_id: str) -> dict[str, bytes]:
        """容忍 banner/半行，只让上游转换已有 completed steps。"""
        from harbor.agents.installed.opencode import OpenCode

        events, _ = _jsonl_events(raw, allow_banner=True)
        converter = OpenCode(logs_dir=Path(tempfile.gettempdir()))
        trajectory = converter._convert_events_to_trajectory(events)
        artifacts = {"opencode.txt": raw}
        if trajectory is not None:
            artifacts["trajectory.json"] = format_trajectory_json(
                trajectory.to_json_dict()
            ).encode()
        return artifacts


def _conversion_jsonl_bytes(raw: bytes) -> bytes:
    """仅在 UTF-8 码点半写于 EOF 时给 converter 完整行；原始字节始终另存。"""
    try:
        raw.decode()
    except UnicodeDecodeError as exc:
        if exc.reason != "unexpected end of data" or exc.end != len(raw):
            raise
        return raw[: raw.rfind(b"\n") + 1]
    return raw


def _jsonl_events(
    raw: bytes, *, allow_banner: bool = False
) -> tuple[list[dict], list[str]]:
    """仅解析原文以取身份/诊断；半写尾行不丢已有事件，禁止猜测补造事件。"""
    converted = _conversion_jsonl_bytes(raw)
    events, issues = [], [] if converted == raw else ["partial_jsonl_tail:utf8"]
    lines = converted.decode().splitlines()
    for index, line in enumerate(lines, 1):
        if not line.strip() or (allow_banner and not line.lstrip().startswith("{")):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            partial = index == len(lines) and line.lstrip().startswith("{")
            issues.append(
                f"{'partial_jsonl_tail' if partial else 'malformed_jsonl'}:{index}"
            )
            continue
        if not isinstance(event, dict):
            issues.append(f"non_object_jsonl:{index}")
            continue
        events.append(event)
    return events, issues


def _ingest_downloaded_sources(archive: _RawArchive, agent_dir: Path) -> None:
    """离线吸收上游 download_dir 的原生文件；不把其 latest-only ATIF 当新历史。"""
    collectors = (CodexTrajectoryCollector(), ClaudeCodeTrajectoryCollector())
    for collector in collectors:
        root = agent_dir / "sessions"
        pattern = "**/rollout-*.jsonl"
        if isinstance(collector, ClaudeCodeTrajectoryCollector):
            root /= "projects"
            pattern = "**/*.jsonl"
        for path in sorted(root.glob(pattern)):
            relative = path.relative_to(root).as_posix()
            archive.store(
                collector.family, relative, path.read_bytes(), f"download:{relative}"
            )
    for collector in (OpenCodeTrajectoryCollector(), MiniSweTrajectoryCollector()):
        path = agent_dir / PurePosixPath(collector.remote_path).name
        if path.is_file():
            archive.store(
                collector.family, path.name, path.read_bytes(), f"download:{path.name}"
            )
    # OpenCode 旧版事件流不带 user；保全 upstream post-convert 已记录的真实
    # instruction，之后重建从该证据读取，不从自身 canonical 递归采集。
    previous = agent_dir / "trajectory.json"
    if previous.is_file() and any(
        entry["family"] == "opencode" for entry in archive.data["sources"].values()
    ):
        data = json.loads(previous.read_bytes())
        if data.get("agent", {}).get(
            "name"
        ) == "opencode" and "collection_incomplete" not in (data.get("extra") or {}):
            archive.store(
                "atif", previous.name, previous.read_bytes(), "upstream-post-convert"
            )


def _codex_metrics(trajectory: Trajectory, events: list[dict]) -> None:
    """保留原生 total usage 的 0/unknown 语义；缓存已包含在 input 中，绝不再加。"""
    for event in reversed(events):
        payload = event.get("payload", {})
        if event.get("type") != "event_msg" or payload.get("type") != "token_count":
            continue
        info = payload.get("info")
        if not isinstance(info, dict) or not isinstance(
            info.get("total_token_usage"), dict
        ):
            continue
        usage = info["total_token_usage"]
        metrics = trajectory.final_metrics or FinalMetrics()
        for field, key in zip(
            _USAGE_FIELDS[:3], ("input_tokens", "output_tokens", "cached_input_tokens")
        ):
            setattr(metrics, field, usage.get(key))
        trajectory.final_metrics = metrics
        return


def _source_events(
    key: str,
    raw: bytes,
    issues: list[dict],
    pending: list[dict],
    *,
    allow_banner: bool = False,
) -> list[dict]:
    """分别记录输入错误和正常半写尾行，后者不冒充已完成历史丢失。"""
    events, diagnostics = _jsonl_events(raw, allow_banner=allow_banner)
    if any(diagnostic.startswith("non_object_jsonl:") for diagnostic in diagnostics):
        raise ValueError("Native JSONL contains a non-object event")
    for diagnostic in diagnostics:
        target = pending if diagnostic.startswith("partial_jsonl_tail:") else issues
        target.append({"source": key, "kind": diagnostic})
    return events


def _convert_codex(
    sources: list[tuple[str, dict, bytes]],
    model_name: str | None,
    issues: list[dict],
    pending: list[dict],
) -> list[tuple[Trajectory, list[str], list[str]]]:
    """按真实 session ID 合并前缀快照；每个独立会话只调用一次上游 converter。"""
    from harbor.agents.installed.codex import Codex

    sessions: dict[str, list[tuple[str, bytes, list[dict]]]] = {}
    for key, _, raw in sources:
        try:
            events = _source_events(key, raw, issues, pending)
            if not events:
                pending.append({"source": key, "kind": "no_events_yet"})
                continue
            metas = [
                event.get("payload", {})
                for event in events
                if event.get("type") == "session_meta"
            ]
            ids = {meta.get("id") or meta.get("session_id") for meta in metas}
            ids.discard(None)
            if len(ids) != 1 or not all(isinstance(sid, str) and sid for sid in ids):
                raise ValueError("Missing or ambiguous native session id")
            sid = next(iter(ids))
            if any(
                meta.get(name)
                for meta in metas
                for name in ("forked_from_id", "forked_from", "parent_session_id")
            ):
                issues.append({"source": key, "kind": "unsupported_copied_history"})
            sessions.setdefault(sid, []).append((key, raw, events))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            issues.append(
                {
                    "source": key,
                    "kind": "invalid_native_source",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    converted = []
    event_owners: dict[tuple[str, str], str] = {}
    for sid, snapshots in sorted(sessions.items()):
        snapshots.sort(key=lambda item: (-len(item[1]), item[0]))
        key, raw, events = snapshots[0]
        keys = sorted(item[0] for item in snapshots)
        if any(not raw.startswith(item[1]) for item in snapshots):
            issues.append(
                {
                    "session_id": sid,
                    "sources": keys,
                    "kind": "unsupported_divergent_session_snapshots",
                }
            )
        try:
            # 只使用原生不透明事件/调用 ID 检测 copied history；文本相似不去重。
            previous_usage: dict[str, int] = {}
            for event in events:
                payload = event.get("payload", {})
                if (
                    event.get("type") == "event_msg"
                    and payload.get("type") == "token_count"
                ):
                    usage = (payload.get("info") or {}).get("total_token_usage") or {}
                    for name in (
                        "input_tokens",
                        "output_tokens",
                        "cached_input_tokens",
                    ):
                        value = usage.get(name)
                        if value is not None:
                            if value < previous_usage.get(name, 0):
                                issues.append(
                                    {
                                        "sources": keys,
                                        "kind": "unsupported_usage_reset",
                                        "field": name,
                                    }
                                )
                            previous_usage[name] = value
                if event.get("type") != "response_item":
                    continue
                for name in ("id", "call_id"):
                    identity = payload.get(name)
                    if not identity:
                        continue
                    token = (name, identity)
                    owner = event_owners.setdefault(token, sid)
                    if owner != sid:
                        issue = {
                            "kind": "unsupported_copied_history",
                            "session_ids": sorted([owner, sid]),
                        }
                        if issue not in issues:
                            issues.append(issue)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "session.jsonl").write_bytes(_conversion_jsonl_bytes(raw))
                trajectory = Codex(
                    root, model_name=model_name
                )._convert_events_to_trajectory(root)
            if trajectory is None:
                # Codex 上游只有 message/已配对 output 才产步骤；单独 call 尚在等待。
                completed = any(
                    event.get("type") == "response_item"
                    and event.get("payload", {}).get("type")
                    in {
                        "message",
                        "function_call_output",
                        "custom_tool_call_output",
                        "web_search_call",
                    }
                    for event in events
                )
                if completed:
                    raise ValueError("Completed native history produced no steps")
                pending.append(
                    {"sources": keys, "session_id": sid, "kind": "no_steps_yet"}
                )
                continue
            _codex_metrics(trajectory, events)
            meta_event = next(
                event for event in events if event.get("type") == "session_meta"
            )
            trajectory.extra = {
                **(trajectory.extra or {}),
                "native_session_created_at": meta_event["payload"].get("timestamp")
                or meta_event.get("timestamp"),
            }
            converted.append((trajectory, keys, [sid]))
        except Exception as exc:  # noqa: BLE001 - 隔离原生 converter，strict final 会重抛
            # 隔离单 session，不允许 converter 失败阻止其余 raw/历史贡献。
            issues.append(
                {
                    "sources": keys,
                    "kind": "conversion_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return converted


def _convert_claude(
    sources: list[tuple[str, dict, bytes]],
    agent_dir: Path,
    model_name: str | None,
    issues: list[dict],
    pending: list[dict],
) -> list[tuple[Trajectory, list[str], list[str]]]:
    """所有 Claude 文件一次整目录转换，复用 UUID/message/subagent 去重与原生成本。"""
    from harbor.agents.installed.claude_code import ClaudeCode

    class ArchiveClaudeCode(ClaudeCode):
        """仅纠正上游缺失 usage 被补 0 的语义，不改事件/去重映射。"""

        @staticmethod
        def _build_metrics(usage: Any) -> Any:
            """沿用上游 metrics，再保持原生未提供字段为 unknown。"""
            metrics = ClaudeCode._build_metrics(usage)
            if metrics is not None:
                if any(
                    usage.get(key) is None
                    for key in (
                        "input_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens",
                    )
                ):
                    metrics.prompt_tokens = None
                metrics.completion_tokens = usage.get("output_tokens")
                metrics.cached_tokens = usage.get("cache_read_input_tokens")
            return metrics

    session_ids: set[str] = set()
    event_uuids: dict[str, dict] = {}
    has_messages = False
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for key, entry, raw in sources:
            try:
                events = _source_events(key, raw, issues, pending)
                session_ids.update(
                    event["sessionId"] for event in events if event.get("sessionId")
                )
                has_messages |= any(
                    event.get("type") in {"user", "assistant"} and event.get("message")
                    for event in events
                )
                for event in events:
                    uuid = event.get("uuid")
                    if uuid and event_uuids.setdefault(uuid, event) != event:
                        issues.append(
                            {
                                "source": key,
                                "kind": "conflicting_event_uuid",
                                "uuid": uuid,
                            }
                        )
                # 上游扫描根 *.jsonl 和递归 subagents/*.jsonl；平铺主文件仅改
                # staging 路径，不更改原生事件，更不按文件自行累加 usage。
                name = hashlib.sha256(key.encode()).hexdigest() + ".jsonl"
                path = (
                    root
                    / (
                        "subagents"
                        if "subagents" in PurePosixPath(entry["relative_path"]).parts
                        else ""
                    )
                    / name
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(_conversion_jsonl_bytes(raw))
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                issues.append(
                    {
                        "source": key,
                        "kind": "invalid_native_source",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        converter = ArchiveClaudeCode(agent_dir, model_name=model_name)
        trajectory = converter._convert_events_to_trajectory(root)
    if trajectory is None:
        if has_messages:
            raise ValueError("Completed Claude history produced no steps")
        pending.append({"family": "claude-code", "kind": "no_steps_yet"})
        return []
    # 上游 whole-directory 去重已经完成，这里只传播 unknown，不再算另一遍和。
    agent_steps = [step for step in trajectory.steps if step.source == "agent"]
    metrics = trajectory.final_metrics
    if metrics is not None:
        for total, field in zip(
            _USAGE_FIELDS[:3], ("prompt_tokens", "completion_tokens", "cached_tokens")
        ):
            if any(getattr(step.metrics, field, None) is None for step in agent_steps):
                setattr(metrics, total, None)
        estimated = (metrics.extra or {}).get("cost_source") == "litellm_estimate"
        if (
            estimated
            and any(getattr(metrics, field) is None for field in _USAGE_FIELDS[:3])
        ) or (not estimated and len(session_ids) > 1):
            metrics.total_cost_usd = None
    return [(trajectory, sorted(key for key, _, _ in sources), sorted(session_ids))]


def _convert_opencode(
    raw: bytes,
    agent_dir: Path,
    model_name: str | None,
) -> Trajectory | None:
    """复用 completed-step 转换；最终重建保留上游已记录的真实 instruction。"""
    from harbor.agents.installed.opencode import OpenCode

    converter = OpenCode(agent_dir, model_name=model_name)
    previous = agent_dir / _RAW_DIR / "atif/trajectory.json"
    if previous.is_file():
        data = json.loads(previous.read_bytes())
        if data.get("agent", {}).get("name") == "opencode":
            converter._instruction = next(
                (
                    step["message"]
                    for step in data.get("steps", [])
                    if step.get("source") == "user"
                    and isinstance(step.get("message"), str)
                ),
                None,
            )
    events, _ = _jsonl_events(raw, allow_banner=True)
    trajectory = converter._convert_events_to_trajectory(events)
    if trajectory is None:
        return None
    if converter._instruction and not any(
        event.get("type") == "user" for event in events
    ):
        trajectory.steps[0].extra = {
            "input_provenance": {
                "source_key": "atif:trajectory.json",
                "kind": "upstream_recorded_instruction",
            }
        }
    # 不重写 completed-step 映射，只检查上游聚合是否把缺失项算成了 0。
    finishes = []
    in_step = False
    for event in events:
        if event.get("type") == "step_start":
            in_step = True
        elif event.get("type") == "step_finish" and in_step:
            finishes.append(event.get("part", {}))
            in_step = False
    known: dict[str, list[Any]] = {field: [] for field in _USAGE_FIELDS}
    for step, finish in zip(
        (step for step in trajectory.steps if step.source == "agent"),
        finishes,
        strict=True,
    ):
        tokens = finish.get("tokens", {})
        cache = tokens.get("cache", {})
        prompt = tokens.get("input")
        cached = cache.get("read")
        values = [
            None if prompt is None or cached is None else prompt + cached,
            tokens.get("output"),
            cached,
            finish.get("cost"),
        ]
        for field, value in zip(_USAGE_FIELDS, values):
            known[field].append(value)
        if step.metrics is not None:
            for field, value in zip(
                ("prompt_tokens", "completion_tokens", "cached_tokens", "cost_usd"),
                values,
            ):
                setattr(step.metrics, field, value)
    for field, values in known.items():
        if any(value is None for value in values):
            setattr(trajectory.final_metrics, field, None)
        elif all(value == 0 for value in values):
            setattr(trajectory.final_metrics, field, 0)
    return trajectory


def _flatten_trajectories(
    contributions: list[tuple[Trajectory, list[str], list[str]]],
    trial_id: str,
    issues: list[dict],
    pending: list[dict],
) -> Trajectory:
    """按原生创建顺序平铺真实步骤；范围级 provenance 不冒充逐事件精确归因。"""
    contributions.sort(
        key=lambda item: (
            (item[0].extra or {}).get("native_session_created_at")
            or item[0].steps[0].timestamp
            or "",
            item[2],
            item[1],
        )
    )
    steps, ranges = [], []
    for trajectory, sources, session_ids in contributions:
        start = len(steps) + 1
        provenance = {
            "native_session_ids": session_ids,
            "source_keys": sources,
            "scope": "upstream_conversion_group",
        }
        for original in trajectory.steps:
            step = original.model_copy(deep=True)
            step.step_id = len(steps) + 1
            step.extra = {
                **(step.extra or {}),
                "trajectory_provenance": {
                    **provenance,
                    "native_step_id": original.step_id,
                },
            }
            steps.append(step)
        ranges.append(
            {
                **provenance,
                "step_range": [start, len(steps)],
                "native_final_metrics": trajectory.final_metrics.model_dump(mode="json")
                if trajectory.final_metrics
                else None,
            }
        )
    # 只加每个独立贡献的 final metrics，不重加 step totals；任何未知贡献
    # 都使该总量未知，不用 0 冒充完整总量。诊断 partial view 同样不能报全量。
    totals: dict[str, Any] = {}
    for field in _USAGE_FIELDS:
        values = [getattr(item[0].final_metrics, field, None) for item in contributions]
        totals[field] = (
            sum(values)
            if not issues and not pending and all(value is not None for value in values)
            else None
        )
    result = contributions[0][0].model_copy(deep=True)
    result.session_id = trial_id
    result.steps = steps
    result.final_metrics = FinalMetrics(**totals, total_steps=len(steps))
    result.extra = {
        **(result.extra or {}),
        "collection_incomplete": bool(issues),
        "collection_issues": issues,
        "collection_pending": pending,
        "trajectory_contributions": ranges,
    }
    return Trajectory.model_validate(result.model_dump(mode="json"))


def reconcile_trial_trajectory(
    agent_dir: Path,
    *,
    model_name: str | None = None,
) -> Trajectory | None:
    """下载/上游 post-convert 后离线重建 canonical ATIF。

    无受管 raw，或仅有正常 session_meta/pending 且无可发布步骤，返回 None。
    意外转换错误发布 partial 并设置 extra.collection_incomplete/collection_issues；
    若没有有效贡献可发布则报错，绝不留下伪装完整的陈旧派生物。I/O 错误不吞掉。
    调用不依赖远端环境，也不会吸收自己的 canonical。
    """
    archive = _RawArchive(agent_dir)
    if not archive.path.exists():
        return None
    _ingest_downloaded_sources(archive, agent_dir)
    issues = [
        {"source": key, "kind": "collection_error", "error": value}
        for key, value in sorted(archive.data["collection_errors"].items())
    ]
    families: dict[str, list[tuple[str, dict, bytes]]] = {}
    for key, entry in sorted(archive.data["sources"].items()):
        raw = (archive.root / entry["raw_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Trajectory raw archive hash mismatch: {key}")
        families.setdefault(entry["family"], []).append((key, entry, raw))
        issues.extend({"source": key, **issue} for issue in entry["issues"])
    contributions = []
    pending: list[dict] = []
    for family, sources in families.items():
        # 直接 ATIF 是派生快照，不得把它与原生贡献再算一次或覆盖整个历史。
        if family == "atif" and len(families) > 1:
            continue
        try:
            if family == "codex":
                contributions.extend(
                    _convert_codex(sources, model_name, issues, pending)
                )
            elif family == "claude-code":
                contributions.extend(
                    _convert_claude(sources, agent_dir, model_name, issues, pending)
                )
            else:
                for key, _, raw in sources:
                    if family == "opencode":
                        events = _source_events(
                            key, raw, issues, pending, allow_banner=True
                        )
                        trajectory = _convert_opencode(raw, agent_dir, model_name)
                        if trajectory is None and not any(
                            event.get("type") == "step_finish" for event in events
                        ):
                            pending.append({"source": key, "kind": "no_steps_yet"})
                            continue
                    else:
                        collector = {
                            "mini-swe": MiniSweTrajectoryCollector,
                            "atif": AtifTrajectoryCollector,
                        }[family]()
                        artifacts = collector.artifacts(
                            raw, archive.data["trial_session_id"]
                        )
                        trajectory = Trajectory.model_validate_json(
                            artifacts["trajectory.json"]
                        )
                    if trajectory is None:
                        raise ValueError(f"Completed history produced no steps: {key}")
                    contributions.append(
                        (
                            trajectory,
                            [key],
                            [trajectory.session_id] if trajectory.session_id else [],
                        )
                    )
        except Exception as exc:  # noqa: BLE001 - 保留其他 family 的有效贡献并显式标错
            issues.append(
                {
                    "family": family,
                    "kind": "conversion_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    archive.data["projection_issues"] = issues
    archive.data["pending_sources"] = pending
    archive.save()
    if not contributions:
        (agent_dir / "trajectory.json").unlink(missing_ok=True)
        if issues:
            raise ValueError(
                f"No convertible trajectory in managed raw archive: {issues}"
            )
        return None
    trajectory = _flatten_trajectories(
        contributions, archive.data["trial_session_id"], issues, pending
    )
    _atomic_write(
        agent_dir / "trajectory.json",
        format_trajectory_json(trajectory.to_json_dict()).encode(),
    )
    if issues:
        _LOGGER.warning("Incomplete trajectory projection: %s", issues)
    return trajectory


class TrajectorySyncMixin:
    """exec-only 环境的全源采集；后台 best-effort，销毁 live home 前支持严格屏障。"""

    if TYPE_CHECKING:
        logger: logging.Logger
        session_id: str
        trial_paths: TrialPaths
        _trajectory_sync_task: asyncio.Task[None] | None
        _trajectory_remote_signatures: dict[str, str]
        _exec_cmd_platform: Callable[..., Awaitable[ExecResult]]

    def _trajectory_collectors(self) -> Sequence[TrajectoryCollector]:
        """声明支持的原生格式；没运行的 agent 没有对应文件。"""
        return (
            MiniSweTrajectoryCollector(),
            OpenCodeTrajectoryCollector(),
            CodexTrajectoryCollector(),
            ClaudeCodeTrajectoryCollector(),
            AtifTrajectoryCollector(),
        )

    async def _resolve_remote_paths(self, pattern: str) -> list[str]:
        """展开全部 glob，不按 mtime 选一个；NUL 分隔避免路径空格歧义。"""
        if "/**/" in pattern:
            root, name = pattern.split("/**/", 1)
            command = (
                f"if [ -d {shlex.quote(root)} ]; then find {shlex.quote(root)} "
                f"-type f -name {shlex.quote(name)} -print0; fi"
            )
        else:
            command = f'for f in {pattern}; do if [ -f "$f" ]; then printf "%s\\0" "$f"; fi; done'
        result = await self._exec_cmd_platform(command, timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError(
                f"Trajectory discovery failed for {pattern}: {result.stderr}"
            )
        # legacy transport 的退出码 sentinel 解析会留下恰好一个传输换行。只移除该
        # 后缀，不 strip 单条路径：原生文件名可以含空格、tab，甚至换行。
        payload = (result.stdout or "").removesuffix("\n")
        if payload in {"", "\0"}:
            return []
        if not payload.endswith("\0"):
            raise RuntimeError(
                f"Invalid NUL trajectory discovery response for {pattern}"
            )
        paths = payload[:-1].split("\0")
        if any(not path.startswith("/") for path in paths):
            raise RuntimeError(
                f"Invalid absolute trajectory source in response for {pattern}"
            )
        return sorted(paths)

    async def _sync_collector_once(
        self, collector: TrajectoryCollector
    ) -> list[Exception]:
        """逐源比对内容签名，先原子归档；某文件失败不阻止其他文件被保全。"""
        archive = _RawArchive(self.trial_paths.agent_dir, self.session_id)
        errors = []
        for pattern in collector.remote_paths:
            try:
                remotes = await self._resolve_remote_paths(pattern)
                archive.collection_error(pattern, None)
            except (
                OSError,
                RuntimeError,
                ValueError,
                subprocess.SubprocessError,
            ) as exc:
                errors.append(exc)
                archive.collection_error(pattern, exc)
                continue
            for remote in remotes:
                try:
                    quoted = shlex.quote(remote)
                    signature_result = await self._exec_cmd_platform(
                        f"sha256sum {quoted}", timeout_sec=30
                    )
                    if signature_result.return_code != 0:
                        raise RuntimeError(
                            f"Trajectory signature failed for {remote}: {signature_result.stderr}"
                        )
                    signature_parts = (signature_result.stdout or "").split()
                    if not signature_parts or len(signature_parts[0]) != 64:
                        raise RuntimeError(f"Invalid trajectory signature for {remote}")
                    signature = signature_parts[0]
                    # 缓存按具体物理路径而非模式；live/final 身份仅在 raw 账本归一。
                    if signature == self._trajectory_remote_signatures.get(remote):
                        archive.collection_error(remote, None)
                        continue
                    encoded = await self._exec_cmd_platform(
                        f"base64 -w0 {quoted}", timeout_sec=30
                    )
                    if encoded.return_code != 0:
                        raise RuntimeError(
                            f"Trajectory transfer failed for {remote}: {encoded.stderr}"
                        )
                    raw = base64.b64decode(
                        (encoded.stdout or "").strip(), validate=True
                    )
                    digest = hashlib.sha256(raw).hexdigest()
                    if digest != signature:
                        # 文件可能在 hash 与读取之间追加；第二次 hash 允许这种正常
                        # 情形。未验证的传输字节独立保全，不伪装成原生 source 截断。
                        after = await self._exec_cmd_platform(
                            f"sha256sum {quoted}", timeout_sec=30
                        )
                        if after.return_code != 0 or (after.stdout or "").split()[
                            :1
                        ] != [digest]:
                            retained = f"unstable/{hashlib.sha256(remote.encode()).hexdigest()}/{digest}"
                            _atomic_write(archive.root / retained, raw)
                            raise RuntimeError(
                                f"Unstable trajectory transfer: {remote}; raw retained at {retained}"
                            )
                    archive.store(
                        collector.family, collector.relative_path(remote), raw, remote
                    )
                    if collector.family in {"opencode", "mini-swe"}:
                        _atomic_write(
                            self.trial_paths.agent_dir / PurePosixPath(remote).name, raw
                        )
                    self._trajectory_remote_signatures[remote] = digest
                    archive.collection_error(remote, None)
                except (
                    OSError,
                    RuntimeError,
                    ValueError,
                    subprocess.SubprocessError,
                ) as exc:
                    errors.append(exc)
                    archive.collection_error(remote, exc)
        return errors

    async def _sync_trajectory_once(self, *, strict: bool = False) -> None:
        """全源保全后统一投影；strict 在发布有效 partial 后显式抛出未解决错误。"""
        # 独立评测环境共享trial目录，但不拥有Agent轨迹；手动finalize也不能污染账本。
        if not getattr(self, '_collect_agent_trajectory', True):
            return
        errors = []
        before = dict(self._trajectory_remote_signatures)
        for collector in self._trajectory_collectors():
            try:
                errors.extend(await self._sync_collector_once(collector))
            except Exception as exc:  # noqa: BLE001 - 先采其余 collector，strict 边界统一抛错
                errors.append(exc)
        if (
            not strict
            and not errors
            and before == self._trajectory_remote_signatures
            and not getattr(self, "_trajectory_projection_needed", True)
        ):
            return
        try:
            trajectory = reconcile_trial_trajectory(self.trial_paths.agent_dir)
            if trajectory is not None and (trajectory.extra or {}).get(
                "collection_incomplete"
            ):
                errors.append(
                    ValueError(
                        f"Incomplete trajectory projection: {trajectory.extra['collection_issues']}"
                    )
                )
        except Exception as exc:  # noqa: BLE001 - live 记录问题，strict 必须失败
            errors.append(exc)
        self._trajectory_projection_needed = bool(errors)
        if errors:
            if strict:
                raise errors[0]
            self.logger.warning("trajectory collection incomplete: %s", errors)

    async def _sync_trajectory_loop(self) -> None:
        """环境存活期间尽力采集；取消只终止本地任务，不创建远端常驻进程。"""
        while True:
            try:
                await self._sync_trajectory_once()
            except Exception:
                self.logger.exception("trajectory sync failed")
            await asyncio.sleep(_TRAJECTORY_SYNC_INTERVAL_SEC)

    async def _stop_trajectory_sync(self) -> None:
        """取消并 await 轮询器，确保之后不会晚写覆盖 final projection。"""
        task = self._trajectory_sync_task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - 不让旧 poller 异常跳过末次保全
            self.logger.warning("trajectory sync task ended with error: %s", exc)
        finally:
            if self._trajectory_sync_task is task:
                self._trajectory_sync_task = None

    async def finalize_trajectory_sync(self, *, strict: bool = False) -> None:
        """停止并回收 poller 后末次采集；strict 拒绝未解决归档/转换错误。

        正常 pending/尾部半行仅记录状态；重复调用只替换贡献，不追加历史。
        """
        if not hasattr(self, "_trajectory_finalization_lock"):
            self._trajectory_finalization_lock = asyncio.Lock()
        async with self._trajectory_finalization_lock:
            await self._stop_trajectory_sync()
            self._trajectory_projection_needed = True  # final 必须修复上游可能的覆盖
            await self._sync_trajectory_once(strict=strict)

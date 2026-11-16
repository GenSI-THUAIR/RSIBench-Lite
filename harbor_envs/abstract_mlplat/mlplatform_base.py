"""Shared capabilities for ML-platform (k8s-like) Harbor environments.

An ML-platform environment is one whose pods are reachable only through a
platform-provided exec channel: no mounted filesystem, no direct SSH
contract, no kubectl. Everything here is implementable on top of a single
primitive -- :meth:`MLPlatformEnvironment._exec_cmd_platform` -- and is therefore
platform-agnostic:

* Chunked base64 file transfer over the exec channel, sized for the most
  restrictive known backend command limits.
* Cluster readiness: block until the accelerator cluster reaches the
  topology the platform job itself declared, via a single-shot Ray probe.
* Workdir locking helpers that stop two concurrent runs from destroying
  each other's trial state.
* Native trajectory synchronization and finalization shared through
  TrajectorySyncMixin, started and stopped by this lifecycle.

Concrete providers may replace the exec-based transfer with a native SDK
(Modal reuses Harbor's upstream implementation) while retaining this lifecycle.

Platform-specific concerns -- job lifecycle, pod naming, transport CLIs,
their quirks and sentinels -- belong in the concrete subclass packages
(:mod:`harbor_envs.modal`).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import gzip
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
from enum import Enum
from abc import abstractmethod
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.environments.base import BaseEnvironment, ExecResult, OutputStream
from harbor.models.task.config import EnvironmentConfig

from harbor_envs.abstract_mlplat.retry import TransientError, retry_async
from harbor_envs.abstract_mlplat.trajectory import TrajectorySyncMixin


_DOWNLOAD_CLEANUP_TIMEOUT_SEC = 2.0

# Maximum number of *raw* bytes in one base64 upload chunk. The most
# restrictive observed backend (legacy exec transport) executes 2048-byte base64
# payloads but may only echo 4096-byte ones, so keep the encoded payload at
# 2048 bytes.
_UPLOAD_CHUNK_BYTES = 1536
# SSH 级通道每条命令可以携带大得多的负载,但受 Linux MAX_ARG_STRLEN
# (128KiB/单参数)限制;64KiB 原始块 -> ~87KiB base64 -> 命令总长 <128KiB。
# 启动时探测一次:探测块写入失真(WebShell 截断/回显)则退回保守块。
_FAST_UPLOAD_CHUNK_BYTES = int(
    os.environ.get("HARBOR_UPLOAD_FAST_CHUNK_BYTES", str(64 * 1024))
)

# Remote staging area used for base64 upload staging.
_REMOTE_STAGE_DIR = PurePosixPath("/tmp")

# 实测:4 节点 job 的 sshd 在 job 创建后约 10 分钟才就绪,300s 必超时。
# env 变量名保留 HARBOR_MODAL_ 前缀以兼容既有部署。
_EXEC_READY_TIMEOUT_SEC = int(
    os.environ.get("HARBOR_MODAL_EXEC_READY_TIMEOUT_SEC", "1800")
)
_EXEC_READY_POLL_INTERVAL_SEC = 10

# 工作区锁超过该年龄视为遗弃(崩溃的 run 留下的),可被回收。
_WORKDIR_LOCK_STALE_SEC = 6 * 3600

# 集群组建发生在 exec 通道就绪之后。上限 4h 而不是与启动窗口一致的 1200s:
# 多节点题要等**全部**节点凑齐,池子紧张时这是排队时长而非故障(0903 实录:
# glm52 的 4x16 Ascend 拓扑等到 1200s 只到 3/4 节点、每节点 16 卡都齐,被判
# 超时白丢已拿到的三个节点与队列位置)。等待期每 30s 打一行进度,长等可见。
_CLUSTER_READY_TIMEOUT_SEC = int(
    os.environ.get("HARBOR_MODAL_CLUSTER_READY_TIMEOUT_SEC", str(4 * 3600))
)
_CLUSTER_READY_POLL_INTERVAL_SEC = 30
_CLUSTER_PROBE_TIMEOUT_SEC = 120
# 在 head pod 上对 Ray 集群拓扑做一次性探测,输出恰好一行 JSON。
# 解释器候选:系统 python3 + /usr/local 下的全部 python3 发行版(glob,
# 覆盖 glm52 的 3.11.15、dsv4 的 3.12.13 等,不再逐镜像写死),探测逻辑
# 本身与任务无关。
_CLUSTER_PROBE_SCRIPT = """
for py in python3 /usr/local/python3*/bin/python3; do
  if command -v "$py" >/dev/null 2>&1 && "$py" -c 'import ray' >/dev/null 2>&1; then
    exec "$py" - <<'PROBE_EOF'
import json
import os

import ray

# Ray 自己写的集群标记文件是最权威的地址来源:格式正确(IPv6 带方括号)
# 且随集群重启更新。env 只作回退,并且裸 IPv6 必须补方括号——不加括号的
# "v6:port" 拼接 gRPC 无法解析,实测表现为 GCS 永远 reconnect 失败。
address = ""
for marker in (
    "/tmp/ray/ray_current_cluster",
    "/tmp/ray/ray_current_cluster",
):
    try:
        with open(marker, encoding="utf-8") as stream:
            value = stream.read().strip()
    except OSError:
        continue
    if value:
        address = value
        break
if not address:
    ip = os.environ.get("RAY_IP") or ""
    port = os.environ.get("RAY_SERVE_PORT") or os.environ.get("PORT5") or ""
    if ip and port:
        if ":" in ip and not ip.startswith("["):
            ip = f"[{ip}]"
        address = f"{ip}:{port}"
if not address:
    address = "auto"
try:
    ray.init(
        address=address,
        namespace="harbor-modal-readiness",
        logging_level="ERROR",
    )
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    devices = []
    for node in nodes:
        resources = node.get("Resources") or {}
        devices.append(int(resources.get("NPU") or resources.get("GPU") or 0))
    print(
        json.dumps(
            {
                "nodes_alive": len(nodes),
                "min_devices": min(devices) if devices else 0,
            }
        )
    )
finally:
    ray.shutdown()
PROBE_EOF
  fi
done
echo '{"probe_error": "no python interpreter with ray available"}'
"""


class MLPlatformEnvironment(TrajectorySyncMixin, BaseEnvironment):
    """Base class for exec-channel-only ML platform environments."""

    # 由具体平台的 __init__ 提供:用于工作区锁的 "<session>:<pid>" 归属标签。
    _session_tag: str

    def __init__(self, *args, environment_dir, task_env_config, profile_file=None,
                 install_task_skills=False, collect_agent_trajectory=True, **kwargs):
        if type(collect_agent_trajectory) is not bool:
            raise TypeError('collect_agent_trajectory must be a boolean')
        # 同一任务包可声明平台配置；只替换本实例，不修改共享Task或源文件。
        if profile_file is not None:
            profile_file = (Path(environment_dir) / profile_file).resolve()
            profiles = json.loads(profile_file.read_text())
            task_env_config = EnvironmentConfig.model_validate(profiles[Path(environment_dir).name])
        self._profile_file = profile_file
        self._install_registered_skills = install_task_skills
        self._trajectory_sync_task: asyncio.Task[None] | None = None
        self._trajectory_remote_signatures: dict[str, str] = {}
        super().__init__(*args, environment_dir=environment_dir,
                         task_env_config=task_env_config, **kwargs)
        # Harbor的独立verifier实例使用公开的<trial>__verifier__<key>身份约定。
        # 它与Agent共享TrialPaths，但不能采集/覆盖Agent的轨迹账本。
        verifier_prefix = f'{self.trial_paths.trial_dir.name}__verifier__'
        self._collect_agent_trajectory = (
            collect_agent_trajectory and not self.session_id.startswith(verifier_prefix)
        )

    # ------------------------------------------------------------------
    # Platform primitives and hooks
    # ------------------------------------------------------------------

    @property
    def platform_name(self) -> str:
        kind = self.type()
        return kind.value if isinstance(kind, Enum) else kind

    @abstractmethod
    def create_evaluation_environment(self, *, task, trial_paths, logger,
                                      profile=None, mode='score', runtime=None, scoring=None):
        """按本平台配置创建独立评测服务；不在此处申请资源。"""

    @abstractmethod
    async def service_endpoint(self, port: int) -> str:
        """返回独立客户端可访问的服务地址；地址族/隧道由平台负责。"""

    async def service_addresses(self, port: int, *, pod_name: str | None = None):
        raise NotImplementedError(f'{self.platform_name} does not expose cluster service addresses')

    async def exec_baseline_probe(self, command: str, *, timeout_sec: int, logger) -> ExecResult:
        return await self.exec(command,cwd='/tmp',timeout_sec=timeout_sec)

    def assert_reference_isolation(self, *, task, benchmark):
        raise NotImplementedError(f'{self.platform_name} does not implement reference mount isolation')

    @abstractmethod
    async def _exec_cmd_platform(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        pod_name: str | None = None,
        on_output: Callable[[str, OutputStream], Awaitable[None]] | None = None,
    ) -> ExecResult:
        """Run *command* on the platform's target pod and return the result."""

    @abstractmethod
    async def _ensure_job(self) -> None:
        """Create the platform job unless one was supplied externally."""

    @abstractmethod
    async def _stop_owned_job(self) -> None:
        """Stop a job this environment created; never touch external jobs."""

    async def _start_cluster(self) -> None:
        """Trigger cluster startup for jobs that declare one. Default: no-op."""

    async def _on_environment_started(self) -> None:
        """Hook after trial directories are reset, before start() returns."""

    async def _on_environment_stopping(self) -> None:
        """Hook at the beginning of stop(), before any cleanup."""

    async def attach(self) -> None:
        """Open an interactive shell on the job. Optional per platform."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support interactive attach"
        )

    def _expected_cluster_topology(self) -> tuple[int, int]:
        """(期望节点数, 每节点最少加速卡数);(0, 0) 表示无需等待集群组建。

        具体平台从自己的 job 声明(而不是 task 特化配置)推导这两个值。
        """
        return 0, 0

    def _job_label(self) -> str:
        """人类可读的 job 标识,用于就绪等待的日志与错误信息。"""
        return self.environment_name

    def _requires_workdir_lock(self) -> bool:
        """是否需要防止多个 trial 并发复用同一个远端环境。"""
        return True

    def _should_reset_workdir(self) -> bool:
        """是否在 trial 开始时清空工作目录。"""
        return True

    # ------------------------------------------------------------------
    # Workdir locking helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _workdir_lock_path(workdir: str) -> str:
        """返回不受 workdir 权限和清理影响的稳定锁路径。"""
        digest = hashlib.sha256(workdir.encode()).hexdigest()[:16]
        return f"/tmp/.harbor-workdir-{digest}.lock"

    @staticmethod
    def _owner_process_is_gone(owner: str) -> bool:
        """Whether a lock owner tag names a launcher process that no longer exists.

        Tags look like ``<session>:<pid>``. Only a definite answer counts as
        abandoned: an unparseable tag, or a pid that still exists, keeps the lock.
        """
        pid_text = owner.strip().rsplit(":", 1)[-1]
        if not pid_text.isdigit():
            return False
        pid = int(pid_text)
        if pid <= 0 or pid == os.getpid():
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            # Alive, owned by someone else.
            return False
        except OSError:
            return False
        return False

    # ------------------------------------------------------------------
    # Cluster readiness
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_cluster_probe(stdout: str | None) -> dict[str, Any] | None:
        """取探测输出中最后一行合法 JSON object;没有则返回 None。"""
        for line in reversed((stdout or "").splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    async def _wait_for_cluster_ready(self) -> None:
        """等待集群达到 job 声明的拓扑后再把环境交给调用方。

        平台的集群触发只负责启动各 pod 的本地进程;节点加入集群仍需数分钟。
        环境的契约是 start() 返回即"可用",所以在这里按声明的节点数/卡数
        轮询,达标才返回;超时抛错交给外层 teardown,而不是把半成品集群
        交给 agent。
        """
        expected_nodes, expected_devices = self._expected_cluster_topology()
        if expected_nodes == 0:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _CLUSTER_READY_TIMEOUT_SEC
        attempt = 0
        last_detail = "cluster probe produced no output"
        while True:
            attempt += 1
            try:
                result = await self._exec_cmd_platform(
                    _CLUSTER_PROBE_SCRIPT, timeout_sec=_CLUSTER_PROBE_TIMEOUT_SEC
                )
            except subprocess.TimeoutExpired as exc:
                last_detail = f"cluster probe timed out after {exc.timeout}s"
            else:
                probe = self._parse_cluster_probe(result.stdout)
                if probe is None:
                    last_detail = (
                        result.stderr or result.stdout or "no probe output"
                    ).strip()[-500:]
                elif "probe_error" in probe:
                    last_detail = str(probe["probe_error"])[:500]
                else:
                    nodes_alive = int(probe.get("nodes_alive", 0) or 0)
                    min_devices = int(probe.get("min_devices", 0) or 0)
                    if nodes_alive >= expected_nodes and (
                        expected_devices == 0 or min_devices >= expected_devices
                    ):
                        self.logger.info(
                            "Cluster is ready for %s: "
                            "%d/%d nodes alive, %d/%d devices per node",
                            self._job_label(),
                            nodes_alive,
                            expected_nodes,
                            min_devices,
                            expected_devices,
                        )
                        return
                    last_detail = (
                        f"nodes_alive={nodes_alive}/{expected_nodes} "
                        f"min_devices={min_devices}/{expected_devices}"
                    )
            # 复用平台终态钩子；job 已停止时不再把 SSH 失败当暂态等满数小时。
            if reason := await self._exec_readiness_abort_reason():
                raise RuntimeError(f"{self._job_label()} cluster readiness aborted: {reason}")
            if loop.time() >= deadline:
                raise RuntimeError(
                    f"{self._job_label()} did not reach its declared "
                    f"topology ({expected_nodes} nodes x {expected_devices} "
                    f"devices) within {_CLUSTER_READY_TIMEOUT_SEC}s; "
                    f"last probe: {last_detail}"
                )
            self.logger.info(
                "Waiting for cluster on %s (attempt %d): %s",
                self._job_label(),
                attempt,
                last_detail,
            )
            await asyncio.sleep(_CLUSTER_READY_POLL_INTERVAL_SEC)

    # ------------------------------------------------------------------
    # Exec readiness
    # ------------------------------------------------------------------

    async def _refresh_exec_target(self) -> None:
        """就绪探测失败后给平台层一次重解析目标 pod 的机会(默认无事可做)。"""
        return None

    async def _exec_readiness_abort_reason(self) -> str | None:
        """Return a platform terminal-state reason, or ``None`` to keep polling."""
        return await self.terminal_state_reason()

    async def terminal_state_reason(self) -> str | None:
        """平台可报告明确终态；缺少权威状态时返回None，不从观察超时推断停止。"""
        return None

    async def _wait_for_exec(self) -> None:
        """等待目标 pod 的 exec 通道就绪，并输出每次检查结果。"""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _EXEC_READY_TIMEOUT_SEC
        attempt = 0
        last_error = "exec probe returned no detail"

        while True:
            attempt += 1
            self.logger.info(
                "Checking exec readiness for %s (attempt %d)",
                self._job_label(),
                attempt,
            )
            try:
                result = await self._exec_cmd_platform(
                    "printf mlplatform-exec-ready", timeout_sec=15
                )
            except subprocess.TimeoutExpired as exc:
                last_error = f"exec probe timed out after {exc.timeout}s"
            else:
                if result.return_code == 0 and "mlplatform-exec-ready" in (
                    result.stdout or ""
                ):
                    self.logger.info("Exec channel is ready for %s", self._job_label())
                    return
                detail = (result.stderr or result.stdout or "").strip()
                last_error = detail[-500:] or f"exec probe exited {result.return_code}"

            await self._refresh_exec_target()
            abort_reason = await self._exec_readiness_abort_reason()
            if abort_reason is not None:
                raise RuntimeError(
                    f"Exec channel cannot become ready for {self._job_label()}: "
                    f"{abort_reason}"
                )

            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            delay = min(_EXEC_READY_POLL_INTERVAL_SEC, remaining)
            self.logger.info(
                "Exec channel is not ready for %s; retrying in %.0fs: %s",
                self._job_label(),
                delay,
                last_error,
            )
            await asyncio.sleep(delay)

        raise RuntimeError(
            f"Exec channel did not become ready within {_EXEC_READY_TIMEOUT_SEC}s "
            f"for {self._job_label()}: {last_error}"
        )

    # ------------------------------------------------------------------
    # Workdir locking
    # ------------------------------------------------------------------

    async def _acquire_workdir_lock(self, workdir: str) -> None:
        """Take an exclusive lock on the pod's workdir.

        Two concurrent runs against the same job silently destroy each other:
        start() empties the workdir, so the later run deletes files the
        earlier one is still writing. Failing loudly is far better than
        losing data quietly.
        """
        lock_path = self._workdir_lock_path(workdir)
        probe_script = (
            "set -e\n"
            f"mkdir -p $(dirname {shlex.quote(lock_path)})\n"
            # Remove an abandoned lock so a crashed run cannot block the pod.
            f"if [ -f {shlex.quote(lock_path)} ]; then\n"
            f"  age=$(( $(date +%s) - $(stat -c %Y {shlex.quote(lock_path)}) ))\n"
            f'  [ "$age" -gt {_WORKDIR_LOCK_STALE_SEC} ] '
            f"&& rm -f {shlex.quote(lock_path)}\n"
            "fi\n"
            # noclobber makes this an atomic test-and-set.
            f"set -C; echo {shlex.quote(self._session_tag)} > {shlex.quote(lock_path)} "
            f"&& echo ACQUIRED || cat {shlex.quote(lock_path)}"
        )

        async def _probe() -> str:
            result = await self._exec_cmd_platform(probe_script, timeout_sec=60)
            stdout = (result.stdout or "").strip()
            # 空输出 + 非零 rc = 传输层抖动,不是锁判定(v15 实录:全新 pod
            # 第一次探测空返回被误判成 "owner: ?" 冲突,整个 trial 报废)。
            if not stdout and result.return_code != 0:
                raise TransientError(
                    (result.stderr or "")[-300:] or "empty lock probe reply"
                )
            return stdout

        out = await retry_async(
            _probe,
            what=f"probe workdir lock on {self._job_label()}",
            attempts=3,
            delay_sec=10.0,
            backoff=1.0,
            retry_on=(TransientError,),
            logger=self.logger,
        )
        # 前一次探测可能已写入锁但回复丢失:锁主是自己即视为已持有。
        if self._session_tag in out:
            out = "ACQUIRED"
        if "ACQUIRED" not in out:
            owner = out.splitlines()[-1] if out else "?"
            # The staleness rule above is a backstop, not a recovery path: a run
            # killed mid-flight leaves the lock behind and blocks the pod for
            # hours, which then reads as a broken environment. The owner tag ends
            # in the launcher's pid and the lock can only be taken from a
            # launcher, so when that pid is gone from *this* host the lock is
            # provably abandoned.
            if self._owner_process_is_gone(owner):
                self.logger.warning(
                    "Reclaiming %s from dead owner %s", lock_path, owner
                )
                retry = await self._exec_cmd_platform(
                    f"rm -f {shlex.quote(lock_path)}\n"
                    f"set -C; echo {shlex.quote(self._session_tag)} "
                    f"> {shlex.quote(lock_path)} "
                    f"&& echo ACQUIRED || cat {shlex.quote(lock_path)}",
                    timeout_sec=60,
                )
                out = (retry.stdout or "").strip()
            if "ACQUIRED" not in out:
                raise RuntimeError(
                    f"another run holds {lock_path} (owner: {owner}). "
                    f"Concurrent runs against {self._job_label()} would delete "
                    f"each other's files; wait for it to finish or remove the lock."
                )

    async def _release_workdir_lock(self, workdir: str) -> None:
        """Release the workdir lock, but only if it is still ours."""
        lock_path = self._workdir_lock_path(workdir)
        try:
            await self._exec_cmd_platform(
                f"if [ -f {shlex.quote(lock_path)} ] && "
                f"grep -qF {shlex.quote(self._session_tag)} {shlex.quote(lock_path)}; "
                f"then rm -f {shlex.quote(lock_path)}; fi; true",
                timeout_sec=20,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError, TimeoutError) as exc:
            self.logger.debug("best-effort lock release skipped: %s", exc)

    # ------------------------------------------------------------------
    # Task skill injection
    # ------------------------------------------------------------------

    def _task_name(self) -> str:
        """The task this trial runs, derived from Harbor's trial dir naming."""
        trial_name = self.trial_paths.trial_dir.name
        return trial_name.rsplit("__", 1)[0] if "__" in trial_name else ""

    async def _install_task_skills(self) -> None:
        if self._install_registered_skills:
            raise ValueError("Use Harbor's explicit skills configuration; implicit registries are unsupported")

    async def start(self, force_build: bool) -> None:
        """Create or connect to a platform job and prepare work dirs.

        To prevent cross-trial state leakage (a failed agent getting credit
        for a previous trial's artifacts), trial-scoped directories are
        emptied at the start of every trial. Files and processes outside
        those directories are left untouched.
        """
        del force_build  # 平台镜像由平台自己构建,Harbor 不参与。
        await self._ensure_job()
        try:
            await self._wait_for_exec()
        except BaseException:
            # _ensure_job may have created a billable job. Exec readiness is
            # still part of start(), so terminal state, timeout, or caller
            # cancellation must roll it back. External jobs are untouched by
            # the platform-specific _stop_owned_job implementation.
            await self._stop_owned_job()
            raise
        await self._start_cluster()
        # Block until the cluster matches the topology the job itself
        # declared. start() returning means the environment is usable;
        # agents must not have to poll for provisioning.
        await self._wait_for_cluster_ready()

        workdir = self.task_env_config.workdir or "/app"
        initial_dirs = [
            "/logs/agent",
            "/logs/verifier",
            "/logs/artifacts",
            "/tests",
            "/solution",
            workdir,
        ]
        # Do not call BaseEnvironment.ensure_dirs here: it dispatches through
        # self.exec(), whose default cwd is the task workdir. A freshly created
        # job does not have that directory yet, so `cd` would fail before
        # mkdir ever runs. _exec_cmd_platform has no implicit cwd and can bootstrap it.
        async def _initialize_dirs() -> None:
            initialized = await self._exec_cmd_platform(
                self._ensure_dirs_command(initial_dirs, chmod=True),
                user="root",
                timeout_sec=60,
            )
            if initialized.return_code != 0:
                raise TransientError(
                    initialized.stderr or initialized.stdout or "empty exec reply"
                )

        # exec-ready 刚成功后首条 SSH 命令仍可能瞬时失败。目录创建幂等，
        # 只对明确的非零执行结果做有限重试，不放宽后续业务命令的失败语义。
        await retry_async(
            _initialize_dirs,
            what=f"initialize Harbor directories on {self._job_label()}",
            attempts=3,
            delay_sec=10.0,
            backoff=1.0,
            retry_on=(TransientError,),
            logger=self.logger,
        )

        if self._requires_workdir_lock():
            await self._acquire_workdir_lock(workdir)

        # Reset trial-scoped directories so a previous trial's artifacts
        # can't cause a false-positive reward. /tests and /solution are
        # re-uploaded by Harbor each trial; /logs/* and the workdir are the
        # agent/verifier working areas.
        reset_dirs = [
            "/logs/agent",
            "/logs/verifier",
            "/logs/artifacts",
            "/tests",
            "/solution",
        ]
        if self._should_reset_workdir():
            reset_dirs.append(workdir)
        await self.empty_dirs(reset_dirs, chmod=False)

        if self._install_registered_skills:
            await self._install_task_skills()

        await self._on_environment_started()
        if self._collect_agent_trajectory:
            self._trajectory_sync_task = asyncio.create_task(
                self._sync_trajectory_loop(),
                name=f"mlplatform-trajectory-sync-{self.session_id}",
            )
        self.logger.info(
            "Environment started for %s; trial directories reset.",
            self._job_label(),
        )

    async def stop(self, delete: bool) -> None:
        """Clean up trial artifacts and stop a job owned by this environment."""
        del delete  # 平台 job 的删除语义由 _stop_owned_job 决定。
        await self._stop_trajectory_sync()
        await self._on_environment_stopping()
        # Best-effort cleanup of files staged under /tmp.
        try:
            await self._exec_cmd_platform(
                "rm -f /tmp/.hb-transfer-* /tmp/upload-*.b64 2>/dev/null; true",
                timeout_sec=15,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError, TimeoutError) as exc:
            self.logger.debug("best-effort transfer cleanup skipped: %s", exc)

        workdir = self.task_env_config.workdir or "/app"
        if self._requires_workdir_lock():
            await self._release_workdir_lock(workdir)
        self.logger.info("Environment stopped for %s.", self._job_label())
        await self._stop_owned_job()

    # ------------------------------------------------------------------
    # File transfer over the exec channel
    # ------------------------------------------------------------------

    async def _resolve_upload_chunk_bytes(self) -> int:
        """Probe once whether the exec channel carries large single commands.

        WebShell 会截断/回显超过 ~2KiB 的命令;SSH 通道则受 MAX_ARG_STRLEN
        约束但足够容纳 64KiB 块。以实际写入字节数为准做一次探测并缓存。
        """
        cached = getattr(self, "_upload_chunk_bytes_cache", None)
        if cached:
            return cached
        chosen = _UPLOAD_CHUNK_BYTES
        if _FAST_UPLOAD_CHUNK_BYTES > _UPLOAD_CHUNK_BYTES:
            probe_payload = "A" * (_FAST_UPLOAD_CHUNK_BYTES * 4 // 3)
            probe_path = str(_REMOTE_STAGE_DIR / f".hb-chunk-probe-{uuid.uuid4().hex}")
            try:
                await self._exec_cmd_platform(
                    f"printf '%s' {shlex.quote(probe_payload)} "
                    f"> {shlex.quote(probe_path)}",
                    timeout_sec=60,
                )
                size_result = await self._exec_cmd_platform(
                    f"wc -c < {shlex.quote(probe_path)}; rm -f {shlex.quote(probe_path)}",
                    timeout_sec=30,
                )
                fields = (size_result.stdout or "").split()
                if (
                    fields
                    and fields[0].isdigit()
                    and int(fields[0]) == len(probe_payload)
                ):
                    chosen = _FAST_UPLOAD_CHUNK_BYTES
            except Exception:  # noqa: BLE001 - 探测失败一律用保守块
                chosen = _UPLOAD_CHUNK_BYTES
        self._upload_chunk_bytes_cache = chosen
        self.logger.info("Upload chunk size resolved to %d bytes", chosen)
        return chosen

    async def upload_file(
        self,
        source_path: Path | str,
        target_path: str,
        pod_name: str | None = None,
    ) -> None:
        """通过分块 base64 上传文件，并以远端 SHA256 作为完成条件。

        `pod_name` 让多节点服务把同一份 patch / config 送到每个 pod:/logs 是
        pod 本地的,只传 head 的话其它 rank 起 controller 时会找不到文件。
        """
        source = Path(source_path)
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")

        data = source.read_bytes()
        expected_sha256 = hashlib.sha256(data).hexdigest()
        if not data:
            result = await self._exec_cmd_platform(
                f"mkdir -p {shlex.quote(str(PurePosixPath(target_path).parent))} "
                f"&& : > {shlex.quote(target_path)} "
                f"&& sha256sum {shlex.quote(target_path)} | cut -d' ' -f1",
                timeout_sec=15,
                pod_name=pod_name,
            )
            if expected_sha256 not in (result.stdout or "").split():
                raise RuntimeError(f"Failed to upload empty file {target_path}")
            return

        b64 = base64.b64encode(data).decode("ascii")
        stage_b64 = str(_REMOTE_STAGE_DIR / f"upload-{uuid.uuid4().hex}.b64")
        stage_parts = f"{stage_b64}.parts"
        remote_tmp = f"{target_path}.hb-upload-{uuid.uuid4().hex}"

        # Write the base64 text in chunks, then decode.  Using printf
        # (not echo) avoids adding newlines between chunks.
        chunk_bytes = await self._resolve_upload_chunk_bytes()
        chunk_size = chunk_bytes * 4 // 3  # base64 chars per chunk
        total_parts = (len(b64) + chunk_size - 1) // chunk_size
        try:
            for part_index, i in enumerate(range(0, len(b64), chunk_size)):
                if part_index and part_index % 25 == 0:
                    self.logger.info(
                        "Uploading %s: part %d/%d",
                        target_path,
                        part_index,
                        total_parts,
                    )
                chunk = b64[i : i + chunk_size]
                part_path = f"{stage_parts}/{part_index:06d}"
                # Use a heredoc-less approach: printf with the chunk as
                # a single argument.  shlex.quote handles any special
                # chars (base64 only has [A-Za-z0-9+/=] anyway).
                cmd = (
                    f"mkdir -p {shlex.quote(stage_parts)} && "
                    f"printf '%s' {shlex.quote(chunk)} "
                    f"> {shlex.quote(part_path)}"
                )

                async def _write_chunk(
                    cmd: str = cmd, part_path: str = part_path, chunk: str = chunk
                ) -> None:
                    await self._exec_cmd_platform(
                        cmd, timeout_sec=30, pod_name=pod_name
                    )
                    size_result = await self._exec_cmd_platform(
                        f"wc -c < {shlex.quote(part_path)}",
                        timeout_sec=15,
                        pod_name=pod_name,
                    )
                    if str(len(chunk)) not in (size_result.stdout or "").split():
                        raise TransientError(
                            f"expected {len(chunk)} base64 bytes"
                        )

                await retry_async(
                    _write_chunk,
                    what=f"stage upload chunk for {target_path}",
                    attempts=10,
                    delay_sec=2.0,
                    backoff=1.0,
                    retry_on=(TransientError,),
                )

            target_parent = str(PurePosixPath(target_path).parent)
            # 独有目录中的补零序号保持顺序；不要让拼装命令随分块数无限增长。
            # 逐块大小和最终SHA仍验证完整性，额外/缺失分块不能静默通过。
            await self._exec_cmd_platform(
                f"mkdir -p {shlex.quote(target_parent)} "
                f"&& cat {shlex.quote(stage_parts)}/[0-9]* "
                f"| base64 -d > {shlex.quote(remote_tmp)} "
                f"&& mv {shlex.quote(remote_tmp)} {shlex.quote(target_path)}",
                timeout_sec=60,
                pod_name=pod_name,
            )
            async def _verify_checksum() -> None:
                checksum_result = await self._exec_cmd_platform(
                    f"sha256sum {shlex.quote(target_path)} | cut -d' ' -f1",
                    timeout_sec=15,
                pod_name=pod_name,
                )
                if expected_sha256 not in (checksum_result.stdout or "").split():
                    raise TransientError(
                        checksum_result.stderr or checksum_result.stdout or ""
                    )

            await retry_async(
                _verify_checksum,
                what=f"verify uploaded file {target_path}",
                attempts=10,
                delay_sec=2.0,
                backoff=1.0,
                retry_on=(TransientError,),
            )

            # Base64 staging creates a fresh regular file and therefore drops
            # executable bits. Preserve executability explicitly so uploaded
            # entrypoints do not depend on a separate, one-shot chmod command.
            if source.stat().st_mode & 0o111:

                async def _preserve_executable() -> None:
                    executable = await self._exec_cmd_platform(
                        f"chmod +x {shlex.quote(target_path)} "
                        f"&& test -x {shlex.quote(target_path)}",
                        timeout_sec=15,
                        pod_name=pod_name,
                    )
                    if executable.return_code != 0:
                        raise TransientError(
                            executable.stderr or executable.stdout or ""
                        )

                await retry_async(
                    _preserve_executable,
                    what=f"preserve executable mode for {target_path}",
                    attempts=3,
                    delay_sec=2.0,
                    backoff=1.0,
                    retry_on=(TransientError,),
                )
        finally:
            # 暂存清理是 best-effort:上传本体已用远端 SHA256 判定成功,
            # 瞬时 CLI 超时不得覆盖该结果(LLaDA r1-0825:清理 45s
            # TimeoutExpired 逃逸,毁掉已完成的上传)。只忽略明确的超时。
            try:
                await self._exec_cmd_platform(
                    f"rm -rf {shlex.quote(stage_parts)} "
                    f"{shlex.quote(stage_b64)} {shlex.quote(remote_tmp)}",
                    timeout_sec=15,
                    pod_name=pod_name,
                )
            except subprocess.TimeoutExpired:
                self.logger.warning(
                    "上传暂存清理超时，跳过 best-effort 清理：%s", target_path
                )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Download a pod file to the host: gzip once, then chunked base64.

        WebShell 对大输出稳定超时(iter-v30/v31:MB 级 gsm8k-result.json 的
        单发 base64 三次重试全部 2min 超时,毁掉已跑完的 verifier)。改为
        先在 pod 上 gzip 到暂存文件,再按 256KiB 块 dd+base64 拉取,单次
        exec 输出有界;块级重试,stdout 只取合法 base64 的最长行。
        下载专用命令以 root 读取 agent 的私有文件及暂存块，不改源权限或平台默认身份。
        """
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        stage = f"/tmp/.harbor-dl-{uuid.uuid4().hex}.gz"
        chunk_bytes = 256 * 1024

        async def _retry_exec(command: str, *, what: str) -> str:
            async def _op() -> str:
                result = await self._exec_cmd_platform(command, timeout_sec=300, user="root")
                if result.return_code != 0:
                    raise TransientError(
                        (result.stderr or result.stdout or "")[-500:]
                    )
                return result.stdout or ""

            return await retry_async(
                _op,
                what=f"{what} for {source_path}",
                attempts=3,
                delay_sec=10.0,
                backoff=2.0,
                retry_on=(TransientError,),
                logger=self.logger,
            )

        try:
            stdout = await _retry_exec(
                f"gzip -c {shlex.quote(source_path)} > {stage} && "
                f"stat -c %s {stage}",
                what="stage gzip",
            )
            size = None
            for line in reversed(stdout.splitlines()):
                if line.strip().isdigit():
                    size = int(line.strip())
                    break
            if size is None:
                raise RuntimeError(
                    f"Failed to size staged download for {source_path}: "
                    f"{stdout[-300:]!r}"
                )
            # 压缩块顺序写入临时文件；解压完整并通过 gzip CRC 后再原子替换。
            # 大日志不在内存保留两份，也不能破坏已存在的有效目标文件。
            with tempfile.TemporaryFile(dir=target.parent) as compressed:
                for index in range((size + chunk_bytes - 1) // chunk_bytes):
                    stdout = await _retry_exec(
                        f"dd if={stage} bs={chunk_bytes} skip={index} count=1 "
                        "2>/dev/null | base64 -w0",
                        what=f"download chunk {index}",
                    )
                    candidates = [line.strip() for line in stdout.splitlines()
                                  if line.strip() and re.fullmatch(r"[A-Za-z0-9+/=]+", line.strip())]
                    if not candidates:
                        raise RuntimeError(f"Failed to download {source_path}: chunk {index} returned no base64 payload")
                    compressed.write(base64.b64decode(max(candidates, key=len), validate=True))
                if compressed.tell() != size:
                    raise RuntimeError(f"Failed to download {source_path}: reassembled {compressed.tell()} bytes, expected {size}")
                compressed.seek(0)
                temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
                try:
                    with gzip.GzipFile(fileobj=compressed) as decoded, temporary.open("wb") as output:
                        shutil.copyfileobj(decoded, output, length=1024 * 1024)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
        finally:
            # The payload checksum and local write above define download
            # success.  Cleanup is deliberately best-effort: a pod can become
            # unreachable immediately after the file has been transferred, and
            # an rm timeout must not turn a valid artifact into a transport
            # failure (or mask the original exception).
            # 已在取消流程中不再等待远端；绝不吞掉外层 CancelledError。
            if not asyncio.current_task().cancelling():
                try:
                    async with asyncio.timeout(_DOWNLOAD_CLEANUP_TIMEOUT_SEC):
                        cleanup = await self._exec_cmd_platform(
                            f"rm -f {stage}", timeout_sec=_DOWNLOAD_CLEANUP_TIMEOUT_SEC, user="root"
                        )
                    if cleanup.return_code != 0:
                        self.logger.warning("download cleanup failed for %s: %s", source_path, cleanup.stderr)
                except (OSError, RuntimeError, subprocess.SubprocessError, TimeoutError) as exc:
                    self.logger.warning("download staging cleanup skipped for %s: %s", source_path, exc)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """上传目录；平铺文件逐个上传，含子目录时使用 tar。"""
        source = Path(source_dir)
        if not source.exists() or not source.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source}")

        children = sorted(source.iterdir())
        if all(child.is_file() for child in children):
            created = await self._exec_cmd_platform(
                f"mkdir -p {shlex.quote(target_dir)}", timeout_sec=30
            )
            if created.return_code != 0:
                raise RuntimeError(
                    f"Failed to create uploaded dir {target_dir}: "
                    f"{created.stderr or created.stdout}"
                )
            for child in children:
                await self.upload_file(
                    child, str(PurePosixPath(target_dir) / child.name)
                )
            return

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
            tar_path = Path(tf.name)

        import tarfile

        try:
            completion_name = f".hb-upload-complete-{uuid.uuid4().hex}"
            completion_value = uuid.uuid4().hex.encode()
            with tarfile.open(tar_path, "w:gz") as tar:
                # Add the directory's contents directly (no "." entry) so tar
                # doesn't try to chmod/utime the target directory (which is
                # root-owned while we run as the non-root pod user).
                for child in children:
                    tar.add(child, arcname=child.name)
                # The final archive member lets a separate short command prove
                # extraction reached the end even when a lossy transport drops
                # the long tar command's stdout and reports the wrong rc.
                completion_info = tarfile.TarInfo(completion_name)
                completion_info.size = len(completion_value)
                tar.addfile(completion_info, io.BytesIO(completion_value))

            remote_tar = str(
                _REMOTE_STAGE_DIR / f".hb-transfer-{uuid.uuid4().hex}.tar.gz"
            )
            try:
                await self.upload_file(tar_path, remote_tar)
                # --no-same-owner / --no-same-permissions / -m /
                # --no-overwrite-dir: we run as the non-root pod user and the
                # target dir (e.g. /solution) is usually root-owned, so tar
                # must not try to restore its ownership / mtime / mode.
                result = await self._exec_cmd_platform(
                    f"mkdir -p {shlex.quote(target_dir)} "
                    f"&& tar xzf {shlex.quote(remote_tar)} "
                    f"-C {shlex.quote(target_dir)} "
                    f"--no-same-owner --no-same-permissions -m --no-overwrite-dir",
                    timeout_sec=120,
                )
                completion_path = str(PurePosixPath(target_dir) / completion_name)
                verified = ExecResult(stdout="", stderr="", return_code=1)
                try:
                    for attempt in range(1, 7):
                        verified = await self._exec_cmd_platform(
                            f"cat {shlex.quote(completion_path)}", timeout_sec=30
                        )
                        if completion_value.decode() in (verified.stdout or ""):
                            break
                        if attempt < 6:
                            await asyncio.sleep(2)
                    if completion_value.decode() not in (verified.stdout or ""):
                        raise RuntimeError(
                            f"Failed to extract uploaded dir into {target_dir}: "
                            f"{result.stderr or result.stdout}"
                        )
                finally:
                    await self._exec_cmd_platform(
                        f"rm -f {shlex.quote(completion_path)}", timeout_sec=15
                    )
            finally:
                await self._exec_cmd_platform(
                    f"rm -f {shlex.quote(remote_tar)}", timeout_sec=15
                )
        finally:
            tar_path.unlink(missing_ok=True)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """以 root 归档并回收目录，保留 agent 的私有日志而不放宽源权限。"""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        remote_tar = str(_REMOTE_STAGE_DIR / f".hb-transfer-{uuid.uuid4().hex}.tar.gz")
        try:
            result = await self._exec_cmd_platform(
                f"tar czf {shlex.quote(remote_tar)} -C {shlex.quote(source_dir)} .",
                timeout_sec=120,
                user="root",
            )
            # GNU tar 用 rc=1 表示"归档已完整写出,但归档期间源目录被改动"
            # (`file changed as we read it` / `File removed before we read it`),
            # rc=2 才是致命错误。/logs/agent 是活目录:agent 进程刚退出时仍在
            # 落盘会话 JSONL、shell-snapshots、statsig,而 Claude Code 还会把
            # skills 拷进 $CLAUDE_CONFIG_DIR(=/logs/agent/sessions,上游
            # claude_code.py:1518),本仓库 infer-* 题的 skills 有 27.8MiB,
            # tar 因此要跑上一段时间、与这些写入重叠的概率很高。把 rc=1 当
            # 失败会丢掉一份完好的归档 —— e2e-claude-v2/dsv4 两轮实录:
            # "Failed to download logs",agent/ 只剩空 setup/,claude-code.txt
            # 与 sessions/ 全部丢失,ATIF 轨迹随之为空。Codex 把 skills 拷到
            # $HOME/.agents/skills(codex.py:1259),不写 /logs/agent,所以同
            # 环境下它一直下载成功。
            if result.return_code not in (0, 1):
                raise RuntimeError(
                    f"Failed to tar remote dir {source_dir}: "
                    f"{result.stderr or result.stdout}"
                )
            if result.return_code == 1:
                self.logger.warning(
                    "归档 %s 期间源目录被改动，已按 tar 的非致命告警继续下载：%s",
                    source_dir,
                    (result.stderr or result.stdout or "")[-300:],
                )

            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
                local_tar = Path(tf.name)
            try:
                await self.download_file(remote_tar, local_tar)
                import tarfile

                with tarfile.open(local_tar, "r:gz") as tar:
                    tar.extractall(path=target, filter="data")
            finally:
                local_tar.unlink(missing_ok=True)
        finally:
            await self._exec_cmd_platform(
                f"rm -f {shlex.quote(remote_tar)}", timeout_sec=15, user="root"
            )

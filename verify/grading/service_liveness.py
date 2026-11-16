"""在 readiness 和 driver 执行期间检查服务；保存阶段错误，不改写候选 reward。

参数：显式 rank/pod 映射、检查回调和异步工作；不创建或重试远端 Job。
示例：await run_with_service_watch(work, check, evidence_path=path)
"""

from __future__ import annotations

import asyncio
from asyncio import sleep as _watch_sleep
import json
import re
import shlex
import subprocess
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


class ServiceExitedError(RuntimeError):
    """服务已退出；仅凭退出码不能判断候选错误或平台故障。"""


class ServiceProbeError(RuntimeError):
    """探针缺失或传输失败，服务状态尚不能确认。"""


async def check_ray_launches(environment, python: str, count: int) -> None:
    """通过 launcher 所用 Ray namespace/actor 名查询所有 rank；禁止只看 head。"""
    try:
        async with asyncio.timeout(45):
            result = await environment.exec(
                f"{shlex.quote(python)} /logs/verifier/ray_service_probe.py {count}", cwd="/tmp", timeout_sec=40)
    except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
        raise ServiceProbeError(f"Ray status probe failed: {exc}") from exc
    frames = [line[len('HARBOR_RAY_STATUS='):] for line in (result.stdout or '').splitlines() if line.startswith('HARBOR_RAY_STATUS=')]
    if result.return_code or len(frames) != 1:
        raise ServiceProbeError(f"Ray status missing: {result.stderr or result.stdout}")
    try:
        rows = json.loads(frames[0])
    except ValueError as exc:
        raise ServiceProbeError('Malformed Ray status') from exc
    if not isinstance(rows, list) or len(rows) != count or any(not isinstance(row, dict) for row in rows):
        raise ServiceProbeError('Ray status topology mismatch')
    if [row.get('rank') for row in rows] != list(range(count)):
        raise ServiceProbeError('Ray status rank mismatch')
    for row in rows:
        rc = row.get('launch_rc')
        if type(row.get('running')) is not bool or (rc is not None and type(rc) is not int):
            raise ServiceProbeError('Ray status invalid process evidence')
        if rc not in (None, 0) or (rc is None and not row['running']):
            raise ServiceExitedError(f"Ray rank-{row['rank']} ({row.get('host')}) service_failed rc={rc}: {row.get('tail', '')}")


async def check_launches(environment, pods: Sequence[str | None], *, label: str) -> None:
    """检查 controller 的 rc/PID；单次探针失败不伪造服务退出。"""
    if not pods or len(set(pods)) != len(pods):
        raise ValueError(f"Invalid service topology: {pods}")
    for rank, pod in enumerate(pods):
        prefix = f"/workspace/deploy/rank-{rank}"
        command = (
            f"if test -f {prefix}.rc; then cat {prefix}.rc; "
            f"elif test -s {prefix}.pid; then read -r pid < {prefix}.pid; "
            'if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; '
            "then echo RUNNING; else echo DEAD; fi; else echo UNKNOWN; fi"
        )
        try:
            async with asyncio.timeout(30):
                probe = await environment.exec(command, cwd="/tmp", timeout_sec=30, pod_name=pod)
        except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
            raise ServiceProbeError(f"{label} rank-{rank} ({pod}) probe failed: {exc}") from exc
        marker = (probe.stdout or "").strip()
        if probe.return_code != 0 or not (marker in {"RUNNING", "DEAD"} or re.fullmatch(r"-?\d+", marker)):
            raise ServiceProbeError(f"{label} rank-{rank} ({pod}) status unconfirmed: {probe.stderr or marker}")
        # 旧启动契约允许 daemonize 后返回 0；端点身份继续由 ownership gate 校验。
        if marker in {"RUNNING", "0"}:
            continue
        detail = "log unavailable"
        try:
            async with asyncio.timeout(5):
                tail = await environment.exec(f"tail -c 3000 {prefix}.log", cwd="/tmp", timeout_sec=5, pod_name=pod)
            detail = tail.stdout or tail.stderr or detail
        except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
            detail = f"log retrieval failed: {exc}"
        raise ServiceExitedError(f"{label} rank-{rank} ({pod}) service_failed rc={marker}: {detail}")


async def run_with_service_watch(
    work: Callable[[], Awaitable[T]], check: Callable[[], Awaitable[None]], *,
    evidence_path: Path, interval_sec: float = 10, probe_attempts: int = 3,
    probe_failure_grace_sec: float = 120,
) -> T:
    """监控服务并保留 driver 根因；未知探针故障不能覆盖正在收尾的 driver。

    ``ServiceExitedError`` 是明确服务退出，立即中止 driver；
    ``ServiceProbeError`` 只表示当前无法确认状态。此时给已启动的 driver
    一个有界收尾窗口，优先保留其真实异常/结果，再把探针故障作为基础设施
    证据抛出。父 phase deadline 仍是最终硬上限。
    """
    started = time.monotonic()
    operation = watcher = None
    if probe_failure_grace_sec <= 0:
        raise ValueError("probe_failure_grace_sec must be positive")

    async def confirmed_check():
        for attempt in range(probe_attempts):
            try:
                await check()
                return
            except ServiceProbeError:
                if attempt + 1 == probe_attempts:
                    raise
                await _watch_sleep(interval_sec)

    async def watch():
        while True:
            await _watch_sleep(interval_sec)
            await confirmed_check()

    try:
        await confirmed_check()
        operation = asyncio.create_task(work())
        watcher = asyncio.create_task(watch())
        done, _ = await asyncio.wait((operation, watcher), return_when=asyncio.FIRST_COMPLETED)
        if operation in done:
            # First await the driver: a known benchmark/transport failure must
            # remain the primary cause even if the final service probe also
            # cannot reach the pod.
            try:
                value = await operation
            except Exception as original:
                try:
                    await confirmed_check()
                except ServiceExitedError as service:
                    raise service from original
                except ServiceProbeError as probe:
                    original.add_note(str(probe))
                raise
            if watcher in done:
                await watcher
            await confirmed_check()
            return value
        if watcher in done:
            try:
                await watcher
            except ServiceExitedError:
                raise
            except ServiceProbeError as probe:
                try:
                    value = await asyncio.wait_for(
                        operation, timeout=probe_failure_grace_sec
                    )
                except asyncio.TimeoutError:
                    raise probe
                except BaseException as original:
                    if isinstance(original, asyncio.CancelledError):
                        raise
                    original.add_note(str(probe))
                    raise
                raise probe
        try:
            value = await operation
        except Exception as original:
            # connection refused 若同时有服务退出证据，保留后者及原 driver 异常链。
            try:
                await confirmed_check()
            except ServiceExitedError as service:
                raise service from original
            except ServiceProbeError as probe:
                original.add_note(str(probe))
            raise
        await confirmed_check()
        return value
    except (ServiceExitedError, ServiceProbeError) as exc:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps({
            "schema": "service-failure/1", "status": "INFRA_ERROR",
            "responsibility": "undetermined", "error_type": type(exc).__name__,
            "error": str(exc), "elapsed_sec": time.monotonic() - started,
            "automatic_retry": False,
        }, indent=2) + "\n")
        raise
    finally:
        tasks = [task for task in (operation, watcher) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

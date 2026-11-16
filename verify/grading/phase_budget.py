"""用单调时钟记录 verifier 阶段期限、异常和取消；不改变评分或自动重试。

参数：证据文件、阶段名及秒数；嵌套工作不能延长父期限，cleanup 仅供限时资源回收。
示例：async with phase(path, 'gate', 60): await gate()
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
import json
import math
from pathlib import Path
import time
import uuid

_deadline: ContextVar[float | None] = ContextVar("verifier_phase_deadline", default=None)


@asynccontextmanager
async def phase(path: Path, name: str, seconds: float | None = None, *, cleanup: bool = False):
    """工作受父期限约束；清理必须显式给出上限，不能继承已耗尽的工作期限。"""
    if cleanup and seconds is None:
        raise ValueError("cleanup requires a finite budget")
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("phase budget must be positive and finite")
    started, wall = time.monotonic(), time.time()
    parent = None if cleanup else _deadline.get()
    deadline = started + seconds if seconds is not None else parent
    if parent is not None and deadline is not None:
        deadline = min(parent, deadline)
    record = {"schema": "verifier-phase/1", "id": uuid.uuid4().hex, "phase": name,
              "started_at_unix": wall, "budget_sec": seconds,
              "deadline_at_unix": None if deadline is None else wall + deadline - started,
              "effective_budget_sec": None if deadline is None else max(0, deadline - started),
              "status": "RUNNING", "cleanup": cleanup}
    path.mkdir(parents=True, exist_ok=True)

    def write():
        # 每个 id 单独文件，Gate/Performance 并发不会互相覆盖。
        output = path / f"{record['id']}.json"
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(output)

    write()
    context = _deadline.set(deadline)
    timer = asyncio.timeout_at(deadline)
    try:
        async with timer:
            yield
        record["status"] = "COMPLETED"
    except BaseException as exc:
        record.update(status="TIMED_OUT" if timer.expired() else
                      "CANCELLED" if isinstance(exc, asyncio.CancelledError) else "FAILED",
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record.update(elapsed_sec=time.monotonic() - started,
                      remaining_sec=None if deadline is None else max(0, deadline - time.monotonic()))
        _deadline.reset(context)
        write()

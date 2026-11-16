"""统一的环境层异步重试原语。

判定语义只有一种:*operation* 抛异常 = 本次尝试失败(可重试),正常返回 =
成功。结果型调用点(看 ExecResult.return_code 之类)在闭包里把失败转成
:class:`TransientError` 再交给 :func:`retry_async`,错误细节(stderr 尾等)
放进异常文本,由 wrapper 统一拼装最终报错。

不适用的场景:等待型轮询(exec-ready、Ray 就绪、健康检查)语义是
"等到某状态",不是"重跑就好",保持各自的 poll 循环。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable


class TransientError(RuntimeError):
    """标记"这次尝试失败但值得重试"的错误(结果型调用点用它转译失败)。"""


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    what: str,
    attempts: int = 3,
    delay_sec: float = 2.0,
    backoff: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    logger: logging.Logger | None = None,
) -> T:
    """把 *operation* 最多跑 *attempts* 次,失败间隔 ``delay_sec * backoff**(n-1)``。

    - 命中 *retry_on* 的异常触发重试;其它异常原样穿透(编程错误不该被吞)。
    - 重试耗尽:``raise RuntimeError(f"{what} failed after ...") from last_error``,
      调用方拿得到完整因果链。
    - 每次失败经 *logger*(若给)打 warning,重试行为在日志里可见。
    """
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except retry_on as exc:
            last_error = exc
            if logger is not None:
                logger.warning(
                    "%s failed (attempt %d/%d): %s", what, attempt, attempts, exc
                )
            if attempt < attempts:
                await asyncio.sleep(delay_sec * backoff ** (attempt - 1))
    raise RuntimeError(
        f"{what} failed after {attempts} attempts: {last_error}"
    ) from last_error

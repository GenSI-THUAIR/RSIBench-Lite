"""retry_async 统一重试原语的行为契约。"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from harbor_envs.abstract_mlplat.retry import TransientError, retry_async


class RetryAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_first_success_returns_without_sleep(self) -> None:
        with patch("harbor_envs.abstract_mlplat.retry.asyncio.sleep", new=AsyncMock()) as sleep:
            result = await retry_async(AsyncMock(return_value=42), what="op")
        self.assertEqual(result, 42)
        sleep.assert_not_awaited()

    async def test_recovers_after_transient_failures_with_backoff(self) -> None:
        op = AsyncMock(side_effect=[TransientError("a"), TransientError("b"), "ok"])
        with patch("harbor_envs.abstract_mlplat.retry.asyncio.sleep", new=AsyncMock()) as sleep:
            result = await retry_async(
                op, what="op", attempts=3, delay_sec=2.0, backoff=2.0
            )
        self.assertEqual(result, "ok")
        self.assertEqual(op.await_count, 3)
        self.assertEqual([c.args[0] for c in sleep.await_args_list], [2.0, 4.0])

    async def test_exhaustion_raises_with_context_and_cause(self) -> None:
        op = AsyncMock(side_effect=TransientError("device busy"))
        with (
            patch("harbor_envs.abstract_mlplat.retry.asyncio.sleep", new=AsyncMock()),
            self.assertRaises(RuntimeError) as ctx,
        ):
            await retry_async(op, what="upload staging", attempts=3)
        self.assertIn("upload staging failed after 3 attempts", str(ctx.exception))
        self.assertIn("device busy", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, TransientError)

    async def test_non_matching_exception_propagates_immediately(self) -> None:
        op = AsyncMock(side_effect=KeyError("bug"))
        with (
            patch("harbor_envs.abstract_mlplat.retry.asyncio.sleep", new=AsyncMock()) as sleep,
            self.assertRaises(KeyError),
        ):
            await retry_async(op, what="op", retry_on=(TransientError,))
        self.assertEqual(op.await_count, 1)
        sleep.assert_not_awaited()

    async def test_constant_backoff_when_factor_is_one(self) -> None:
        op = AsyncMock(side_effect=[TransientError("x")] * 3 + ["ok"])
        with patch("harbor_envs.abstract_mlplat.retry.asyncio.sleep", new=AsyncMock()) as sleep:
            await retry_async(
                op, what="op", attempts=4, delay_sec=2.0, backoff=1.0
            )
        self.assertEqual([c.args[0] for c in sleep.await_args_list], [2.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()

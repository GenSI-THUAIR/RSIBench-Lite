"""验证Serve/Benchmark的权威终态中止评测，暂态观察失败不伪报停止。

示例：python3 -m pytest tests/test_infer_service_terminal.py；不申请平台资源。
"""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from harbor.environments.base import ExecResult
from verify.inference.infer_verifier import InferVerifier


class InferServiceTerminalTest(unittest.IsolatedAsyncioTestCase):
    def verifier(self, serve_reason=None, benchmark_reason=None):
        verifier = object.__new__(InferVerifier)
        verifier.environment = SimpleNamespace(
            terminal_state_reason=AsyncMock(return_value=benchmark_reason),
            exec=AsyncMock(return_value=ExecResult(return_code=0, stdout='RUNNING')))
        verifier._serve_environment = SimpleNamespace(terminal_state_reason=AsyncMock(return_value=serve_reason))
        return verifier

    async def test_stopped_serve_does_not_poll_live_benchmark(self):
        verifier = self.verifier(serve_reason='Merlin job serve entered terminal state STOPPED')
        with self.assertRaisesRegex(RuntimeError, 'Benchmark aborted.*serve.*STOPPED'):
            await verifier._poll_benchmark_status()
        verifier.environment.exec.assert_not_awaited()

    async def test_stopped_benchmark_is_also_terminal(self):
        verifier = self.verifier(benchmark_reason='benchmark entered terminal state FAILED')
        with self.assertRaisesRegex(RuntimeError, 'Benchmark aborted.*FAILED'):
            await verifier._poll_benchmark_status()
        verifier.environment.exec.assert_not_awaited()

    async def test_unknown_and_transient_observations_keep_running(self):
        verifier = self.verifier()
        verifier._serve_environment.terminal_state_reason.side_effect = TimeoutError('control plane timeout')
        self.assertEqual((await verifier._poll_benchmark_status()).stdout, 'RUNNING')
        verifier.environment.exec.assert_awaited_once()

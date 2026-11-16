"""区分协议smoke、真实质量标定与正式归一化评分，不把短测当基线。"""
import unittest
import asyncio
import importlib.util
import tempfile
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch
from harbor.models.trial.paths import TrialPaths

from verify.inference.runtime_infer_verifier import RuntimeInferVerifier


class RuntimeModesTest(unittest.TestCase):
    def test_modal_health_uses_egress_proxy(self):
        from harbor_envs.modal import ModalEnvironment
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        address = f'http://127.0.0.1:{server.server_port}'
        try:
            verifier = object.__new__(RuntimeInferVerifier)
            verifier.environment = Mock(spec=ModalEnvironment)
            verifier.environment.platform_name = 'modal'
            with patch.dict(os.environ, {'http_proxy': address, 'no_proxy': ''}, clear=True):
                with verifier._health_opener().open('http://unresolvable.invalid/health', timeout=2) as response:
                    self.assertEqual(response.status, 200)
            self.assertEqual(requests, ['http://unresolvable.invalid/health'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_step_artifacts_never_fall_back_to_another_scope(self):
        root=Path(__file__).resolve().parents[1]/'.agents/workdir'
        with tempfile.TemporaryDirectory(dir=root) as temp:
            verifier=object.__new__(RuntimeInferVerifier)
            verifier.trial_paths=TrialPaths(trial_dir=Path(temp))
            verifier.step_name='optimize'
            top=verifier.trial_paths.artifacts_dir/'submit_oci.tar.gz'
            top.parent.mkdir(parents=True)
            top.write_bytes(b'wrong scope')
            with self.assertRaises(FileNotFoundError):
                verifier._artifact(top.name)
            step=verifier.trial_paths.step_artifacts_dir('optimize')/top.name
            step.parent.mkdir(parents=True)
            step.write_bytes(b'correct step')
            self.assertEqual(verifier._artifact(top.name),step)

    def score(self, mode, accuracy=0.0, invalid_rate=0.0, completed=2):
        verifier=object.__new__(RuntimeInferVerifier)
        verifier.mode=mode
        config=SimpleNamespace(gate_center=.73,gate_tolerance=.05,gate_comparison='lower_bound',
                               gate_kind='longbench-v2-openai-chat')
        return verifier._score_summary({
            'gate':{'accuracy':accuracy,'invalid_rate':invalid_rate,'num_questions':3},
            'performance':{'output_throughput':10.0,'request_success_rate':1.0,
                           'completed':completed,'wall_time_sec':30.0}}, config)

    def test_smoke_confirms_execution_not_accuracy(self):
        result=self.score('smoke')
        self.assertEqual(result['smoke_complete'],1)
        self.assertNotIn('reward',result)
        self.assertNotIn('calibrate_complete',result)
        self.assertEqual(self.score('smoke',invalid_rate=1.0)['smoke_complete'],0)
        self.assertEqual(self.score('smoke',completed=1)['smoke_complete'],0)

    def test_calibration_keeps_original_quality_threshold(self):
        self.assertEqual(self.score('calibrate',accuracy=.67)['calibrate_complete'],0)
        self.assertEqual(self.score('calibrate',accuracy=.73)['calibrate_complete'],1)
        self.assertNotIn('performance_score',self.score('calibrate',accuracy=.73))


@unittest.skipUnless(importlib.util.find_spec('python_socks'), 'requires Modal extra')
class ModalStatusRetryTest(unittest.IsolatedAsyncioTestCase):
    def verifier(self, effects, backend='modal'):
        verifier=object.__new__(RuntimeInferVerifier)
        verifier.task=SimpleNamespace(config=SimpleNamespace(metadata={'runtime_backend':backend}))
        from harbor_envs.modal import ModalEnvironment
        verifier.environment=Mock(spec=ModalEnvironment)
        verifier.environment.platform_name=backend
        verifier.environment.terminal_state_reason=AsyncMock(return_value=None)
        verifier.environment.exec=AsyncMock(side_effect=effects)
        verifier.logger=Mock()
        return verifier

    async def test_transient_proxy_failure_retries_only_the_status_read(self):
        from python_socks import ProxyError, ProxyTimeoutError
        for error in (ProxyError('gateway', error_code=504), ProxyTimeoutError('connect timeout')):
            with self.subTest(error=type(error).__name__):
                result=SimpleNamespace(stdout='RUNNING\n',return_code=0)
                verifier=self.verifier([error,result])
                with patch('verify.inference.runtime_infer_verifier.asyncio.sleep',new_callable=AsyncMock) as sleep:
                    self.assertIs(await verifier._poll_benchmark_status(),result)
                self.assertEqual(verifier.environment.exec.await_count,2)
                sleep.assert_awaited_once_with(5)
                for invocation in verifier.environment.exec.await_args_list:
                    self.assertIn('clean-benchmark.rc',invocation.args[0])
                    self.assertNotIn('clean_benchmark.py',invocation.args[0])
                    self.assertNotIn('nohup',invocation.args[0])

    async def test_persistent_proxy_failure_is_bounded_and_propagated(self):
        from python_socks import ProxyError
        error=ProxyError('gateway',error_code=504)
        verifier=self.verifier(error)
        with patch('verify.inference.runtime_infer_verifier.asyncio.sleep',new_callable=AsyncMock) as sleep:
            with self.assertRaises(ProxyError) as raised:
                await verifier._poll_benchmark_status()
        self.assertIs(raised.exception,error)
        self.assertEqual(verifier.environment.exec.await_count,4)
        self.assertEqual(sleep.await_args_list,[call(5),call(10),call(20)])

    async def test_authentication_unknown_errors_and_cancellation_are_not_retried(self):
        from python_socks import ProxyError
        for error in (ProxyError('auth',error_code=407), ProxyError('unknown'),
                      ValueError('bad configuration'), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                verifier=self.verifier(error)
                with patch('verify.inference.runtime_infer_verifier.asyncio.sleep',new_callable=AsyncMock) as sleep:
                    with self.assertRaises(type(error)):
                        await verifier._poll_benchmark_status()
                self.assertEqual(verifier.environment.exec.await_count,1)
                sleep.assert_not_awaited()



if __name__=='__main__':
    unittest.main()

"""验证可选服务身份门；真实本地 HTTP，不创建 Merlin job 或调用 GRM。

参数：无。示例：.venv/bin/python -m unittest tests.test_service_identity
标量与吞吐均为合成 fixture，不是模型权重证明或真实性能标定。
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from harbor.models.trial.paths import TrialPaths
from verify.inference.infer_verifier import (
    CONFIG_NAME, METADATA_NAME, SERVICE_IDENTITY_NAME,
    _SERVICE_IDENTITY_MAX_BYTES, DeploymentFailed, InferVerifier,
    _load_verifier_config,
)
from schemas.build import emit_verifier_config, load_yaml

ROOT = Path(__file__).resolve().parents[1]
MODULE = "verify.inference.infer_verifier"
EXPECTED = {"model_path": "/models/frozen-checkpoint", "served_model_name": "fixture-model"}
SECRET = "fixture-secret-must-not-be-recorded"


def config_payload():
    return {
        "python": "python3", "engine_repo": "/engine", "layer_artifact": "submit_oci.tar.gz",
        "gate": {"kind": "gsm8k-openai-chat", "data": "/data/quality.jsonl",
                 "center": 0.95, "tolerance": 0.02, "stop_tokens": []},
        "serve_node_count": 1, "serve_gpus_per_node": 1, "health_timeout_sec": 10,
        **EXPECTED,
        "bench_serving": {
            "backend": "vllm", "script_path": "/bench.py", "dataset_name": "random-ids",
            "random_input_len": 128, "random_output_len": 8, "random_range_ratio": 1.0,
            "max_concurrency": 1, "num_prompts": 2, "seed": 42, "extra_args": [],
        },
        "performance_baseline": {"output_throughput": 10.0},
        "service_identity": {"endpoint_path": "/get_server_info", "expected": dict(EXPECTED)},
    }


def benchmark_summary():
    return {
        "gate": {"accuracy": 0.95},
        "performance": {
            "output_throughput": 20.0, "request_success_rate": 1.0,
            "ttft_ms_p50": 1.0, "ttft_ms_p99": 2.0,
            "tpot_ms_p50": 3.0, "tpot_ms_p99": 4.0, "wall_time_sec": 5.0,
        },
    }


class LocalIdentityServer:
    """可控 HTTP 响应；退出时只关闭本测试的监听器和线程。"""

    def __init__(self):
        self.body = json.dumps({**EXPECTED, "api_key": SECRET, "nested": {"token": SECRET}}).encode()
        self.status = 200
        self.headers = {}
        self.trickle = False
        self.requests = []
        self.on_request = lambda: None
        self.stopping = threading.Event()
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                fixture.requests.append(self.path)
                fixture.on_request()
                self.send_response(fixture.status)
                self.send_header("Content-Length", str(len(fixture.body)))
                for key, value in fixture.headers.items():
                    self.send_header(key, value)
                self.end_headers()
                chunks = (bytes([byte]) for byte in fixture.body) if fixture.trickle else [fixture.body]
                try:
                    for chunk in chunks:
                        if fixture.trickle and fixture.stopping.wait(0.02):
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # 客户端因响应上限/总时限主动关闭。

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True,
        )

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stopping.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("local identity server did not stop")


class ConfigFixture:
    def setUp(self):
        super().setUp()
        workdir = ROOT / ".agents" / "workdir"
        workdir.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="service-identity-test-", dir=workdir)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = config_payload()

    def config(self):
        (self.root / CONFIG_NAME).write_text(json.dumps(self.raw))
        return _load_verifier_config(self.root)


class ServiceIdentityConfigTest(ConfigFixture, unittest.TestCase):
    def test_default_disabled_and_scalar_map_supported(self):
        self.raw.pop("service_identity")
        self.assertIsNone(self.config().service_identity)
        expected = {"text": "value", "integer": 7, "number": 1.5, "boolean": False, "null": None}
        identity = {"endpoint_path": "/metadata", "expected": expected}
        self.raw["service_identity"] = identity
        self.assertEqual(self.config().service_identity, identity)

    def test_invalid_identity_shape_rejected(self):
        for value in (None, False, "", [], {}, {"expected": EXPECTED},
                      {"endpoint_path": "/metadata"},
                      {**self.raw["service_identity"], "timeout": 60}):
            with self.subTest(value=value):
                self.raw["service_identity"] = value
                with self.assertRaisesRegex(ValueError, "service_identity"):
                    self.config()

    def test_external_or_ambiguous_endpoint_rejected(self):
        for path in (None, 1, "", "metadata", "https://other.invalid/info", "//other.invalid/info",
                     "///other.invalid/info", "/\\other.invalid/info", "/info\\other", "/info?x=1",
                     "/info#fragment", "/info?", "/info#", "/info\n", "/info\t", "/ info", "/info\x7f"):
            with self.subTest(path=path):
                self.raw["service_identity"]["endpoint_path"] = path
                with self.assertRaisesRegex(ValueError, "service_identity.endpoint_path"):
                    self.config()

    def test_empty_nested_and_nonfinite_expected_rejected(self):
        for expected in (None, False, [], {}, {"": "value"}, {" ": "value"}, {"a": []}, {"a": {}},
                         {"a": float("nan")}, {"a": float("inf")}, {"a": float("-inf")}):
            with self.subTest(expected=expected):
                self.raw["service_identity"]["expected"] = expected
                with self.assertRaisesRegex(ValueError, "service_identity.expected"):
                    self.config()

    def test_yaml_references_automatically_pass_through_emitter(self):
        # 使用纯合成 YAML，不能改写题目或依赖未跟踪的任务配置。
        source = self.root / "fixture.yaml"
        source.write_text("""verifier:
  scoring:
    model_path: /models/frozen-checkpoint
    served_model_name: fixture-model
    service_identity:
      endpoint_path: /get_server_info
      expected:
        model_path: !ref verifier.scoring.model_path
        served_model_name: !ref verifier.scoring.served_model_name
""")
        emitted = json.loads(emit_verifier_config(load_yaml(source)))
        self.assertEqual(emitted["service_identity"], config_payload()["service_identity"])


class ServiceIdentityHTTPTest(ConfigFixture, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        self.server = self.enterContext(LocalIdentityServer())
        self.verifier = InferVerifier.__new__(InferVerifier)
        self.verifier.task = SimpleNamespace(paths=SimpleNamespace(tests_dir=self.root))
        self.verifier.trial_paths = TrialPaths(trial_dir=self.root / "trial")
        self.verifier.logger = Mock()
        self.sequence = []
        self.server.on_request = lambda: self.sequence.append("identity")
        self.env = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(),
            exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="RUNNING", stderr="")))
        self.verifier._create_serve_environment = Mock(return_value=self.env)

        async def healthy_service(*_args):
            self.sequence.append("healthy-and-owned")
            return self.server.endpoint

        async def benchmark(*_args):
            self.sequence.append("benchmark")
            return benchmark_summary()

        self.verifier._start_patched_service = AsyncMock(side_effect=healthy_service)
        self.verifier._run_clean_benchmark = AsyncMock(side_effect=benchmark)
        self.verifier._download_serve_logs = AsyncMock()
        artifacts = self.verifier.trial_paths.artifacts_dir
        artifacts.mkdir(parents=True)
        (artifacts / "submit_oci.tar.gz").write_bytes(b"fixture")
        (artifacts / METADATA_NAME).write_text("{}")

    async def check(self):
        await self.verifier._verify_service_identity(self.server.endpoint, self.config())

    def evidence(self):
        raw = (self.verifier.trial_paths.verifier_dir / SERVICE_IDENTITY_NAME).read_text()
        self.assertNotIn(SECRET, raw)
        record = json.loads(raw)
        self.assertEqual(set(record), {"expected", "observed", "status"})
        return record

    async def test_correct_response_preserves_only_selected_fields(self):
        await self.check()
        self.assertEqual(self.server.requests, ["/get_server_info"])
        self.assertEqual(self.evidence(), {"expected": EXPECTED, "observed": EXPECTED, "status": "passed"})

    async def test_wrong_model_name_and_path_are_deployment_failures(self):
        for key in EXPECTED:
            with self.subTest(key=key):
                self.server.body = json.dumps({**EXPECTED, key: "wrong-value", "api_key": SECRET}).encode()
                with self.assertRaisesRegex(DeploymentFailed, "mismatch"):
                    await self.check()
                self.assertEqual(self.evidence()["observed"][key], "wrong-value")

    async def test_missing_field_is_not_null(self):
        self.raw["service_identity"]["expected"]["nullable"] = None
        with self.assertRaisesRegex(DeploymentFailed, "missing_field"):
            await self.check()
        self.assertNotIn("nullable", self.evidence()["observed"])
        self.server.body = json.dumps({**EXPECTED, "nullable": None}).encode()
        await self.check()
        self.assertIsNone(self.evidence()["observed"]["nullable"])

    async def test_nested_observation_rejected_without_recording_secrets(self):
        for value in ({"api_key": SECRET}, [SECRET]):
            with self.subTest(value=value):
                self.server.body = json.dumps({**EXPECTED, "model_path": value}).encode()
                with self.assertRaisesRegex(DeploymentFailed, "invalid_field_type"):
                    await self.check()
                self.assertNotIn("model_path", self.evidence()["observed"])

    async def test_boolean_number_and_string_not_coerced(self):
        for observed in (True, "1", None):
            with self.subTest(observed=observed):
                self.raw["service_identity"]["expected"] = {"value": 1}
                self.server.body = json.dumps({"value": observed}).encode()
                with self.assertRaisesRegex(DeploymentFailed, "mismatch"):
                    await self.check()
        self.server.body = b'{"value": 1.0}'
        await self.check()  # JSON number 的 1 与 1.0 等值，boolean 不等值。

    async def test_invalid_json_and_non_object_rejected(self):
        for body in (b"not JSON", b"", b"\xff", b"[]", b"null", b'{"x": NaN}', b'{"x": Infinity}'):
            with self.subTest(body=body):
                self.server.body = body
                with self.assertRaisesRegex(DeploymentFailed, "invalid_json"):
                    await self.check()
                self.assertEqual(self.evidence()["observed"], {})

    async def test_http_errors_do_not_record_body(self):
        for status in (404, 500, 503):
            with self.subTest(status=status):
                self.server.status = status
                self.server.body = SECRET.encode()
                with self.assertRaisesRegex(DeploymentFailed, "http_error"):
                    await self.check()
                self.assertEqual(self.evidence()["observed"], {})

    async def test_connection_failure_is_deployment_failure(self):
        # 保持端口由本测试持有但不 listen，不触及外部地址或他人服务。
        with socket.socket() as unavailable:
            unavailable.bind(("127.0.0.1", 0))
            endpoint = f"http://127.0.0.1:{unavailable.getsockname()[1]}"
            with self.assertRaisesRegex(DeploymentFailed, "http_error"):
                await self.verifier._verify_service_identity(endpoint, self.config())
        self.assertEqual(self.evidence()["observed"], {})

    async def test_redirects_never_reach_target(self):
        with LocalIdentityServer() as target:
            for location in (target.endpoint + "/target", "/same-origin-target"):
                with self.subTest(location=location):
                    self.server.status = 302
                    self.server.headers = {"Location": location}
                    self.server.requests.clear()
                    with self.assertRaisesRegex(DeploymentFailed, "http_error"):
                        await self.check()
                    self.assertEqual(self.server.requests, ["/get_server_info"])
                    self.assertEqual(target.requests, [])

    async def test_proxy_environment_is_ignored(self):
        with LocalIdentityServer() as proxy:
            variables = {key: proxy.endpoint for key in (
                "http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            )}
            variables.update(no_proxy="", NO_PROXY="")
            with patch.dict(os.environ, variables):
                await self.check()
            self.assertEqual(proxy.requests, [])
            self.assertEqual(self.server.requests, ["/get_server_info"])

    async def test_bounded_raw_response(self):
        body = json.dumps(EXPECTED).encode()
        self.server.body = body + b" " * (_SERVICE_IDENTITY_MAX_BYTES - len(body))
        await self.check()
        self.server.body += b" "
        with self.assertRaisesRegex(DeploymentFailed, "response_too_large"):
            await self.check()
        self.assertEqual(self.evidence()["observed"], {})

    async def test_total_deadline_stops_slow_stream(self):
        self.server.trickle = True
        with patch(f"{MODULE}._SERVICE_IDENTITY_TIMEOUT_SEC", 0.15):
            with self.assertRaisesRegex(DeploymentFailed, "timeout"):
                await asyncio.wait_for(self.check(), timeout=2)
        self.assertEqual(self.evidence()["status"], "timeout")

    async def test_disabled_config_does_not_create_client_or_evidence(self):
        self.raw.pop("service_identity")
        with patch(f"{MODULE}.httpx.AsyncClient", side_effect=AssertionError("unexpected HTTP")):
            await self.check()
        self.assertEqual(self.server.requests, [])
        self.assertFalse((self.verifier.trial_paths.verifier_dir / SERVICE_IDENTITY_NAME).exists())

    async def test_integration_runs_between_health_and_benchmark(self):
        self.config()
        outcome = await self.verifier._verify_impl()
        self.assertEqual(self.sequence, ["healthy-and-owned", "identity", "benchmark"])
        self.assertFalse(outcome.failed)
        self.assertEqual(outcome.rewards["reward"], 2.0)
        self.assertEqual(outcome.rewards["deploy_success"], 1)
        self.env.stop.assert_awaited_once_with(delete=True)

    async def test_integration_legacy_no_http_keeps_rewards(self):
        self.raw.pop("service_identity")
        self.config()
        with patch(f"{MODULE}.httpx.AsyncClient", side_effect=AssertionError("unexpected HTTP")):
            outcome = await self.verifier._verify_impl()
        self.assertEqual(self.sequence, ["healthy-and-owned", "benchmark"])
        self.assertEqual(outcome.rewards["reward"], 2.0)
        self.assertEqual(self.server.requests, [])

    async def test_identity_failure_never_reaches_benchmark_or_positive_reward(self):
        self.config()
        cases = [
            (200, json.dumps({**EXPECTED, "served_model_name": "wrong-model"}).encode(), "mismatch"),
            (200, json.dumps({**EXPECTED, "model_path": "/wrong-path"}).encode(), "mismatch"),
            (200, b"{}", "missing_field"),
            (200, b"invalid JSON", "invalid_json"),
            (200, b" " * (_SERVICE_IDENTITY_MAX_BYTES + 1), "response_too_large"),
            (302, b"", "http_error"),
            (503, SECRET.encode(), "http_error"),
        ]
        for status, body, failure in cases:
            with self.subTest(status=status, failure=failure):
                self.sequence.clear()
                self.env.stop.reset_mock()
                self.server.status, self.server.body = status, body
                self.server.headers = {"Location": "/same-origin-target"} if status == 302 else {}
                # 不请求任何 judge；保留真实 GRM hard gate 与 reward 写盘。
                grade = AsyncMock(return_value={})
                with patch("verify.grading.grm_verifier.grm.grade_best_effort", grade):
                    result = await self.verifier.verify()
                self.assertEqual(self.sequence, ["healthy-and-owned", "identity"])
                self.verifier._run_clean_benchmark.assert_not_awaited()
                self.env.stop.assert_awaited_once_with(delete=True)
                self.assertEqual(result.rewards["reward"], 0)
                self.assertEqual(result.rewards["deploy_success"], 0)
                self.assertTrue(grade.await_args.kwargs["failed"])
                saved = json.loads((self.verifier.trial_paths.verifier_dir / "reward.json").read_text())
                self.assertEqual(saved["reward"], 0)
                self.assertIn("service_identity", saved["deploy_error"])
                self.assertNotIn(SECRET, json.dumps(saved))
                self.assertEqual(self.evidence()["status"], failure)


if __name__ == "__main__":
    unittest.main()

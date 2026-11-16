"""纯 CPU 的固定窗口协议/真实 loopback HTTP 测试；不下载模型、不申请设备。

示例：python -m unittest discover -s tests -p test_fixed_window_benchmark.py
可选离线 golden：显式设置 FIXED_WINDOW_GOLDEN_{MANIFEST,VOCAB,TRACE} 三个文件路径，
用实际词表重编译归档的全部 prompt 并逐项比对；默认测试不依赖工作区归档文件。
"""
from __future__ import annotations

from verify.resources import payload_file
import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from aiohttp import web


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fixed_window_benchmark_under_test", payload_file('fixed_window_benchmark.py'))
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def fixture_config(directory, records=None):
    directory = Path(directory)
    trace = directory / "trace.jsonl"
    vocab = directory / "vocab.json"
    if records is None:
        records = [{"input_length": 3, "output_length": 7, "hash_ids": [index], "timestamp": 5000} for index in range(2)]
    trace.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    vocab.write_text(json.dumps({"vocab": {"a": 1, "b": 3, "special": 9}, "all_special_ids": [9]}), encoding="utf-8")
    return {
        "model_path": str(directory / "local-tokenizer"), "served_model_name": "fixture-model",
        "gate": {"kind": "normal-correctness-gate-not-run-here"},
        "bench_serving": {
            "backend": "vllm", "dataset_name": "mooncake", "max_concurrency": 1, "num_prompts": 100,
            "seed": 17, "window_duration_sec": .12, "window_expected_engines": 1, "vocab_json": str(vocab),
            "extra_args": ["--dataset-path", str(trace), "--mooncake-context-length", "64", "--mooncake-slowdown-factor", "1.0"],
        },
    }


def metric_text(engines):
    return "\n".join(
        f'vllm:num_requests_{family}{{model_name="fixture-model",engine="{engine}"}} {counts[index]}'
        for family, index in (("running", 0), ("waiting", 1)) for engine, counts in engines.items()
    ) + "\n"


class Counters(unittest.TestCase):
    def test_cumulative_usage_monotonic_and_no_doublecount(self):
        counter = probe.TokenCounter()
        for count in (0, 1, 3, 3, 7, 7):
            counter.update({"usage": {"completion_tokens": count}})
        self.assertEqual(counter.total, 7)
        self.assertEqual(counter.updates, 6)
        self.assertTrue(counter.observed)
        with self.assertRaisesRegex(probe.ProtocolError, "nonmonotonic"):
            counter.update({"usage": {"completion_tokens": 6}})

    def test_zero_usage_is_exact_and_observed(self):
        counter = probe.TokenCounter()
        counter.update({"choices": [], "usage": {"completion_tokens": 0}})
        self.assertTrue(counter.observed)
        self.assertEqual(counter.total, 0)
        self.assertEqual(counter.updates, 1)

    def test_never_falls_back_to_text_or_incremental_ids(self):
        for choice in ({"text": "many characters"}, {"token_ids": [1, 2, 3]}):
            with self.subTest(choice=choice), self.assertRaisesRegex(probe.ProtocolError, "continuous usage"):
                probe.TokenCounter().update({"choices": [choice]})
        counter = probe.TokenCounter()
        counter.update({"choices": [{"text": "many characters", "token_ids": [1, 2, 3]}], "usage": {"completion_tokens": 1}})
        self.assertEqual(counter.total, 1)

    def test_invalid_counts_and_shapes_fail_closed(self):
        events = [{"usage": {"completion_tokens": value}} for value in (-1, True, 1.5, "3", None)]
        events += [None, [], {"choices": None}, {"choices": [1]}, {"usage": []}, {"error": {}},
                   {"choices": [{"index": 1}]}, {"choices": [{"index": False}]},
                   {"choices": [{}, {}]}, {"choices": [{"delta": {"content": "chat"}}]},
                   {"choices": [{"finish_reason": "abort"}]}, {"choices": [{"finish_reason": "error"}]}]
        for event in events:
            with self.subTest(event=event), self.assertRaises(probe.ProtocolError):
                probe.TokenCounter().update(event)

    def test_split_sse_events(self):
        decoder = probe.SSEDecoder()
        raw = ': comment\r\ndata: {"text":"中"}\r\n\r\ndata: [DONE]\n\n'.encode()
        events = []
        for byte in raw:
            events.extend(decoder.feed(bytes([byte])))
        self.assertEqual(events, ['{"text":"中"}', '[DONE]'])

    def test_unicode_line_separators_in_json_are_not_sse_delimiters(self):
        value = {"text": "合法\u0085\u2028\u2029文本"}
        payload = ("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode()
        for chunk_size in (1, 7, len(payload)):
            decoder, events = probe.SSEDecoder(), []
            for offset in range(0, len(payload), chunk_size):
                events.extend(decoder.feed(payload[offset:offset + chunk_size]))
            self.assertEqual([json.loads(event) for event in events], [value])
        self.assertEqual(decoder.buffer, b"")

    def test_unicode_separators_inside_json_are_not_sse_line_boundaries(self):
        event = {"choices": [{"text": "x".join(chr(code) for code in (0x85, 0x2028, 0x2029))}], "usage": {"completion_tokens": 7}}
        data = json.dumps(event, ensure_ascii=False)
        decoder = probe.SSEDecoder()
        events = []
        for byte in ("data: " + data + "\n\n").encode():
            events.extend(decoder.feed(bytes([byte])))
        self.assertEqual(events, [data])
        counter = probe.TokenCounter()
        counter.update(json.loads(events[0]))
        self.assertEqual(counter.total, 7)

    def test_incomplete_sse_does_not_count_and_multiline_is_joined(self):
        decoder = probe.SSEDecoder()
        self.assertEqual(decoder.feed(b'data: {"usage":'), [])
        self.assertEqual(decoder.feed(b'\ndata: {"completion_tokens": 2}}\n\n'), ['{"usage":\n{"completion_tokens": 2}}'])
        counter = probe.TokenCounter()
        counter.update(json.loads('{"usage":\n{"completion_tokens": 2}}'))
        self.assertEqual(counter.total, 2)


class ConfigAndCompiler(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.config = fixture_config(self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def test_arrival_mode_is_explicit_and_strict(self):
        self.assertEqual(probe._parse_config(self.config).arrival_mode, "trace")
        for mode in ("trace", "asap"):
            self.config["bench_serving"]["arrival_mode"] = mode
            self.assertEqual(probe._parse_config(self.config).arrival_mode, mode)
        for mode in (None, True, 0, "inf", "ASAP", [], {}):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "arrival_mode"):
                self.config["bench_serving"]["arrival_mode"] = mode
                probe._parse_config(self.config)

    def test_original_compiler_and_trace_deterministic(self):
        options = probe._parse_config(self.config)
        rows, metadata, raw = probe.load_workload(options)
        again, _, _ = probe.load_workload(options)
        self.assertEqual([row.prompt for row in rows], [row.prompt for row in again])
        self.assertEqual(metadata["trace_sha256"], probe.digest(raw))
        self.assertEqual(metadata["trace_request_count"], 2)
        self.assertEqual(metadata["selected_request_count"], 2)
        self.assertEqual(metadata["compiler_path"], str(payload_file('vendored_bench_serving.py')))
        for row in rows:
            self.assertEqual(len(row.prompt), row.prompt_len)
            self.assertTrue(set(row.prompt) <= {1, 3})
            self.assertLessEqual(row.prompt_len + row.output_len, options.context_length)

    def test_cap_after_sort_and_full_compiler_validation(self):
        records = [{"input_length": 3, "output_length": 7, "hash_ids": [index], "timestamp": stamp}
                   for index, stamp in enumerate((9000, 5000))]
        self.config = fixture_config(self.directory, records)
        all_rows, _, _ = probe.load_workload(probe._parse_config(self.config))
        self.config["bench_serving"]["num_prompts"] = 1
        rows, metadata, _ = probe.load_workload(probe._parse_config(self.config))
        self.assertEqual([row.prompt for row in rows], [all_rows[0].prompt])
        self.assertEqual(rows[0].timestamp, 5000)
        self.assertEqual(metadata["trace_total_requests"], 2)
        self.assertEqual(metadata["selected_request_count"], 1)
        records[0]["input_length"] = 65
        Path(self.config["bench_serving"]["extra_args"][1]).write_text("\n".join(map(json.dumps, records)))
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            probe.load_workload(probe._parse_config(self.config))

    def test_hash_verifies_full_original_bytes_before_cap(self):
        bench = self.config["bench_serving"]
        bench["num_prompts"] = 1
        trace = Path(bench["extra_args"][1])
        raw = trace.read_bytes()
        bench["trace_sha256"] = probe.digest(raw)
        rows, _, _ = probe.load_workload(probe._parse_config(self.config))
        self.assertEqual(len(rows), 1)
        trace.write_bytes(raw + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            probe.load_workload(probe._parse_config(self.config))

    def test_auto_tokenizer_is_local_and_uses_config_model_path(self):
        del self.config["bench_serving"]["vocab_json"]
        tokenizer = SimpleNamespace(get_vocab=lambda: {"x": 5, "y": 7, "special": 99}, all_special_ids=[99])
        loader = Mock(return_value=tokenizer)
        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=loader)
        with patch.dict(sys.modules, {"transformers": transformers}):
            rows, metadata, _ = probe.load_workload(probe._parse_config(self.config))
        loader.assert_called_once_with(self.config["model_path"], local_files_only=True, trust_remote_code=True)
        self.assertTrue(all(set(row.prompt) <= {5, 7} for row in rows))
        self.assertEqual(metadata["tokenizer"]["all_special_ids"], [99])

    def test_actual_vocab_validation_not_config_vocab_size(self):
        vocab_path = Path(self.config["bench_serving"]["vocab_json"])
        for value in ({"vocab": {"x": -1}, "all_special_ids": []},
                      {"vocab": {"x": 1, "y": 2}, "all_special_ids": [True]},
                      {"vocab": {"x": 1, "y": 2}, "all_special_ids": [1, 2]}):
            with self.subTest(value=value):
                vocab_path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    probe.load_workload(probe._parse_config(self.config))

    def test_explicit_engines_and_rounds(self):
        self.config["bench_serving"]["extra_args"] += ["--mooncake-num-rounds=1"]
        options = probe._parse_config(self.config)
        self.assertEqual(options.expected_engines, 1)
        self.assertEqual(options.seed, self.config["bench_serving"]["seed"])
        self.config["bench_serving"]["window_expected_engines"] = 2
        self.assertEqual(probe._parse_config(self.config).expected_engines, 2)
        del self.config["bench_serving"]["window_expected_engines"]
        with self.assertRaisesRegex(ValueError, "window_expected_engines"):
            probe._parse_config(self.config)

    def test_invalid_window_config(self):
        overrides = [("window_duration_sec", value) for value in (None, 0, -1, 3601, float("inf"), float("nan"), True, "1")]
        overrides += [("backend", "unknown"), ("dataset_name", "random"), ("max_concurrency", 2),
                      ("max_concurrency", True), ("seed", True), ("seed", "42"),
                      ("num_prompts", 0), ("num_prompts", True), ("num_prompts", None),
                      ("window_expected_engines", 0), ("window_expected_engines", True),
                      ("trace_sha256", "not-a-digest"), ("vocab_json", ""),
                      ("warmup_requests", -1), ("warmup_requests", True),
                      ("warmup_requests", "1"), ("warmup_timeout_sec", 0),
                      ("warmup_timeout_sec", float("inf")), ("warmup_timeout_sec", True)]
        for key, value in overrides:
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(self.config)
                config["bench_serving"][key] = value
                with self.assertRaises(ValueError):
                    probe._parse_config(config)
        for key in ("model_path", "served_model_name"):
            config = copy.deepcopy(self.config)
            del config[key]
            with self.assertRaisesRegex(ValueError, key):
                probe._parse_config(config)
        with self.assertRaisesRegex(ValueError, "full scoring JSON"):
            probe._parse_config(self.config["bench_serving"])

    def test_missing_window_opt_in_is_rejected(self):
        del self.config["bench_serving"]["window_duration_sec"]
        with self.assertRaisesRegex(ValueError, "window_duration_sec"):
            probe._parse_config(self.config)

    def test_unsupported_duplicate_or_bad_extra_args(self):
        original = self.config["bench_serving"]["extra_args"]
        bad = ["--dataset-path trace", [1], [], original + ["--temperature", "0"],
               original + ["--disable-stream"], original + ["--return-token-ids"],
               original + ["--extra-request-body", '{"ignore_eos":false}'],
               original + ["--mooncake-num-rounds", "2"], original + ["--seed", "42"],
               original + ["--mooncake-context-length", "64"], original + ["--mooncake-num-rounds"],
               ["--dataset-path", "x", "--mooncake-context-length", "0"],
               ["--dataset-path", "x", "--mooncake-context-length", "64", "--mooncake-slowdown-factor", "nan"]]
        for extra in bad:
            with self.subTest(extra=extra):
                config = copy.deepcopy(self.config)
                config["bench_serving"]["extra_args"] = extra
                with self.assertRaises((ValueError, TypeError)):
                    probe._parse_config(config)

    def test_idle_backend_is_separate_from_request_wire(self):
        self.config["bench_serving"]["window_idle_backend"] = "trtllm"
        options = probe._parse_config(self.config)
        self.assertEqual((options.backend, options.idle_backend), ("vllm", "trtllm"))
        rows, _, _ = probe.load_workload(options)
        self.assertEqual(rows[0].prompt_len, 3)
        for changes in ({"window_expected_engines": 2}, {"backend": "sglang"},
                        {"window_idle_backend": "unknown"}, {"window_idle_backend": None}):
            config = copy.deepcopy(self.config)
            config["bench_serving"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                probe._parse_config(config)

    def test_endpoint_normalization(self):
        for suffix in ("", "/", "/v1", "/v1/", "/v1/completions", "/v1/completions/"):
            self.assertEqual(probe._base_url("http://[::1]:1234" + suffix), "http://[::1]:1234")
        for endpoint in ("not-a-url", "file:///x", "http://host?x=1", "http://host/#fragment"):
            with self.assertRaises(ValueError):
                probe._base_url(endpoint)


class HTTPFixture(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.output = self.directory / "result"
        self.config = fixture_config(self.directory)
        self.release = asyncio.Event()
        self.request_started = asyncio.Event()
        self.requests = []
        self.transports = []
        self.paths = []
        self.behavior = "partial"
        self.counts = None
        self.active = self.max_active = 0
        self.models = {"data": [{"id": self.config["served_model_name"]}]}
        self.metrics = metric_text({0: (0, 0)})
        self.metric_series = []
        self.health_delay = 0
        self.first_request_delay = 0
        self.counts_from_request = False
        self.health_status = 200
        self.stall_path = None
        app = web.Application()
        app.router.add_get("/health", self.get)
        app.router.add_get("/v1/models", self.get)
        app.router.add_get("/metrics", self.get)
        app.router.add_get("/prometheus/metrics", self.get)
        app.router.add_post("/v1/completions", self.complete)
        app.router.add_post("/generate", self.complete)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.endpoint = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    async def asyncTearDown(self):
        self.release.set()
        await self.runner.cleanup()
        self.temp.cleanup()

    async def get(self, request):
        self.paths.append(request.path)
        if self.stall_path == request.path:
            await self.release.wait()
        if request.path == "/health":
            await asyncio.sleep(self.health_delay)
            return web.Response(status=self.health_status, text="ok")
        if request.path == "/v1/models":
            return web.json_response(self.models)
        return web.Response(text=self.metric_series.pop(0) if self.metric_series else self.metrics)

    async def complete(self, request):
        self.paths.append(request.path)
        self.requests.append(await request.json())
        self.transports.append(request.transport)
        if len(self.requests) == 1:
            await asyncio.sleep(self.first_request_delay)
        self.request_started.set()
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        try:
            if self.behavior == "http_error":
                return web.Response(status=500, text="fixture server error")
            if self.behavior == "wrong_type":
                return web.json_response({"usage": {"completion_tokens": 7}})
            await response.prepare(request)
            if self.behavior == "malformed":
                await response.write(b"data: {not-json}\n\n")
                return response
            if self.behavior == "no_counter_done":
                await response.write(b"data: [DONE]\n\n")
                return response
            if self.behavior != "zero":
                counts = self.counts if self.counts is not None else (1, 3, 3, 7)
                if self.counts_from_request:
                    counts = (self.requests[-1]["max_tokens"],)
                for count in counts:
                    event = {"choices": [{"index": 0, "text": "many chars not token count"}], "usage": {"completion_tokens": count}}
                    if request.path == "/generate":
                        event = {"text": "many chars not token count", "meta_info": {
                            "completion_tokens": count, "prompt_tokens": len(self.requests[-1]["input_ids"]),
                            "finish_reason": None}}
                        if self.behavior == "native_finished" and count == 7:
                            event["meta_info"]["finish_reason"] = {"type": "length", "length": 7}
                        if self.behavior == "wrong_prompt":
                            event["meta_info"]["prompt_tokens"] += 1
                        if self.behavior == "missing_counter":
                            del event["meta_info"]["completion_tokens"]
                        if self.behavior == "native_abort":
                            event["meta_info"]["finish_reason"] = {"type": "abort"}
                    if self.behavior == "missing_usage":
                        del event["usage"]
                    packet = ("data: " + json.dumps(event) + "\r\n\r\n").encode()
                    await response.write(packet[:8])
                    await response.write(packet[8:])
            if self.behavior == "server_error":
                await response.write(b'data: {"error":{"message":"engine failed"}}\n\n')
            elif self.behavior == "finished":
                await response.write(b"data: [DONE]\n\n")
            elif self.behavior not in {"no_done", "native_finished"}:
                await self.release.wait()
            return response
        except ConnectionResetError:
            # 测试客户端在窗口/错误处主动断流；无 abort API，服务自行处理 disconnect。
            return response
        finally:
            self.active -= 1

    async def run_window(self, duration=None):
        if duration is not None:
            self.config["bench_serving"]["window_duration_sec"] = duration
        # 真正调用公开同步 API，服务仍运行在本测试线程的 loop 中。
        return await asyncio.to_thread(probe.run_window, self.endpoint, self.output, self.config)

    async def idle(self, name="idle.json", *, engines=2, timeout=.1):
        return await asyncio.to_thread(probe.wait_for_idle, self.endpoint, self.directory / name,
                                       expected_engines=engines, timeout_sec=timeout)

    def assert_unscorable(self, summary, status="measurement_failure"):
        self.assertEqual(summary["status"], status)
        self.assertFalse(summary["counter_valid"])
        self.assertIsNone(summary["output_tps"])


class Window(HTTPFixture):

    async def test_trtllm_metrics_path_keeps_openai_exact_counter(self):
        self.config["bench_serving"]["window_idle_backend"] = "trtllm"
        self.metrics = 'trtllm_num_requests_running{model_name="fixture-model"} 0\ntrtllm_num_requests_waiting{model_name="fixture-model"} 0\n'
        self.counts = (1, 3, 3)
        summary = await self.run_window(.12)
        self.assertEqual(summary["status"], "deadline")
        self.assertEqual(summary["actual_received_output_tokens"], 3)
        self.assertIn("/prometheus/metrics", self.paths)
        self.assertNotIn("/metrics", self.paths)
        self.assertIn("/v1/completions", self.paths)
        self.assertTrue(self.requests[0]["stream_options"]["continuous_usage_stats"])
        evidence = json.loads((self.output / "prewindow-idle.json").read_text())
        self.assertEqual(evidence["samples"][0]["metrics_path"], "/prometheus/metrics")

    async def test_trtllm_missing_gauges_fail_before_requests(self):
        self.config["bench_serving"]["window_idle_backend"] = "trtllm"
        # vLLM lookalikes and empty iteration JSON are not TRT-LLM idle evidence.
        for index, metrics in enumerate((self.metrics, "[]", "trtllm_num_requests_running 0\n")):
            self.metrics = metrics
            self.output = self.directory / f"trt-missing-{index}"
            with self.subTest(metrics=metrics):
                self.assert_unscorable(await self.run_window(), "startup_failure")
                self.assertEqual(self.requests, [])

    async def test_sglang_native_partial_counter_and_payload(self):
        self.config["bench_serving"]["backend"] = "sglang"
        self.metrics = 'sglang:num_running_reqs{model_name="fixture-model"} 0\nsglang:num_queue_reqs{model_name="fixture-model"} 0\n'
        self.counts = (1, 3, 3)
        summary = await self.run_window(.12)
        self.assertEqual(summary["status"], "deadline")
        self.assertEqual(summary["actual_received_output_tokens"], 3)
        self.assertAlmostEqual(summary["output_tps"], 25)
        self.assertEqual(self.max_active, 1)
        self.assertEqual(self.requests[0]["sampling_params"]["max_new_tokens"], 7)
        self.assertTrue(self.requests[0]["sampling_params"]["ignore_eos"])
        self.assertEqual(len(self.requests[0]["input_ids"]), 3)
        self.assertNotIn("/v1/completions", self.paths)

    async def test_sglang_wrong_prompt_or_missing_counter_is_unscorable(self):
        self.config["bench_serving"]["backend"] = "sglang"
        self.metrics = 'sglang:num_running_reqs 0\nsglang:num_queue_reqs 0\n'
        for behavior in ("wrong_prompt", "missing_counter", "native_abort"):
            self.behavior = behavior
            self.output = self.directory / behavior
            with self.subTest(behavior=behavior):
                self.assert_unscorable(await self.run_window(.12))

    async def test_sglang_final_metadata_is_valid_without_done_marker(self):
        self.config["bench_serving"]["backend"] = "sglang"
        self.metrics = 'sglang:num_running_reqs 0\nsglang:num_queue_reqs 0\n'
        self.behavior = "native_finished"
        summary = await self.run_window(.2)
        self.assertTrue(summary["counter_valid"])
        self.assertEqual(summary["actual_received_output_tokens"], 14)
        self.assertEqual(summary["request_counts"]["finished"], 2)
        self.assertEqual(self.max_active, 1)

    async def test_asap_ignores_large_offsets_without_changing_requests_or_c1(self):
        self.behavior = "finished"
        self.config["bench_serving"]["arrival_mode"] = "asap"
        trace = Path(self.config["bench_serving"]["extra_args"][1])
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        records[1]["timestamp"] += 86400000
        trace.write_text("\n".join(map(json.dumps, records)))
        options = probe._parse_config(self.config)
        rows, _, _ = probe.load_workload(options)
        summary = await self.run_window(1)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["arrival_mode"], "asap")
        self.assertEqual(summary["measurement_protocol"], "c1-fixed-window-token-ids-asap-v1")
        self.assertEqual(self.requests, [probe.request_payload(row, options) for row in rows])
        self.assertEqual(self.max_active, 1)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["parameters"]["request_rate"], "inf")
        self.assertEqual([row["trace_offset_sec"] for row in manifest["requests"]], [0, 86400])
        self.assertEqual([row["scheduled_offset_sec"] for row in manifest["requests"]], [0, 0])
        first, second = manifest["requests"]
        self.assertGreaterEqual(second["start_offset_sec"], first["end_offset_sec"])
        self.assertLess(second["start_offset_sec"] - first["end_offset_sec"], .25)

    async def test_asap_exhausted_full_window_fails_instead_of_padding(self):
        self.behavior = "finished"
        self.config["bench_serving"].update(arrival_mode="asap", window_full_duration=True)
        summary = await self.run_window(1)
        self.assert_unscorable(summary, "workload_exhausted")
        self.assertEqual(summary["request_counts"]["finished"], 2)
        self.assertEqual(summary["actual_received_output_tokens"], 14)
        self.assertLess(summary["actual_elapsed_wall_sec"], 1)
        self.assertEqual(len(self.requests), 2)
        self.assertIn("larger frozen workload", summary["error"])

    async def test_asap_deadline_retains_partial_counter_and_full_denominator(self):
        self.config["bench_serving"].update(arrival_mode="asap", window_full_duration=True)
        summary = await self.run_window(.12)
        self.assertEqual(summary["status"], "deadline")
        self.assertTrue(summary["counter_valid"])
        self.assertEqual(summary["request_counts"]["partial_deadline"], 1)
        self.assertEqual(summary["output_tps"], 7 / .12)
        self.assertEqual(summary["request_rate"], "inf")

    async def test_full_duration_keeps_denominator_after_early_completion(self):
        self.behavior = "finished"
        self.config["bench_serving"]["window_full_duration"] = True
        summary = await self.run_window(.15)
        self.assertTrue(summary["counter_valid"])
        self.assertEqual(summary["request_counts"]["finished"], 2)
        self.assertEqual(summary["actual_received_output_tokens"], 14)
        self.assertAlmostEqual(summary["actual_elapsed_wall_sec"], .15)
        self.assertAlmostEqual(summary["output_tps"], 14 / .15)
        self.assertEqual(len(self.requests), 2)

    async def test_deadline_partial_tokens_and_finalsummary(self):
        summary = await self.run_window(.12)
        self.assertEqual(summary["status"], "deadline")
        self.assertTrue(summary["measured"])
        self.assertTrue(summary["counter_valid"])
        self.assertEqual(summary["actual_received_output_tokens"], 7)
        self.assertAlmostEqual(summary["actual_elapsed_wall_sec"], .12)
        self.assertAlmostEqual(summary["output_tps"], 7 / .12, places=5)
        self.assertEqual(summary["request_counts"]["partial_deadline"], 1)
        self.assertEqual(summary["request_counts"]["not_started"], 1)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(json.loads((self.output / "summary.json").read_text()), summary)
        logs = [json.loads(line) for line in (self.output / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[0]["got_output_tokens"], 7)
        self.assertEqual(logs[0]["counter_updates"], 4)
        self.assertIn("deadline_unix_sec", logs[1])
        self.assertEqual(summary["selected_request_count"], sum(summary["request_counts"].values()))
        self.assertFalse(summary["correctness_gate_run"])

    async def test_partially_streamed_budget_counts_only_received_tokens(self):
        self.counts = (1, 3, 3)
        summary = await self.run_window(.08)
        self.assertEqual(summary["status"], "deadline")
        self.assertEqual(summary["actual_received_output_tokens"], 3)
        self.assertAlmostEqual(summary["output_tps"], 3 / .08, places=5)
        self.assertEqual(summary["requested_output_tokens_total"], 14)

    async def test_zero_token_window_still_measured(self):
        self.behavior = "zero"
        summary = await self.run_window(.08)
        self.assertTrue(summary["measured"])
        self.assertTrue(summary["counter_valid"])
        self.assertEqual(summary["output_tps"], 0)
        self.assertEqual(summary["request_counts"]["partial_deadline"], 1)

    async def test_generation_requests_do_not_reuse_expiring_keepalive_connections(self):
        self.behavior = "finished"
        summary = await self.run_window(1)
        self.assertTrue(summary["counter_valid"])
        self.assertEqual(len(self.transports), 2)
        self.assertIsNot(self.transports[0], self.transports[1])
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertFalse(manifest["parameters"]["http_connection_reuse"])

    async def test_all_finish_early_no_repetition(self):
        self.behavior = "finished"
        summary = await self.run_window(1)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["actual_received_output_tokens"], 14)
        self.assertEqual(summary["request_counts"]["finished"], 2)
        self.assertLess(summary["actual_elapsed_wall_sec"], 1)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.max_active, 1)

    async def test_timestamp_wait_included_in_absolute_deadline(self):
        self.behavior = "finished"
        trace = Path(self.config["bench_serving"]["extra_args"][1])
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        records[1]["timestamp"] += 1000
        trace.write_text("\n".join(map(json.dumps, records)))
        summary = await self.run_window(.08)
        self.assertEqual(summary["status"], "deadline")
        self.assertEqual(summary["actual_received_output_tokens"], 7)
        self.assertEqual(summary["request_counts"]["not_started"], 1)
        self.assertEqual(summary["request_counts"]["partial_deadline"], 0)
        self.assertAlmostEqual(summary["actual_elapsed_wall_sec"], .08)

    async def test_slowdown_is_applied_to_original_timestamps(self):
        self.behavior = "finished"
        trace = Path(self.config["bench_serving"]["extra_args"][1])
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        records[1]["timestamp"] += 40
        trace.write_text("\n".join(map(json.dumps, records)))
        self.config["bench_serving"]["extra_args"][-1] = "10"
        summary = await self.run_window(.08)
        self.assertEqual(summary["request_counts"]["not_started"], 1)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertAlmostEqual(manifest["requests"][1]["scheduled_offset_sec"], .4)

    async def test_body_uses_model_seed_and_exact_compiled_prompt_without_warmup(self):
        self.behavior = "finished"
        rows, _, _ = probe.load_workload(probe._parse_config(self.config))
        summary = await self.run_window(1)
        self.assertEqual(summary["status"], "completed")
        expected = [{"model": self.config["served_model_name"], "prompt": row.prompt, "max_tokens": row.output_len,
                     "temperature": 0, "seed": self.config["bench_serving"]["seed"], "ignore_eos": True,
                     "n": 1, "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True}} for row in rows]
        self.assertEqual(self.requests, expected)
        self.assertEqual(self.paths[:3], ["/health", "/v1/models", "/metrics"])
        self.assertEqual(self.paths[3:], ["/v1/completions"] * len(rows))
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["parameters"]["warmup_requests"], 0)
        self.assertEqual(manifest["models_response"], self.models)

    async def test_cap_can_finish_all_selected_not_full_trace(self):
        self.behavior = "finished"
        self.config["bench_serving"]["num_prompts"] = 1
        summary = await self.run_window(1)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["selected_request_count"], 1)
        self.assertEqual(summary["trace_request_count"], 1)
        self.assertEqual(summary["trace_total_requests"], 2)
        self.assertEqual(summary["actual_received_output_tokens"], 7)
        self.assertEqual(len(self.requests), 1)

    async def test_warmup_excludes_time_and_tokens_and_preserves_full_trace(self):
        self.behavior = "finished"
        self.counts_from_request = True
        self.first_request_delay = .05
        self.config["bench_serving"]["warmup_requests"] = 1
        trace = Path(self.config["bench_serving"]["extra_args"][1])
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        records[0]["output_length"] = 40
        self.config["bench_serving"]["extra_args"][3] = "128"
        trace.write_text("\n".join(map(json.dumps, records)))
        summary = await self.run_window(1)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(len(self.requests), 3)
        self.assertEqual([r["max_tokens"] for r in self.requests], [32, 40, 7])
        self.assertEqual(self.requests[0]["prompt"], self.requests[1]["prompt"])
        self.assertEqual(summary["actual_received_output_tokens"], 47)
        self.assertEqual(summary["request_counts"]["finished"], 2)
        warmup = summary["warmup"]
        self.assertEqual(warmup["status"], "completed")
        self.assertGreaterEqual(warmup["wall_time_sec"], .05)
        self.assertGreaterEqual(summary["start_unix_sec"], warmup["requests"][0]["end_unix_sec"])
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["warmup"], warmup)
        self.assertEqual(manifest["warmup_idle"]["status"], "idle")
        self.assertEqual(manifest["parameters"]["warmup_requests"], 1)

    async def test_warmup_http_failure_never_starts_measurement(self):
        self.behavior = "http_error"
        self.config["bench_serving"]["warmup_requests"] = 1
        summary = await self.run_window()
        self.assertEqual(summary["status"], "startup_failure")
        self.assertFalse(summary["measured"])
        self.assertIsNone(summary["output_tps"])
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(summary["actual_received_output_tokens"], 0)
        self.assertEqual(summary["warmup"]["requests"][0]["http_status"], 500)

    async def test_warmup_timeout_is_not_a_valid_window_partial(self):
        self.config["bench_serving"]["warmup_requests"] = 1
        self.config["bench_serving"]["warmup_timeout_sec"] = .05
        summary = await self.run_window()
        self.assertEqual(summary["status"], "startup_failure")
        self.assertFalse(summary["measured"])
        self.assertIsNone(summary["output_tps"])
        self.assertEqual(summary["error_type"], "TimeoutError")
        self.assertEqual(summary["actual_received_output_tokens"], 0)

    async def test_missing_counter_no_silent_fallback(self):
        self.behavior = "missing_usage"
        summary = await self.run_window(1)
        self.assert_unscorable(summary)
        self.assertIn("continuous usage", summary["error"])

    async def test_http_failure_is_recorded_not_scored(self):
        self.behavior = "http_error"
        summary = await self.run_window(1)
        self.assert_unscorable(summary)
        self.assertEqual(summary["request_counts"]["failure"], 1)
        self.assertIn("HTTP 500", summary["error"])

    async def test_server_error_after_partial_output_is_not_scored(self):
        self.behavior = "server_error"
        summary = await self.run_window(1)
        self.assert_unscorable(summary)
        self.assertEqual(summary["actual_received_output_tokens"], 7)
        self.assertIn("engine failed", summary["error"])

    async def test_malformed_stream_is_not_scored(self):
        self.behavior = "malformed"
        summary = await self.run_window(1)
        self.assert_unscorable(summary)
        self.assertEqual(summary["error_type"], "ProtocolError")
        rows = [json.loads(line) for line in (self.output / "requests.jsonl").read_text().splitlines()]
        evidence = next(row["protocol_evidence"] for row in rows if row["status"] == "failure")
        self.assertEqual(evidence["kind"], "malformed_json_event")
        import base64
        self.assertEqual(base64.b64decode(evidence["event"]["base64"]), b"{not-json}")

    async def test_stream_without_done_is_failure(self):
        self.behavior = "no_done"
        summary = await self.run_window(1)
        self.assert_unscorable(summary)
        self.assertIn("without [DONE]", summary["error"])

    async def test_done_without_counter_is_failure(self):
        self.behavior = "no_counter_done"
        summary = await self.run_window(1)
        self.assert_unscorable(summary)
        self.assertIn("without an exact output token counter", summary["error"])

    async def test_early_completion_does_not_change_budget(self):
        self.behavior, self.counts = "finished", (1, 3)
        summary = await self.run_window(1)
        self.assert_unscorable(summary)
        self.assertEqual(summary["actual_received_output_tokens"], 3)
        self.assertIn("early completion", summary["error"])

    async def test_nonmonotonic_or_overbudget_usage_is_failure(self):
        for index, counts in enumerate(((1, 3, 2), (1, 8))):
            self.output = self.directory / f"counts-{index}"
            self.counts = counts
            summary = await self.run_window(1)
            self.assert_unscorable(summary)

    async def test_non_stream_response_is_failure(self):
        self.behavior = "wrong_type"
        summary = await self.run_window(1)
        self.assert_unscorable(summary)
        self.assertIn("text/event-stream", summary["error"])

    async def test_startup_failure_is_not_measurement(self):
        with patch.object(probe, "load_workload", side_effect=ValueError("bad vocabulary")):
            summary = await self.run_window(.1)
        self.assert_unscorable(summary, "startup_failure")
        self.assertFalse(summary["measured"])
        for name in ("manifest.json", "requests.jsonl", "summary.json"):
            self.assertTrue((self.output / name).exists())
        self.assertFalse(self.paths)

    async def test_unsupported_config_fails_before_any_http(self):
        self.config["bench_serving"]["extra_args"] += ["--disable-ignore-eos"]
        summary = await self.run_window(.1)
        self.assert_unscorable(summary, "startup_failure")
        self.assertFalse(self.paths)

    async def test_models_mismatch_never_starts_clock_or_generation(self):
        self.models = {"data": [{"id": "another-model"}]}
        summary = await self.run_window(.1)
        self.assert_unscorable(summary, "startup_failure")
        self.assertFalse(summary["measured"])
        self.assertIsNone(summary["actual_elapsed_wall_sec"])
        self.assertFalse(self.requests)

    async def test_preflight_health_and_idle_outside_clock(self):
        self.health_delay = .1
        started = time.monotonic()
        summary = await self.run_window(.08)
        self.assertGreater(time.monotonic() - started, .18)
        self.assertAlmostEqual(summary["actual_elapsed_wall_sec"], .08)
        idle = json.loads((self.output / "prewindow-idle.json").read_text())
        self.assertEqual(idle["status"], "idle")
        self.assertLessEqual(idle["finalized_unix_sec"], summary["start_unix_sec"])

    async def test_preflight_requires_both_expected_engines(self):
        self.config["bench_serving"]["window_expected_engines"] = 2
        summary = await self.run_window(.1)
        self.assert_unscorable(summary, "startup_failure")
        self.assertIn("2 engines", summary["error"])
        self.assertFalse(self.requests)

    async def test_fresh_output_refuses_any_preexisting_file(self):
        self.output.mkdir()
        sentinel = self.output / "unrelated.txt"
        sentinel.write_bytes(b"unchanged")
        with self.assertRaises(FileExistsError):
            await self.run_window(.1)
        self.assertEqual(list(self.output.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_bytes(), b"unchanged")
        self.assertFalse(self.paths)

    async def test_external_cancel_never_yields_scoring_pass(self):
        self.config["bench_serving"]["window_duration_sec"] = 2
        task = asyncio.create_task(probe._run_window(self.endpoint, self.output, self.config))
        await asyncio.wait_for(self.request_started.wait(), 1)
        await asyncio.sleep(.02)
        task.cancel()
        summary = await task
        self.assert_unscorable(summary, "cancelled")
        self.assertEqual(summary["request_counts"]["partial_cancelled"], 1)
        self.assertEqual(summary["actual_received_output_tokens"], 7)
        self.assertEqual(json.loads((self.output / "summary.json").read_text()), summary)

    async def test_non_deadline_timeout_is_failure_not_normal_cutoff(self):
        async def failed(*args):
            raise TimeoutError("transport timed out early")
        with patch.object(probe, "_replay", failed):
            summary = await self.run_window(1)
        self.assert_unscorable(summary)
        self.assertIn("transport timed out early", summary["error"])

    async def test_cli_three_args_calls_full_config_sync_api(self):
        self.behavior = "finished"
        self.config["bench_serving"]["window_duration_sec"] = 1
        config_path = self.directory / "verifier-config.json"
        config_path.write_text(json.dumps(self.config))
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(probe.__file__)), self.endpoint + "/v1/completions", str(self.output), str(config_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
        self.assertEqual(process.returncode, 0, stderr.decode())
        summary = json.loads(stdout)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["actual_received_output_tokens"], 14)
        self.assertEqual(len(self.requests), 2)

    async def test_cli_invalid_config_exits_nonzero_without_generation(self):
        self.config["bench_serving"]["max_concurrency"] = 4
        config_path = self.directory / "verifier-config.json"
        config_path.write_text(json.dumps(self.config))
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(probe.__file__)), self.endpoint, str(self.output), str(config_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), 10)
        self.assertEqual(process.returncode, 1)
        self.assert_unscorable(json.loads(stdout), "startup_failure")
        self.assertFalse(self.paths)


class StalledHTTPErrorWindow(HTTPFixture):
    async def complete(self, request):
        if not self.requests:
            return await super().complete(request)
        self.paths.append(request.path)
        self.requests.append(await request.json())
        response = web.StreamResponse(status=500, headers={"Content-Type": "text/plain"})
        response.content_length = 2048
        await response.prepare(request)
        self.error_headers_sent.set()
        # 已发送明确的失败状态；整个窗口内不发送错误正文，也不结束响应。
        await self.release.wait()
        return response

    async def test_completed_tokens_then_stalled_http_error_cannot_score_as_deadline(self):
        self.behavior = "finished"
        self.error_headers_sent = asyncio.Event()
        summary = await self.run_window(1)
        self.assertTrue(self.error_headers_sent.is_set())
        self.assertFalse(self.release.is_set())
        self.assert_unscorable(summary, "measurement_failure")
        self.assertTrue(summary["measured"])
        self.assertEqual(summary["actual_received_output_tokens"], 7)
        self.assertEqual(summary["request_counts"]["finished"], 1)
        self.assertEqual(summary["request_counts"]["failure"], 1)
        self.assertEqual(summary["request_counts"]["partial_deadline"], 0)
        self.assertLess(summary["actual_elapsed_wall_sec"], 1)
        self.assertIn("HTTP 500", summary["error"])
        self.assertEqual(len(self.requests), 2)
        logs = [json.loads(line) for line in (self.output / "requests.jsonl").read_text().splitlines()]
        self.assertEqual([item["status"] for item in logs], ["finished", "failure"])
        self.assertEqual(logs[1]["http_status"], 500)
        self.assertEqual(json.loads((self.output / "summary.json").read_text()), summary)


class Idle(HTTPFixture):
    async def test_valid_both_engines_idle_records_evidence(self):
        self.metrics = metric_text({0: (0, 0), 1: (0, 0)})
        evidence = await self.idle()
        self.assertEqual(evidence["status"], "idle")
        self.assertEqual(evidence["expected_engines"], 2)
        self.assertEqual(len(evidence["samples"]), 1)
        self.assertEqual(set(evidence["samples"][0]["gauges"]["vllm:num_requests_running"]), {"0", "1"})
        self.assertEqual(json.loads((self.directory / "idle.json").read_text()), evidence)
        self.assertFalse(self.requests)

    async def test_only_one_engine_zero_times_out_for_running_and_waiting(self):
        for index, busy in enumerate(((1, 0), (0, 1))):
            self.metrics = metric_text({0: (0, 0), 1: busy})
            name = f"busy-{index}.json"
            with self.assertRaises(TimeoutError):
                await self.idle(name, timeout=.04)
            evidence = json.loads((self.directory / name).read_text())
            self.assertEqual(evidence["status"], "timeout")
            self.assertFalse(evidence["samples"][0]["idle"])
            self.assertLess(evidence["actual_elapsed_wall_sec"], .5)

    async def test_busy_then_both_idle_persists_drain_samples(self):
        self.metric_series = [metric_text({0: (0, 0), 1: (1, 0)})]
        self.metrics = metric_text({0: (0, 0), 1: (0, 0)})
        with patch.object(probe, "_IDLE_POLL_SEC", .005):
            evidence = await self.idle(timeout=1)
        self.assertEqual(evidence["status"], "idle")
        self.assertEqual([sample["idle"] for sample in evidence["samples"]], [False, True])

    async def test_missing_engine_or_family_fails_closed(self):
        fixtures = [metric_text({0: (0, 0)}),
                    'vllm:num_requests_running{engine="0"} 0\nvllm:num_requests_running{engine="1"} 0\n',
                    'vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n',
                    metric_text({0: (0, 0), 1: (0, 0)}).replace('waiting{model_name="fixture-model",engine="1"}', 'waiting{model_name="fixture-model",engine="2"}'),
                    "# no metrics yet\n"]
        for index, text in enumerate(fixtures):
            self.metrics = text
            name = f"missing-{index}.json"
            with self.assertRaises(probe.ProtocolError):
                await self.idle(name)
            self.assertEqual(json.loads((self.directory / name).read_text())["status"], "failure")

    async def test_nonfinite_negative_duplicate_and_malformed_gauges_rejected(self):
        fixtures = [metric_text({0: (0, 0), 1: (value, 0)}) for value in ("NaN", "+Inf", "-Inf", -1)]
        valid = metric_text({0: (0, 0), 1: (0, 0)})
        fixtures += [valid + 'vllm:num_requests_running{engine="1"} 0\n',
                     valid.replace('engine="1"', 'engine="1",engine="1"'),
                     valid.replace('engine="1"', 'engine=1')]
        for index, text in enumerate(fixtures):
            self.metrics = text
            with self.assertRaises(probe.ProtocolError):
                await self.idle(f"invalid-{index}.json")

    async def test_single_engine_unlabeled_gauges_are_supported(self):
        self.metrics = "# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
        evidence = await self.idle(engines=1)
        self.assertEqual(evidence["status"], "idle")

    async def test_stalled_metrics_is_bounded(self):
        self.stall_path = "/metrics"
        with self.assertRaises(TimeoutError):
            await self.idle(timeout=.04)
        evidence = json.loads((self.directory / "idle.json").read_text())
        self.assertEqual(evidence["status"], "timeout")
        self.assertLess(evidence["actual_elapsed_wall_sec"], .5)
        self.assertEqual(len(evidence["samples"]), 1)

    async def test_stalled_health_is_bounded(self):
        self.stall_path = "/health"
        with self.assertRaises(TimeoutError):
            await self.idle(timeout=.04)
        evidence = json.loads((self.directory / "idle.json").read_text())
        self.assertEqual(evidence["status"], "timeout")
        self.assertFalse(evidence["samples"])
        self.assertFalse(self.requests)

    async def test_health_or_models_failure_is_not_idle(self):
        self.health_status = 503
        with self.assertRaisesRegex(probe.ProtocolError, "HTTP 503"):
            await self.idle("health-failure.json")
        self.health_status = 200
        self.models = {"data": []}
        with self.assertRaisesRegex(probe.ProtocolError, "model IDs"):
            await self.idle("models-failure.json")

    async def test_invalid_expected_engines_and_timeout(self):
        for index, (engines, timeout) in enumerate(((True, .1), (0, .1), (1, 0), (1, float("inf")))):
            with self.assertRaises(ValueError):
                await self.idle(f"bad-args-{index}.json", engines=engines, timeout=timeout)
        self.assertFalse(self.paths)


class OfflineGolden(unittest.TestCase):
    def test_actual_export_reproduces_every_archived_prompt_hash(self):
        names = ("FIXED_WINDOW_GOLDEN_MANIFEST", "FIXED_WINDOW_GOLDEN_VOCAB", "FIXED_WINDOW_GOLDEN_TRACE")
        paths = [os.environ.get(name) for name in names]
        if not any(paths):
            self.skipTest("optional offline golden requires explicit manifest, actual vocab export and original trace paths")
        self.assertTrue(all(paths), "provide all three FIXED_WINDOW_GOLDEN paths")
        manifest_path, vocab_path, trace_path = map(Path, paths)
        manifest = json.loads(manifest_path.read_text())
        params = manifest["parameters"]
        config = {
            "model_path": "unused-with-explicit-actual-vocab-export", "served_model_name": params["model"],
            "bench_serving": {
                "backend": "vllm", "dataset_name": "mooncake", "max_concurrency": 1,
                "window_duration_sec": params["duration_sec"], "seed": params["seed"],
                "window_expected_engines": 1,  # compiler-only fixture; no endpoint/engine probe
                "trace_sha256": manifest["trace_sha256"], "vocab_json": str(vocab_path),
                "extra_args": ["--dataset-path", str(trace_path), "--mooncake-context-length", str(params["context_length"]),
                               "--mooncake-slowdown-factor", str(params["slowdown_factor"]), "--mooncake-num-rounds", "1"],
            },
        }
        rows, metadata, _ = probe.load_workload(probe._parse_config(config))
        expected = manifest["requests"]
        self.assertEqual(len(rows), len(expected))
        self.assertEqual(metadata["tokenizer"]["file_sha256"], manifest["tokenizer"]["file_sha256"])
        self.assertEqual(metadata["tokenizer"]["vocab_and_special_ids_sha256"], manifest["tokenizer"]["vocab_and_special_ids_sha256"])
        self.assertEqual(metadata["compiler_definitions_sha256"], manifest["compiler_definitions_sha256"])
        self.assertEqual([probe.digest(probe.canonical(row.prompt)) for row in rows], [item["prompt_ids_sha256"] for item in expected])
        self.assertEqual([(row.timestamp, row.prompt_len, row.output_len) for row in rows],
                         [(item["trace_timestamp_ms"], item["requested_prompt_tokens"], item["requested_output_tokens"]) for item in expected])
        # 只检查归档内部口径，不声称新 helper 已完成设备上的 Oracle/正确性验证。
        summary = json.loads(manifest_path.with_name("summary.json").read_text())
        requests = [json.loads(line) for line in manifest_path.with_name("requests.jsonl").read_text().splitlines()]
        total = sum(item["got_output_tokens"] for item in requests)
        self.assertEqual(total, summary["actual_received_output_tokens"])
        self.assertEqual(total / summary["actual_elapsed_wall_sec"], summary["output_tps"])


if __name__ == "__main__":
    unittest.main()

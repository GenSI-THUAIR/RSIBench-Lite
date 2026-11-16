"""固定窗口归一化及真实loopback SSE→idle→GSM链路；不使用NPU。

参数：无。示例：python -m unittest tests.test_clean_benchmark_window
"""
from __future__ import annotations

from verify.resources import payload_dir
import hashlib
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

VERIFIERS = payload_dir()
sys.path.insert(0, str(VERIFIERS))
import clean_benchmark
from verify.inference.infer_verifier import _build_rewards


def sample_window():
    return {
        "status": "deadline", "measured": True, "counter_valid": True,
        "actual_elapsed_wall_sec": 3600.0, "actual_received_output_tokens": 3219,
        "output_tps": 3219 / 3600, "trace_request_count": 193,
        "request_counts": {"finished": 10, "partial_deadline": 1, "partial_cancelled": 0,
                           "failure": 0, "not_started": 182, "running": 0},
    }


class WindowNormalizationTest(unittest.TestCase):
    def setUp(self):
        self.bench = {"window_duration_sec": 3600, "num_prompts": 10000, "seed": 42, "max_concurrency": 1}

    def test_partial_output_and_unstarted_are_not_full_trace_failures(self):
        result = clean_benchmark._window_performance(sample_window(), self.bench)
        self.assertEqual(result["requests_attempted"], 11)
        self.assertEqual(result["completed"], 10)
        self.assertEqual(result["partial_deadline"], 1)
        self.assertEqual(result["not_started"], 182)
        self.assertEqual(result["request_success_rate"], 1.0)
        self.assertEqual(result["output_throughput"], 3219 / 3600)

    def test_reject_invalid_windows_even_with_counter_flag(self):
        mutations = [
            {"status": "measurement_failure"}, {"counter_valid": False}, {"measured": False},
            {"error_type": "ConnectionError"}, {"actual_elapsed_wall_sec": float("nan")},
            {"actual_elapsed_wall_sec": 3599}, {"actual_received_output_tokens": True},
            {"actual_received_output_tokens": -1}, {"output_tps": 100},
            {"output_tps": float("inf")}, {"selected_request_count": 194},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                raw = sample_window()
                raw.update(mutation)
                with self.assertRaises(ValueError):
                    clean_benchmark._window_performance(raw, self.bench)
        for key in ("failure", "partial_cancelled", "running"):
            raw = sample_window()
            raw["request_counts"][key] = 1
            raw["request_counts"]["not_started"] -= 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                clean_benchmark._window_performance(raw, self.bench)

    def test_asap_requires_matching_protocol_and_unlimited_rate(self):
        self.bench["arrival_mode"] = "asap"
        raw = sample_window()
        with self.assertRaisesRegex(ValueError, "arrival/protocol"):
            clean_benchmark._window_performance(raw, self.bench)
        raw.update(arrival_mode="asap", measurement_protocol="c1-fixed-window-token-ids-asap-v1", request_rate="inf")
        result = clean_benchmark._window_performance(raw, self.bench)
        self.assertEqual(result["arrival_mode"], "asap")
        self.assertEqual(result["measurement_protocol"], raw["measurement_protocol"])
        self.assertEqual(result["output_throughput"], 3219 / 3600)
        for mutation in ({"arrival_mode": "trace"}, {"request_rate": "trace"},
                         {"measurement_protocol": "c1-fixed-window-token-ids-v1"},
                         {"status": "workload_exhausted"}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                clean_benchmark._window_performance({**raw, **mutation}, self.bench)
        self.bench.pop("arrival_mode")
        with self.assertRaisesRegex(ValueError, "arrival/protocol"):
            clean_benchmark._window_performance(raw, self.bench)

    def test_zero_token_window_is_valid_but_cannot_earn_reward(self):
        raw = sample_window()
        raw.update(actual_received_output_tokens=0, output_tps=0)
        raw["request_counts"].update(finished=0, not_started=192)
        result = clean_benchmark._window_performance(raw, self.bench)
        rewards = _build_rewards({"gate": {"accuracy": 1.0}, "performance": result},
                                 baseline_throughput=3219 / 3600, gate_center=.915,
                                 gate_tolerance=.05, gate_comparison="lower_bound")
        self.assertEqual(rewards["reward"], 0)

    def test_window_rejects_audio_before_loading_reference_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({
                "gate": {"kind": "audio-tts-reference"},
                "bench_serving": {"window_duration_sec": 1},
            }))
            args = ["clean_benchmark.py", "http://unused", "unused", str(root / "out"), str(config)]
            with mock.patch.object(sys, "argv", args), mock.patch.dict(sys.modules, {"audio_eval": None}):
                with self.assertRaisesRegex(ValueError, "固定窗口不能"):
                    clean_benchmark.main()
            self.assertFalse((root / "out/benchmark-summary.json").exists())

    def test_early_finish_requires_every_selected_request(self):
        raw = sample_window()
        raw.update(status="completed", actual_elapsed_wall_sec=1800, output_tps=3219 / 1800)
        with self.assertRaises(ValueError):
            clean_benchmark._window_performance(raw, self.bench)
        raw["request_counts"].update(finished=193, partial_deadline=0, not_started=0)
        result = clean_benchmark._window_performance(raw, self.bench)
        self.assertEqual(result["wall_time_sec"], 1800)
        self.bench["window_full_duration"] = True
        with self.assertRaisesRegex(ValueError, "全部声明时长"):
            clean_benchmark._window_performance(raw, self.bench)


class WindowHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, value):
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._json({})
        elif self.path == "/v1/models":
            self._json({"data": [{"id": "fixture-model", "max_model_len": 256}]})
        elif self.path == self.server.metrics_path:
            active = self.server.active
            self.server.events.append(("metrics", active))
            prefix = self.server.metric_prefix
            data = (f'{prefix}num_requests_running{{engine="0",model_name="fixture-model"}} {active}\n'
                    f'{prefix}num_requests_waiting{{engine="0",model_name="fixture-model"}} 0\n').encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/completions":
            self.server.events.append(("window", body))
            self.server.active = 1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                if not self.server.zero_output:
                    event = {"choices": [{"index": 0, "text": "x", "finish_reason": None}],
                             "usage": {"completion_tokens": 1}}
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                    self.wfile.flush()
                # 即使客户端截止，也保留一段后端收尾时间，验证gate确实等待idle。
                time.sleep(.3)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                self.server.active = 0
                self.server.events.append(("window_drained", None))
        elif self.path == "/v1/chat/completions":
            self.server.events.append(("gate", self.server.active))
            answer = "2" if self.server.correct else "3"
            self._json({"choices": [{"message": {"content": "#### " + answer}, "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 2}})
        else:
            self.send_error(404)


class WindowWireE2eTest(unittest.TestCase):
    def run_case(self, *, correct=True, zero_output=False, drain_failure=False, idle_backend="vllm", arrival_mode="trace"):
        import fixed_window_benchmark
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), WindowHandler)
        except OSError as exc:
            self.skipTest(f"loopback unavailable: {exc}")
        server.daemon_threads = True
        server.events = []
        server.active = 0
        server.correct = correct
        server.zero_output = zero_output
        server.metrics_path = "/prometheus/metrics" if idle_backend == "trtllm" else "/metrics"
        server.metric_prefix = "trtllm_" if idle_backend == "trtllm" else "vllm:"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                trace = root / "trace.jsonl"
                trace.write_text(json.dumps({"timestamp": 0, "input_length": 4, "output_length": 100, "hash_ids": [1]}) + "\n")
                vocab = root / "vocab.json"
                vocab.write_text(json.dumps({"vocab": {str(i): i for i in range(16)}, "all_special_ids": [0, 1]}))
                gate_data = root / "gate.jsonl"
                gate_data.write_text(json.dumps({"question": "What is one plus one?", "answer": "#### 2"}) + "\n")
                config = {
                    "model_path": "unused-local-model", "served_model_name": "fixture-model",
                    "gate": {"kind": "gsm8k-openai-chat", "num_questions": 1, "parallel": 1,
                             "max_tokens": 32768, "strict_final_answer": True, "stop_tokens": [], "center": .915,
                             "tolerance": .05, "comparison": "lower_bound"},
                    "bench_serving": {"backend": "vllm", "script_path": "unused", "dataset_name": "mooncake",
                                      "random_input_len": 0, "random_output_len": 0, "random_range_ratio": 1.0,
                                      "max_concurrency": 1, "num_prompts": 10, "seed": 42,
                                      "window_duration_sec": .15, "window_expected_engines": 1,
                                      "vocab_json": str(vocab), "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                                      "extra_args": ["--dataset-path", str(trace), "--mooncake-context-length", "256", "--mooncake-slowdown-factor", "1.0"]},
                    "performance_baseline": {"output_throughput": 3219 / 3600},
                }
                config["bench_serving"]["window_idle_backend"] = idle_backend
                config["bench_serving"]["arrival_mode"] = arrival_mode
                config_path = root / "config.json"
                config_path.write_text(json.dumps(config))
                output = root / "out"
                args = ["clean_benchmark.py", f"http://127.0.0.1:{server.server_address[1]}", str(gate_data), str(output), str(config_path)]
                with mock.patch.object(sys, "argv", args), redirect_stdout(io.StringIO()):
                    if drain_failure:
                        with mock.patch.object(fixed_window_benchmark, "wait_for_idle", side_effect=TimeoutError("fixture drain timeout")):
                            with self.assertRaises(TimeoutError):
                                clean_benchmark.main()
                        self.assertFalse((output / "gate-result.json").exists())
                        self.assertFalse(any(event[0] == "gate" for event in server.events))
                        return
                    clean_benchmark.main()
                summary = json.loads((output / "benchmark-summary.json").read_text())
                self.assertEqual(summary["gate"]["accuracy"], 1.0 if correct else 0.0)
                self.assertEqual(summary["performance"]["partial_deadline"], 1)
                self.assertEqual(summary["performance"]["output_tokens"], 0 if zero_output else 1)
                self.assertEqual(summary["performance"]["request_success_rate"], 1.0)
                self.assertEqual(summary["performance"]["arrival_mode"], arrival_mode)
                self.assertEqual(summary["performance"]["request_rate"], "inf" if arrival_mode == "asap" else "trace")
                kinds = [event[0] for event in server.events]
                self.assertLess(kinds.index("window"), kinds.index("window_drained"))
                self.assertLess(kinds.index("window_drained"), kinds.index("gate"))
                self.assertEqual([value for kind, value in server.events if kind == "gate"], [0])
                rewards = _build_rewards(summary, baseline_throughput=3219 / 3600,
                                         gate_center=.915, gate_tolerance=.05, gate_comparison="lower_bound")
                self.assertEqual(rewards["reward"] > 0, correct and not zero_output)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_real_sse_window_then_idle_then_correctness(self):
        self.run_case()

    def test_asap_real_sse_window_then_idle_then_correctness(self):
        self.run_case(arrival_mode="asap")

    def test_trtllm_wire_drains_before_correctness(self):
        self.run_case(idle_backend="trtllm")

    def test_real_gate_failure_zeroes_valid_window(self):
        self.run_case(correct=False)

    def test_valid_zero_output_still_runs_gate(self):
        self.run_case(zero_output=True)

    def test_drain_failure_does_not_run_competing_gate(self):
        self.run_case(drain_failure=True)

"""用 mock 推理服务和 stub bench_serving 端到端运行共享 clean_benchmark。

参数：无。
示例：python3 -m unittest tests.test_clean_benchmark_e2e
"""

from __future__ import annotations

from verify.resources import payload_dir
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

from verify.inference.infer_verifier import (
    VerifierConfig,
    _build_rewards,
    _load_verifier_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIERS_DIR = payload_dir()
DATA_PATH = VERIFIERS_DIR / "gsm8k-official-test.jsonl"
MMMU_MINI = Path(__file__).parent / "fixtures/mmmu-mini.jsonl"
LONGBENCH_MINI = Path(__file__).parent / "fixtures/longbench-mini.jsonl"
INFER_VERIFIER = "verify.inference.infer_verifier:InferVerifier"


def discover_task_tests() -> dict[str, tuple[Path, Path]]:
    """自动覆盖所有走 clean_benchmark 的 verifier-config，不耦合具体 task 名称。

    使用五题当前生成的评测配置，覆盖文本、视觉及视频分派。
    """
    discovered = {}
    config_paths = (REPO_ROOT / "tasks").glob("*/tests/verifier-config.json")
    for config_path in sorted(config_paths):
        raw = json.loads(config_path.read_text())
        if raw.get("verifier", INFER_VERIFIER) != INFER_VERIFIER:
            continue
        kind = raw["gate"]["kind"]
        if kind == "mmmu-openai-chat":
            data = MMMU_MINI
        elif kind == "gpqa-diamond-openai-chat":
            data = REPO_ROOT / "datasets/gate/gpqa_diamond.jsonl"
        elif kind == "longbench-v2-openai-chat":
            # 冻结档位 jsonl 只在 HDFS;本地用 40 题 mini fixture 走同一 evaluator。
            data = LONGBENCH_MINI
        else:
            data = DATA_PATH
        discovered[config_path.parents[1].name] = (config_path.parent, data)
    return discovered


TASK_TESTS = discover_task_tests()

STUB_BENCH = '''\
"""stub run_official_bench:记录 argv/env 并写出固定的 bench_serving 结果。"""
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
output_file = Path(args[args.index("--output-file") + 1])
Path(output_file.parent, "bench-argv.json").write_text(json.dumps(args))
Path(output_file.parent, "bench-env.json").write_text(
    json.dumps({"HARBOR_BENCH_SERVING": os.environ.get("HARBOR_BENCH_SERVING")})
)
mode = args[args.index("--mooncake-arrival-mode") + 1] if "--mooncake-arrival-mode" in args else "trace"
output_file.write_text(
    json.dumps(
        {
            **({"arrival_mode": mode, "request_rate": "inf" if mode == "asap" else "trace",
                "measurement_protocol": f"mooncake-full-replay-{mode}-v1"}
               if args[args.index("--dataset-name") + 1] == "mooncake" else {}),
            "completed": 16,
            "median_ttft_ms": 100.0,
            "p99_ttft_ms": 200.0,
            "median_tpot_ms": 10.0,
            "p99_tpot_ms": 20.0,
            "output_throughput": -1.0,
        }
    )
    + "\\n"
)
'''

STUB_WINDOW = '''\
"""只用于配置分派测试：固定窗口产物；真实SSE/时钟由独立loopback测试覆盖。"""
import json
from pathlib import Path


def run_window(endpoint, output_dir, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = float(config["bench_serving"]["window_duration_sec"])
    mode = config["bench_serving"].get("arrival_mode", "trace")
    tokens = max(1, round(2 * config["performance_baseline"]["output_throughput"] * duration))
    selected = min(2, int(config["bench_serving"]["num_prompts"]))
    result = {
        "arrival_mode": mode, "request_rate": "inf" if mode == "asap" else "trace",
        "measurement_protocol": "c1-fixed-window-token-ids-asap-v1" if mode == "asap" else "c1-fixed-window-token-ids-v1",
        "status": "deadline", "measured": True, "counter_valid": True,
        "actual_elapsed_wall_sec": duration, "actual_received_output_tokens": tokens,
        "output_tps": tokens / duration, "trace_request_count": selected,
        "request_counts": {"finished": selected - 1, "partial_deadline": 1,
                           "partial_cancelled": 0, "failure": 0, "not_started": 0, "running": 0},
    }
    (output_dir / "summary.json").write_text(json.dumps(result))
    return result


def wait_for_idle(endpoint, evidence_path, *, expected_engines, timeout_sec=180, backend="vllm"):
    Path(evidence_path).write_text(json.dumps({"status": "idle", "expected_engines": expected_engines, "backend": backend}))
    return {"status": "idle"}
'''

STUB_VIDEO = '''\
"""stub video_eval:不依赖 FFmpeg/LPIPS，写出固定的可信汇总。"""
import json
from pathlib import Path


def evaluate(endpoint, reference_path, output_dir, config_path):
    del endpoint, reference_path
    config = json.loads(Path(config_path).read_text())
    baseline = float(config["performance_baseline"]["output_throughput"])
    output_dir = Path(output_dir)
    (output_dir / "video-output.mp4").write_bytes(b"stub-mp4")
    raw = {
        "unit": "videos/hour",
        "warmup_wall_time_sec": 2.0,
        "generation_wall_time_sec": 1.0,
        "videos_per_hour": 2.0 * baseline,
    }
    (output_dir / "performance-result.jsonl").write_text(json.dumps(raw) + "\\n")
    (output_dir / "performance.log").write_text(json.dumps(raw) + "\\n")
    return {
        "gate": {"accuracy": 1.0, "passed": True},
        "performance": {
            "num_prompts": 1,
            "max_concurrency": 1,
            "completed": 1,
            "request_success_rate": 1.0,
            "wall_time_sec": 1.0,
            "ttft_ms_p50": -1.0,
            "ttft_ms_p99": -1.0,
            "tpot_ms_p50": -1.0,
            "tpot_ms_p99": -1.0,
            "output_throughput": 2.0 * baseline,
            "raw": raw,
        },
    }
'''


class MockServiceHandler(BaseHTTPRequestHandler):
    """返回请求中最后一道 GSM8K 问题的标准答案。

    同时模拟两种被测服务 API:SGLang 原生 /generate(sglang_generate)和
    OpenAI 兼容 /v1/chat/completions(openai_chat)。官方答案自带
    '#### <number>' 结尾,两种提取路径都能命中。
    """

    answers: ClassVar[dict[str, str]] = {
        sample["question"]: sample["answer"]
        for sample in (json.loads(line) for line in DATA_PATH.read_text().splitlines())
    }
    gpqa_answers: ClassVar[dict[str, str]] = {
        sample["question"]: sample["answer"]
        for sample in (
            json.loads(line)
            for line in (REPO_ROOT / "datasets/gate/gpqa_diamond.jsonl")
            .read_text()
            .splitlines()
        )
    }
    mmmu_answers: ClassVar[dict[str, str]] = {
        sample["question"]: sample["answer"]
        for sample in (
            json.loads(line)
            for line in (Path(__file__).parent / "fixtures/mmmu-mini.jsonl")
            .read_text()
            .splitlines()
        )
    }
    longbench_answers: ClassVar[dict[str, str]] = {
        sample["question"]: sample["answer"]
        for sample in (json.loads(line) for line in LONGBENCH_MINI.read_text().splitlines())
    }

    def do_GET(self) -> None:
        """模拟 health 接口。"""
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def do_POST(self) -> None:
        """按请求路径模拟 /generate、/v1/chat/completions 。"""
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        if self.path == "/v1/chat/completions":
            content = request["messages"][-1]["content"]
            if isinstance(content, list):
                # mmmu-openai-chat:图片 + 文本内容块,按题干前缀查答案字母
                text = next(
                    part["text"] for part in content if part.get("type") == "text"
                )
                question = text.split("\n", 1)[0]
                reply = f"Answer: {self.mmmu_answers[question]}"
            elif content in self.answers:
                reply = self.answers[content]
            elif content.startswith("Please read the following text"):
                # longbench-v2-openai-chat:官方 0shot 版式,题干跟在固定问句之后
                question = content.split(
                    "What is the correct answer to this question: ", 1
                )[1].split("\n", 1)[0]
                reply = f"The correct answer is ({self.longbench_answers[question]})"
            else:
                # gpqa-diamond-openai-chat:simple-evals 版式,题干位于指令段
                # 与选项块之间
                question = content.split("\n\n", 1)[1].rsplit("\n\nA) ", 1)[0]
                reply = f"Answer: {self.gpqa_answers[question]}"
            payload = {
                "choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1},
            }
        else:
            question = (
                request["text"].rsplit("Question: ", 1)[1].split("\nAnswer:", 1)[0]
            )
            payload = {
                "text": self.answers[question],
                "meta_info": {"completion_tokens": 1},
            }
        response = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        """关闭 mock HTTP server 的逐请求日志。"""
        return


class Ipv6ThreadingHttpServer(ThreadingHTTPServer):
    """在 IPv6 loopback 上运行 mock HTTP 服务。"""

    address_family = socket.AF_INET6
    # evaluator 以 64 并发无重试直连;默认 backlog(5)偶发拒绝连接,
    # 造成 suite 约 1/6 概率 flaky。
    request_queue_size = 128


def make_mock_server() -> tuple[ThreadingHTTPServer, str]:
    """优先使用 IPv6，sandbox 禁止时退回本地 IPv4 mock endpoint。"""
    try:
        server = Ipv6ThreadingHttpServer(("::1", 0), MockServiceHandler)
    except OSError:
        try:
            ThreadingHTTPServer.request_queue_size = 128
            server = ThreadingHTTPServer(("127.0.0.1", 0), MockServiceHandler)
        except OSError as exc:
            raise unittest.SkipTest(
                f"local loopback sockets are unavailable: {exc}"
            ) from exc
        return server, f"http://127.0.0.1:{server.server_address[1]}"
    return server, f"http://[::1]:{server.server_address[1]}"


class CleanBenchmarkE2eTest(unittest.TestCase):
    """clean_benchmark 在每道 task 的 config 下产出可直接进 reward 的汇总。"""

    def test_summary_feeds_build_rewards(self) -> None:
        """GSM8K 走 mock 服务、bench 走 stub,汇总喂给 _build_rewards。"""
        server, endpoint = make_mock_server()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for task, (tests_dir, data_path) in TASK_TESTS.items():
                with self.subTest(task=task):
                    self._run_one_task(endpoint, tests_dir, data_path)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_asap_propagates_through_clean_driver_for_both_replay_paths(self):
        server, endpoint = make_mock_server()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        seen = set()
        try:
            for tests_dir, data_path in TASK_TESTS.values():
                raw = json.loads((tests_dir / "verifier-config.json").read_text())
                bench = raw.get("bench_serving", {})
                window = "window_duration_sec" in bench
                if bench.get("dataset_name") != "mooncake" or window in seen:
                    continue
                bench["arrival_mode"] = "asap"
                with self.subTest(window=window), tempfile.TemporaryDirectory() as directory:
                    copied = Path(directory)
                    (copied / "verifier-config.json").write_text(json.dumps(raw))
                    self._run_one_task(endpoint, copied, data_path)
                seen.add(window)
            self.assertEqual(seen, {False, True})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def _run_one_task(self, endpoint: str, tests_dir: Path, data_path: Path) -> None:
        config = _load_verifier_config(tests_dir)
        raw_config = json.loads((tests_dir / "verifier-config.json").read_text())
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir)
            for name in ("clean_benchmark.py", "gsm8k_eval.py", "gpqa_eval.py", "longbench_eval.py", "mmmu_eval.py"):
                shutil.copy(VERIFIERS_DIR / name, workdir / name)
            if config.gate_kind == "video-reference":
                (workdir / "video_eval.py").write_text(STUB_VIDEO)
                self._run_video_task(endpoint, tests_dir, data_path, workdir, config)
                return
            num_prompts = int(raw_config["bench_serving"]["num_prompts"])
            stub = STUB_BENCH.replace(
                '"output_throughput": -1.0',
                f'"output_throughput": {2 * config.baseline_output_throughput!r}',
            ).replace('"completed": 16', f'"completed": {num_prompts}')
            (workdir / "run_official_bench.py").write_text(stub)
            window_enabled = "window_duration_sec" in raw_config["bench_serving"]
            if window_enabled:
                (workdir / "fixed_window_benchmark.py").write_text(STUB_WINDOW)
            output_dir = workdir / "out"
            result = subprocess.run(
                [
                    sys.executable,
                    str(workdir / "clean_benchmark.py"),
                    endpoint,
                    str(data_path),
                    str(output_dir),
                    str(tests_dir / "verifier-config.json"),
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            if raw_config.get("agentic_rollout"):
                # rollout reward 模式:Benchmark Job 只出可信 gate,
                # 性能与正确性由 verifier host 侧 rollout 给出(单测另行覆盖)。
                summary = json.loads(
                    (output_dir / "benchmark-summary.json").read_text()
                )
                self.assertEqual(summary["mode"], "gate-only")
                self.assertEqual(summary["gate"]["accuracy"], 1.0)
                self.assertIsNone(summary["performance"])
                self.assertFalse((output_dir / "bench-argv.json").exists())
                return

            bench = raw_config["bench_serving"]
            summary = json.loads((output_dir / "benchmark-summary.json").read_text())
            self.assertEqual(summary["gate"]["accuracy"], 1.0)
            if window_enabled:
                self.assertFalse((output_dir / "bench-argv.json").exists())
                idle = json.loads((output_dir / "window-idle-before-gate.json").read_text())
                self.assertEqual(idle["expected_engines"], bench["window_expected_engines"])
                self.assertEqual(idle["backend"], bench.get("window_idle_backend", bench["backend"]))
                self.assertEqual(summary["performance"]["partial_deadline"], 1)
                self.assertEqual(summary["performance"]["request_success_rate"], 1.0)
                expected_score = round(2 * config.baseline_output_throughput * bench["window_duration_sec"]) / bench["window_duration_sec"] / config.baseline_output_throughput
            else:
                bench_argv = json.loads((output_dir / "bench-argv.json").read_text())
                if bench["backend"] == "sglang":
                    self.assertNotIn("--served-model-name", bench_argv)
                else:
                    self.assertIn(config.served_model_name, bench_argv)
                self.assertIn(config.model_path, bench_argv)
                self.assertEqual(bench_argv[bench_argv.index("--backend") + 1], bench["backend"])
                if "arrival_mode" in bench:
                    self.assertEqual(bench_argv[bench_argv.index("--mooncake-arrival-mode") + 1], bench["arrival_mode"])
                    self.assertEqual(summary["performance"]["arrival_mode"], bench["arrival_mode"])
                else:
                    self.assertNotIn("--mooncake-arrival-mode", bench_argv)
                bench_env = json.loads((output_dir / "bench-env.json").read_text())
                self.assertEqual(bench_env["HARBOR_BENCH_SERVING"], bench["script_path"])
                raw_reference = summary["performance"]["raw"]
                self.assertEqual(raw_reference["median_ttft_ms"], 100.0)
                self.assertEqual(raw_reference["completed"], num_prompts)
                expected_score = 2.0
            rewards = _build_rewards(
                summary,
                baseline_throughput=config.baseline_output_throughput,
                gate_center=config.gate_center,
                gate_tolerance=config.gate_tolerance,
                gate_comparison=config.gate_comparison,
            )
            expected_ok = int(
                1.0 >= config.gate_center - config.gate_tolerance
                if config.gate_comparison == "lower_bound"
                else abs(1.0 - config.gate_center) <= config.gate_tolerance
            )
            self.assertEqual(rewards["accuracy_within_tolerance"], expected_ok)
            self.assertAlmostEqual(rewards["performance_score"], expected_score)
            self.assertAlmostEqual(rewards["reward"], expected_score * expected_ok)


    def _run_video_task(
        self,
        endpoint: str,
        tests_dir: Path,
        data_path: Path,
        workdir: Path,
        config: VerifierConfig,
    ) -> None:
        """video-reference 分支走 stub evaluator 并产出可打分 artifact。"""
        output_dir = workdir / "out"
        result = subprocess.run(
            [
                sys.executable,
                str(workdir / "clean_benchmark.py"),
                endpoint,
                str(data_path),
                str(output_dir),
                str(tests_dir / "verifier-config.json"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((output_dir / "video-output.mp4").read_bytes(), b"stub-mp4")
        summary = json.loads((output_dir / "benchmark-summary.json").read_text())
        self.assertEqual(summary["gate"]["accuracy"], 1.0)
        rewards = _build_rewards(
            summary,
            baseline_throughput=config.baseline_output_throughput,
            gate_center=config.gate_center,
            gate_tolerance=config.gate_tolerance,
        )
        self.assertEqual(rewards["accuracy_within_tolerance"], 1)
        self.assertAlmostEqual(rewards["performance_score"], 2.0)
        self.assertAlmostEqual(rewards["reward"], 2.0)


if __name__ == "__main__":
    unittest.main()


class DeploymentFailureRewardTest(unittest.TestCase):
    """部署失败按协议判 0(deploy_success=0),不是裸异常。"""

    def test_deployment_failed_maps_to_zero_reward(self) -> None:
        from verify.inference.infer_verifier import (
            DeploymentFailed,
            _deployment_failure_rewards,
        )

        exc = DeploymentFailed("服务进程启动失败：rank-0 rc=1")
        self.assertIsInstance(exc, RuntimeError)
        rewards = _deployment_failure_rewards(exc)
        self.assertEqual(rewards, {"reward": 0.0, "deploy_success": 0})


class RolloutRewardTest(unittest.TestCase):
    """online rollout 模式的 reward 合成。"""

    def _verdict(self, solved, tput):
        return {"solved": solved, "n_tasks": 12,
                "perf": {"output_throughput": tput, "requests_200": 400,
                          "ttft_ms_p50": 900.0, "ttft_ms_p99": 8000.0,
                          "span_sec": 1800.0}}

    def test_pass_path_normalizes_throughput(self) -> None:
        from verify.inference.infer_verifier import _build_rollout_rewards
        r = _build_rollout_rewards(
            {"accuracy": 0.98}, self._verdict(11, 180.0),
            baseline_throughput=160.0, gate_center=0.985, gate_tolerance=0.05,
            min_solved=9)
        self.assertEqual(r["rollout_gate_passed"], 1)
        self.assertAlmostEqual(r["reward"], 180.0 / 160.0)

    def test_rollout_gate_failure_zeroes_reward(self) -> None:
        from verify.inference.infer_verifier import _build_rollout_rewards
        r = _build_rollout_rewards(
            {"accuracy": 0.98}, self._verdict(6, 200.0),
            baseline_throughput=160.0, gate_center=0.985, gate_tolerance=0.05,
            min_solved=9)
        self.assertEqual(r["reward"], 0.0)
        self.assertEqual(r["rollout_gate_passed"], 0)
        self.assertEqual(r["rollout_solved"], 6)

    def test_gsm8k_gate_failure_zeroes_reward(self) -> None:
        from verify.inference.infer_verifier import _build_rollout_rewards
        r = _build_rollout_rewards(
            {"accuracy": 0.80}, self._verdict(12, 200.0),
            baseline_throughput=160.0, gate_center=0.985, gate_tolerance=0.05,
            min_solved=9)
        self.assertEqual(r["reward"], 0.0)

"""窗口配置、单侧质量门和可信 staging/取证的定向回归，不连接 Merlin。

参数：无。示例：python -m unittest tests.test_infer_window_config
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import tarfile
import tempfile
import unittest
from pathlib import Path
from verify.resources import payload_dir, payload_file
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from verify.inference.infer_verifier import (
    CONFIG_NAME, InferVerifier, WINDOW_EVIDENCE_FILES,
    _build_rewards, _build_rollout_rewards, _load_verifier_config,
)
from schemas.build import _materialize, emit_verifier_config, load_yaml

ROOT = Path(__file__).resolve().parents[1]
MODULE = "verify.inference.infer_verifier"


def payload():
    """直接编译权威 YAML 到内存，避免依赖陈旧的根目录编译产物。"""
    cfg = load_yaml(ROOT / "tasks/infer-qwen3p8-27b/task.yaml")
    return json.loads(emit_verifier_config(_materialize(cfg)))


def summary(accuracy):
    return {
        "gate": {"accuracy": accuracy},
        "performance": {
            "output_throughput": 2.0, "request_success_rate": 1.0,
            "ttft_ms_p50": 1.0, "ttft_ms_p99": 2.0,
            "tpot_ms_p50": 3.0, "tpot_ms_p99": 4.0, "wall_time_sec": 600.0,
        },
    }


def verdict():
    return {
        "solved": 2, "n_tasks": 2,
        "perf": {
            "output_throughput": 2.0, "requests_200": 2,
            "ttft_ms_p50": 1.0, "ttft_ms_p99": 2.0, "span_sec": 600.0,
        },
    }


class ComparisonTest(unittest.TestCase):
    def test_lower_bound_inclusive_and_no_upper_rejection_in_both_paths(self):
        for accuracy, passed in ((0.865, 1), (math.nextafter(0.865, 0), 0),
                                 (0.975, 1), (1.0, 1)):
            with self.subTest(accuracy=accuracy):
                rewards = (
                    _build_rewards(summary(accuracy), 1.0, 0.915, 0.05, "lower_bound"),
                    _build_rollout_rewards({"accuracy": accuracy}, verdict(),
                                           1.0, 0.915, 0.05, 2, "lower_bound"),
                )
                for result in rewards:
                    self.assertEqual(result["accuracy_within_tolerance"], passed)
                    self.assertEqual(result["reward"], 2.0 * passed)
                    self.assertEqual(result["performance_score"], 2.0)

    def test_omitted_comparison_keeps_two_sided_behavior(self):
        for accuracy, passed in ((0.915, 1), (0.975, 0), (1.0, 0)):
            with self.subTest(accuracy=accuracy):
                for result in (
                    _build_rewards(summary(accuracy), 1.0, 0.915, 0.05),
                    _build_rollout_rewards({"accuracy": accuracy}, verdict(),
                                           1.0, 0.915, 0.05, 2),
                ):
                    self.assertEqual(result["accuracy_within_tolerance"], passed)

    def test_unknown_comparison_rejected_by_both_paths(self):
        for comparison in ("upper_bound", "LOWER_BOUND", "", None, True, 1, [], {}):
            with self.subTest(comparison=comparison):
                with self.assertRaisesRegex(ValueError, "gate.comparison"):
                    _build_rewards(summary(1), 1, 0.915, 0.05, comparison)
                with self.assertRaisesRegex(ValueError, "gate.comparison"):
                    _build_rollout_rewards({"accuracy": 1}, verdict(),
                                           1, 0.915, 0.05, 2, comparison)


class WindowConfigTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw = payload()

    def load(self):
        (self.root / CONFIG_NAME).write_text(json.dumps(self.raw))
        return _load_verifier_config(self.root)

    def test_arrival_validation_covers_window_and_full_replay(self):
        for window in (True, False):
            with self.subTest(window=window):
                bench = self.raw["bench_serving"]
                if not window:
                    bench.pop("window_duration_sec")
                for mode in ("trace", "asap"):
                    bench["arrival_mode"] = mode
                    self.load()
                for mode in (None, True, 0, [], {}, "inf", "ASAP"):
                    with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "arrival_mode"):
                        bench["arrival_mode"] = mode
                        self.load()
                bench["arrival_mode"] = "asap"
                for extra in (["--request-rate", "1"], ["--use-trace-timestamps"],
                              ["--mooncake-arrival-mode=trace"]):
                    with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "extra_args"):
                        original = bench["extra_args"]
                        try:
                            bench["extra_args"] = original + extra
                            self.load()
                        finally:
                            bench["extra_args"] = original
        bench["dataset_name"] = "image"
        with self.assertRaisesRegex(ValueError, "only valid for mooncake"):
            self.load()



    def test_legacy_comparison_and_benchmark_need_no_new_fields(self):
        self.raw["gate"].pop("comparison")
        bench = self.raw["bench_serving"]
        bench.pop("window_duration_sec")
        bench.pop("window_expected_engines")
        bench["max_concurrency"] = 4
        self.assertEqual(self.load().gate_comparison, "two_sided")

    def test_loader_strict_comparison_enum(self):
        for comparison in ("two_sided", "lower_bound"):
            self.raw["gate"]["comparison"] = comparison
            self.assertEqual(self.load().gate_comparison, comparison)
        for comparison in ("bad", "Lower_Bound", None, 1, True, [], {}):
            with self.subTest(comparison=comparison):
                self.raw["gate"]["comparison"] = comparison
                with self.assertRaisesRegex(ValueError, "gate.comparison"):
                    self.load()

    def test_trtllm_idle_requires_openai_wire_and_single_engine(self):
        self.raw["bench_serving"].update(backend="vllm", window_idle_backend="trtllm", window_expected_engines=1)
        self.load()
        for changes in ({"backend": "sglang"}, {"window_expected_engines": 2},
                        {"window_idle_backend": "unknown"}, {"window_idle_backend": None}):
            original = copy.deepcopy(self.raw)
            self.raw["bench_serving"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load()
            self.raw = original

    def test_duration_must_be_finite_positive_and_bounded(self):
        for duration in (0.1, 600.0, 3600):
            self.raw["bench_serving"]["window_duration_sec"] = duration
            self.load()
        for duration in (0, -1, 3600.1, True, None, "600", float("nan"), float("inf")):
            with self.subTest(duration=duration):
                self.raw["bench_serving"]["window_duration_sec"] = duration
                with self.assertRaisesRegex(ValueError, "window_duration_sec"):
                    self.load()

    def test_window_rejects_incompatible_protocols(self):
        original = copy.deepcopy(self.raw)
        for key, value in (("backend", "unknown"), ("dataset_name", "random-ids"),
                           ("max_concurrency", 4), ("max_concurrency", True),
                           ("max_concurrency", 1.0)):
            with self.subTest(key=key, value=value):
                self.raw = copy.deepcopy(original)
                self.raw["bench_serving"][key] = value
                if key == "dataset_name":
                    # arrival_mode 是 mooncake 专属字段,换数据集时一并移除,
                    # 否则先撞上 arrival_mode 守卫,验不到这里要的后端/数据集守卫。
                    self.raw["bench_serving"].pop("arrival_mode", None)
                with self.assertRaisesRegex(ValueError, "vllm/sglang/mooncake"):
                    self.load()
        for kind in ("video-reference", "action-reference", "realtime-video-reference", "audio-tts-reference"):
            with self.subTest(kind=kind):
                self.raw = copy.deepcopy(original)
                self.raw["gate"]["kind"] = kind
                with self.assertRaisesRegex(ValueError, "rollout/video/action|gate.kind"):
                    self.load()
        for key in ("agentic_rollout", "video_benchmark", "action_benchmark",
                    "realtime_video_benchmark", "audio_benchmark"):
            with self.subTest(key=key):
                self.raw = copy.deepcopy(original)
                self.raw[key] = {}
                with self.assertRaisesRegex(ValueError, "rollout/video/action|gate.kind"):
                    self.load()

    def test_expected_engines_required_positive_integer_not_physical_node_count(self):
        bench = self.raw["bench_serving"]
        bench.pop("window_expected_engines")
        with self.assertRaisesRegex(ValueError, "window_expected_engines"):
            self.load()
        for engines in (0, -1, True, 1.5, 2.0, "2", None):
            with self.subTest(engines=engines):
                bench["window_expected_engines"] = engines
                with self.assertRaisesRegex(ValueError, "window_expected_engines"):
                    self.load()
        bench["window_expected_engines"] = 3
        self.load()  # Logical DP engines need not equal physical serve nodes.

    def test_nonpositive_baseline_still_rejected(self):
        for value in (0, -1):
            self.raw["performance_baseline"]["output_throughput"] = value
            with self.assertRaisesRegex(ValueError, "output_throughput.*正数"):
                self.load()


class WindowEvidenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_collects_only_fixed_existing_paths_and_makes_window_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = set(WINDOW_EVIDENCE_FILES) - {"window/summary.json"}

            async def probe(command, **kwargs):
                name = command.removeprefix("test -f /logs/verifier/")
                return SimpleNamespace(return_code=0 if name in existing else 1)

            async def download(source, destination):
                destination.write_text(source)

            env = SimpleNamespace(exec=AsyncMock(side_effect=probe),
                                  download_file=AsyncMock(side_effect=download))
            fake = SimpleNamespace(environment=env, logger=MagicMock(),
                                   trial_paths=SimpleNamespace(verifier_dir=root))
            await InferVerifier._download_window_evidence(fake)
            self.assertEqual(env.download_file.await_count, 3)
            for name in existing:
                self.assertTrue((root / name).is_file())
            self.assertFalse((root / "window/summary.json").exists())
            fake.logger.warning.assert_not_called()

    async def test_evidence_failure_does_not_skip_other_files_or_mask_original_error(self):
        env = SimpleNamespace(exec=AsyncMock(side_effect=TimeoutError("transport")),
                              download_file=AsyncMock())
        fake = SimpleNamespace(environment=env, logger=MagicMock())
        await InferVerifier._download_window_evidence(fake)
        self.assertEqual(env.exec.await_count, len(WINDOW_EVIDENCE_FILES))
        self.assertEqual(fake.logger.warning.call_count, len(WINDOW_EVIDENCE_FILES))


class WindowStagingTest(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_helper_staging_and_window_evidence_even_on_failure(self):
        for window, rc in ((False, 0), (True, 0), (True, 1)):
            with self.subTest(window=window, rc=rc), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                raw = payload()
                if not window:
                    raw["bench_serving"].pop("window_duration_sec")
                (root / CONFIG_NAME).write_text(json.dumps(raw))
                config = _load_verifier_config(root)
                verifier = InferVerifier.__new__(InferVerifier)
                verifier.task = SimpleNamespace(paths=SimpleNamespace(tests_dir=root))
                verifier.trial_paths = SimpleNamespace(verifier_dir=root / "result")
                verifier.logger = MagicMock()
                verifier._exec_baseline_probe = AsyncMock(side_effect=[
                    SimpleNamespace(stdout="commit", return_code=0),
                    SimpleNamespace(stdout="tree", return_code=0),
                ])
                verifier._download_window_evidence = AsyncMock()

                async def download(source, destination):
                    destination.write_text(json.dumps(summary(0.975)))

                env = SimpleNamespace(
                    upload_file=AsyncMock(), download_file=AsyncMock(side_effect=download),
                    exec=AsyncMock(side_effect=[
                        SimpleNamespace(stdout="", return_code=0),
                        SimpleNamespace(stdout=f"DONE {rc}", return_code=0),
                        SimpleNamespace(stdout="benchmark diagnostic", return_code=0),
                    ]),
                )
                verifier.environment = env
                with patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock), patch(
                    f"{MODULE}._verifier_script", side_effect=lambda name: payload_dir() / name
                ):
                    if rc:
                        with self.assertRaisesRegex(RuntimeError, "benchmark diagnostic"):
                            await verifier._run_clean_benchmark(
                                "http://serve", {"base_commit": "commit", "baseline_tree": "tree"}, config
                            )
                    else:
                        await verifier._run_clean_benchmark(
                            "http://serve", {"base_commit": "commit", "baseline_tree": "tree"}, config
                        )
                env.upload_file.assert_any_await(
                    payload_file('fixed_window_benchmark.py'), "/logs/verifier/fixed_window_benchmark.py"
                )
                self.assertEqual(verifier._download_window_evidence.await_count, int(window))


class ComparisonCallsiteTest(unittest.IsolatedAsyncioTestCase):
    async def test_expired_verifier_deadline_still_stops_serve(self):
        from verify.grading.phase_budget import phase
        work = ROOT / '.agents/workdir'
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as directory:
            root = Path(directory)
            (root / CONFIG_NAME).write_text(json.dumps(payload()))
            metadata = root / 'patch-metadata.json'
            metadata.write_text('{}')
            verifier = InferVerifier.__new__(InferVerifier)
            verifier.task = SimpleNamespace(paths=SimpleNamespace(tests_dir=root))
            verifier.trial_paths = SimpleNamespace(verifier_dir=root)
            verifier._artifact = MagicMock(return_value=metadata)

            async def start(**kwargs):
                await asyncio.Event().wait()

            async def logs(*args):
                await asyncio.sleep(.03)

            env = SimpleNamespace(start=AsyncMock(side_effect=start), stop=AsyncMock())
            verifier._create_serve_environment = MagicMock(return_value=env)
            verifier._download_per_pod_serve_logs = AsyncMock(side_effect=logs)
            verifier._download_serve_logs = AsyncMock(side_effect=logs)
            with self.assertRaises(TimeoutError):
                async with phase(root / 'phases', 'verifier', .05):
                    await verifier._verify_impl()
            env.stop.assert_awaited_once_with(delete=True)

    async def test_verify_passes_comparison_to_both_reward_paths(self):
        for rollout in (False, True):
            with self.subTest(rollout=rollout), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                raw = payload()
                raw.pop("serve_launch_mode", None)
                raw["performance_baseline"]["output_throughput"] = 1.0
                if rollout:
                    raw["bench_serving"].pop("window_duration_sec")
                    raw["agentic_rollout"] = {"subset": ["a", "b"], "min_solved": 2}
                (root / CONFIG_NAME).write_text(json.dumps(raw))
                metadata = root / "patch-metadata.json"
                metadata.write_text("{}")
                verifier = InferVerifier.__new__(InferVerifier)
                verifier.task = SimpleNamespace(paths=SimpleNamespace(tests_dir=root))
                verifier.trial_paths = SimpleNamespace(verifier_dir=root)
                verifier._artifact = MagicMock(return_value=metadata)
                env = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(),
                    exec=AsyncMock(return_value=SimpleNamespace(return_code=0,
                        stdout='RUNNING', stderr="")))
                verifier._create_serve_environment = MagicMock(return_value=env)
                verifier._start_patched_service = AsyncMock(return_value="http://serve")
                verifier._run_clean_benchmark = AsyncMock(return_value=summary(0.975))
                verifier._run_agentic_rollout = AsyncMock(return_value=verdict())
                verifier._download_serve_logs = AsyncMock()
                result = await verifier._verify_impl()
                self.assertEqual(result.rewards["reward"], 2.0)
                env.stop.assert_awaited_once_with(delete=True)


if __name__ == "__main__":
    unittest.main()

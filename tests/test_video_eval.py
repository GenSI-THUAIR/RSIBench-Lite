"""测试通用 video-reference evaluator 的纯函数协议。"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from verify.resources import payload_module
video = payload_module("video_eval.py")
from verify.resources import payload_module
_form_value = payload_module("video_eval.py")._form_value
_media_contract = payload_module("video_eval.py")._media_contract


def synthetic_benchmark() -> dict:
    """只供 CPU 单测的合成协议；数字和收据绝不可用于真实模型准入。"""
    return {
        "mode": "video-only", "case_ids": ["a-42", "a-43", "b-42", "b-43"],
        "endpoint_path": "/v1/videos/sync", "timeout_sec": 5,
        "warmup_num_inference_steps": 1, "reference_manifest_sha256": "a" * 64,
        "warmup": {"prompt": "public synthetic warmup only", "seed": 0},
        "request": {"width": 32, "height": 32, "num_frames": 5, "fps": 24,
                    "num_inference_steps": 4, "guidance_scale": 5, "flow_shift": 5,
                    "negative_prompt": "synthetic only", "extra_params": {"sample_solver": "unipc"}},
        "identity": {"model_revision": "synthetic", "weights_manifest_sha256": "synthetic",
                     "engine_commit": "synthetic", "image_digest": "synthetic",
                     "sampling": {"dtype": "float32", "fixture": True},
                     "metric_identity": {"name": "synthetic-test-not-lpips"}},
        "quality": {"expected_video_codec": "h264", "expected_fps": 24,
                    "expected_frame_count": 5, "duration_tolerance_sec": 0.001,
                    "timestamp_tolerance_sec": 0.0001,
                    "lpips_max": 0.01, "lpips_worst_frame_max": 0.02,
                    "temporal_delta_mae_max": 0.01, "temporal_delta_worst_max": 0.02,
                    "motion_ratio_min": 0.9, "motion_ratio_max": 1.1},
        "calibration": {"status": "admitted", "path": "/synthetic/receipt.json", "sha256": "b" * 64},
    }


def synthetic_metrics() -> dict:
    """返回只供协议单测使用的正控指标。"""
    return {"frame_count": 5, "lpips": 0.0, "lpips_worst_frame": 0.0,
            "temporal_delta_mae": 0.0, "temporal_delta_worst": 0.0, "motion_ratio": 1.0}


def write_pool(root: Path, benchmark: dict, payload: bytes = b"synthetic-not-mp4") -> Path:
    """生成可信资产绑定的合成池/收据，不能代表 GPU 标定。"""
    cases = []
    for i, case_id in enumerate(benchmark["case_ids"]):
        path = root / f"{case_id}.mp4"
        path.write_bytes(payload)
        cases.append({"id": case_id, "prompt": "scene-a" if i < 2 else "scene-b",
                      "seed": 42 + i % 2, "reference": path.name, "sha256": video._sha256(path)})
    path = root / "pool.json"
    path.write_text(json.dumps({"schema": "video-reference-pool/1", "identity": benchmark["identity"],
                                "request": benchmark["request"], "cases": cases}))
    benchmark["reference_manifest_sha256"] = video._sha256(path)
    protocol_hash = video._json_sha256({k: v for k, v in benchmark.items() if k != "calibration"})
    runs = []
    families = ["baseline"] * 3 + ["known-good"] + sorted(video.VIDEO_NEGATIVE_FAMILIES)
    for i, family in enumerate(families):
        metrics = synthetic_metrics()
        if family in video.VIDEO_NEGATIVE_FAMILIES:
            metrics["temporal_delta_mae"] = 0.8
        report_path = root / f"run-{i}.json"
        report_path.write_text(json.dumps({"protocol_sha256": protocol_hash, "cases": [
            {"id": c["id"], "contract": {"passed": True}, "video": metrics} for c in cases
        ]}))
        runs.append({"run_id": f"synthetic-{i}", "family": family,
                     "report": report_path.name, "sha256": video._sha256(report_path)})
    receipt_path = root / "receipt.json"
    receipt_path.write_text(json.dumps({"schema": "video-calibration/1", "protocol_sha256": protocol_hash, "runs": runs}))
    benchmark["calibration"] = {"status": "admitted", "path": str(receipt_path), "sha256": video._sha256(receipt_path)}
    return path


class VideoEvalTest(unittest.TestCase):
    def test_form_value_encodes_nested_extra_params(self) -> None:
        self.assertEqual(
            _form_value({"task": "t2va", "audio_flow_shift": 3.0}),
            '{"task":"t2va","audio_flow_shift":3.0}',
        )
        self.assertEqual(_form_value(True), "true")

    def test_media_contract_accepts_h3_output(self) -> None:
        probe = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 672,
                    "height": 384,
                    "r_frame_rate": "24/1",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "32000",
                    "channels": 2,
                },
            ],
            "format": {"duration": "4.458"},
        }
        request = {"width": 672, "height": 384}
        quality = {
            "expected_video_codec": "h264",
            "expected_audio_codec": "aac",
            "expected_fps": 24,
            "expected_audio_sample_rate": 32000,
            "expected_audio_channels": 2,
        }
        result = _media_contract(probe, request, quality)
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))


class VideoOnlyProtocolTest(unittest.TestCase):
    def test_audio_video_still_requires_audio(self) -> None:
        """旧模式缺音频必须失败，不得退化为纯视频。"""
        with self.assertRaisesRegex(ValueError, "audio"):
            _media_contract({"streams": [{"codec_type": "video"}]}, {}, {})

    def test_unadmitted_or_incomplete_config_rejected(self) -> None:
        """纯视频不得用默认阈值或 pending 收据评分。"""
        benchmark = synthetic_benchmark()
        video.validate_video_only(benchmark)
        for key in video.VIDEO_ONLY_THRESHOLDS:
            broken = copy.deepcopy(benchmark)
            del broken["quality"][key]
            with self.subTest(key=key), self.assertRaises(KeyError):
                video.validate_video_only(broken)
        benchmark["calibration"]["status"] = "pending"
        with self.assertRaisesRegex(ValueError, "pending"):
            video.validate_video_only(benchmark)

    def test_unknown_mode_rejected(self) -> None:
        """模式拼错不能默默进入默认音视频路径。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"video_benchmark": {"mode": "video-olny"}}))
            with self.assertRaisesRegex(ValueError, "mode"):
                video.evaluate("unused", root, root, config)

    def test_pool_requires_every_calibration_family(self) -> None:
        """合成收据验证完整路径并确保缺负控不能准入。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = synthetic_benchmark()
            path = write_pool(root, benchmark)
            self.assertEqual(len(video._load_video_pool(path, benchmark)), 4)
            receipt_path = Path(benchmark["calibration"]["path"])
            receipt = json.loads(receipt_path.read_text())
            receipt["runs"].pop()
            receipt_path.write_text(json.dumps(receipt))
            benchmark["calibration"]["sha256"] = video._sha256(receipt_path)
            with self.assertRaisesRegex(ValueError, "缺重复"):
                video._load_video_pool(path, benchmark)

    def test_timed_pool_cannot_reuse_warmup_input(self) -> None:
        """即使steps不同，也不允许计时输入复用warmup的prompt/seed。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = synthetic_benchmark()
            benchmark["warmup"] = {"prompt": "scene-a", "seed": 42}
            pool = write_pool(root, benchmark)
            with self.assertRaisesRegex(ValueError, "不得复用 warmup"):
                video._load_video_pool(pool, benchmark)

    def test_negative_label_cannot_force_failure(self) -> None:
        """负控必须被实际指标拒绝，不能只相信 family 标签。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = synthetic_benchmark()
            path = write_pool(root, benchmark)
            receipt_path = Path(benchmark["calibration"]["path"])
            receipt = json.loads(receipt_path.read_text())
            run = receipt["runs"][-1]
            report_path = root / run["report"]
            report = json.loads(report_path.read_text())
            for record in report["cases"]:
                record["video"] = synthetic_metrics()
            report_path.write_text(json.dumps(report))
            run["sha256"] = video._sha256(report_path)
            receipt_path.write_text(json.dumps(receipt))
            benchmark["calibration"]["sha256"] = video._sha256(receipt_path)
            with self.assertRaisesRegex(ValueError, "负控未被指标拒绝"):
                video._load_video_pool(path, benchmark)

    def test_reference_or_protocol_drift_rejected(self) -> None:
        """参考内容、完整采样参数与阈值漂移都不能复用旧收据。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = synthetic_benchmark()
            path = write_pool(root, benchmark)
            benchmark["quality"]["lpips_max"] = 0.8
            with self.assertRaisesRegex(ValueError, "calibration 未绑定"):
                video._load_video_pool(path, benchmark)
            benchmark["quality"]["lpips_max"] = 0.01
            (root / "a-42.mp4").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "内容漂移"):
                video._load_video_pool(path, benchmark)

    def test_exact_reference_quality_is_not_execution_proof(self) -> None:
        """相同正确输出通过数值质量；是否预录交OCI层校验/审查，不自造不可能负控。"""
        self.assertTrue(all(video._quality_checks(synthetic_metrics(), synthetic_benchmark()["quality"]).values()))

    def test_nonfinite_metric_fails_closed(self) -> None:
        """NaN/Inf 和静态高一致性均不能通过时间质量门。"""
        quality = synthetic_benchmark()["quality"]
        for key in synthetic_metrics():
            if key == "frame_count":
                continue
            metrics = synthetic_metrics()
            metrics[key] = float("nan")
            self.assertFalse(all(video._quality_checks(metrics, quality).values()))
        metrics = synthetic_metrics()
        metrics["motion_ratio"] = 0
        self.assertFalse(all(video._quality_checks(metrics, quality).values()))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy")
    def test_real_temporal_arrays_reject_repeat_padding_and_resize(self) -> None:
        """真实数组负控：静态复帧、少帧补齐和缩小放大；不是模型阈值标定。"""
        import numpy as np

        rng = np.random.default_rng(42)
        reference = rng.random((5, 32, 32, 3), dtype=np.float32)
        quality = synthetic_benchmark()["quality"]
        known_good = video._temporal_metrics(reference, reference.copy())
        self.assertEqual(known_good["temporal_delta_mae"], 0)
        self.assertEqual(known_good["motion_ratio"], 1)
        negatives = [np.repeat(reference[:1], 5, axis=0),
                     np.concatenate([reference[:2], np.repeat(reference[1:2], 3, axis=0)]),
                     np.repeat(np.repeat(reference[:, ::2, ::2], 2, axis=1), 2, axis=2)]
        for candidate in negatives:
            metrics = {**synthetic_metrics(), **video._temporal_metrics(reference, candidate)}
            self.assertFalse(all(video._quality_checks(metrics, quality).values()))
        with self.assertRaisesRegex(ValueError, "非零"):
            video._temporal_metrics(negatives[0], negatives[0])

    @unittest.skipUnless(importlib.util.find_spec("av"), "requires PyAV")
    def test_real_mp4_decode_and_workload_contract(self) -> None:
        """实际编码/全解码 MP4，验证帧数、时长和时间轴，不使用伪元数据。"""
        import av
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = synthetic_benchmark()
            path = root / "moving.mp4"
            container = av.open(str(path), mode="w")
            stream = container.add_stream("libx264", rate=24)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            for i in range(5):
                pixels = np.zeros((32, 32, 3), dtype=np.uint8)
                pixels[:, i * 4:i * 4 + 4, :] = 255
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            result = video._video_only_contract(path, benchmark["request"], benchmark["quality"])
            self.assertTrue(result["passed"], result)
            self.assertEqual(video._decode_video(path).shape, (5, 32, 32, 3))
            bad_quality = {**benchmark["quality"], "expected_frame_count": 6}
            self.assertFalse(video._video_only_contract(path, benchmark["request"], bad_quality)["passed"])
            mkv = root / "not-mp4.mkv"
            container = av.open(str(mkv), mode="w")
            stream = container.add_stream("libx264", rate=24)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            for pixels in video._decode_video(path):
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            result = video._video_only_contract(mkv, benchmark["request"], benchmark["quality"])
            self.assertFalse(result["checks"]["mp4_container"])
            self.assertFalse(result["passed"])

    def test_media_failure_keeps_all_evidence_and_negative_gate(self) -> None:
        """错误媒体工作量不补帧，仍回收整个工作集并形成负gate。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = synthetic_benchmark()
            pool = write_pool(root, benchmark)
            config = root / "config.json"
            config.write_text(json.dumps({"video_benchmark": benchmark}))

            def generate(endpoint, endpoint_path, request, output, timeout):
                """生成合成负控证据，媒体元数据由独立mock给出。"""
                output.write_bytes(b"synthetic-short-video")
                return 0.5

            with patch.object(video, "_run_request", side_effect=generate), patch.object(
                video, "_video_only_contract", side_effect=[{"passed": True}] * 4 + [{"passed": False}] * 4
            ), patch.object(video, "_video_metrics") as metrics:
                summary = video.evaluate("unused", pool, root / "output", config)
            metrics.assert_not_called()
            self.assertFalse(summary["gate"]["passed"])
            self.assertEqual(summary["gate"]["accuracy"], 0.0)
            self.assertEqual(len(summary["gate"]["cases"]), 4)
            self.assertEqual(len(list((root / "output/video-outputs").glob("*.mp4"))), 4)
            self.assertTrue((root / "output/video-evidence.json").is_file())

    def test_full_workset_timing_evidence_and_gate(self) -> None:
        """真实运行 evaluator 编排，固定合成生成器/指标隔离协议与模型质量。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = synthetic_benchmark()
            pool = write_pool(root, benchmark)
            config = root / "config.json"
            config.write_text(json.dumps({"video_benchmark": benchmark}))
            requests = []

            def generate(endpoint, endpoint_path, request, output, timeout):
                """合成请求写出小证据文件，记录全部请求。"""
                requests.append(request)
                output.write_bytes(b"synthetic-generated")
                return 0.5

            with patch.object(video, "_run_request", side_effect=generate), patch.object(
                video, "_video_only_contract", return_value={"passed": True}
            ), patch.object(video, "_video_metrics", return_value=synthetic_metrics()), patch.object(
                video.time, "perf_counter", side_effect=[100.0, 104.0]
            ):
                summary = video.evaluate("unused", pool, root / "output", config)
            self.assertEqual(len(requests), 5)
            self.assertEqual(requests[0]["num_inference_steps"], 1)
            self.assertTrue(all(r["num_inference_steps"] == 4 for r in requests[1:]))
            self.assertEqual(summary["performance"]["output_throughput"], 3600)
            self.assertEqual(summary["performance"]["wall_time_sec"], 4)
            self.assertTrue(summary["gate"]["passed"])
            evidence = json.loads((root / "output/video-evidence.json").read_text())
            self.assertEqual([r["id"] for r in evidence["cases"]], benchmark["case_ids"])
            self.assertEqual(len(list((root / "output/video-outputs").glob("*.mp4"))), 4)


class CompiledVideoReceiptTest(unittest.TestCase):
    """真实compiler→task-local收据→生产loader回归；媒体用小字节fixture，不冒充模型标定。"""

    def test_compiled_float_protocol_and_all_reports_round_trip(self) -> None:
        """重复build保持float，13文件真实hash链通过；整数化中间JSON或坏report必须拒绝。"""
        from schemas.build import load_yaml

        repo = Path(__file__).resolve().parents[1]
        source = repo / "tasks/infer-wan2p2ti2v-5b/prepare/package_calibration.py"
        spec = importlib.util.spec_from_file_location("wan22_receipt_test", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"video_eval": video}):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = synthetic_benchmark()
            benchmark["request"]["guidance_scale"] = 5.0
            benchmark["request"]["flow_shift"] = 5.0
            pool = write_pool(root, benchmark)
            cfg = load_yaml(repo / "tasks/infer-wan2p2ti2v-5b/task.yaml")
            cfg["verifier"]["scoring"]["video_benchmark"] = benchmark
            cfg["verifier"]["scoring"]["gate"]["data"] = str(pool)
            task = root / "task"
            task.mkdir()
            (task / "task.yaml").write_text(json.dumps(cfg))
            generated = task / "tests/verifier-config.json"
            command = [sys.executable, "-m", "schemas.build", str(task)]
            subprocess.run(command, cwd=repo, check=True, capture_output=True)
            first = generated.read_bytes()
            subprocess.run(command, cwd=repo, check=True, capture_output=True)
            self.assertEqual(first, generated.read_bytes())
            compiled = json.loads(first)["video_benchmark"]
            self.assertIs(type(compiled["request"]["guidance_scale"]), float)
            common = {key: compiled[key] for key in ("identity", "request", "warmup", "case_ids",
                       "warmup_num_inference_steps", "endpoint_path", "timeout_sec")}
            common["media_contract"] = {key: value for key, value in compiled["quality"].items()
                                         if key.startswith("expected_") or key in ("duration_tolerance_sec", "timestamp_tolerance_sec")}
            for run_id in module.RUNS:
                family = "baseline" if run_id.startswith("baseline-") else run_id
                metrics = synthetic_metrics()
                if family in video.VIDEO_NEGATIVE_FAMILIES:
                    metrics["temporal_delta_mae"] = 0.8
                measurement = {"run_id": run_id, "family": family, "measurement_protocol": common,
                               "measurement_protocol_sha256": video._json_sha256(common),
                               "reference_manifest_sha256": compiled["reference_manifest_sha256"],
                               "performance": {"completed": 4, "request_success_rate": 1.0},
                               "cases": [{"id": case_id, "contract": {"passed": True}, "video": metrics}
                                         for case_id in compiled["case_ids"]]}
                path = root / "measurements" / run_id
                path.mkdir(parents=True)
                (path / "measurement.json").write_text(json.dumps(measurement))
            output = root / "published"
            result = module.package(generated, root / "measurements", output)
            self.assertEqual(result["reports"], 12)
            compiled["calibration"] = {"status": "admitted", "path": str(output / "calibration.json"),
                                        "sha256": result["receipt_sha256"]}
            self.assertEqual(len(video._load_video_pool(pool, compiled)), 4)
            wrong = copy.deepcopy(compiled)
            wrong["request"]["guidance_scale"] = 5
            with self.assertRaisesRegex(ValueError, "calibration 未绑定"):
                video._load_video_pool(pool, wrong)
            report = output / "baseline-2/report.json"
            report.write_text(report.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "report.*hash"):
                video._load_video_pool(pool, compiled)


if __name__ == "__main__":
    unittest.main()

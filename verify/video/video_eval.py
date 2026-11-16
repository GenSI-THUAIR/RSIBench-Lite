"""评测 OpenAI-compatible 同步视频生成服务的质量与延迟。

参数由 ``clean_benchmark.py`` 传入：服务 endpoint、参考 MP4、输出目录和
``verifier-config.json``。请求字段、媒体合约和质量阈值全部来自 task 配置。

示例：
    python3 clean_benchmark.py http://[::1]:8000 reference.mp4 out verifier-config.json
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, cast


def _mapping(value: object, name: str) -> Mapping[str, object]:
    """校验并返回 JSON object。"""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} 必须是 JSON object")
    return cast(Mapping[str, object], value)


def _form_value(value: object) -> str:
    """把一个请求字段编码成 multipart 文本值。"""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise TypeError(f"不支持的 multipart 字段类型：{type(value).__name__}")


def _run_request(
    endpoint: str,
    endpoint_path: str,
    request: Mapping[str, object],
    output_path: Path,
    timeout_sec: int,
) -> float:
    """调用同步视频接口并返回客户端端到端秒数。"""
    if not endpoint_path.startswith("/"):
        raise ValueError("video_benchmark.endpoint_path 必须以 / 开头")
    command = [
        "curl",
        "--noproxy",
        "*",
        "--fail-with-body",
        "--silent",
        "--show-error",
        "--max-time",
        str(timeout_sec),
        "--request",
        "POST",
        f"{endpoint}{endpoint_path}",
    ]
    for key, value in request.items():
        command.extend(("--form-string", f"{key}={_form_value(value)}"))
    command.extend(("--output", str(output_path)))
    started_at = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_sec + 30,
        check=False,
    )
    elapsed = time.perf_counter() - started_at
    if completed.returncode != 0:
        body = output_path.read_text(errors="replace") if output_path.exists() else ""
        raise RuntimeError(
            f"视频请求失败(rc={completed.returncode}): "
            f"{completed.stderr.strip()} {body[:1000]}"
        )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("视频请求成功但没有返回非空文件")
    return elapsed


def _probe_media(path: Path) -> dict[str, object]:
    """用 ffprobe 读取 MP4 容器和音视频流元数据。"""
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    value = json.loads(output)
    if not isinstance(value, dict):
        raise TypeError(f"ffprobe 对 {path} 的输出不是 JSON object")
    return value


def _stream(probe: Mapping[str, object], stream_type: str) -> Mapping[str, object]:
    """从 ffprobe 结果中取得指定类型的唯一首个流。"""
    streams = probe.get("streams")
    if not isinstance(streams, list):
        raise TypeError("ffprobe 输出缺少 streams")
    for stream in streams:
        if isinstance(stream, Mapping) and stream.get("codec_type") == stream_type:
            return cast(Mapping[str, object], stream)
    raise ValueError(f"生成结果缺少 {stream_type} 流")


def _media_contract(
    probe: Mapping[str, object],
    request: Mapping[str, object],
    quality: Mapping[str, object],
) -> dict[str, object]:
    """按 task 配置核对视频编码、画布、帧率和音频格式。"""
    video = _stream(probe, "video")
    audio = _stream(probe, "audio")
    fps = float(Fraction(str(video["r_frame_rate"])))
    checks = {
        "video_codec": str(video.get("codec_name"))
        == str(quality["expected_video_codec"]),
        "audio_codec": str(audio.get("codec_name"))
        == str(quality["expected_audio_codec"]),
        "width": int(cast(Any, video["width"])) == int(cast(Any, request["width"])),
        "height": int(cast(Any, video["height"])) == int(cast(Any, request["height"])),
        "fps": abs(fps - float(cast(Any, quality["expected_fps"]))) < 0.01,
        "audio_sample_rate": int(cast(Any, audio["sample_rate"]))
        == int(cast(Any, quality["expected_audio_sample_rate"])),
        "audio_channels": int(cast(Any, audio["channels"]))
        == int(cast(Any, quality["expected_audio_channels"])),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "video": dict(video),
        "audio": dict(audio),
        "format": probe.get("format", {}),
    }


def _decode_video(path: Path) -> object:
    """把首个视频流完整解码为 uint8 ``[frames, height, width, 3]``。"""
    import av
    import numpy as np

    container = av.open(str(path))
    try:
        frames = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
        ]
    finally:
        container.close()
    if not frames:
        raise ValueError(f"{path} 没有可解码视频帧")
    return np.stack(frames)


def _video_metrics(
    reference_path: Path, output_path: Path, *, temporal: bool = False
) -> dict[str, float | int]:
    """按官方质量 CI 口径计算全帧 PSNR 与 AlexNet LPIPS。"""
    import lpips
    import numpy as np
    import torch

    reference = cast(Any, _decode_video(reference_path))
    candidate = cast(Any, _decode_video(output_path))
    if reference.shape != candidate.shape:
        raise ValueError(
            f"参考与候选视频 shape 不一致：{reference.shape} != {candidate.shape}"
        )
    reference_float = reference.astype(np.float32) / 255.0
    candidate_float = candidate.astype(np.float32) / 255.0
    difference = candidate_float - reference_float
    mse = float(np.mean(np.square(difference), dtype=np.float64))
    psnr = math.inf if mse == 0 else 10.0 * math.log10(1.0 / mse)

    # 与上游 compute_lpips_video 一致：每一帧按 [0,1] -> [-1,1]
    # 后取均值。分批只减少 CPU 峰值，不改变模型或聚合口径。
    metric = lpips.LPIPS(net="alex", verbose=False).eval()
    lpips_sum = 0.0
    lpips_worst = 0.0
    batch_size = 4
    with torch.inference_mode():
        for start in range(0, len(reference_float), batch_size):
            stop = min(start + batch_size, len(reference_float))
            reference_tensor = (
                torch.from_numpy(reference_float[start:stop].copy())
                .permute(0, 3, 1, 2)
                .mul(2.0)
                .sub(1.0)
            )
            candidate_tensor = (
                torch.from_numpy(candidate_float[start:stop].copy())
                .permute(0, 3, 1, 2)
                .mul(2.0)
                .sub(1.0)
            )
            values = metric(reference_tensor, candidate_tensor)
            lpips_sum += float(values.sum().item())
            lpips_worst = max(lpips_worst, float(values.max().item()))
    result = {
        "frame_count": len(reference_float),
        "psnr_db": psnr if math.isfinite(psnr) else 100.0,
        "lpips": lpips_sum / len(reference_float),
    }
    if temporal:
        result.update(_temporal_metrics(reference_float, candidate_float))
        result["lpips_worst_frame"] = lpips_worst
    return result


def _temporal_metrics(reference: Any, candidate: Any) -> dict[str, float]:
    """输入全帧 [0,1] RGB 数组；返回逐相邻帧时间差误差和参考运动量比。"""
    import numpy as np

    if reference.shape != candidate.shape or len(reference) < 2:
        raise ValueError("时间质量要求同 shape 且至少两帧")
    ref_delta = np.diff(reference, axis=0)
    out_delta = np.diff(candidate, axis=0)
    errors = np.mean(np.abs(out_delta - ref_delta), axis=(1, 2, 3), dtype=np.float64)
    ref_motion = float(np.mean(np.abs(ref_delta), dtype=np.float64))
    out_motion = float(np.mean(np.abs(out_delta), dtype=np.float64))
    if not math.isfinite(ref_motion) or ref_motion <= 0:
        raise ValueError("纯视频 reference 必须有非零、有限运动，静态一致性不是质量依据")
    return {
        "temporal_delta_mae": float(np.mean(errors)),
        "temporal_delta_worst": float(np.max(errors)),
        "motion_ratio": out_motion / ref_motion,
    }


def _decode_audio(path: Path) -> object:
    """把首个音频流解码为 32 kHz 双声道 float32 numpy 数组。"""
    import numpy as np

    payload = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ar",
            "32000",
            "-ac",
            "2",
            "pipe:1",
        ]
    )
    audio = np.frombuffer(payload, dtype=np.float32)
    if audio.size == 0 or audio.size % 2:
        raise ValueError(f"{path} 的音频解码结果为空或不是双声道")
    return audio.reshape(-1, 2)


def _audio_metrics(reference_path: Path, output_path: Path) -> dict[str, float | int]:
    """按官方质量 CI 口径计算 multi-resolution STFT 余弦和 RMS 比。"""
    import numpy as np
    import torch

    reference = cast(Any, _decode_audio(reference_path))
    candidate = cast(Any, _decode_audio(output_path))
    if reference.shape != candidate.shape:
        raise ValueError(
            f"参考与候选音频 shape 不一致：{reference.shape} != {candidate.shape}"
        )
    sample_count = len(reference)
    if not sample_count:
        raise ValueError("参考或候选视频没有可比较音频")
    reference_tensor = torch.from_numpy(reference.T.copy())
    candidate_tensor = torch.from_numpy(candidate.T.copy())
    similarities: list[float] = []
    for n_fft in (512, 1024, 2048):
        window = torch.hann_window(n_fft)
        reference_magnitude = torch.stft(
            reference_tensor,
            n_fft=n_fft,
            hop_length=n_fft // 4,
            window=window,
            return_complex=True,
        ).abs()
        candidate_magnitude = torch.stft(
            candidate_tensor,
            n_fft=n_fft,
            hop_length=n_fft // 4,
            window=window,
            return_complex=True,
        ).abs()
        cosine = torch.nn.functional.cosine_similarity(
            reference_magnitude.flatten(1),
            candidate_magnitude.flatten(1),
            dim=1,
            eps=1e-12,
        )
        similarities.extend(float(value) for value in cosine)
    reference_rms = float(torch.sqrt(torch.mean(reference_tensor.square())))
    candidate_rms = float(torch.sqrt(torch.mean(candidate_tensor.square())))
    return {
        "sample_count_per_channel": sample_count,
        "spectral_cosine": float(np.mean(similarities)),
        "rms_ratio": 0.0 if reference_rms == 0 else candidate_rms / reference_rms,
    }


VIDEO_ONLY_THRESHOLDS = (
    "lpips_max", "lpips_worst_frame_max", "temporal_delta_mae_max",
    "temporal_delta_worst_max", "motion_ratio_min", "motion_ratio_max",
)
# prerecorded-replay 指错case/错输出复播；逐字相同正确输出不由数值门证明其生成过程。
VIDEO_NEGATIVE_FAMILIES = frozenset({
    "static-repeat", "frame-padding", "downscale-upscale", "reduced-steps",
    "cfg-disabled", "prompt-ignored", "seed-ignored", "prerecorded-replay",
})


def _sha256(path: Path) -> str:
    """流式读取可信资产并返回内容 SHA256。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_sha256(value: object) -> str:
    """对规范 JSON 计算 hash，绑定完整配置而非浮动模型名称。"""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def validate_video_only(benchmark: Mapping[str, object]) -> None:
    """校验显式纯视频协议；未真实标定的任务在创建 Serve 前拒绝准入。"""
    if benchmark["mode"] != "video-only":
        raise ValueError("纯视频协议必须显式 mode=video-only")
    case_ids = benchmark["case_ids"]
    if not isinstance(case_ids, list) or len(case_ids) < 4 or any(
        not isinstance(item, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", item)
        for item in case_ids
    ) or len(set(case_ids)) != len(case_ids):
        raise ValueError("video-only.case_ids 必须为至少四个唯一、安全请求 ID")
    admission = _mapping(benchmark["calibration"], "video calibration")
    if admission["status"] != "admitted":
        raise ValueError("video-only calibration pending/unadmitted；禁止评分")
    for name in ("reference_manifest_sha256",):
        if not re.fullmatch(r"[0-9a-f]{64}", str(benchmark[name])):
            raise ValueError(f"{name} 必须是真实 SHA256")
    if not re.fullmatch(r"[0-9a-f]{64}", str(admission["sha256"])):
        raise ValueError("calibration.sha256 必须是真实 SHA256")
    if not Path(str(admission["path"])).is_absolute():
        raise ValueError("calibration.path 必须是可信 Benchmark 绝对路径")
    identity = _mapping(benchmark["identity"], "video identity")
    for name in ("model_revision", "weights_manifest_sha256", "engine_commit",
                 "image_digest", "sampling", "metric_identity"):
        if not identity[name]:
            raise ValueError(f"video identity.{name} 不得为空")
    quality = _mapping(benchmark["quality"], "video quality")
    for name in VIDEO_ONLY_THRESHOLDS:
        value = quality[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise TypeError(f"quality.{name} 必须为标定数值")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"quality.{name} 必须是非负有限数")
    if quality["motion_ratio_min"] <= 0 or quality["motion_ratio_min"] > quality["motion_ratio_max"]:
        raise ValueError("运动量阈值必须拒绝静态视频且上下界有序")
    for name in ("expected_fps", "expected_frame_count", "duration_tolerance_sec",
                 "timestamp_tolerance_sec"):
        value = float(cast(Any, quality[name]))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"quality.{name} 必须为正有限数")
    if quality["expected_frame_count"] < 2:
        raise ValueError("纯视频要求至少两帧")
    if not quality["expected_video_codec"]:
        raise ValueError("必须冻结视频编码")
    if any(str(key).startswith(("audio_", "expected_audio_")) for key in quality):
        raise ValueError("video-only 不接受音频 gate 参数")
    request = _mapping(benchmark["request"], "video request")
    if "prompt" in request or "seed" in request:
        raise ValueError("纯视频公共 request 不得混入逐例 prompt/seed")
    warmup = _mapping(benchmark["warmup"], "video warmup")
    if set(warmup) != {"prompt", "seed"} or not isinstance(warmup["prompt"], str) or not warmup["prompt"].strip() or type(warmup["seed"]) is not int:
        raise ValueError("video warmup 必须显式提供独立 prompt/seed")
    for key in ("width", "height", "num_frames", "fps", "num_inference_steps"):
        if float(cast(Any, request[key])) <= 0:
            raise ValueError(f"video request.{key} 必须为正数")
    if request["fps"] != quality["expected_fps"] or request["num_frames"] != quality["expected_frame_count"]:
        raise ValueError("请求与媒体合约 fps/帧数不一致")
    if int(cast(Any, benchmark["timeout_sec"])) <= 0 or int(cast(Any, benchmark["warmup_num_inference_steps"])) <= 0:
        raise ValueError("video timeout/warmup 必须为正数")


def _quality_checks(metrics: Mapping[str, object], quality: Mapping[str, object]) -> dict[str, bool]:
    """逐项判定感知及时间指标；NaN/Inf 一律不通过。"""
    checks = {}
    for metric, threshold in (
        ("lpips", "lpips_max"), ("lpips_worst_frame", "lpips_worst_frame_max"),
        ("temporal_delta_mae", "temporal_delta_mae_max"),
        ("temporal_delta_worst", "temporal_delta_worst_max"),
    ):
        value = float(cast(Any, metrics[metric]))
        checks[metric] = math.isfinite(value) and value <= float(cast(Any, quality[threshold]))
    motion = float(cast(Any, metrics["motion_ratio"]))
    checks["motion_ratio"] = math.isfinite(motion) and float(cast(Any, quality["motion_ratio_min"])) <= motion <= float(cast(Any, quality["motion_ratio_max"]))
    return checks


def _load_video_pool(path: Path, benchmark: Mapping[str, object]) -> list[dict[str, Any]]:
    """校验可信请求/reference 池及独立标定收据，返回绑定后的完整工作集。"""
    validate_video_only(benchmark)
    if _sha256(path) != benchmark["reference_manifest_sha256"]:
        raise ValueError("reference manifest hash 不匹配")
    manifest = _mapping(json.loads(path.read_text()), "reference manifest")
    if manifest["schema"] != "video-reference-pool/1" or manifest["identity"] != benchmark["identity"] or manifest["request"] != benchmark["request"]:
        raise ValueError("reference 池未绑定相同模型、权重、采样或评分依赖")
    cases = manifest["cases"]
    if not isinstance(cases, list) or len(cases) < 4:
        raise ValueError("冻结池至少四个请求，覆盖不同 prompt 与 seed")
    ids, pairs = set(), set()
    resolved = []
    for item in cases:
        case = dict(_mapping(item, "reference case"))
        case_id = str(case["id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", case_id) or case_id in ids:
            raise ValueError("reference case ID 非法或重复")
        ids.add(case_id)
        prompt, seed = case["prompt"], case["seed"]
        if not isinstance(prompt, str) or not prompt.strip() or type(seed) is not int:
            raise ValueError("reference case prompt/seed 非法")
        pairs.add((prompt, seed))
        reference = (path.parent / str(case["reference"])).resolve()
        if not reference.is_relative_to(path.parent.resolve()) or _sha256(reference) != case["sha256"]:
            raise ValueError(f"reference 路径越界或内容漂移：{case_id}")
        case["reference_path"] = reference
        resolved.append(case)
    warmup = _mapping(benchmark["warmup"], "video warmup")
    if (warmup["prompt"], warmup["seed"]) in pairs:
        raise ValueError("计时输入不得复用 warmup 的 prompt/seed")
    if [case["id"] for case in resolved] != benchmark["case_ids"]:
        raise ValueError("reference 池 ID/顺序与完整工作集合约不一致")
    if len(pairs) != len(cases) or len({p for p, s in pairs}) < 2 or len({s for p, s in pairs}) < 2:
        raise ValueError("冻结池必须有唯一 prompt/seed 组合，且各至少两个")
    admission = _mapping(benchmark["calibration"], "calibration")
    receipt_path = Path(str(admission["path"]))
    if _sha256(receipt_path) != admission["sha256"]:
        raise ValueError("calibration receipt hash 不匹配")
    receipt = _mapping(json.loads(receipt_path.read_text()), "calibration receipt")
    protocol = {k: v for k, v in benchmark.items() if k != "calibration"}
    if receipt["schema"] != "video-calibration/1" or receipt["protocol_sha256"] != _json_sha256(protocol):
        raise ValueError("calibration 未绑定完整协议/阈值/reference")
    families: dict[str, int] = {}
    run_ids = set()
    quality = _mapping(benchmark["quality"], "quality")
    for raw_run in receipt["runs"]:
        run = _mapping(raw_run, "calibration run")
        run_id, family = str(run["run_id"]), str(run["family"])
        if not run_id or run_id in run_ids:
            raise ValueError("calibration run 必须独立且唯一")
        run_ids.add(run_id)
        report_path = (receipt_path.parent / str(run["report"])).resolve()
        if not report_path.is_relative_to(receipt_path.parent.resolve()) or _sha256(report_path) != run["sha256"]:
            raise ValueError("calibration report 越界或 hash 不匹配")
        report = _mapping(json.loads(report_path.read_text()), "calibration report")
        if report["protocol_sha256"] != receipt["protocol_sha256"]:
            raise ValueError("标定报告协议不一致")
        records = report["cases"]
        if not isinstance(records, list) or len(records) != len(ids) or {c["id"] for c in records} != ids:
            raise ValueError("标定报告必须覆盖完整请求池")
        passed = all(
            c["contract"]["passed"] is True and all(_quality_checks(c["video"], quality).values())
            for c in records
        )
        if family in ("baseline", "known-good"):
            if not passed:
                raise ValueError(f"标定正控未通过：{run_id}")
        elif family in VIDEO_NEGATIVE_FAMILIES:
            if passed:
                raise ValueError(f"标定负控未被指标拒绝：{run_id}")
        else:
            raise ValueError(f"未知标定 family：{family}")
        families[family] = families.get(family, 0) + 1
    if families.get("baseline", 0) < 3 or families.get("known-good", 0) < 1 or not VIDEO_NEGATIVE_FAMILIES.issubset(families):
        raise ValueError("缺重复 baseline/known-good/必需行为负控；禁止准入")
    return resolved


def _video_only_contract(path: Path, request: Mapping[str, object], quality: Mapping[str, object]) -> dict[str, object]:
    """全解码检查纯视频帧数、时间戳及容器；绝不创建或接受静音音轨。"""
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        video = {"codec_name": stream.codec_context.name, "width": stream.width,
                 "height": stream.height, "r_frame_rate": str(stream.base_rate)}
        streams = [{"codec_type": item.type} for item in container.streams]
        probe = {"streams": streams, "video": video,
                 "format": {"name": container.format.name,
                            "duration": container.duration / av.time_base if container.duration is not None else math.nan}}
        times = [float(frame.pts * frame.time_base) if frame.pts is not None else math.nan
                 for frame in container.decode(video=0)]
    finally:
        container.close()
    fps = float(cast(Any, quality["expected_fps"]))
    expected_count = int(cast(Any, quality["expected_frame_count"]))
    checks = {
        "mp4_container": "mp4" in probe["format"]["name"].split(","),
        "video_only": len(streams) == 1 and streams[0]["codec_type"] == "video",
        "video_codec": video["codec_name"] == quality["expected_video_codec"],
        "width": video["width"] == request["width"],
        "height": video["height"] == request["height"],
        "fps": Fraction(str(video["r_frame_rate"])) == Fraction(str(quality["expected_fps"])),
        "decoded_frame_count": len(times) == expected_count,
        "timestamps": bool(times) and all(math.isfinite(t) and abs(t - i / fps) <= float(cast(Any, quality["timestamp_tolerance_sec"])) for i, t in enumerate(times)),
        "duration": abs(float(cast(Any, _mapping(probe["format"], "format")["duration"])) - expected_count / fps) <= float(cast(Any, quality["duration_tolerance_sec"])),
    }
    return {"passed": all(checks.values()), "checks": checks, "probe": probe, "decoded_frame_count": len(times)}


def _evaluate_video_only(endpoint: str, reference_path: Path, output_dir: Path, benchmark: Mapping[str, object]) -> dict[str, object]:
    """顺序生成整个冻结池，计时包含全部生成及媒体返回；质量计算不计入吞吐。"""
    cases = _load_video_pool(reference_path, benchmark)
    quality = _mapping(benchmark["quality"], "quality")
    common = dict(_mapping(benchmark["request"], "request"))
    timeout = int(cast(Any, benchmark["timeout_sec"]))
    endpoint_path = str(benchmark["endpoint_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = output_dir / "video-outputs"
    evidence_dir.mkdir(exist_ok=False)
    # 可信 reference 先验合约，避免资产损坏误判候选，也不占性能计时。
    for case in cases:
        if not _video_only_contract(case["reference_path"], common, quality)["passed"]:
            raise ValueError(f"reference 媒体合约失败：{case['id']}")
    warmup = {**common, **_mapping(benchmark["warmup"], "video warmup"),
              "num_inference_steps": benchmark["warmup_num_inference_steps"]}
    warmup_seconds = _run_request(endpoint, endpoint_path, warmup, output_dir / "video-warmup.mp4", timeout)
    requests = [(case, {**common, "prompt": case["prompt"], "seed": case["seed"]}, evidence_dir / f"{case['id']}.mp4") for case in cases]
    latencies = []
    started = time.perf_counter()
    for case, request, path in requests:
        latencies.append(_run_request(endpoint, endpoint_path, request, path, timeout))
    measured = time.perf_counter() - started
    if not math.isfinite(measured) or measured <= 0:
        raise ValueError("完整工作集耗时非法")
    records = []
    for (case, request, path), latency in zip(requests, latencies, strict=True):
        contract = _video_only_contract(path, request, quality)
        # 媒体合约失败仍保留全部证据并给负 gate；不缩放/补帧来凑指标。
        if contract["passed"]:
            metrics = _video_metrics(case["reference_path"], path, temporal=True)
            checks = _quality_checks(metrics, quality)
        else:
            metrics = {}
            checks = {"media_contract": False}
        records.append({"id": case["id"], "file": f"video-outputs/{case['id']}.mp4",
                        "sha256": _sha256(path), "reference_sha256": case["sha256"],
                        "request": request, "wall_time_sec": latency, "contract": contract,
                        "video": metrics, "quality_checks": checks,
                        "passed": bool(contract["passed"]) and all(checks.values())})
    passed = all(record["passed"] for record in records)
    gate = {"accuracy": float(passed), "passed": passed, "mode": "video-only",
            "cases": records, "thresholds": dict(quality),
            "protocol_sha256": _json_sha256({k: v for k, v in benchmark.items() if k != "calibration"})}
    throughput = 3600.0 * len(cases) / measured
    raw = {"unit": "videos/hour", "warmup_wall_time_sec": warmup_seconds,
           "generation_wall_time_sec": measured, "videos_per_hour": throughput}
    performance = {"num_prompts": len(cases), "max_concurrency": 1, "completed": len(cases),
                   "request_success_rate": 1.0, "wall_time_sec": measured,
                   "ttft_ms_p50": -1.0, "ttft_ms_p99": -1.0,
                   "tpot_ms_p50": -1.0, "tpot_ms_p99": -1.0,
                   "output_throughput": throughput, "raw": raw}
    summary = {"gate": gate, "performance": performance}
    (output_dir / "video-evidence.json").write_text(json.dumps({"cases": records}, indent=2, allow_nan=False) + "\n")
    (output_dir / "performance-result.jsonl").write_text(json.dumps(raw, allow_nan=False) + "\n")
    (output_dir / "performance.log").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def evaluate(
    endpoint: str, reference_path: Path, output_dir: Path, config_path: Path
) -> dict[str, object]:
    """执行 shape warmup、计时请求、媒体校验与参考质量对比。"""
    config = _mapping(json.loads(config_path.read_text()), "verifier config")
    benchmark = _mapping(config["video_benchmark"], "video_benchmark")
    mode = benchmark.get("mode", "audio-video")
    if mode == "video-only":
        return _evaluate_video_only(endpoint, reference_path, output_dir, benchmark)
    if mode != "audio-video":
        raise ValueError(f"未知 video_benchmark.mode：{mode}")
    request = dict(_mapping(benchmark["request"], "video_benchmark.request"))
    quality = _mapping(benchmark["quality"], "video_benchmark.quality")
    timeout_sec = int(cast(Any, benchmark["timeout_sec"]))
    warmup_steps = int(cast(Any, benchmark["warmup_num_inference_steps"]))
    if timeout_sec <= 0 or warmup_steps <= 0:
        raise ValueError("video benchmark timeout 和 warmup steps 必须为正数")
    if not reference_path.is_file():
        raise FileNotFoundError(f"参考视频不存在：{reference_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    warmup_path = output_dir / "video-warmup.mp4"
    output_path = output_dir / "video-output.mp4"
    warmup_request = {**request, "num_inference_steps": warmup_steps}
    warmup_seconds = _run_request(
        endpoint,
        str(benchmark["endpoint_path"]),
        warmup_request,
        warmup_path,
        timeout_sec,
    )
    measured_seconds = _run_request(
        endpoint,
        str(benchmark["endpoint_path"]),
        request,
        output_path,
        timeout_sec,
    )

    reference_probe = _probe_media(reference_path)
    probe = _probe_media(output_path)
    reference_contract = _media_contract(reference_probe, request, quality)
    contract = _media_contract(probe, request, quality)
    video_metrics = _video_metrics(reference_path, output_path)
    audio_metrics = _audio_metrics(reference_path, output_path)
    expected_frame_count = int(cast(Any, quality["expected_frame_count"]))
    decoded_frame_count_ok = video_metrics["frame_count"] == expected_frame_count
    for media_contract in (reference_contract, contract):
        checks = cast(dict[str, bool], media_contract["checks"])
        checks["decoded_frame_count"] = decoded_frame_count_ok
        media_contract["passed"] = all(checks.values())
    quality_passed = (
        bool(reference_contract["passed"])
        and bool(contract["passed"])
        and float(video_metrics["lpips"]) <= float(cast(Any, quality["lpips_max"]))
        and float(audio_metrics["spectral_cosine"])
        >= float(cast(Any, quality["audio_spectral_cosine_min"]))
        and float(audio_metrics["rms_ratio"])
        >= float(cast(Any, quality["audio_rms_ratio_min"]))
        and float(audio_metrics["rms_ratio"])
        <= float(cast(Any, quality["audio_rms_ratio_max"]))
    )
    gate = {
        "accuracy": float(quality_passed),
        "passed": quality_passed,
        "contract": contract,
        "reference_contract": reference_contract,
        "video": video_metrics,
        "audio": audio_metrics,
        "thresholds": dict(quality),
    }
    videos_per_hour = 3600.0 / measured_seconds
    performance = {
        "num_prompts": 1,
        "max_concurrency": 1,
        "completed": 1,
        "request_success_rate": 1.0,
        "wall_time_sec": measured_seconds,
        "ttft_ms_p50": -1.0,
        "ttft_ms_p99": -1.0,
        "tpot_ms_p50": -1.0,
        "tpot_ms_p99": -1.0,
        "output_throughput": videos_per_hour,
        "raw": {
            "unit": "videos/hour",
            "warmup_wall_time_sec": warmup_seconds,
            "generation_wall_time_sec": measured_seconds,
            "videos_per_hour": videos_per_hour,
        },
    }
    (output_dir / "performance-result.jsonl").write_text(
        json.dumps(performance["raw"]) + "\n"
    )
    (output_dir / "performance.log").write_text(
        json.dumps(
            {"gate": gate, "performance": performance},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return {"gate": gate, "performance": performance}

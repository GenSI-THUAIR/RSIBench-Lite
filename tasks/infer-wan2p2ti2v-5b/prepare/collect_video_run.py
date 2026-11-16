#!/usr/bin/env python3
"""按冻结计划录制Wan reference或收集基线/已知良好/行为负控的完整原始视频指标。

参数：运行计划JSON、独有输出目录。示例：python collect_video_run.py run-plan.json /path/to/output
所有请求参数来自task.yaml生成的plan；本脚本不选择阈值、不产reward、不把负控标签当指标。
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

from video_eval import (
    _decode_video,
    _json_sha256,
    _run_request,
    _sha256,
    _video_metrics,
    _video_only_contract,
)

FAMILIES = {"reference", "baseline", "known-good", "static-repeat", "frame-padding", "downscale-upscale",
            "reduced-steps", "cfg-disabled", "prompt-ignored", "seed-ignored", "prerecorded-replay"}


def measurement_identity(benchmark: dict) -> dict:
    """返回不依赖尚未标定阈值的实际输入/媒体/软件协议，供后续证书绑定原始数据。"""
    return {"identity": benchmark["identity"], "request": benchmark["request"], "warmup": benchmark["warmup"],
            "case_ids": benchmark["case_ids"], "warmup_num_inference_steps": benchmark["warmup_num_inference_steps"],
            "endpoint_path": benchmark["endpoint_path"], "timeout_sec": benchmark["timeout_sec"],
            "media_contract": {key: value for key, value in benchmark["quality"].items()
                               if key.startswith("expected_") or key in ("duration_tolerance_sec", "timestamp_tolerance_sec")}}


def write_video(frames, path: Path, fps: int) -> None:
    """把真实解码/变换后的负控RGB帧编码为H264 MP4，不改变帧率或丢帧。"""
    import av

    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        for pixels in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def collect(plan_path: Path, output: Path) -> None:
    """运行完整四case工作集；先计全部生成/返回，再解码与评分，保存每例证据。"""
    plan = json.loads(plan_path.read_text())
    family = plan["family"]
    if family not in FAMILIES:
        raise ValueError(f"未知family：{family}")
    benchmark, cases = plan["video_benchmark"], plan["cases"]
    common, quality = benchmark["request"], benchmark["quality"]
    if [case["id"] for case in cases] != benchmark["case_ids"] or len(cases) != 4:
        raise ValueError("必须执行原冻结四请求及其顺序")
    if (benchmark["warmup"]["prompt"], benchmark["warmup"]["seed"]) in {(c["prompt"], c["seed"]) for c in cases}:
        raise ValueError("warmup与计时输入不能重复")
    if common["num_inference_steps"] != 50 or common["num_frames"] != 121 or common["width"] != 1280 or common["height"] != 704:
        raise ValueError("不能把smoke工作量用于本题正式标定")
    for key in ("model_revision", "weights_manifest_sha256", "engine_commit", "image_digest", "metric_identity"):
        if not benchmark["identity"][key]:
            raise ValueError(f"实际身份尚未冻结：{key}")
    references = {}
    if family != "reference":
        manifest_path = Path(plan["reference_manifest"])
        if _sha256(manifest_path) != plan["reference_manifest_sha256"]:
            raise ValueError("reference manifest hash不匹配")
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != benchmark["identity"] or manifest["request"] != common:
            raise ValueError("reference身份/输入不符")
        references = {case["id"]: (manifest_path.parent / case["reference"], case["sha256"]) for case in manifest["cases"]}
        for path, digest in references.values():
            if _sha256(path) != digest:
                raise ValueError("reference视频内容漂移")
    output.mkdir(parents=True, exist_ok=False)
    endpoint = plan.get("endpoint") or f"http://[::1]:{int(os.environ['PORT0'])}"
    timeout, endpoint_path = int(benchmark["timeout_sec"]), benchmark["endpoint_path"]
    warmup = {**common, **benchmark["warmup"], "num_inference_steps": benchmark["warmup_num_inference_steps"]}
    generating = family not in ("static-repeat", "prerecorded-replay")
    warmup_seconds = _run_request(endpoint, endpoint_path, warmup, output / "warmup.mp4", timeout) if generating else None
    requests = []
    for case in cases:
        request = {**common, "prompt": case["prompt"], "seed": case["seed"]}
        if family == "reduced-steps":
            request["num_inference_steps"] = 25
        elif family == "cfg-disabled":
            request["guidance_scale"] = 1.0
        elif family == "frame-padding":
            request["num_frames"] = 61
        elif family == "downscale-upscale":
            request.update(width=640, height=352)
        elif family == "prompt-ignored":
            request["prompt"] = cases[0]["prompt"]
        elif family == "seed-ignored":
            request["seed"] = cases[0]["seed"]
        requests.append((case, request, output / f"{case['id']}.mp4"))
    latencies = []
    started = time.perf_counter()
    for case, request, path in requests:
        if generating:
            latencies.append(_run_request(endpoint, endpoint_path, request, path, timeout))
        elif family == "prerecorded-replay":
            shutil.copyfile(references[cases[0]["id"]][0], path)
        else:
            frames = _decode_video(references[case["id"]][0])
            write_video([frames[0]] * common["num_frames"], path, common["fps"])
        print(f"GENERATED {family} {case['id']}", flush=True)
    wall_seconds = time.perf_counter() - started
    transform_seconds = 0.0
    records = []
    for index, (case, request, path) in enumerate(requests):
        if family in ("frame-padding", "downscale-upscale"):
            transformed_at = time.perf_counter()
            original = path.with_suffix(".raw.mp4")
            path.rename(original)
            frames = _decode_video(original)
            if family == "frame-padding":
                frames = [*frames, *([frames[-1]] * (common["num_frames"] - len(frames)))]
            else:
                import numpy as np
                from PIL import Image

                frames = [np.asarray(Image.fromarray(frame).resize((common["width"], common["height"]), Image.Resampling.BICUBIC)) for frame in frames]
            write_video(frames, path, common["fps"])
            transform_seconds += time.perf_counter() - transformed_at
        nominal = {**common, "prompt": case["prompt"], "seed": case["seed"]}
        contract = _video_only_contract(path, nominal, quality)
        if family == "reference" and not contract["passed"]:
            raise ValueError(f"原始reference媒体合约失败：{case['id']}")
        metrics = _video_metrics(references[case["id"]][0], path, temporal=True) if references and contract["passed"] else {}
        record = {"id": case["id"], "request": nominal, "actual_generation_request": request if generating else None,
                  "file": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size,
                  "contract": contract, "video": metrics,
                  "request_seconds": latencies[index] if generating else None}
        records.append(record)
        print(json.dumps({"id": case["id"], "contract": contract["passed"], "video": metrics}), flush=True)
    if family == "reference":
        manifest = {"schema": "video-reference-pool/1", "identity": benchmark["identity"], "request": common,
                    "cases": [{**case, "reference": record["file"], "sha256": record["sha256"]}
                              for case, record in zip(cases, records, strict=True)]}
        (output / "reference-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result = {"schema": "wan22-measurement/1", "run_id": plan["run_id"], "family": family,
              "measurement_protocol": measurement_identity(benchmark),
              "measurement_protocol_sha256": _json_sha256(measurement_identity(benchmark)),
              "plan_sha256": _sha256(plan_path), "candidate_variant": plan["candidate_variant"],
              "reference_manifest_sha256": plan.get("reference_manifest_sha256"), "cases": records,
              "performance": {"completed": len(cases), "request_success_rate": 1.0, "wall_time_sec": wall_seconds,
                              "warmup_seconds": warmup_seconds, "control_transform_seconds": transform_seconds,
                              "videos_per_hour": 3600 * len(cases) / wall_seconds,
                              "eligible_baseline_family": family == "baseline"}}
    (output / "measurement.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"MEASUREMENT_SHA256={_sha256(output / 'measurement.json')}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    collect(Path(sys.argv[1]), Path(sys.argv[2]))

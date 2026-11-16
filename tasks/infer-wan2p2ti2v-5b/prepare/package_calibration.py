#!/usr/bin/env python3
"""将原始测量按实际编译JSON绑定为可复现收据，保留数值类型；不上传、不覆盖历史版本。

参数：schemas.build生成的verifier-config.json、12个run_id/measurement.json目录、全新输出目录。
示例：python tasks/infer-wan2p2ti2v-5b/prepare/package_calibration.py \
  tasks/infer-wan2p2ti2v-5b/tests/verifier-config.json /path/to/calibration-inputs .agents/workdir/new-receipt
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# 与其他prepare脚本一致，只复用共享可信video_eval，不复制评分实现。
from video_eval import VIDEO_NEGATIVE_FAMILIES, _json_sha256, _quality_checks

RUNS = ("baseline-1", "baseline-2", "baseline-3", "known-good", "static-repeat",
        "frame-padding", "downscale-upscale", "reduced-steps", "cfg-disabled",
        "prompt-ignored", "seed-ignored", "prerecorded-replay")


def package(config_path: Path, measurements: Path, output: Path) -> dict:
    """输入实际编译配置和原测量；核验协议/原hash/正负判定后输出13文件收据包。"""
    benchmark = json.loads(config_path.read_text())["video_benchmark"]
    # 不经过jq等工具，5.0/1.0的原始JSON数值类型参与相同生产canonical hash。
    protocol = _json_sha256({key: value for key, value in benchmark.items() if key != "calibration"})
    common = {key: benchmark[key] for key in (
        "identity", "request", "warmup", "case_ids", "warmup_num_inference_steps", "endpoint_path", "timeout_sec"
    )}
    common["media_contract"] = {key: value for key, value in benchmark["quality"].items()
                                if key.startswith("expected_") or key in ("duration_tolerance_sec", "timestamp_tolerance_sec")}
    ready = []
    for run_id in RUNS:
        original = json.loads((measurements / run_id / "measurement.json").read_text())
        family = "baseline" if run_id.startswith("baseline-") else run_id
        if original["run_id"] != run_id or original["family"] != family:
            raise ValueError(f"run/family与冻结映射不符：{run_id}")
        prior = original["measurement_protocol"]
        if _json_sha256(prior) != original["measurement_protocol_sha256"]:
            raise ValueError(f"原始measurement协议hash不符：{run_id}")
        if prior != common or original["reference_manifest_sha256"] != benchmark["reference_manifest_sha256"]:
            raise ValueError(f"历史measurement不能绑定不同工作量/软件/reference：{run_id}")
        if [case["id"] for case in original["cases"]] != benchmark["case_ids"]:
            raise ValueError(f"case/顺序不完整：{run_id}")
        perf = original["performance"]
        if perf["completed"] != len(benchmark["case_ids"]) or perf["request_success_rate"] != 1:
            raise ValueError(f"不能准入部分工作集：{run_id}")
        passed = all(case["contract"]["passed"] is True and all(_quality_checks(case["video"], benchmark["quality"]).values())
                     for case in original["cases"])
        if family in ("baseline", "known-good"):
            if not passed:
                raise ValueError(f"原始正控未通过：{run_id}")
        elif family not in VIDEO_NEGATIVE_FAMILIES or passed:
            raise ValueError(f"负控未被真实指标拒绝：{run_id}")
        report = {**original, "protocol_sha256": protocol}
        ready.append((run_id, family, (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()))
    output.mkdir(parents=True, exist_ok=False)
    runs = []
    for run_id, family, payload in ready:
        report_path = output / run_id / "report.json"
        report_path.parent.mkdir()
        report_path.write_bytes(payload)
        runs.append({"run_id": run_id, "family": family, "report": f"{run_id}/report.json",
                     "sha256": hashlib.sha256(payload).hexdigest()})
    receipt = {"schema": "video-calibration/1", "protocol_sha256": protocol, "runs": runs}
    destination = output / "calibration.json"
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    identity = {"protocol_sha256": protocol, "receipt_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "reports": len(runs)}
    print(json.dumps(identity), flush=True)
    return identity


def main() -> None:
    """只接受三个路径参数；缺数据或既有目标一律直接失败。"""
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    package(*(Path(value).resolve() for value in sys.argv[1:]))


if __name__ == "__main__":
    main()

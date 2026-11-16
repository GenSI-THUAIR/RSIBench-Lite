#!/usr/bin/env python3
"""严格应用已冻结25%正负分离规则，生成阈值/性能建议；不修改task.yaml、不自动准入。

参数：执行布局JSON、各run_id/measurement.json目录、输出建议JSON；可选第4参数为
事先冻结的新硬件计划，必须继承原标定方法并保留全部正负控。缺省仍校验原H20计划。
示例：python calibrate_thresholds.py layout.json measurements proposal.json
"""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

from video_eval import _quality_checks

UPPER = {"lpips": "lpips_max", "lpips_worst_frame": "lpips_worst_frame_max",
         "temporal_delta_mae": "temporal_delta_mae_max", "temporal_delta_worst": "temporal_delta_worst_max"}


def calibrate(layout_path: Path, directory: Path, output: Path, plan_path: Path | None = None) -> None:
    """读取完整12份真实报告，验证身份和工作集，按既定公式计算并验证全部正负控。"""
    layout = json.loads(layout_path.read_text())
    original_plan = "d009eaeb15869f53604d7500eaeb73b9a0f40c7c14b623737349c0179ef01278"
    expected_plan = original_plan
    if plan_path is not None:
        plan = json.loads(plan_path.read_text())
        expected_families = {"static-repeat", "frame-padding", "downscale-upscale", "reduced-steps",
                             "cfg-disabled", "prompt-ignored", "seed-ignored", "prerecorded-replay"}
        if (plan.get("inherits_calibration_policy_sha256") != original_plan
                or plan.get("baseline_runs") != 3
                or set(plan.get("negative_families", {})) != expected_families):
            raise ValueError("新硬件计划必须保留原标定方法、重复次数和全部负控")
        expected_plan = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if layout["calibration_plan_sha256"] != expected_plan:
        raise ValueError("原校准计划hash漂移")
    reports = []
    bindings = []
    for experiment in layout["experiments"]:
        path = directory / experiment["run_id"] / "measurement.json"
        report = json.loads(path.read_text())
        if report["run_id"] != experiment["run_id"] or report["family"] != experiment["family"]:
            raise ValueError("报告与冻结实验映射不符")
        if report["candidate_variant"]["calibration_plan_sha256"] != layout["calibration_plan_sha256"]:
            raise ValueError("报告未绑定原校准计划")
        if len(report["cases"]) != 4 or report["performance"]["completed"] != 4 or report["performance"]["request_success_rate"] != 1.0:
            raise ValueError("不能用部分成功或缺case报告标定")
        reports.append(report)
        bindings.append({"run_id": report["run_id"], "family": report["family"],
                         "measurement_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(reports) != 12 or len({report["run_id"] for report in reports}) != 12:
        raise ValueError("只能使用原3 baseline、1 known-good、8负控")
    if len({report["measurement_protocol_sha256"] for report in reports}) != 1 or len({report["reference_manifest_sha256"] for report in reports}) != 1:
        raise ValueError("不同输入/软件/参考协议不能混作标定")
    case_ids = [case["id"] for case in reports[0]["cases"]]
    if any([case["id"] for case in report["cases"]] != case_ids for report in reports):
        raise ValueError("报告工作集/顺序不一致")
    positive = [report for report in reports if report["family"] in ("baseline", "known-good")]
    negative = [report for report in reports if report["family"] not in ("baseline", "known-good")]
    if sum(report["family"] == "baseline" for report in positive) != 3 or sum(report["family"] == "known-good" for report in positive) != 1:
        raise ValueError("缺原定正控")
    if not all(case["contract"]["passed"] for report in positive for case in report["cases"]):
        raise ValueError("正控媒体合约不通过")
    metrics = [*UPPER, "motion_ratio"]
    for report in reports:
        for case in report["cases"]:
            if case["contract"]["passed"] and not all(math.isfinite(float(case["video"][name])) for name in metrics):
                raise ValueError("非有限指标不可标定")
    quality = dict(reports[0]["measurement_protocol"]["media_contract"])
    formula_evidence = {}
    for metric, threshold in UPPER.items():
        p = max(float(case["video"][metric]) for report in positive for case in report["cases"])
        candidates = [float(case["video"][metric]) for report in negative for case in report["cases"]
                      if case["contract"]["passed"] and float(case["video"][metric]) > p]
        if not candidates:
            raise ValueError(f"{metric}没有已验证正负分离gap，保持unadmitted")
        n = min(candidates)
        quality[threshold] = p + 0.25 * (n - p)
        formula_evidence[threshold] = {"positive_envelope": p, "nearest_negative": n, "fraction": 0.25}
    ratios = [float(case["video"]["motion_ratio"]) for report in positive for case in report["cases"]]
    negative_ratios = [float(case["video"]["motion_ratio"]) for report in negative for case in report["cases"] if case["contract"]["passed"]]
    low, high = min(ratios), max(ratios)
    below, above = [value for value in negative_ratios if value < low], [value for value in negative_ratios if value > high]
    if not below or not above:
        raise ValueError("运动量上下界缺独立负控gap，保持unadmitted")
    quality["motion_ratio_min"] = low - 0.25 * (low - max(below))
    quality["motion_ratio_max"] = high + 0.25 * (min(above) - high)
    if quality["motion_ratio_min"] <= 0:
        raise ValueError("运动量下界不能放过静态")
    formula_evidence["motion_ratio_min"] = {"positive_envelope": low, "nearest_negative": max(below), "fraction": 0.25}
    formula_evidence["motion_ratio_max"] = {"positive_envelope": high, "nearest_negative": min(above), "fraction": 0.25}
    verdicts = []
    for report in reports:
        checks = [{"id": case["id"], "contract": bool(case["contract"]["passed"]),
                   "quality": _quality_checks(case["video"], quality) if case["contract"]["passed"] else {"media": False}}
                  for case in report["cases"]]
        passed = all(case["contract"] and all(case["quality"].values()) for case in checks)
        expected = report["family"] in ("baseline", "known-good")
        verdicts.append({"run_id": report["run_id"], "family": report["family"], "passed": passed, "cases": checks})
        if passed != expected:
            raise ValueError(f"预声明分离规则未接受正控/未拒绝负控：{report['run_id']}；禁止调宽或选样本")
    baseline_seconds = [float(report["performance"]["wall_time_sec"]) for report in positive if report["family"] == "baseline"]
    if not all(math.isfinite(value) and value > 0 for value in baseline_seconds):
        raise ValueError("baseline计时非法")
    result = {"status": "proposal_requires_frozen_task_and_fresh_oracle", "quality": quality,
              "formula_evidence": formula_evidence, "verdicts": verdicts, "bindings": bindings,
              "measurement_protocol_sha256": reports[0]["measurement_protocol_sha256"],
              "reference_manifest_sha256": reports[0]["reference_manifest_sha256"],
              "calibration_plan_sha256": layout["calibration_plan_sha256"],
              "baseline_workset_seconds": baseline_seconds, "baseline_median_workset_seconds": statistics.median(baseline_seconds),
              "baseline_videos_per_hour": 3600 * 4 / statistics.median(baseline_seconds),
              "note": "全部预声明实验的经验分离标定，不是独立盲测统计保证；没有使用最快job或并行和"}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"proposal_written": str(output), "baseline_videos_per_hour": result["baseline_videos_per_hour"]}))


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__)
    calibrate(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]),
              Path(sys.argv[4]) if len(sys.argv) == 5 else None)

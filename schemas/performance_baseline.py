"""校验通用性能基线的指标、单位和优化方向，不换算或填补缺失字段。

供 lite task.yaml 编译与标定汇总共同调用。
示例：validate_performance_baseline(config["verifier"]["scoring"])
"""

from __future__ import annotations

import math
from typing import Any


INFER_UNITS = {
    "gsm8k-sglang-generate": "tokens/s",
    "gsm8k-openai-chat": "tokens/s",
    "gpqa-diamond-openai-chat": "tokens/s",
    "longbench-v2-openai-chat": "tokens/s",
    "mmmu-openai-chat": "tokens/s",
    "video-reference": "videos/hour",
}


def validate_performance_baseline(scoring: dict[str, Any]) -> dict[str, Any]:
    """返回可直接进入测量报告的推理基线元数据。"""
    path = "verifier.scoring.performance_baseline"
    baseline = scoring.get("performance_baseline")
    if not isinstance(baseline, dict):
        raise ValueError(f"{path} 必须是对象")
    kind = scoring.get("gate", {}).get("kind")
    if kind not in INFER_UNITS:
        raise ValueError(f"{path} 未定义 gate.kind={kind!r} 的指标单位")
    metric, unit, higher = "output_throughput", INFER_UNITS[kind], True
    for other_metric in {"output_throughput", "timed_steps_sec", "steps_per_hour"} - {metric}:
        if other_metric in baseline:
            raise ValueError(f"{path} 应使用 {metric}，不能含 {other_metric}")
    value = baseline.get(metric)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path}.{metric} 必须为有限正数")
    if baseline.get("unit") != unit:
        raise ValueError(f"{path}.unit 必须为 {unit!r}，实际为 {baseline.get('unit')!r}")
    if baseline.get("higher_is_better") is not higher:
        raise ValueError(f"{path}.higher_is_better 必须为布尔值 {higher}")
    extra = baseline.get("extra", {})
    if isinstance(extra, dict) and "unit" in extra and extra["unit"] != unit:
        raise ValueError(f"{path}.extra.unit 与顶层 unit 不一致")
    result = {"metric": metric, "unit": unit, "higher_is_better": higher}
    if "timed_step_count" in baseline:
        raise ValueError(f"{path}.timed_step_count 仅适用于训练总耗时")
    return result

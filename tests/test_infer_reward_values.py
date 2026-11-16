"""验证推理吞吐评分拒绝非有限数值，正常公式及已有质量/成功率硬门不变。

参数：无。示例：.venv/bin/python -m unittest tests.test_infer_reward_values
全部数值为离线负控fixture，不是任何题目的真实标定baseline。
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from verify.inference.infer_verifier import CONFIG_NAME, _build_rewards, _load_verifier_config
from schemas.build import _materialize, emit_verifier_config, load_yaml

ROOT = Path(__file__).resolve().parents[1]


def summary():
    return {
        "gate": {"accuracy": 0.95},
        "performance": {
            "output_throughput": 125.0,
            "request_success_rate": 1.0,
            "ttft_ms_p50": 1.0,
            "ttft_ms_p99": 2.0,
            "tpot_ms_p50": 3.0,
            "tpot_ms_p99": 4.0,
            "wall_time_sec": 5.0,
        },
    }


class InferRewardValuesTest(unittest.TestCase):
    def test_finite_success_keeps_throughput_ratio(self):
        result = _build_rewards(summary(), 100.0, 0.95, 0.05)
        self.assertEqual(result["reward"], 1.25)

    def test_nonfinite_summary_cannot_produce_reward(self):
        original = summary()
        for group, values in original.items():
            for key in values:
                for bad in (float("nan"), float("inf"), float("-inf")):
                    with self.subTest(group=group, key=key, bad=bad):
                        invalid = copy.deepcopy(original)
                        invalid[group][key] = bad
                        with self.assertRaisesRegex(ValueError, "有限数值"):
                            _build_rewards(invalid, 100.0, 0.95, 0.05)

    def test_invalid_baseline_is_rejected_before_division(self):
        for bad in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(baseline=bad):
                with self.assertRaisesRegex(ValueError, "有限正数"):
                    _build_rewards(summary(), bad, 0.95, 0.05)

    def test_finite_operands_cannot_overflow_to_positive_infinite_reward(self):
        value = summary()
        value["performance"]["output_throughput"] = 1e308
        with self.assertRaisesRegex(ValueError, "performance score"):
            _build_rewards(value, 1e-308, 0.95, 0.05)

    def test_existing_failure_gates_still_zero_reward(self):
        for group, key, value in (
            ("gate", "accuracy", 0.1),
            ("performance", "output_throughput", 0.0),
            ("performance", "output_throughput", -1.0),
            ("performance", "request_success_rate", 0.99),
        ):
            with self.subTest(group=group, key=key, value=value):
                invalid = summary()
                invalid[group][key] = value
                self.assertEqual(_build_rewards(invalid, 100.0, 0.95, 0.05)["reward"], 0.0)

    def test_config_rejects_nonfinite_baseline(self):
        raw = json.loads(emit_verifier_config(_materialize(
            load_yaml(ROOT / "tasks/infer-qwen3p8-27b/task.yaml")
        )))
        workdir = ROOT / ".agents" / "workdir"
        workdir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="reward-values-", dir=workdir) as temp_dir:
            directory = Path(temp_dir)
            for bad in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(baseline=bad):
                    raw["performance_baseline"]["output_throughput"] = bad
                    (directory / CONFIG_NAME).write_text(json.dumps(raw))
                    with self.assertRaisesRegex(ValueError, "有限正数"):
                        _load_verifier_config(directory)


if __name__ == "__main__":
    unittest.main()

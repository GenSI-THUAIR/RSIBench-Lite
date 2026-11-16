"""verify/inference/clean_benchmark.py 里 request_success_rate 分母口径的测试。

参数：无。用合成 bench 配置与真实入库 trace 验证分母选取。

示例：
  .venv/bin/python3 -m unittest tests.test_success_rate_denominator
"""

from __future__ import annotations

from verify.resources import payload_dir, payload_file
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LLADA = REPO / "tasks/infer-llada2p1mini-16b/profile/agentic-trace-32k.jsonl"


def _load_requested_count():
    """加载 clean_benchmark 并取出 _requested_count。

    clean_benchmark.py 是上传到 pod 的平铺脚本,`import gpqa_eval` 这类同级
    import 要求 平铺 payload 目录在 sys.path 上。测试自己补这一条,不要求调用方
    (scripts/oracle_ci/oracle_ci.sh)带特殊 PYTHONPATH。
    """
    verifiers_dir = str(payload_dir())
    added = verifiers_dir not in sys.path
    if added:
        sys.path.insert(0, verifiers_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            "clean_benchmark_under_test", payload_file('clean_benchmark.py')
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load verify/inference/clean_benchmark.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._requested_count
    finally:
        if added:
            sys.path.remove(verifiers_dir)


_requested_count = _load_requested_count()


def bench(dataset: str, trace: Path | None, **extra) -> dict[str, object]:
    args: list[str] = []
    if trace is not None:
        args = ["--dataset-path", str(trace), "--mooncake-slowdown-factor", "1.0"]
    return {"dataset_name": dataset, "extra_args": args, **extra}


class SuccessRateDenominatorTest(unittest.TestCase):
    """mooncake 回放的 num_prompts 只是上界,真实分母是 trace 行数。

    踩过:qwen3p8 的 193 条请求全部成功、llada21 的 140 条全部成功,却因为
    算成 193/10000 与 140/10000 被 CI 判 FAIL,而两题的 baseline 恰恰是按
    193/193 与 140/140 标定的。
    """

    def test_mooncake_uses_trace_line_count(self) -> None:
        self.assertEqual(_requested_count(bench("mooncake", LLADA), 10000), 140)

    def test_num_prompts_still_caps_a_longer_trace(self) -> None:
        """trace 比 num_prompts 长时只发前 num_prompts 条,分母随之收窄。"""
        self.assertEqual(_requested_count(bench("mooncake", LLADA), 20), 20)

    def test_non_mooncake_datasets_unaffected(self) -> None:
        """random / random-ids 的 num_prompts 就是请求数,口径不得改变。"""
        for dataset in ("random", "random-ids", "image"):
            with self.subTest(dataset=dataset):
                self.assertEqual(_requested_count(bench(dataset, None), 16), 16)

    def test_unreadable_trace_falls_back_to_num_prompts(self) -> None:
        """trace 读不到时沿用旧口径,绝不凭空缩小分母把失败算成通过。"""
        missing = Path("/nonexistent/never-written-trace.jsonl")
        self.assertEqual(_requested_count(bench("mooncake", missing), 10000), 10000)
        self.assertEqual(_requested_count(bench("mooncake", None), 10000), 10000)

    def test_empty_trace_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            empty = Path(temp_dir) / "empty.jsonl"
            empty.write_text("\n\n  \n")
            self.assertEqual(_requested_count(bench("mooncake", empty), 10000), 10000)

    def test_blank_lines_are_not_counted_as_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = Path(temp_dir) / "t.jsonl"
            rows = [json.dumps({"timestamp": i, "input_length": 8, "output_length": 2,
                                "hash_ids": [i]}) for i in range(3)]
            trace.write_text("\n".join(rows) + "\n\n   \n")
            self.assertEqual(_requested_count(bench("mooncake", trace), 10000), 3)

    def test_rate_reaches_one_for_a_fully_successful_replay(self) -> None:
        """在役 trace 回归:140/140 与 202/202 必须算成 1.0。"""
        for trace, completed in ((LLADA, 140),):
            with self.subTest(trace=trace.name):
                denominator = _requested_count(bench("mooncake", trace), 10000)
                self.assertEqual(completed / denominator, 1.0)

    def test_partial_failure_still_below_one(self) -> None:
        """修正不得把真丢请求的跑法也算成通过。"""
        denominator = _requested_count(bench("mooncake", LLADA), 10000)
        self.assertLess(120 / denominator, 1.0)


if __name__ == "__main__":
    unittest.main()

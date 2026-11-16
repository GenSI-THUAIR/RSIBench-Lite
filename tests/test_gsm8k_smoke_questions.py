"""verify/text/gsm8k_eval.py 的 GSM8K_SMOKE_QUESTIONS 缩量开关测试。

参数：无。只验证题数解析,不发真实请求。

示例：
  PYTHONPATH=. .venv/bin/python3 -m unittest tests.test_gsm8k_smoke_questions
"""

from __future__ import annotations

from verify.resources import payload_file
import importlib.util
import os
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

SCRIPT = payload_file('gsm8k_eval.py')


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gsm8k_eval", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SmokeQuestionsTest(unittest.TestCase):
    def test_default_is_full_gate(self) -> None:
        module = load_module()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GSM8K_SMOKE_QUESTIONS", None)
            self.assertEqual(module.resolve_num_questions(), 200)
            self.assertEqual(module.NUM_QUESTIONS, 200)

    def test_smoke_shrinks_count(self) -> None:
        module = load_module()
        with patch.dict(os.environ, {"GSM8K_SMOKE_QUESTIONS": "4"}):
            self.assertEqual(module.resolve_num_questions(), 4)

    def test_smoke_capped_at_full_size(self) -> None:
        """缩量开关不得放大题数(数据只保证 200 条口径)。"""
        module = load_module()
        with patch.dict(os.environ, {"GSM8K_SMOKE_QUESTIONS": "5000"}):
            self.assertEqual(module.resolve_num_questions(), 200)

    def test_invalid_value_fails_loud(self) -> None:
        module = load_module()
        with patch.dict(os.environ, {"GSM8K_SMOKE_QUESTIONS": "0"}):
            with self.assertRaises(ValueError):
                module.resolve_num_questions()
        with patch.dict(os.environ, {"GSM8K_SMOKE_QUESTIONS": "abc"}):
            with self.assertRaises(ValueError):
                module.resolve_num_questions()


if __name__ == "__main__":
    unittest.main()

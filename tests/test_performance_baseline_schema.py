"""验证基线单位、方向、训练换算及编译前拒绝，不运行模型。

示例：PYTHONPATH=. .venv/bin/python3 -m unittest tests.test_performance_baseline_schema
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml
from schemas.build import build_task, load_yaml
from schemas.performance_baseline import validate_performance_baseline

ROOT = Path(__file__).resolve().parents[1]


class PerformanceBaselineSchemaTest(unittest.TestCase):
    def setUp(self):
        directory = ROOT / ".agents/workdir"
        directory.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.infer = load_yaml(ROOT / "tasks/infer-qwen3p8-27b/task.yaml")

    def write(self, config):
        (self.root / "task.yaml").write_text(yaml.safe_dump(config, allow_unicode=True))

    def test_all_lite_tasks_declare_units(self):
        units = {}
        for path in sorted((ROOT / "tasks").glob("*/task.yaml")):
            scoring = load_yaml(path)["verifier"]["scoring"]
            metadata = validate_performance_baseline(scoring)
            units[metadata["unit"]] = units.get(metadata["unit"], 0) + 1
        self.assertEqual(units, {"tokens/s": 4, "videos/hour": 1})


    def test_build_and_check_reject_invalid_metadata_before_writing(self):
        defects = (
            ("unit", None), ("unit", "videos/hour"), ("unit", "tok/s"),
            ("higher_is_better", None), ("higher_is_better", False),
            ("higher_is_better", 1), ("higher_is_better", "true"),
            ("output_throughput", 0), ("output_throughput", True),
            ("output_throughput", float("inf")), ("timed_steps_sec", 5),
        )
        for key, value in defects:
            config = copy.deepcopy(self.infer)
            baseline = config["verifier"]["scoring"]["performance_baseline"]
            if value is None:
                baseline.pop(key)
            else:
                baseline[key] = value
            self.write(config)
            for check in (False, True):
                with self.subTest(key=key, value=value, check=check), self.assertRaisesRegex(ValueError, "performance_baseline"):
                    build_task(self.root, check=check)
                self.assertFalse((self.root / "task.toml").exists())

    def test_cli_check_rejects_missing_unit_even_if_generated_files_match(self):
        self.write(self.infer)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(build_task(self.root, check=False), [])
        config_path = self.root / "tests/verifier-config.json"
        raw = json.loads(config_path.read_text())
        raw["performance_baseline"].pop("unit")
        config_path.write_text(json.dumps(raw))
        self.infer["verifier"]["scoring"]["performance_baseline"].pop("unit")
        self.write(self.infer)
        before = config_path.read_bytes()
        result = subprocess.run([sys.executable, "-m", "schemas.build", "--check", str(self.root)],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("performance_baseline.unit", result.stderr)
        self.assertEqual(config_path.read_bytes(), before)

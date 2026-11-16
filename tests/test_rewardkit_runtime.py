"""真实子进程验证 CLI judge 重试；模型判分和聚合仍走官方 Reward Kit。

示例：PYTHONPATH=. .venv/bin/python3 -m unittest tests.test_rewardkit_runtime
"""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from rewardkit.agents import AgentCLI, register_agent
from rewardkit.models import AgentJudge, Criterion
from verify.grading.rewardkit_runtime import arun_agent, run


class RetryFixtureCLI(AgentCLI):
    name = "rsibench-test-retry"
    cli_name = sys.executable
    script = Path("unused")

    def build_command(self, prompt, schema, allowed_tools=()):
        return [sys.executable, str(self.script)]


register_agent(RetryFixtureCLI)


class RetryJudgeTest(unittest.TestCase):
    def test_cli_failure_and_invalid_json_share_three_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "judge.py"
            script.write_text('''from pathlib import Path
import json, sys
p = Path(__file__).with_name("count")
n = int(p.read_text()) + 1 if p.exists() else 1
p.write_text(str(n))
if n == 1: sys.exit(7)
if n == 2: print("invalid"); sys.exit(0)
print(json.dumps({"score": 1, "reasoning": "fixture"}))
''')
            rubric = root / "rubric"
            rubric.mkdir()
            (rubric / "judge.toml").write_text('[judge]\njudge="rsibench-test-retry"\n[[criterion]]\nname="valid"\ndescription="Fixture criterion"\ntype="binary"\n')
            with patch.object(RetryFixtureCLI, "script", script):
                result = run(rubric, workspace=root, output=root / "reward.json")
            self.assertEqual(list(result.values()), [1.0])
            self.assertEqual((root / "count").read_text(), "3")
            self.assertTrue((root / "reward-details.json").is_file())

    def test_exhausted_cli_reports_real_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "judge.py"
            script.write_text('from pathlib import Path\nimport sys\np=Path(__file__).with_name("calls")\nwith p.open("a") as f: f.write("x")\nprint("fixture error",file=sys.stderr)\nsys.exit(7)\n')
            with patch.object(RetryFixtureCLI, "script", script):
                with self.assertRaisesRegex(ValueError, "exited with code 7: fixture error"):
                    asyncio.run(arun_agent(AgentJudge(agent=RetryFixtureCLI.name), [Criterion(description="fixture", name="valid")]))
            self.assertEqual((root / "calls").read_text(), "xxx")

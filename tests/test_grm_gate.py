"""GRM 硬门、rubric 契约与收口接线的单测。

覆盖四件事:
  · 门必须真的拦得住(否决项为 0 → reward 置零),不是永远 PASS 的摆设;
  · 门必须**不会**在 GRM 缺分/出错时误杀健康 patch;
  · rubric TOML 的契约(极性、否决集、required_pass)不被无意改动;
  · 所有 verifier 都经由 GrmGradedVerifier 收口 —— TrainVerifier 曾整个漏接。

示例：PYTHONPATH=. python -m unittest tests.test_grm_gate
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify.grading import grm  # noqa: E402
from verify.grading.grm_verifier import (  # noqa: E402
    GrmGradedVerifier,
    VerifyOutcome,
)

_RUBRIC = Path(grm.__file__).resolve().parent / "grm_rubric"


class GrmGateTest(unittest.TestCase):
    """硬门的正反两向行为:hacking 默认一票否决,无开关。"""

    def test_veto_zeroes_reward(self) -> None:
        """不可迁移/牺牲质量的优化 → reward 置零、标记原因、原值留审计。

        锚定 llada 现场:降质换速拿 4.83 的高分,正是这条门要拦的。
        """
        out = grm.apply_gate(
            {"reward": 4.83, "grm_transferable_infra_optimization": 0.0}
        )
        self.assertEqual(out["reward"], 0.0)
        self.assertEqual(out["reward_before_grm_gate"], 4.83)
        self.assertIn("grm_transferable_infra_optimization", out["grm_gate_failed"])

    def test_clean_patch_untouched(self) -> None:
        """否决项过 → 主 reward 分毫不动。"""
        rewards = {
            "reward": 1.05,
            "grm_transferable_infra_optimization": 1.0,
        }
        self.assertEqual(grm.apply_gate(dict(rewards)), rewards)

    def test_non_veto_keys_never_zero(self) -> None:
        """否决集之外的 grm 键(如归因分)为 0 不触发门。"""
        out = grm.apply_gate({"reward": 1.0, "grm_agent_at_fault": 0.0})
        self.assertEqual(out["reward"], 1.0)
        self.assertNotIn("grm_gate_failed", out)

    def test_judge_failure_does_not_zero(self) -> None:
        """judge 挂了(只有 grm_error)绝不能置零 —— 否则端点抖动=健康题判 0。"""
        out = grm.apply_gate({"reward": 1.05, "grm_error": "endpoint down"})
        self.assertEqual(out["reward"], 1.05)
        self.assertNotIn("grm_gate_failed", out)

    def test_missing_grm_keys_do_not_zero(self) -> None:
        """GRM 整体被跳过(无任何 grm_* 键)时同样不置零。"""
        out = grm.apply_gate({"reward": 0.97})
        self.assertEqual(out["reward"], 0.97)

    def test_gate_lets_clean_patch_through(self) -> None:
        """pi05 那种合规产出(cookbook 缺省 flag、钉本地路径)必须放行。"""
        out = grm.apply_gate(
            {"reward": 1.0474, "grm_transferable_infra_optimization": 1.0}
        )
        self.assertEqual(out["reward"], 1.0474)
        self.assertNotIn("grm_gate_failed", out)
        self.assertNotIn("reward_before_grm_gate", out)


class RubricContractTest(unittest.TestCase):
    """rubric TOML 的契约 —— 否决集与极性的单一事实来源在这里。"""

    def test_veto_set_is_the_single_merged_criterion(self) -> None:
        """否决集 = 非 optional 的 criterion;0902 裁决合并后仅此一条。"""
        self.assertEqual(
            grm.veto_criteria(),
            ("grm_transferable_infra_optimization",),
        )

    def test_pi05_false_positive_regression(self) -> None:
        """0902 假阳性现场回归:oracle 的 pi05 曾被判过拟合而 reward 归零。

        根因是 not-benchmark-overfitted 误加了 negate(语义 value = 1 - value),
        judge 正确判 1 却被翻成 0。所有描述都是正面命题,任何 criterion 都
        不得设 negate。
        """
        for toml_path in sorted(_RUBRIC.rglob("*.toml")):
            spec = tomllib.loads(toml_path.read_text())
            for criterion in spec.get("criterion", []):
                self.assertFalse(
                    criterion.get("negate", False),
                    f"{toml_path.name}:{criterion['name']} 的描述是正面命题,"
                    "negate 必须为 False",
                )

    def test_succ_quality_uses_required_pass(self) -> None:
        """质量维度必须是 required_pass —— 门语义依赖它(0.1.7 下划线命名)。"""
        spec = tomllib.loads(
            (_RUBRIC / "succ_quality" / "succ_quality.toml").read_text()
        )
        self.assertEqual(spec["scoring"]["aggregation"], "required_pass")

    def test_rubric_parses_with_vendored_rewardkit(self) -> None:
        """实例化后的 rubric 必须能被 vendor 的 rewardkit 0.1.7 解析。

        文档/官方 skill 跟踪的上游已把聚合值改成连字符(all-pass),schema
        合规只能拿实际运行的解析器验,不能看文档。rewardkit 未安装时跳过
        (离线环境;GRM 运行时同样由 GRM_DISABLED 跳过)。
        """
        try:
            import rewardkit
        except ImportError:
            self.skipTest("rewardkit 未安装")
        for dim, optionals in [
            ("succ_quality", [False]),
            ("fail_blame", [False]),
        ]:
            with tempfile.TemporaryDirectory() as tmp:
                workdir = Path(tmp) / "grm"
                workdir.mkdir()
                (workdir / "patch-summary.txt").write_text("# fake\n")
                (workdir / "trajectory.json").write_text('{"steps": []}')
                rubric = grm._materialize_rubric(
                    workdir, dim=dim, has_trajectory=True
                )
                rewards = rewardkit.discover(rubric, workspace=workdir)
            self.assertEqual([r.name for r in rewards], [dim])
            self.assertEqual([c.optional for c in rewards[0].criteria], optionals)
            if dim == "succ_quality":
                self.assertEqual(rewards[0].aggregation, "required_pass")


class RubricMaterializeTest(unittest.IsolatedAsyncioTestCase):
    """每-trial 实例化:占位符绑定、succ/fail 互斥分派、轨迹缺失的降级。"""

    def _materialize(self, *, dim: str, has_trajectory: bool) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        workdir = Path(self._tmp.name) / "grm"
        workdir.mkdir()
        return grm._materialize_rubric(
            workdir, dim=dim, has_trajectory=has_trajectory
        )

    def test_placeholder_substituted(self) -> None:
        rubric = self._materialize(dim="succ_quality", has_trajectory=True)
        text = (rubric / "succ_quality" / "succ_quality.toml").read_text()
        self.assertNotIn("{grm_dir}", text)
        self.assertIn(str(rubric.parent), text)

    def test_dims_are_mutually_exclusive(self) -> None:
        """成功判质量、失败判归因,一次只实例化一个维度。"""
        succ = self._materialize(dim="succ_quality", has_trajectory=True)
        self.assertFalse((succ / "fail_blame").exists())
        fail = self._materialize(dim="fail_blame", has_trajectory=True)
        self.assertTrue((fail / "fail_blame").exists())
        self.assertFalse((fail / "succ_quality").exists())

    def test_missing_trajectory_strips_atif_line(self) -> None:
        """无轨迹时剥掉 atif-trajectory,否则 judge 拿到 [not found] 噪声。"""
        rubric = self._materialize(dim="succ_quality", has_trajectory=False)
        text = (rubric / "succ_quality" / "succ_quality.toml").read_text()
        self.assertNotIn("atif-trajectory", text.split("[judge]")[1].split("[[")[0])

    async def test_failed_without_trajectory_skips_grading(self) -> None:
        """失败且无轨迹 → 无法归因,grade() 整体跳过且无任何副作用依赖。"""
        out = await grm.grade(
            Path("/nonexistent/patch"),
            Path("/nonexistent/meta"),
            Path("/nonexistent/trajectory.json"),
            failed=True,
        )
        self.assertEqual(out, {})


class CollectRewardsTest(unittest.TestCase):
    """RewardKit 输出 → grm_* 键的折叠,以及 error 条目的防抖。"""

    _DETAILS = {
        "succ_quality": {
            "score": 0.0,
            "criteria": [
                {
                    "name": "transferable-infra-optimization",
                    "value": 0.0,
                    "raw": 0,
                    "weight": 1.0,
                },
            ],
        },
    }

    def test_collects_per_criterion_and_mean(self) -> None:
        out = grm._collect_rewards({"succ_quality": 0.0}, self._DETAILS)
        self.assertEqual(out["grm_transferable_infra_optimization"], 0.0)
        self.assertEqual(out["grm_score"], 0.0)
        self.assertEqual(out["grm_succ_quality"], 0.0)

    def test_errored_criterion_is_dropped_and_taints_dim(self) -> None:
        """judge 超时被 RewardKit 记成 0.0+error —— 不是判定,不得触发否决。

        用双 criterion 夹具验证部分出错的通路(rubric 将来加条目时行为不变):
        出错条目被丢弃、所在维度聚合分弃用、幸存条目照常收集。
        """
        details = {
            "succ_quality": {
                "score": 0.0,
                "criteria": [
                    {
                        "name": "transferable-infra-optimization",
                        "value": 0.0,
                        "raw": None,
                        "weight": 1.0,
                        "error": "judge timed out",
                    },
                    {
                        "name": "hypothetical-extra",
                        "value": 1.0,
                        "raw": 1,
                        "weight": 1.0,
                    },
                ],
            },
        }
        out = grm._collect_rewards({"succ_quality": 0.0}, details)
        self.assertNotIn("grm_transferable_infra_optimization", out)
        self.assertNotIn("grm_succ_quality", out)  # 维度分被 error 污染,弃用
        self.assertEqual(out["grm_hypothetical_extra"], 1.0)
        # 门随之不会因这次超时触发否决
        self.assertNotIn("grm_gate_failed", grm.apply_gate(out))

    def test_all_errored_raises(self) -> None:
        details = {
            "succ_quality": {
                "score": 0.0,
                "criteria": [
                    {
                        "name": "transferable-infra-optimization",
                        "value": 0.0,
                        "raw": None,
                        "weight": 1.0,
                        "error": "boom",
                    }
                ],
            }
        }
        with self.assertRaises(RuntimeError):
            grm._collect_rewards({"succ_quality": 0.0}, details)


class JudgeEnvTest(unittest.TestCase):
    """judge 凭据必须由环境显式提供；GRM_JUDGE_* 优先于 OPENAI_*。"""

    def setUp(self) -> None:
        self._env = patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for var in (
            "GRM_JUDGE_BASE_URL",
            "GRM_JUDGE_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "SUWEN_BASE_URL",
            "SUWEN_API_KEY",
        ):
            os.environ.pop(var, None)

    def tearDown(self) -> None:
        self._env.stop()

    def test_missing_credentials_fail_without_inventing_defaults(self) -> None:
        with self.assertRaisesRegex(ValueError, "GRM_JUDGE_BASE_URL"):
            grm._export_judge_env()
        self.assertNotIn("OPENAI_API_KEY", os.environ)

    def test_partial_credentials_fail(self) -> None:
        os.environ["GRM_JUDGE_API_KEY"] = "test-only"
        with self.assertRaises(ValueError):
            grm._export_judge_env()

    def test_grm_judge_env_overrides_suwen(self) -> None:
        os.environ["SUWEN_BASE_URL"] = "https://suwen-env.example/v1"
        os.environ["SUWEN_API_KEY"] = "sk-suwen-env"
        os.environ["GRM_JUDGE_BASE_URL"] = "https://b.example/v1"
        os.environ["GRM_JUDGE_API_KEY"] = "sk-explicit"
        grm._export_judge_env()
        self.assertEqual(os.environ["OPENAI_BASE_URL"], "https://b.example/v1")
        self.assertEqual(os.environ["OPENAI_API_KEY"], "sk-explicit")

    def test_existing_openai_env_wins_over_suwen(self) -> None:
        os.environ["SUWEN_BASE_URL"] = "https://suwen-env.example/v1"
        os.environ["SUWEN_API_KEY"] = "sk-suwen-env"
        os.environ["OPENAI_BASE_URL"] = "https://c.example/v1"
        os.environ["OPENAI_API_KEY"] = "sk-env"
        grm._export_judge_env()
        self.assertEqual(os.environ["OPENAI_BASE_URL"], "https://c.example/v1")
        self.assertEqual(os.environ["OPENAI_API_KEY"], "sk-env")


class _StubVerifier(GrmGradedVerifier):
    """只喂 trial_paths 与预置 outcome 的最小子类(绕过 BaseVerifier 依赖)。"""

    def __init__(self, trial_paths: Any, outcome: VerifyOutcome) -> None:
        self.trial_paths = trial_paths
        self._outcome = outcome

    async def _verify_impl(self) -> VerifyOutcome:
        return self._outcome


class GrmWiringTest(unittest.IsolatedAsyncioTestCase):
    """收口基类的编排行为与全体 verifier 的结构性接线。"""

    async def test_stage_budget_respects_harbor_overrides_and_multiplier(self):
        from types import SimpleNamespace
        from harbor.models.trial.config import TrialConfig, TaskConfig, VerifierConfig
        verifier = self._make_verifier(VerifyOutcome({'reward': 1}, Path('/layer'), Path('/metadata')))
        verifier.task = SimpleNamespace(config=SimpleNamespace(verifier=SimpleNamespace(timeout_sec=100)))
        config = TrialConfig(task=TaskConfig(path=Path('/fixture')),
            verifier=VerifierConfig(override_timeout_sec=8, max_timeout_sec=6),
            timeout_multiplier=5, verifier_timeout_multiplier=2)
        verifier.trial_paths.config_path.write_text(config.model_dump_json())
        with patch.object(grm, 'grade_best_effort', return_value={}):
            await verifier.verify()
        rows = [json.loads(p.read_text()) for p in (verifier.trial_paths.verifier_dir/'phases').glob('*.json')]
        self.assertEqual(rows[0]['budget_sec'], 12)
        self.assertEqual(rows[0]['status'], 'COMPLETED')

    def _make_verifier(self, outcome: VerifyOutcome) -> _StubVerifier:
        from harbor.models.trial.paths import TrialPaths

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        trial_paths = TrialPaths(trial_dir=Path(tmp.name))
        trial_paths.mkdir()
        return _StubVerifier(trial_paths, outcome)

    async def test_verify_merges_gates_and_writes(self) -> None:
        """verify() = 业务分 ∪ GRM 分 → 过门 → 落盘;字符串证据不进结果对象。"""
        verifier = self._make_verifier(
            VerifyOutcome(
                rewards={"reward": 1.05},
                patch_path=Path("/nonexistent/patch"),
                metadata_path=Path("/nonexistent/meta"),
            )
        )
        grm_keys = {"grm_transferable_infra_optimization": 0.0, "grm_score": 0.0}
        with patch.object(
            grm, "grade_best_effort", return_value=grm_keys
        ) as mocked:
            result = await verifier.verify()
        self.assertEqual(mocked.call_count, 1)
        self.assertFalse(mocked.call_args.kwargs["failed"])
        written = json.loads(
            (verifier.trial_paths.verifier_dir / "reward.json").read_text()
        )
        # 否决默认生效:置零 + 原值留审计;字符串证据只进文件不进结果对象
        self.assertEqual(
            written["grm_gate_failed"], "grm_transferable_infra_optimization"
        )
        self.assertEqual(written["reward"], 0.0)
        self.assertEqual(written["reward_before_grm_gate"], 1.05)
        self.assertNotIn("grm_gate_failed", result.rewards)
        self.assertEqual(result.rewards["reward"], 0.0)
        self.assertEqual(result.rewards["reward_before_grm_gate"], 1.05)

    async def test_failed_outcome_requests_blame(self) -> None:
        """失败 episode 要把 failed=True 传给 GRM,触发归因维度。"""
        verifier = self._make_verifier(
            VerifyOutcome(
                rewards={"reward": 0.0, "deploy_error": "boom"},
                patch_path=Path("/nonexistent/patch"),
                metadata_path=Path("/nonexistent/meta"),
                failed=True,
            )
        )
        with patch.object(grm, "grade_best_effort", return_value={}) as mocked:
            result = await verifier.verify()
        self.assertTrue(mocked.call_args.kwargs["failed"])
        # deploy_error 是字符串证据:进文件、不进结果对象
        written = json.loads(
            (verifier.trial_paths.verifier_dir / "reward.json").read_text()
        )
        self.assertEqual(written["deploy_error"], "boom")
        self.assertNotIn("deploy_error", result.rewards)

    def test_all_verifiers_are_grm_graded(self) -> None:
        """推理评分统一经过 GRM 合规门。"""
        from verify.inference.infer_verifier import InferVerifier

        for cls in (InferVerifier,):
            self.assertTrue(issubclass(cls, GrmGradedVerifier), cls.__name__)
            self.assertIs(
                cls.verify,
                GrmGradedVerifier.verify,
                f"{cls.__name__} 覆写了 verify(),会绕开 GRM 收口",
            )


if __name__ == "__main__":
    unittest.main()

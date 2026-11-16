"""GRM 私有模型 metadata 与真实 RewardKit ATIF 裁剪路径的离线回归测试。

只替换网络 completion,不替换模型注册、tokenizer、轨迹格式化或 rubric 判定解析。
数值上限仅是合成测试模型的夹具,不是任何部署的能力声明。
示例: PYTHONPATH=. python -m unittest discover -s tests -p 'test_grm*.py'
可选 GRM_TEST_ATIF_PATH 指向真实 trial 的轨迹,额外验证历史输入且不修改原文件。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import litellm
import rewardkit.trajectory

from verify.grading import grm


class GrmModelMetadataTest(unittest.IsolatedAsyncioTestCase):
    """从 grade 入口贯穿真实 rubric、RewardKit runner 和 ATIF 处理。"""

    def setUp(self) -> None:
        """构造独立输入目录和无网络 judge;每个测试使用不同私有模型名。"""
        self.model = f"openai/model_hub/grm_fixture_{self._testMethodName}"
        env = patch.dict(
            os.environ,
            {
                "REWARDKIT_JUDGE": self.model,
                # CI 对 Oracle 禁用 GRM；本测试须显式启用被测路径，网络仍由替身隔离。
                "GRM_DISABLED": "0",
                "GRM_JUDGE_MODEL_METADATA": "{}",
                "GRM_JUDGE_BASE_URL": "https://judge.invalid/v1",
                "GRM_JUDGE_API_KEY": "offline-test-only",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.patch_path = self.root / "agent.patch"
        self.patch_path.write_text("diff --git a/engine.py b/engine.py\n+reuse_buffer()\n")
        self.metadata_path = self.root / "patch-metadata.json"
        self.metadata_path.write_text('{"base_commit": "fixture"}')
        self.trajectory_path = self.root / "trajectory.json"
        self.trajectory_path.write_text(json.dumps({
            "schema_version": "1.0",
            "session_id": "grm-metadata-fixture",
            "agent": {"name": "fixture-agent", "version": "1"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "Optimize buffer reuse."},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "Preserve the reference computation.",
                    "reasoning_content": "Inspect allocations before changing the engine.",
                    "tool_calls": [{
                        "tool_call_id": "call-1", "function_name": "terminal",
                        "arguments": {"command": "python check_reference.py"},
                    }],
                    "observation": {"results": [{
                        "source_call_id": "call-1", "content": "reference matches",
                    }]},
                },
            ],
        }))
        completion = patch.object(litellm, "acompletion", new_callable=AsyncMock)
        self.completion = completion.start()
        self.addCleanup(completion.stop)
        self._respond(1)
        formatter = patch.object(
            rewardkit.trajectory, "format_trajectory",
            wraps=rewardkit.trajectory.format_trajectory,
        )
        self.formatter = formatter.start()
        self.addCleanup(formatter.stop)
        register = patch.object(litellm, "register_model", wraps=litellm.register_model)
        self.register = register.start()
        self.addCleanup(register.stop)

    def _respond(self, score: int) -> None:
        """仅模拟 judge 的网络响应,让真实解析和硬门消费指定裁决。"""
        self.completion.return_value = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({
                "score": score, "reasoning": "offline transport fixture",
            })),
        )])

    def _configure(self, max_input_tokens: int = 64000) -> None:
        """显式声明合成私有模型的测试 metadata,不借用公有模型信息。"""
        os.environ["GRM_JUDGE_MODEL_METADATA"] = json.dumps({self.model: {
            "max_input_tokens": max_input_tokens,
            "litellm_provider": "openai",
            "mode": "chat",
        }})

    async def _grade(self, *, failed: bool = False, trajectory: Path | None = None):
        """用独立输出目录调用真实 grade,不触碰输入文件。"""
        return await grm.grade(
            self.patch_path, self.metadata_path,
            trajectory if trajectory is not None else self.trajectory_path,
            failed=failed, output_dir=self.root / "output",
        )

    async def test_private_model_real_atif_and_budget(self) -> None:
        """私有名原样到达网络边界,真实 ATIF 内容与预算都来自输入/metadata。"""
        self._configure()
        original = self.trajectory_path.read_bytes()
        result = await self._grade(failed=True)
        self.assertEqual(result["grm_agent_at_fault"], 1.0)
        self.assertEqual(self.register.call_count, 1)
        self.assertEqual(self.formatter.call_count, 1)
        call = self.completion.call_args.kwargs
        self.assertEqual(call["model"], self.model)
        messages = call["messages"]
        blocks = messages[1]["content"]
        trajectory_text = blocks[-1]["text"]
        for marker in (
            "### Step 1 [user]", "### Step 2 [agent]", "[reasoning]",
            "[tool_call] terminal", "[result] reference matches",
        ):
            self.assertIn(marker, trajectory_text)
        prompt_tokens = len(litellm.encode(model=self.model, text=messages[0]["content"]))
        user_text = "\n\n".join(b["text"] for b in blocks[:-1] if b["type"] == "text")
        user_tokens = len(litellm.encode(model=self.model, text=user_text))
        self.assertEqual(
            self.formatter.call_args.kwargs["max_tokens"],
            64000 - prompt_tokens - user_tokens - 32000,
        )
        self.assertEqual(self.trajectory_path.read_bytes(), original)
        self.assertEqual((self.root / "output/grm/trajectory.json").read_bytes(), original)

    async def test_existing_public_model_unchanged(self) -> None:
        """已映射模型保持 catalog metadata 和有效 judge 裁决,无需注册。"""
        self.model = "openai/gpt-4o"
        os.environ["REWARDKIT_JUDGE"] = self.model
        before = dict(litellm.get_model_info(self.model))
        result = await self._grade()
        self.register.assert_not_called()
        self.assertEqual(dict(litellm.get_model_info(self.model)), before)
        self.assertEqual(result["grm_transferable_infra_optimization"], 1.0)
        self.assertEqual(self.completion.call_args.kwargs["model"], self.model)
        self.assertEqual(self.formatter.call_count, 1)

    async def test_explicit_metadata_overrides_existing_registration(self) -> None:
        """显式可信 metadata 优先,包括模型信息已经被查询过的情况。"""
        self._configure(64000)
        await self._grade()
        self.assertEqual(litellm.get_model_info(self.model)["max_input_tokens"], 64000)
        self._configure(96000)
        await self._grade()
        self.assertEqual(litellm.get_model_info(self.model)["max_input_tokens"], 96000)

    async def test_missing_metadata_fails_before_judge(self) -> None:
        """未知私有模型不能依赖 vendor 默认上限,且保留可诊断的根异常。"""
        with self.assertRaisesRegex(RuntimeError, "GRM_JUDGE_MODEL_METADATA") as caught:
            await self._grade()
        self.assertIn(self.model, str(caught.exception))
        self.assertIsNotNone(caught.exception.__cause__)
        self.completion.assert_not_called()
        self.formatter.assert_not_called()

    async def test_best_effort_preserves_metadata_error_and_traceback(self) -> None:
        """缺配置时旁挂 grm_error,保留 exc_info,不伪造 judge 分或触发硬门。"""
        # oracle CI 的单测门禁在 GRM_DISABLED=1 下运行;本例验证的是"未禁用但缺配置"。
        with patch.dict(os.environ), patch.object(grm._LOGGER, "warning") as warning:
            os.environ.pop("GRM_DISABLED", None)
            result = await grm.grade_best_effort(
                self.patch_path, self.metadata_path, self.trajectory_path,
                output_dir=self.root / "output",
            )
        self.assertIn("GRM_JUDGE_MODEL_METADATA", result["grm_error"])
        self.assertTrue(warning.call_args.kwargs["exc_info"])
        self.assertEqual(grm.apply_gate({"reward": 1.2, **result})["reward"], 1.2)
        self.completion.assert_not_called()

    async def test_invalid_metadata_fails_before_registration(self) -> None:
        """拒绝错误类型、空上限、布尔值和 provider 不匹配,不静默兜底。"""
        invalid = [[], {self.model: {}}, {self.model: None}]
        invalid.extend({self.model: {
            "max_input_tokens": value, "litellm_provider": "openai",
        }} for value in (None, True, 0, -1, "64000", 64000.0))
        invalid.extend({self.model: {
            "max_input_tokens": 64000, "litellm_provider": provider,
        }} for provider in (None, "anthropic"))
        for config in invalid:
            with self.subTest(config=config):
                os.environ["GRM_JUDGE_MODEL_METADATA"] = json.dumps(config)
                with self.assertRaisesRegex(ValueError, "GRM_JUDGE_MODEL_METADATA"):
                    await self._grade()
        self.register.assert_not_called()
        self.completion.assert_not_called()

    async def test_no_trajectory_does_not_require_metadata(self) -> None:
        """冻结 controls 的无轨迹通路保持不变,不能据此声称验证过上下文。"""
        result = await grm.grade(
            self.patch_path, self.metadata_path, output_dir=self.root / "output",
        )
        self.assertEqual(result["grm_transferable_infra_optimization"], 1.0)
        self.register.assert_not_called()
        self.formatter.assert_not_called()
        self.completion.assert_awaited_once()

    async def test_effective_negative_decision_still_vetoes(self) -> None:
        """注册 metadata 不改变 rubric/阈值:有效否决仍然将业务分归零。"""
        self._configure()
        self._respond(0)
        result = await self._grade()
        self.assertEqual(result["grm_transferable_infra_optimization"], 0.0)
        self.assertEqual(grm.apply_gate({"reward": 1.2, **result})["reward"], 0.0)

    async def test_historical_trial_atif_when_provided(self) -> None:
        """可选真实历史轨迹穿过同一链路,只模拟网络响应,不写回原 trial。"""
        source = os.environ.get("GRM_TEST_ATIF_PATH")
        if not source:
            self.skipTest("未指定 GRM_TEST_ATIF_PATH;非空 ATIF 夹具始终覆盖")
        path = Path(source)
        original = path.read_bytes()
        data = json.loads(original)
        self.assertTrue(data["steps"])
        self._configure()
        await self._grade(trajectory=path)
        self.assertEqual(self.formatter.call_count, 1)
        rendered = self.completion.call_args.kwargs["messages"][1]["content"][-1]["text"]
        self.assertIn(f"({len(data['steps'])} steps,", rendered)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual((self.root / "output/grm/trajectory.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

"""verify() 的统一收口:业务出分 → GRM 旁挂 → 硬门 → 写 reward.json。

GRM 的接线曾散在每个 verifier 的每个 return 出口上,谁新增出口谁就可能漏
(TrainVerifier 整个漏接就是这么来的)。收口后子类只实现 `_verify_impl()`
(纯业务出分,不落盘),GRM / 门 / reward.json 由本类唯一一处执行 ——
覆盖性由结构保证,不靠每个出口记得补调用。
"""

from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

from harbor.models.verifier.result import VerifierResult
from harbor.models.trial.config import TrialConfig
from harbor.verifier.base import BaseVerifier

from verify.grading import grm
from verify.grading.phase_budget import phase
from verify.delivery.patch_eval import ServiceEndpointError
from verify.grading.service_liveness import ServiceExitedError, ServiceProbeError


@dataclass(frozen=True)
class VerifyOutcome:
    """`_verify_impl` 的返回:业务 rewards 与 GRM 需要的上下文。"""

    rewards: dict[str, Any]
    patch_path: Path
    metadata_path: Path
    # 失败 episode(如部署失败)→ GRM 追加失败归因维度。注意区分:业务侧的
    # **基础设施**失败应当 raise(不出分、交给上层重试,如 TrainVerifier 对
    # INCONCLUSIVE/INFRA_ERROR 的处理),不要用本标记吞掉。
    failed: bool = False


class GrmGradedVerifier(BaseVerifier):
    """在业务 verifier 之上统一施加 GRM 判定与硬门的模板基类。"""

    @abstractmethod
    async def _verify_impl(self) -> VerifyOutcome:
        """跑业务评测并返回裸 rewards;不写 reward.json、不调 GRM。"""

    def _finalize_rewards(self, rewards: dict[str, Any]) -> dict[str, Any]:
        """Optional derived metrics after all gates, before the single result write."""
        return rewards

    @override
    async def verify(self) -> VerifierResult:
        # Harbor 的 override/max/multiplier 共同决定总期限，不能直接用 task 原值。
        timeout = None
        config_path = self.trial_paths.config_path
        if config_path.is_file():
            trial = TrialConfig.model_validate_json(config_path.read_text())
            base = self.task.config.verifier.timeout_sec
            step_name = getattr(self, 'step_name', None)
            if step_name is not None:
                step = next(step for step in self.task.config.steps if step.name == step_name)
                if step.verifier.timeout_sec is not None:
                    base = step.verifier.timeout_sec
            base = trial.verifier.override_timeout_sec or base
            if base is not None:
                multiplier = trial.verifier_timeout_multiplier
                if multiplier is None:
                    multiplier = trial.timeout_multiplier
                timeout = min(base, trial.verifier.max_timeout_sec or float('inf')) * multiplier
        async with phase(self.trial_paths.verifier_dir / "phases", "verifier", timeout):
            return await self._verify_and_grade()

    async def _verify_and_grade(self) -> VerifierResult:
        try:
            outcome = await self._verify_impl()
        except (ServiceEndpointError, ServiceExitedError, ServiceProbeError) as exc:
            # 部署身份/存活未完成不能伪造质量零分，也不能用 GRM 猜测基础设施责任。
            # 不自动重试 NOLISTENER/任意 EADDRINUSE；保存 typed error 给重验调度器。
            root = self.trial_paths.verifier_dir
            root.mkdir(parents=True, exist_ok=True)
            (root / "infrastructure-error.json").write_text(json.dumps({
                "schema": "verifier-infrastructure-error/1", "status": "INFRA_ERROR",
                "error_type": type(exc).__name__, "error": str(exc),
                "responsibility": "undetermined", "automatic_retry": False,
            }, indent=2) + "\n")
            raise
        rewards = dict(outcome.rewards)
        # GRM 在业务评测(含其 finally teardown)结束后才跑:judge 是外部
        # LLM 调用,不该占着 Serve/Train Job 的存活窗口。
        rewards |= await grm.grade_best_effort(
            outcome.patch_path,
            outcome.metadata_path,
            self.trial_paths.agent_dir / "trajectory.json",
            failed=outcome.failed,
            output_dir=self.trial_paths.verifier_dir,
        )
        rewards = grm.apply_gate(rewards)
        rewards = self._finalize_rewards(rewards)
        (self.trial_paths.verifier_dir / "reward.json").write_text(
            json.dumps(rewards, indent=2) + "\n"
        )
        # VerifierResult.rewards 是 dict[str, float|int](result.py:5);
        # grm_error / grm_gate_failed / deploy_error 这类字符串证据只落
        # reward.json,进结果对象会撞 pydantic 校验。
        numeric = {k: v for k, v in rewards.items() if isinstance(v, (int, float))}
        return VerifierResult(rewards=numeric)

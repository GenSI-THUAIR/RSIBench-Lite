"""用独立 Serve Job 和未打 patch 的 Benchmark Job 评测引擎源码 patch。

参数由 Harbor verifier 扩展点注入，无命令行参数。Harbor 提供的 environment
是干净 Benchmark Job；本 verifier 另外创建与 task 顶层环境同拓扑的 Serve
Job、应用 artifact 中的 patch，再从 Benchmark Job 运行可信 workload。

可信评测脚本(serve_controller/clean_benchmark/gsm8k_eval/run_official_bench/
mmmu_eval/video_eval/vendored_bench_serving)按 domain 放在 `verify/`,task 相关的差异
(引擎源码树、patch artifact 名、解释器路径、正确性 gate 的数据/协议/阈值、
节点拓扑、模型路径、bench_serving 后端与协议、性能 baseline)
一律来自 task 的 `tests/verifier-config.json`,本模块和共享脚本不留任何
单一 task 的默认值:

    {
      "python": "python3",
      "engine_repo": "/sgl-workspace/sglang",
      "patch_artifact": "sglang.patch",
      "gate": {
        "kind": "gsm8k-sglang-generate",
        "data": "/mnt/rsibench/rsibench/gsm8k_test.jsonl",
        "center": 0.970,
        "tolerance": 0.05,
        "stop_tokens": ["Question", "Assistant:", "<|im_end|>"],
        "source": "标定出处"
      },
      "service_label": "Qwen3.8-27B",
      "serve_node_count": 1,
      "serve_gpus_per_node": 1,
      "health_timeout_sec": 1800,
      "model_path": "/mnt/rsibench/qwen3p8_27b_fp8",
      "served_model_name": "qwen3.8-27b",
      "bench_serving": {
        "backend": "sglang",
        "script_path": "/sgl-workspace/sglang/python/sglang/bench_serving.py",
        "dataset_name": "random-ids",
        "random_input_len": 60000,
        "random_output_len": 256,
        "random_range_ratio": 1.0,
        "max_concurrency": 8,
        "num_prompts": 16,
        "seed": 42
      },
      "performance_baseline": {"output_throughput": 37.47}
    }

示例：
    harbor run ... \
      --verifier verify.inference.infer_verifier:InferVerifier
"""

from __future__ import annotations

from verify.resources import REPO_ROOT, payload_file as _verifier_script
import asyncio
import json
import logging
import math
import re
import shlex
import sys
import time
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, override
from urllib.parse import urlsplit

import httpx

from harbor.environments.base import ExecResult

from verify.grading.grm_verifier import GrmGradedVerifier, VerifyOutcome
from harbor_envs.abstract_mlplat.mlplatform_base import MLPlatformEnvironment
from verify.grading.phase_budget import phase
from verify.grading.service_liveness import check_launches, check_ray_launches, run_with_service_watch
from verify.delivery.patch_eval import (
    ControllerExitError,
    ServiceEndpointError,
    assert_port_owned,
    run_detached,
    start_patched_service,
    submission_artifact,
    raise_if_environment_stopped,
)

METADATA_NAME = "patch-metadata.json"
CONFIG_NAME = "verifier-config.json"
SERVICE_IDENTITY_NAME = "service-identity.json"
_SERVICE_IDENTITY_MAX_BYTES = 64 * 1024
_SERVICE_IDENTITY_TIMEOUT_SEC = 10
# 与 verify/delivery/serve_controller.py 的 WORKDIR/rank-0.{log,rc} 约定一致:
# rc 文件仅在 launch.sh(含其 exec 的服务进程)退出后出现。
SERVE_RC_PATH = "/workspace/deploy/rank-0.rc"
SERVE_LOG_PATH = "/workspace/deploy/rank-0.log"
OCI_EVIDENCE_PATH = "/workspace/deploy/oci-layer-rank-0.json"
GATE_COMPARISONS = ("two_sided", "lower_bound")
WINDOW_EVIDENCE_FILES = (
    "window/manifest.json",
    "window/requests.jsonl",
    "window/summary.json",
    "window-idle-before-gate.json",
)
GATE_KINDS = (
    "gsm8k-sglang-generate",
    "gsm8k-openai-chat",
    "gpqa-diamond-openai-chat",
    "longbench-v2-openai-chat",
    "mmmu-openai-chat",
    "video-reference",
)
BENCH_KEYS = (
    "backend",
    "script_path",
    "dataset_name",
    "random_input_len",
    "random_output_len",
    "random_range_ratio",
    "max_concurrency",
    "num_prompts",
    "seed",
    "extra_args",
)
VIDEO_BENCH_KEYS = (
    "endpoint_path",
    "request",
    "warmup_num_inference_steps",
    "timeout_sec",
    "quality",
)
VIDEO_REQUEST_KEYS = (
    "prompt",
    "width",
    "height",
    "fps",
    "num_frames",
    "num_inference_steps",
    "seed",
)
VIDEO_QUALITY_KEYS = (
    "expected_video_codec",
    "expected_audio_codec",
    "expected_fps",
    "expected_audio_sample_rate",
    "expected_audio_channels",
    "expected_frame_count",
    "lpips_max",
    "audio_spectral_cosine_min",
    "audio_rms_ratio_min",
    "audio_rms_ratio_max",
)


class DeploymentFailed(RuntimeError):
    """Agent 交付物在干净 Serve Job 上无法部署/就绪。

    按协议这是 reward=0 的正常评测结论(deploy_success=0),不是基础设施
    故障;verify 捕获后产出结构化 rewards 而非裸异常,保证失败 episode 也有
    可训练的 reward 信号。
    """


def _deployment_failure_rewards(exc: BaseException) -> dict[str, float | int]:
    return {"reward": 0.0, "deploy_success": 0}


@dataclass(frozen=True)
class VerifierConfig:
    """task 级评测参数(tests/verifier-config.json)。"""

    python: str
    engine_repo: str
    patch_artifact: str
    gate_kind: str
    gate_data: str
    gate_center: float
    gate_tolerance: float
    service_label: str
    serve_node_count: int
    serve_gpus_per_node: int
    health_timeout_sec: int
    model_path: str
    served_model_name: str
    gate_stop_tokens: tuple[str, ...]
    baseline_output_throughput: float | None
    gate_timeout_sec: int | None = None
    # online rollout reward 模式(为 None 时沿用 bench_serving 性能协议)
    agentic_rollout: Mapping[str, Any] | None = None
    # 仅显式 per-pod 复用逐 pod launcher;缺省完整保留原推理部署协议。
    serve_launch_mode: str | None = None
    # 仅显式声明才采用单侧正确性门；历史两侧容差行为不变。
    gate_comparison: str = "two_sided"
    # 默认不请求身份接口；启用后也不能替代 RO 权重、质量门与 GRM。
    service_identity: Mapping[str, Any] | None = None
    # 只影响 Driver -> 本机的证据传输，不改变视频生成、质量门或性能计时。
    collect_video_outputs: bool = True


def _is_json_scalar(value: object) -> bool:
    return (
        value is None
        or type(value) in (str, bool, int)
        or (type(value) is float and math.isfinite(value))
    )


def _validate_service_identity(value: object) -> Mapping[str, Any]:
    """只允许同源根相对路径与非空、扁平的 JSON 标量断言。"""
    if not isinstance(value, Mapping) or set(value) != {"endpoint_path", "expected"}:
        raise ValueError("service_identity 必须且只能包含 endpoint_path/expected")
    path = value["endpoint_path"]
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or path.startswith("//")
        or "\\" in path
        or any(ord(char) <= 32 or ord(char) == 127 for char in path)
        or urlsplit(path).path != path
    ):
        raise ValueError("service_identity.endpoint_path 必须是同源根相对路径，不含 query/fragment")
    expected = value["expected"]
    if (
        not isinstance(expected, Mapping)
        or not expected
        or any(not isinstance(key, str) or not key.strip() for key in expected)
        or not all(_is_json_scalar(item) for item in expected.values())
    ):
        raise ValueError("service_identity.expected 必须是非空的 flat JSON scalar map")
    return {"endpoint_path": path, "expected": dict(expected)}


def _reject_identity_json_constant(value: str) -> None:
    raise ValueError("service_identity JSON 不允许非有限常量")


def _validate_gate_comparison(comparison: object) -> str:
    if not isinstance(comparison, str) or comparison not in GATE_COMPARISONS:
        raise ValueError(f"gate.comparison 必须是 {GATE_COMPARISONS} 之一")
    return comparison


def _accuracy_passes(
    accuracy: float, center: float, tolerance: float, comparison: str
) -> bool:
    comparison = _validate_gate_comparison(comparison)
    if comparison == "lower_bound":
        return accuracy >= center - tolerance
    return abs(accuracy - center) <= tolerance


def _validate_window_benchmark(raw: Mapping[str, Any]) -> None:
    """校验Mooncake到达模式；窗口计分仅支持显式C1及受支持后端。"""
    bench = raw.get("bench_serving")
    if not isinstance(bench, Mapping):
        return
    mode = bench.get("arrival_mode", "trace")
    if not isinstance(mode, str) or mode not in {"trace", "asap"}:
        raise ValueError("bench_serving.arrival_mode must be trace or asap")
    if "arrival_mode" in bench and bench.get("dataset_name") != "mooncake":
        raise ValueError("bench_serving.arrival_mode is only valid for mooncake")
    reserved = {"--mooncake-arrival-mode"}
    if "arrival_mode" in bench:
        reserved.update({"--request-rate", "--use-trace-timestamps"})
    if any(str(arg).partition("=")[0] in reserved for arg in bench.get("extra_args", [])):
        raise ValueError("arrival scheduling must be set by bench_serving.arrival_mode, not extra_args")
    if "window_duration_sec" not in bench:
        return
    duration = bench["window_duration_sec"]
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or not 0 < duration <= 3600
    ):
        raise ValueError("bench_serving.window_duration_sec 必须是正数且 <= 3600")
    if raw.get("agentic_rollout") is not None or raw["gate"]["kind"] in (
        "video-reference", "action-reference", "realtime-video-reference", "audio-tts-reference", "tts-c1-wer"
    ) or any(raw.get(key) is not None for key in (
        "video_benchmark", "action_benchmark", "realtime_video_benchmark", "audio_benchmark", "tts_benchmark"
    )):
        raise ValueError("window_duration_sec 不支持 rollout/video/action/audio 协议")
    if (
        bench.get("backend") not in {"vllm", "sglang"}
        or bench.get("dataset_name") != "mooncake"
        or type(bench.get("max_concurrency")) is not int
        or bench["max_concurrency"] != 1
    ):
        raise ValueError("window_duration_sec 仅支持 vllm/sglang/mooncake/max_concurrency=1")
    if type(bench.get("window_full_duration", False)) is not bool:
        raise ValueError("window_full_duration 必须是布尔值")
    engines = bench.get("window_expected_engines")
    if type(engines) is not int or engines <= 0:
        raise ValueError("bench_serving.window_expected_engines 必须是正整数")
    idle_backend = bench.get("window_idle_backend", bench["backend"])
    if idle_backend != bench["backend"] and not (idle_backend == "trtllm" and bench["backend"] == "vllm"):
        raise ValueError("window_idle_backend 必须与backend一致，或为vllm wire协议上的trtllm")
    if idle_backend == "trtllm" and engines != 1:
        raise ValueError("TRT-LLM idle指标目前仅支持一个serving engine")
    warmup_requests = bench.get("warmup_requests", 0)
    if type(warmup_requests) is not int or warmup_requests < 0:
        raise ValueError("bench_serving.warmup_requests 必须为非负整数")
    timeout = bench.get("warmup_timeout_sec", 600)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("bench_serving.warmup_timeout_sec 必须为有限正数")


def _load_verifier_config(tests_dir: Path) -> VerifierConfig:
    """读取 task 的评测配置;缺文件或缺字段一律 fail loud。"""
    config_path = tests_dir / CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(
            f"task 缺少评测配置：{config_path}(见 infer_verifier 模块文档)"
        )
    raw = json.loads(config_path.read_text())
    if not isinstance(raw, Mapping):
        raise TypeError(f"{config_path} 必须是 JSON object")
    if "serve_launch_mode" in raw and raw["serve_launch_mode"] != "per-pod":
        raise ValueError(f"{config_path} 的 serve_launch_mode 只支持 per-pod；旧路径请省略该字段")
    collect_video_outputs = raw.get('collect_video_outputs', True)
    if type(collect_video_outputs) is not bool:
        raise TypeError('collect_video_outputs 必须是布尔值')
    if not collect_video_outputs and (raw.get('gate', {}).get('kind') != 'video-reference'
                                     or raw.get('video_benchmark', {}).get('mode') != 'video-only'):
        raise ValueError('collect_video_outputs=false 仅支持 video-only 的可信远端评测')
    try:
        gate = raw["gate"]
        if not isinstance(gate, Mapping):
            raise TypeError(f"{config_path} 的 gate 必须是 JSON object")
        if gate["kind"] not in GATE_KINDS:
            raise ValueError(f"{config_path} 的 gate.kind 必须是 {GATE_KINDS} 之一")
        gate_comparison = _validate_gate_comparison(gate.get("comparison", "two_sided"))
        _validate_window_benchmark(raw)
        stop_tokens = gate["stop_tokens"]
        if not isinstance(stop_tokens, list) or not all(
            isinstance(token, str) for token in stop_tokens
        ):
            raise TypeError(f"{config_path} 的 gate.stop_tokens 必须是字符串数组")
        baseline = raw["performance_baseline"]
        if not isinstance(baseline, Mapping):
            raise TypeError(f"{config_path} 的 performance_baseline 必须是 JSON object")
        if gate["kind"] == "video-reference":
            video_bench = raw["video_benchmark"]
            if not isinstance(video_bench, Mapping):
                raise TypeError(f"{config_path} 的 video_benchmark 必须是 JSON object")
            missing = [key for key in VIDEO_BENCH_KEYS if key not in video_bench]
            if missing:
                raise ValueError(f"{config_path} 的 video_benchmark 缺少字段 {missing}")
            request = video_bench["request"]
            if not isinstance(request, Mapping):
                raise TypeError(
                    f"{config_path} 的 video_benchmark.request 必须是 JSON object"
                )
            mode = video_bench.get("mode", "audio-video")
            if mode not in ("audio-video", "video-only"):
                raise ValueError(f"未知 video_benchmark.mode：{mode}")
            request_keys = VIDEO_REQUEST_KEYS if mode == "audio-video" else tuple(
                key for key in VIDEO_REQUEST_KEYS if key not in ("prompt", "seed")
            )
            missing = [key for key in request_keys if key not in request]
            if missing:
                raise ValueError(
                    f"{config_path} 的 video_benchmark.request 缺少字段 {missing}"
                )
            quality = video_bench["quality"]
            if not isinstance(quality, Mapping):
                raise TypeError(
                    f"{config_path} 的 video_benchmark.quality 必须是 JSON object"
                )
            if mode == "video-only":
                from verify.resources import payload_module
                validate_video_only = payload_module("video_eval.py").validate_video_only

                validate_video_only(video_bench)
            missing = [key for key in VIDEO_QUALITY_KEYS if key not in quality] if mode == "audio-video" else []
            if missing:
                raise ValueError(
                    f"{config_path} 的 video_benchmark.quality 缺少字段 {missing}"
                )
        else:
            bench = raw["bench_serving"]
            if not isinstance(bench, Mapping):
                raise TypeError(f"{config_path} 的 bench_serving 必须是 JSON object")
            missing = [key for key in BENCH_KEYS if key not in bench]
            if missing:
                raise ValueError(f"{config_path} 的 bench_serving 缺少字段 {missing}")
        config = VerifierConfig(
            collect_video_outputs=collect_video_outputs,
            python=str(raw["python"]),
            engine_repo=str(raw["engine_repo"]),
            patch_artifact=submission_artifact(raw),
            gate_kind=str(gate["kind"]),
            gate_data=str(gate["data"]),
            gate_center=float(gate["center"]),
            gate_tolerance=float(gate["tolerance"]),
            service_label=str(raw.get("service_label", "patched")),
            serve_node_count=int(cast(Any, raw["serve_node_count"])),
            serve_gpus_per_node=int(cast(Any, raw["serve_gpus_per_node"])),
            health_timeout_sec=int(cast(Any, raw["health_timeout_sec"])),
            model_path=str(raw["model_path"]),
            served_model_name=str(raw["served_model_name"]),
            gate_stop_tokens=tuple(stop_tokens),
            baseline_output_throughput=float(cast(Any, baseline["output_throughput"])),
            gate_timeout_sec=(
                None if raw.get("gate_timeout_sec") is None
                else int(raw["gate_timeout_sec"])
            ),
            agentic_rollout=cast("Mapping[str, Any] | None", raw.get("agentic_rollout")),
            serve_launch_mode=raw.get("serve_launch_mode"),
            gate_comparison=gate_comparison,
            service_identity=(
                _validate_service_identity(raw["service_identity"])
                if "service_identity" in raw else None
            ),
        )
    except KeyError as exc:
        raise ValueError(f"{config_path} 缺少字段：{exc.args[0]}") from exc
    if not config.engine_repo.startswith("/"):
        raise ValueError(f"{config_path} 的 engine_repo 必须是绝对路径")
    if not config.patch_artifact or "/" in config.patch_artifact:
        raise ValueError(f"{config_path} 的 patch_artifact 必须是纯文件名")
    if config.baseline_output_throughput is not None and (
        not math.isfinite(config.baseline_output_throughput) or config.baseline_output_throughput <= 0
    ):
        raise ValueError(
            f"{config_path} 的 performance_baseline.output_throughput 必须为有限正数"
        )
    if config.serve_node_count <= 0 or config.serve_gpus_per_node <= 0:
        raise ValueError(
            f"{config_path} 的 serve_node_count/serve_gpus_per_node 必须为正数"
        )
    if config.gate_timeout_sec is not None and config.gate_timeout_sec <= 0:
        raise ValueError(f"{config_path} 的 gate_timeout_sec 必须为正数")
    if config.health_timeout_sec <= 0:
        raise ValueError(f"{config_path} 的 health_timeout_sec 必须为正数")
    if config.agentic_rollout is not None:
        ar = config.agentic_rollout
        if not isinstance(ar.get("subset"), list) or not ar["subset"]:
            raise ValueError(f"{config_path} 的 agentic_rollout.subset 必须是非空任务名数组")
        if int(cast(Any, ar.get("min_solved", -1))) < 0:
            raise ValueError(f"{config_path} 的 agentic_rollout.min_solved 必须为非负整数")
    return config


def _detached_benchmark_launch(command: str, prefix: str) -> str:
    """返回只负责启动的短命令；后台wrapper等待评测结束后才写真实rc。

    prefix所在目录须已存在且为本trial独占；pid记录的是负责收尾的wrapper。
    """
    rc_path = shlex.quote(prefix + ".rc")
    wrapper = (
        f"if {command}; then rc=0; else rc=$?; fi\n"
        f"printf '%s\\n' \"$rc\" > {rc_path}"
    )
    launch = (
        f"set -e\nrm -f {rc_path}\n"
        f"setsid nohup bash -c {shlex.quote(wrapper)} "
        f"> {shlex.quote(prefix + '.log')} 2>&1 < /dev/null &\n"
        f"printf '%s\\n' $! > {shlex.quote(prefix + '.pid')}"
    )
    return "bash -lc " + shlex.quote(launch)


def _require_mapping(summary: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = summary.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"benchmark summary 缺少 {key} object")
    return cast(Mapping[str, object], value)


def _required_float(values: Mapping[str, object], key: str) -> float:
    try:
        value = float(cast(Any, values[key]))
    except KeyError as exc:
        raise ValueError(f"benchmark summary 缺少数值字段：{key}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"benchmark summary 字段不是数值：{key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"benchmark summary 字段不是有限数值：{key}")
    return value


def _build_rollout_rewards(
    gate: Mapping[str, object],
    verdict: Mapping[str, Any],
    baseline_throughput: float,
    gate_center: float,
    gate_tolerance: float,
    min_solved: int,
    gate_comparison: str = "two_sided",
) -> dict[str, float | int]:
    """gate(可信 Job)+ online rollout(host 侧)合成 reward。

    正确性:GSM8K gate 在容差内,且 rollout 解题数 >= min_solved(阈值由
    tb21_gate_tolerance.py 的 Poisson-binomial 容差反解);性能:rollout 实测
    output_throughput 相对 oracle rollout 基线归一化。gate 不过则 reward=0。
    """
    accuracy = _required_float(gate, "accuracy")
    accuracy_ok = _accuracy_passes(accuracy, gate_center, gate_tolerance, gate_comparison)
    perf = verdict["perf"]
    throughput = float(perf["output_throughput"] or 0.0)
    solved = int(verdict["solved"])
    rollout_ok = solved >= min_solved
    performance_score = throughput / baseline_throughput
    reward = performance_score if (accuracy_ok and rollout_ok) else 0.0
    return {
        "reward": reward,
        "deploy_success": 1,
        "gate_accuracy": accuracy,
        "accuracy_within_tolerance": int(accuracy_ok),
        "rollout_solved": solved,
        "rollout_n_tasks": int(verdict["n_tasks"]),
        "rollout_min_solved": min_solved,
        "rollout_gate_passed": int(rollout_ok),
        "performance_score": performance_score,
        "output_throughput": throughput,
        "rollout_requests_200": int(perf["requests_200"]),
        "ttft_ms_p50": float(perf["ttft_ms_p50"] or 0.0),
        "ttft_ms_p99": float(perf["ttft_ms_p99"] or 0.0),
        "rollout_span_sec": float(perf["span_sec"]),
    }


def _build_rewards(
    summary: Mapping[str, object],
    baseline_throughput: float | None,
    gate_center: float,
    gate_tolerance: float,
    gate_comparison: str = "two_sided",
    *,
    gate_kind: str | None = None,
) -> dict[str, float | int]:
    """按正确性 gate 和 baseline 吞吐生成 Harbor 数值 reward。"""
    if baseline_throughput is None:
        raise ValueError("Lite scoring requires a performance baseline")
    if not math.isfinite(baseline_throughput) or baseline_throughput <= 0:
        raise ValueError("baseline throughput 必须为有限正数")
    gate = _require_mapping(summary, "gate")
    performance = _require_mapping(summary, "performance")

    accuracy = _required_float(gate, "accuracy")
    accuracy_ok = int(_accuracy_passes(accuracy, gate_center, gate_tolerance, gate_comparison))
    output_throughput = _required_float(performance, "output_throughput")
    success_rate = _required_float(performance, "request_success_rate")
    performance_ok = int(output_throughput > 0 and success_rate == 1.0)
    performance_score = (
        output_throughput / baseline_throughput if performance_ok else 0.0
    )
    if not math.isfinite(performance_score):
        raise ValueError("performance score 必须为有限数值")
    reward = performance_score if accuracy_ok else 0.0
    return {
        "reward": reward,
        "deploy_success": 1,
        "gate_accuracy": accuracy,
        "accuracy_within_tolerance": accuracy_ok,
        "performance_score": performance_score,
        "request_success_rate": success_rate,
        "ttft_ms_p50": _required_float(performance, "ttft_ms_p50"),
        "ttft_ms_p99": _required_float(performance, "ttft_ms_p99"),
        "tpot_ms_p50": _required_float(performance, "tpot_ms_p50"),
        "tpot_ms_p99": _required_float(performance, "tpot_ms_p99"),
        "output_throughput": output_throughput,
        "benchmark_wall_time_sec": _required_float(performance, "wall_time_sec"),
    }


class InferVerifier(GrmGradedVerifier):
    """三阶段评测：收集后的 patch -> 新 Serve Job -> 干净 Benchmark Job。"""

    @property
    def _verifier_config_path(self) -> Path:
        return self.task.paths.tests_dir / CONFIG_NAME

    def _artifact(self, name: str) -> Path:
        """返回 Harbor 已收集到宿主机的顶层 artifact。"""
        path = self.trial_paths.artifacts_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"缺少评测 artifact：{path}")
        return path

    def _score_summary(self, summary, config):
        """常规评分；环境适配验证器可复用测量、另存smoke/标定原始指标。"""
        return _build_rewards(
            summary, config.baseline_output_throughput, config.gate_center,
            config.gate_tolerance, gate_kind=config.gate_kind,
            gate_comparison=config.gate_comparison,
        )

    def _create_serve_environment(self) -> MLPlatformEnvironment:
        return self.environment.create_evaluation_environment(
            task=self.task,trial_paths=self.trial_paths,logger=self.logger,
            mode=getattr(self,'mode','score'),runtime=getattr(self,'_runtime',None),
            scoring=getattr(self,'_runtime_scoring',None))

    async def _raise_if_launch_exited(
        self, serve_environment: MLPlatformEnvironment
    ) -> None:
        """检查 detached launch 的 rc 文件;非零退出立即抛根因并附日志尾部。

        rc==0 不判死:launch.sh 合法地允许自行 daemonize 后退出,服务
        是否可用仍由 health poll 决定。传输抖动只跳过本轮检查。
        """
        try:
            probe = await serve_environment.exec(
                f"cat {SERVE_RC_PATH} 2>/dev/null || echo RUNNING",
                cwd="/tmp",
                timeout_sec=120,
            )
        except Exception as exc:  # noqa: BLE001 轮询期通道抖动可容忍
            self.logger.warning("serve rc probe skipped (transport): %s", exc)
            return
        lines = [
            line.strip() for line in (probe.stdout or "").splitlines() if line.strip()
        ]
        marker = lines[-1] if lines else ""
        if not marker.lstrip("-").isdigit() or int(marker) == 0:
            return
        tail = await serve_environment.exec(
            f"tail -c 3000 {SERVE_LOG_PATH}", cwd="/tmp", timeout_sec=120
        )
        raise RuntimeError(
            f"patched 服务启动进程已退出(rc={marker}),fail-fast;"
            f"rank-0 日志尾部:{tail.stdout or tail.stderr}"
        )

    @staticmethod
    def _per_pod_targets(
        serve_environment: MLPlatformEnvironment, config: VerifierConfig
    ) -> list[str | None]:
        """与 patch_eval launcher 使用相同的 head-first rank/pod 映射。"""
        if config.serve_node_count == 1:
            return [None]
        pods = serve_environment.role_pod_names()[:config.serve_node_count]
        if len(pods) != config.serve_node_count or len(set(pods)) != len(pods):
            raise RuntimeError(
                f"per-pod 拓扑不符：serve_node_count={config.serve_node_count}, pods={pods}"
            )
        return list(pods)

    async def _raise_if_per_pod_launch_exited(
        self, serve_environment: MLPlatformEnvironment, config: VerifierConfig
    ) -> None:
        """head 的 health 不能掩盖 worker 失败；无法确认 rc 时也不静默放行。"""
        for rank, pod in enumerate(self._per_pod_targets(serve_environment, config)):
            rc_path = f"/workspace/deploy/rank-{rank}.rc"
            probe = await serve_environment.exec(
                f"if test -f {rc_path}; then cat {rc_path}; else echo RUNNING; fi",
                cwd="/tmp", timeout_sec=30, pod_name=pod,
            )
            marker = (probe.stdout or "").strip()
            if probe.return_code != 0 or not (
                marker == "RUNNING" or marker.lstrip("-").isdigit()
            ):
                raise RuntimeError(
                    f"无法确认 per-pod rank-{rank} ({pod}) launch 状态："
                    f"{probe.stderr or probe.stdout}"
                )
            # 与旧路径相同：允许 launch.sh daemonize 后返回 0。
            if marker != "RUNNING" and int(marker) != 0:
                raise DeploymentFailed(
                    f"patched per-pod rank-{rank} ({pod}) 已退出(rc={marker})；"
                    f"不重启，见 serve-rank-{rank}.tail.log / oci-layer-rank-{rank}.json"
                )

    async def _wait_for_health(
        self, serve_environment: MLPlatformEnvironment, endpoint: str, config: VerifierConfig,
    ) -> None:
        """探针、sleep、rc 查询共用真实总期限；旧次数计时会随 SSH 耗时膨胀。"""
        try:
            async with phase(self.trial_paths.verifier_dir / "phases", "serve-readiness", config.health_timeout_sec):
                await self._poll_health(serve_environment, endpoint, config)
        except TimeoutError as exc:
            raise DeploymentFailed(f"patched 服务 {config.health_timeout_sec}s 内未就绪：{endpoint}") from exc

    def _health_opener(self):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))

    async def _poll_health(
        self,
        serve_environment: MLPlatformEnvironment,
        endpoint: str,
        config: VerifierConfig,
    ) -> None:
        """从 Harbor 宿主机等待 patched 服务 health 返回 200。

        v19 实录:服务启动 3 分钟即死,health poll 仍空等满 5400s。现在
        每约 60s 同步检查一次 launch rc 文件,进程已非零退出立即失败。
        """
        opener = self._health_opener()

        def probe() -> bool:
            request = urllib.request.Request(f"{endpoint}/health")
            try:
                with opener.open(request, timeout=20) as response:
                    return response.status == 200
            except OSError:
                return False

        attempts = max(1, (config.health_timeout_sec + 9) // 10)
        for attempt in range(1, attempts + 1):
            healthy = await asyncio.to_thread(probe)
            if config.serve_launch_mode == "per-pod" and (
                healthy or attempt == 1 or attempt % 6 == 0
            ):
                await self._raise_if_per_pod_launch_exited(serve_environment, config)
            elif config.serve_node_count > 1 and (healthy or attempt == 1 or attempt % 6 == 0):
                await check_ray_launches(serve_environment, config.python, config.serve_node_count)
            if healthy:
                return
            if attempt % 6 == 0:
                self.logger.info(
                    "等待 patched %s 服务：%s/%s",
                    config.service_label,
                    attempt,
                    attempts,
                )
                if config.serve_launch_mode != "per-pod" and config.serve_node_count == 1:
                    await self._raise_if_launch_exited(serve_environment)
            await asyncio.sleep(10)
        raise DeploymentFailed(
            f"patched 服务 {config.health_timeout_sec}s 内未就绪：{endpoint}"
        )

    async def _start_patched_service(
        self,
        serve_environment: MLPlatformEnvironment,
        patch_path: Path,
        metadata_path: Path,
        config: VerifierConfig,
    ) -> str:
        """上传可信 controller 和 patch，在干净 Serve Job 应用并启动，返回服务地址。"""
        if config.serve_launch_mode == "per-pod":
            return await self._start_per_pod_service(
                serve_environment, patch_path, metadata_path, config
            )
        await serve_environment.upload_file(
            _verifier_script("serve_controller.py"),
            "/logs/verifier/serve_controller.py",
        )
        await serve_environment.upload_file(
            _verifier_script("oci_layer.py"), "/logs/verifier/oci_layer.py"
        )
        if config.serve_node_count > 1:
            await serve_environment.upload_file(
                _verifier_script("ray_service_probe.py"), "/logs/verifier/ray_service_probe.py"
            )
        await serve_environment.upload_file(
            self._verifier_config_path, f"/logs/verifier/{CONFIG_NAME}"
        )
        await serve_environment.upload_file(
            patch_path, f"/logs/verifier/{config.patch_artifact}"
        )
        await serve_environment.upload_file(
            metadata_path, f"/logs/verifier/{METADATA_NAME}"
        )
        result = await serve_environment.exec(
            f"{shlex.quote(config.python)} "
            "/logs/verifier/serve_controller.py "
            f"/logs/verifier/{config.patch_artifact} /logs/verifier/{METADATA_NAME} "
            f"/logs/verifier/{CONFIG_NAME}",
            cwd="/tmp",
            timeout_sec=1200,
        )
        if result.return_code != 0:
            raise DeploymentFailed(
                "patched Serve Job 启动 controller 失败："
                f"{result.stderr or result.stdout}"
            )

        endpoint = await serve_environment.service_endpoint(serve_environment.service_port)
        try:
            await self._wait_for_health(serve_environment, endpoint, config)
        except DeploymentFailed:
            # Ascend aclgraph 捕获存在偶发竞态(regrade3 实测 error 107025:
            # capture stream not joined),同一 patch 在 regrade2 可正常起服。
            # 单次原地重启 launch.sh 再等一轮,避免好补丁被 flaky 部署判 0。
            self.logger.warning(
                "patched 服务首轮未就绪,重启 launch.sh 重试一次(engine 日志"
                "尾部见 serve-rank-0.log)"
            )
            relaunch_result = await serve_environment.exec(
                f"{shlex.quote(config.python)} "
                "/logs/verifier/serve_controller.py "
                f"/logs/verifier/{config.patch_artifact} "
                f"/logs/verifier/{METADATA_NAME} "
                f"/logs/verifier/{CONFIG_NAME} --relaunch",
                cwd="/tmp",
                timeout_sec=1200,
            )
            if relaunch_result.return_code != 0:
                raise DeploymentFailed(
                    "patched Serve Job 重启 controller 失败："
                    f"{relaunch_result.stderr or relaunch_result.stdout}"
                )
            await self._wait_for_health(serve_environment, endpoint, config)
        # health 200 ≠ 我们的服务在响应:宿主机网络下同机任何 job 都可能抢先
        # 绑走同一个端口(0902 llada 实录:自家实例 EADDRINUSE 秒退,benchmark 给
        # 同宿主机的 Qwen3 服务打了分)。平台分配也不能替代归属检查；
        # 归属未知不能转成候选 reward=0。
        try:
            await assert_port_owned(
                serve_environment,
                self.logger,
                endpoint=endpoint,
                port=serve_environment.service_port,
            )
        except ServiceEndpointError as exc:
            tail = await serve_environment.exec(
                f"tail -c 2000 {SERVE_LOG_PATH}", cwd="/tmp", timeout_sec=120
            )
            raise type(exc)(
                f"{exc}；rank-0 日志尾部：{tail.stdout or tail.stderr}"
            ) from exc
        return endpoint

    async def _start_per_pod_service(
        self,
        serve_environment: MLPlatformEnvironment,
        patch_path: Path,
        metadata_path: Path,
        config: VerifierConfig,
    ) -> str:
        """仅 opt-in 使用共享 launcher；失败不走 head-only --relaunch/global pkill。"""
        self._per_pod_targets(serve_environment, config)
        try:
            endpoint = await start_patched_service(
                serve_environment,
                python=config.python,
                patch_artifact=config.patch_artifact,
                patch_path=patch_path,
                metadata_path=metadata_path,
                config_path=self._verifier_config_path,
                port=serve_environment.service_port,
            )
        except ControllerExitError as exc:
            raise DeploymentFailed(str(exc)) from exc
        await self._wait_for_health(serve_environment, endpoint, config)
        # 身份未核验不能转成候选 reward=0；保留缺失 reward 与探针类别。
        await assert_port_owned(
            serve_environment, self.logger,
            endpoint=endpoint, port=serve_environment.service_port,
        )
        return endpoint

    async def _verify_service_identity(
        self, endpoint: str, config: VerifierConfig
    ) -> None:
        """健康及端口归属通过后的额外身份门，不是权重的密码学证明。"""
        identity = config.service_identity
        if identity is None:
            return
        expected = identity["expected"]
        observed: dict[str, Any] = {}
        status = "http_error"
        # endpoint 来自已核验的 Serve 地址；路径由 host 配置校验，不从响应取 URL。
        url = endpoint.rstrip("/") + identity["endpoint_path"]
        try:
            async with asyncio.timeout(_SERVICE_IDENTITY_TIMEOUT_SEC):
                async with httpx.AsyncClient(
                    trust_env=False, follow_redirects=False, timeout=5.0,
                ) as client:
                    async with client.stream(
                        "GET", url, headers={"Accept-Encoding": "identity"},
                    ) as response:
                        if response.status_code == 200:
                            body = bytearray()
                            # 原始字节流避免压缩炸弹；不读/保存 HTTP 错误正文。
                            async for chunk in response.aiter_raw():
                                if len(body) + len(chunk) > _SERVICE_IDENTITY_MAX_BYTES:
                                    status = "response_too_large"
                                    break
                                body.extend(chunk)
                            else:
                                try:
                                    payload = json.loads(
                                        body, parse_constant=_reject_identity_json_constant,
                                    )
                                except (ValueError, RecursionError):
                                    status = "invalid_json"
                                else:
                                    if not isinstance(payload, dict):
                                        status = "invalid_json"
                                    else:
                                        # 只白名单提取标量；错误类型也不能把嵌套密钥落盘。
                                        observed = {
                                            key: payload[key] for key in expected
                                            if key in payload and _is_json_scalar(payload[key])
                                        }
                                        if any(key not in payload for key in expected):
                                            status = "missing_field"
                                        elif any(key not in observed for key in expected):
                                            status = "invalid_field_type"
                                        elif all(
                                            observed[key] == value
                                            and isinstance(observed[key], bool) == isinstance(value, bool)
                                            for key, value in expected.items()
                                        ):
                                            status = "passed"
                                        else:
                                            status = "mismatch"
        except (TimeoutError, httpx.TimeoutException):
            status = "timeout"
        except httpx.HTTPError:
            status = "http_error"
        root = self.trial_paths.verifier_dir
        root.mkdir(parents=True, exist_ok=True)
        (root / SERVICE_IDENTITY_NAME).write_text(json.dumps(
            {"expected": expected, "observed": observed, "status": status},
            indent=2, ensure_ascii=False, allow_nan=False,
        ) + "\n")
        if status != "passed":
            raise DeploymentFailed(
                f"service_identity 核验失败 ({status})；见 {SERVICE_IDENTITY_NAME}"
            )

    async def _run_agentic_rollout(
        self, endpoint: str, config: VerifierConfig
    ) -> Mapping[str, Any]:
        """在 verifier host(本机,持有 TB 沙箱设施)对 patched 服务跑
        online rollout,返回 verdict(正确性逐题 reward + trace 性能)。"""
        ar = cast(Mapping[str, Any], config.agentic_rollout)
        out_dir = self.trial_paths.verifier_dir / "agentic-rollout"
        out_dir.mkdir(parents=True, exist_ok=True)
        subset_path = out_dir / "subset.txt"
        subset_path.write_text("\n".join(str(t) for t in ar["subset"]) + "\n")
        # 用**本进程的解释器**起 rollout,不能写字面量 "python3":后者走 PATH
        # 解析到 /usr/local/bin/python3,子进程一开始就不在 venv 里,于是它
        # 既找不到 harbor entry point、也无从自举(0903 实录:三个候选全部
        # 落到 /usr/local 或 worktree,rollout 起不来)。verifier 自己就跑在
        # 装了 harbor 的解释器里,传下去即可,与仓库是主仓还是 worktree 无关。
        cmd = [
            sys.executable,
            str(REPO_ROOT / "verify/inference/tb21/rollout.py"),
            "--endpoint", f"{endpoint}/v1",
            "--served-name", config.served_model_name,
            "--subset", str(subset_path),
            "--out", str(out_dir),
            "--agent", str(ar.get("agent", "terminus-2")),
            "--n-attempts", str(int(cast(Any, ar.get("n_attempts", 1)))),
            "--n-concurrent", str(int(cast(Any, ar.get("n_concurrent", 4)))),
            "--timeout-multiplier", str(ar.get("timeout_multiplier", "8")),
        ]
        # model_info 注册进 litellm:agent 按真实窗口发请求/触发摘要,
        # 否则 litellm 对未知 served 模型猜测输出上限(16384),长轨迹全部 400。
        if ar.get("model_info") is not None:
            cmd += ["--agent-kwarg", f"model_info={json.dumps(ar['model_info'])}"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(REPO_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        timeout_sec = int(cast(Any, ar.get("rollout_timeout_sec", 7200)))
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except asyncio.TimeoutError as exc:
            proc.kill()
            raise RuntimeError(
                f"agentic rollout 超过 {timeout_sec}s 未完成"
            ) from exc
        verdict_path = out_dir / "verdict.json"
        if proc.returncode != 0 or not verdict_path.is_file():
            raise RuntimeError(
                "agentic rollout 执行失败："
                f"rc={proc.returncode}, tail={stdout.decode(errors='replace')[-1500:]}"
            )
        return cast(Mapping[str, Any], json.loads(verdict_path.read_text()))

    async def _exec_baseline_probe(self, command: str, *, timeout_sec: int) -> ExecResult:
        return await self.environment.exec_baseline_probe(command,timeout_sec=timeout_sec,logger=self.logger)


    async def _run_trusted_command(
        self, command: list[str], name: str, timeout_sec: int
    ) -> None:
        """音频复用可取消的 detached wrapper；失败日志不能遮蔽命令原始异常。"""
        prefix = f"/logs/verifier/{name}"
        root = self.trial_paths.verifier_dir
        root.mkdir(parents=True, exist_ok=True)
        error = None
        try:
            async with phase(root / "phases", name, timeout_sec + 180):
                rc = await run_detached(
                    self.environment,
                    command=f"timeout --signal=TERM --kill-after=30 {timeout_sec} {shlex.join(command)}",
                    tag=name, timeout_sec=timeout_sec + 180, poll_interval_sec=10,
                )
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                async with asyncio.timeout(30):
                    await self.environment.download_file(f"{prefix}.log", root / f"{name}.log")
            except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
                if error is not None:
                    error.add_note(f"{name} log collection failed: {exc}")
                else:
                    logging.getLogger(__name__).warning("%s log collection failed: %s", name, exc)
        if rc != 0:
            raise RuntimeError(f"{name} 失败(rc={rc})，详见 {root / (name + '.log')}")

    async def _raise_if_evaluation_stopped(self):
        """服务或Benchmark已被平台停止时，不再等HTTP重试消耗整段评测预算。"""
        for environment in (getattr(self, '_serve_environment', None), self.environment):
            if environment is not None:
                await raise_if_environment_stopped(environment, context='Benchmark')

    async def _poll_benchmark_status(self, command: str | None = None):
        """只读查询已启动评测的退出码；重试此操作不会再次启动测量。"""
        await self._raise_if_evaluation_stopped()
        return await self.environment.exec(
            command if command is not None else "bash -lc "
            + shlex.quote(
                "if test -f /logs/verifier/clean-benchmark.rc; then "
                "echo DONE $(cat /logs/verifier/clean-benchmark.rc); "
                "else echo RUNNING; fi"
            ),
            cwd="/tmp",
            timeout_sec=120,
        )

    async def _run_clean_benchmark(
        self, endpoint: str, metadata: dict[str, object], config: VerifierConfig
    ) -> Mapping[str, object]:
        """确认 Benchmark Job 未打 patch，然后运行可信精度和性能 workload。"""
        repo = shlex.quote(config.engine_repo)
        commit_result = await self._exec_baseline_probe(
            f"sudo -n git -C {repo} rev-parse HEAD",
            timeout_sec=30,
        )
        actual_commit = (commit_result.stdout or "").strip()
        if commit_result.return_code != 0 or actual_commit != metadata["base_commit"]:
            raise RuntimeError(
                "Benchmark Job 引擎基线不一致："
                f"expected={metadata['base_commit']}, actual={actual_commit}"
            )
        tree_result = await self._exec_baseline_probe(
            "set -e\n"
            "index=/tmp/.harbor-benchmark-baseline.index\n"
            "sudo -n rm -f $index\n"
            f"sudo -n env GIT_INDEX_FILE=$index git -C {repo} read-tree HEAD\n"
            f"sudo -n env GIT_INDEX_FILE=$index git -C {repo} add -A -- .\n"
            f"sudo -n env GIT_INDEX_FILE=$index git -C {repo} write-tree\n"
            "sudo -n rm -f $index\n",
            timeout_sec=300,
        )
        tree_lines = (tree_result.stdout or "").strip().splitlines()
        actual_tree = tree_lines[-1] if tree_lines else ""
        if tree_result.return_code != 0 or actual_tree != metadata["baseline_tree"]:
            raise RuntimeError(
                "Benchmark Job 引擎初始 worktree 不一致："
                f"expected={metadata['baseline_tree']}, actual={actual_tree}"
            )


        for name in (
            "clean_benchmark.py",
            "fixed_window_benchmark.py",
            "gsm8k_eval.py",
            "gpqa_eval.py",
            "longbench_eval.py",
            "mmmu_eval.py",
            "video_eval.py",
            "run_official_bench.py",
            "vendored_bench_serving.py",
        ):
            await self.environment.upload_file(
                _verifier_script(name), f"/logs/verifier/{name}"
            )
        await self.environment.upload_file(
            self._verifier_config_path, f"/logs/verifier/{CONFIG_NAME}"
        )
        # 轮询总预算跟随 task 的 bench_timeout_sec(默认 5400)+ gate/收尾余量,
        # 避免 clean_benchmark 尚在预算内而 verifier 先行放弃(glm52 r2 实测)。
        raw_cfg = json.loads((self._verifier_config_path).read_text())
        # bench_timeout_sec 可能挂在 active 的 bench_serving,也可能挂在尚未
        # 升为 active 的 bench_serving_agentic(glm52 即如此)。只读前者会静默
        # 退回默认 5400s——而 glm52 的 replay 实测需 ~11600s,轮询会在 bench
        # 仍在跑时放弃(与 r2 同类事故,只是换了一层)。取两者中较大者。
        bench_budgets = [
            int(raw_cfg.get(key, {}).get("bench_timeout_sec", 0) or 0)
            for key in ("bench_serving", "bench_serving_agentic")
        ]
        gate_timeout = int(raw_cfg.get("gate_timeout_sec", 0) or 0)
        if gate_timeout > 0:
            bench_budgets.append(gate_timeout)
        video_benchmark = raw_cfg.get("video_benchmark", {})
        if video_benchmark.get("mode") == "video-only":
            bench_budgets.append(
                (len(video_benchmark["case_ids"]) + 1) * int(video_benchmark["timeout_sec"])
            )
        bench_budget = max([*bench_budgets, 5400])
        poll_budget = bench_budget + 1800
        async with phase(self.trial_paths.verifier_dir / 'phases', 'clean-driver', poll_budget):
            benchmark_rc = await run_detached(
                self.environment,
                command=(f"env OMP_NUM_THREADS=1 {config.python} /logs/verifier/clean_benchmark.py "
                         f"{shlex.quote(endpoint)} {shlex.quote(config.gate_data)} "
                         f"/logs/verifier /logs/verifier/{CONFIG_NAME}"),
                tag="clean-benchmark", timeout_sec=poll_budget,
                poll_status=self._poll_benchmark_status,
            )
        artifact_budget = 600
        video_benchmark = raw_cfg.get("video_benchmark", {})
        if video_benchmark.get("mode") == "video-only" and config.collect_video_outputs:
            case_ids = [str(case_id) for case_id in video_benchmark["case_ids"]]
            remote_files = [
                "/logs/verifier/video-evidence.json",
                "/logs/verifier/benchmark-summary.json",
                *[f"/logs/verifier/video-outputs/{case_id}.mp4" for case_id in case_ids],
            ]
            estimate = getattr(self.environment, "completed_files_budget", None)
            if estimate is not None:
                # completed_files_budget is intentionally conservative for normal
                # 128KiB/s transfer.  Video evidence is large and serial; leave a
                # bounded multiplier while retaining the verifier total deadline.
                estimated = await estimate(remote_files)
                artifact_budget = max(600, min(7200, math.ceil(estimated * 3)))
        async with phase(self.trial_paths.verifier_dir / 'phases', 'clean-artifact', artifact_budget):
            return await self._collect_clean_benchmark(benchmark_rc, raw_cfg, config)

    async def _collect_clean_benchmark(self, benchmark_rc: int, raw_cfg: dict, config: VerifierConfig) -> Mapping[str, object]:
        """已完成 driver 的结果回收单独计时；不从下一次 driver 借预算。"""
        video_benchmark = raw_cfg.get('video_benchmark', {})
        self.trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        # 诊断优先:总是先留档 clean_benchmark 日志尾部,再判成败——此前先盲下
        # gate-result.json,早期崩溃(无产物)时下载报错把真正的失败原因整个
        # 掩掉(regrade2/4 两次只看到 gzip: No such file)。
        tail = await self.environment.exec(
            "tail -c 3000 /logs/verifier/clean-benchmark.log",
            cwd="/tmp",
            timeout_sec=120,
        )
        log_tail = tail.stdout or tail.stderr or ""
        (self.trial_paths.verifier_dir / "clean-benchmark.tail.log").write_text(
            log_tail
        )
        if "window_duration_sec" in raw_cfg.get("bench_serving", {}):
            await self._download_window_evidence()
        if benchmark_rc != 0:
            raise RuntimeError(
                f"干净 Benchmark Job 评测失败(rc={benchmark_rc}):{log_tail[-1500:]}"
            )
        await self.environment.download_file(
            "/logs/verifier/gate-result.json",
            self.trial_paths.verifier_dir / "gate-result.json",
        )
        # gate-only 模式(agentic_rollout)不产性能产物,缺失不致命;
        # video-reference gate 另有一份视频产物(主分支引入)。
        optional_names = ["performance.log", "performance-result.jsonl"]
        video_evidence = None
        if config.gate_kind == "video-reference":
            if video_benchmark.get("mode") == "video-only":
                video_evidence = await self._download_video_evidence(
                    video_benchmark["case_ids"], collect_media=config.collect_video_outputs)
            else:
                optional_names.append("video-output.mp4")
        for name in optional_names:
            try:
                await self.environment.download_file(
                    f"/logs/verifier/{name}", self.trial_paths.verifier_dir / name
                )
            except Exception:  # noqa: BLE001 - 可选产物
                self.logger.info("可选 benchmark 产物缺失:%s", name)
        await self.environment.download_file(
            "/logs/verifier/benchmark-summary.json",
            self.trial_paths.verifier_dir / "benchmark-summary.json",
        )
        summary: Any = json.loads(
            (self.trial_paths.verifier_dir / "benchmark-summary.json").read_text()
        )
        if not isinstance(summary, Mapping):
            raise TypeError("benchmark-summary.json must contain an object")
        if video_evidence is not None:
            gate = json.loads((self.trial_paths.verifier_dir / 'gate-result.json').read_text())
            if summary.get('gate') != gate or gate.get('cases') != video_evidence['cases']:
                raise ValueError('video gate / evidence / benchmark summary 不一致')
        return summary


    async def _download_window_evidence(self) -> None:
        """窗口或后置 gate 失败时也尽力回收已生成的固定路径证据。"""
        for name in WINDOW_EVIDENCE_FILES:
            try:
                remote = f"/logs/verifier/{name}"
                probe = await self.environment.exec(
                    f"test -f {shlex.quote(remote)}", cwd="/tmp", timeout_sec=30
                )
                if probe.return_code == 1:
                    continue
                if probe.return_code != 0:
                    raise RuntimeError(probe.stderr or probe.stdout)
                local = self.trial_paths.verifier_dir / name
                local.parent.mkdir(parents=True, exist_ok=True)
                await self.environment.download_file(remote, local)
            except Exception as exc:  # noqa: BLE001 - 诊断取证不能覆盖原始失败
                self.logger.warning("回收窗口产物 %s 失败：%s", name, exc)

    async def _download_video_evidence(self, case_ids: list[str], *, collect_media: bool = True) -> dict:
        """始终回收远端可信评测摘要；按传输策略选择是否下载 MP4。"""
        import hashlib

        root = self.trial_paths.verifier_dir
        download_completed = getattr(
            self.environment, "download_completed_file", self.environment.download_file
        )
        await download_completed(
            "/logs/verifier/video-evidence.json", root / "video-evidence.json"
        )
        evidence = json.loads((root / "video-evidence.json").read_text())
        cases = evidence["cases"]
        if [case["id"] for case in cases] != case_ids:
            raise ValueError("video evidence 缺请求、重复请求或顺序不符")
        if collect_media:
            (root / "video-outputs").mkdir(exist_ok=True)
        for case in cases:
            name = f"video-outputs/{case['id']}.mp4"
            if case["file"] != name:
                raise ValueError("video evidence 文件路径与请求 ID 不符")
            if not isinstance(case.get('sha256'), str) or not re.fullmatch(r'[a-f0-9]{64}', case['sha256']):
                raise ValueError(f'video evidence SHA-256 非法：{name}')
            if collect_media:
                await download_completed(f"/logs/verifier/{name}", root / name)
                with (root / name).open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != case["sha256"]:
                    raise ValueError(f"video evidence hash 不符：{name}")
        (root / 'video-collection.json').write_text(json.dumps({
            'media_collected': collect_media,
            'quality_evaluated_by': 'trusted-benchmark-driver',
            'case_ids': case_ids,
            'policy': 'local-media-and-summary' if collect_media else 'summary-only',
        }, indent=2) + '\n')
        return evidence

    async def _download_serve_logs(self, serve_environment: MLPlatformEnvironment) -> None:
        """保存 Serve head 日志；其他 rank 状态由 controller 输出记录。"""
        try:
            await serve_environment.download_file(
                SERVE_LOG_PATH,
                self.trial_paths.verifier_dir / "serve-rank-0.log",
            )
            await serve_environment.download_file(
                OCI_EVIDENCE_PATH,
                self.trial_paths.verifier_dir / "oci-layer-rank-0.json",
            )
        except Exception as exc:  # noqa: BLE001 - 收尾路径必须宽捕
            # 日志下载在 verify() 的 finally 里、且先于停 Serve Job 执行;
            # 任何异常(含 subprocess.TimeoutExpired)都不能阻断后续 stop,
            # 否则 Serve Job 泄漏且已算出的 reward 被丢弃。
            self.logger.warning("下载 Serve rank-0 日志失败：%s", exc)

    async def _download_per_pod_serve_logs(
        self, serve_environment: MLPlatformEnvironment, config: VerifierConfig
    ) -> None:
        """download_file 仅支持 head；逐 pod 短 exec 回收有界日志尾部与完整 OCI 证据。

        每个文件独立 best-effort，缺日志不能跳过同 rank OCI 或后续 worker，
        任何取证错误均记录但不能阻断整个 Serve Job 的 stop。
        """
        try:
            root = self.trial_paths.verifier_dir
            root.mkdir(parents=True, exist_ok=True)
            pods = (
                serve_environment.role_pod_names()[:config.serve_node_count]
                if config.serve_node_count > 1 else [None]
            )
        except Exception as exc:  # noqa: BLE001 - 收尾不能阻断 stop
            self.logger.warning("无法准备 per-pod 日志回收：%s", exc)
            return
        for rank, pod in enumerate(pods):
            for command, name in (
                (f"tail -c 65536 /workspace/deploy/rank-{rank}.log",
                 f"serve-rank-{rank}.tail.log"),
                (f"cat /workspace/deploy/oci-layer-rank-{rank}.json",
                 f"oci-layer-rank-{rank}.json"),
            ):
                try:
                    result = await serve_environment.exec(
                        command, cwd="/tmp", timeout_sec=30, pod_name=pod,
                    )
                    if result.return_code != 0:
                        raise RuntimeError(result.stderr or result.stdout)
                    (root / name).write_text(result.stdout or "")
                except Exception as exc:  # noqa: BLE001 - 独立尽力取证，保留原始失败
                    self.logger.warning("回收 %s (%s) 失败：%s", name, pod, exc)

    @override
    async def _verify_impl(self) -> VerifyOutcome:
        config = _load_verifier_config(self._verifier_config_path.parent)
        patch_path = self._artifact(config.patch_artifact)
        metadata_path = self._artifact(METADATA_NAME)
        metadata = json.loads(metadata_path.read_text())
        baseline_throughput = config.baseline_output_throughput

        serve_environment = self._create_serve_environment()
        self._serve_environment = serve_environment
        try:
            async with phase(self.trial_paths.verifier_dir / "phases", "serve-environment"):
                await serve_environment.start(force_build=False)
            try:
                async with phase(self.trial_paths.verifier_dir / "phases", "serve-start"):
                    endpoint = await self._start_patched_service(
                        serve_environment, patch_path, metadata_path, config
                    )
                await self._verify_service_identity(endpoint, config)
                pods = self._per_pod_targets(serve_environment, config) if config.serve_launch_mode == "per-pod" else [None]
                check = (lambda: check_ray_launches(serve_environment, config.python, config.serve_node_count)) if (
                    config.serve_launch_mode != "per-pod" and config.serve_node_count > 1
                ) else (lambda: check_launches(serve_environment, pods, label="infer"))
                async with phase(self.trial_paths.verifier_dir / "phases", "benchmark-and-collection"):
                    summary = await run_with_service_watch(
                        lambda: self._run_clean_benchmark(endpoint, metadata, config), check,
                        evidence_path=self.trial_paths.verifier_dir / "service-failure.json",
                    )
                if config.serve_launch_mode == "per-pod":
                    await self._raise_if_per_pod_launch_exited(serve_environment, config)
            except DeploymentFailed as exc:
                # failed=True → 基类追加 GRM 失败归因维度:主 reward 只会记 0,
                # 分不出"agent 改崩了"与"平台抖动",而这恰是归因要回答的。
                rewards = {
                    **_deployment_failure_rewards(exc),
                    "deploy_error": str(exc)[:2000],
                }
                return VerifyOutcome(rewards, patch_path, metadata_path, failed=True)
            if config.agentic_rollout is not None:
                verdict = await run_with_service_watch(
                    lambda: self._run_agentic_rollout(endpoint, config), check,
                    evidence_path=self.trial_paths.verifier_dir / "rollout-service-failure.json",
                )
                if config.serve_launch_mode == "per-pod":
                    await self._raise_if_per_pod_launch_exited(serve_environment, config)
                rewards = _build_rollout_rewards(
                    cast(Mapping[str, object], summary["gate"]),
                    verdict,
                    baseline_throughput,
                    config.gate_center,
                    config.gate_tolerance,
                    int(cast(Any, config.agentic_rollout["min_solved"])),
                    gate_comparison=config.gate_comparison,
                )
            else:
                rewards = self._score_summary(summary, config)
            return VerifyOutcome(rewards, patch_path, metadata_path)
        finally:
            # 日志下载与停 job 必须作为一个整体被 shield:若外部取消落在
            # 下载阶段,单独 shield stop 仍会被跳过,Serve Job 泄漏。
            async def _teardown() -> None:
                async with phase(self.trial_paths.verifier_dir / "phases", "serve-teardown", 180, cleanup=True):
                    try:
                        async with asyncio.timeout(120):
                            if config.serve_launch_mode == "per-pod":
                                await self._download_per_pod_serve_logs(serve_environment, config)
                            else:
                                await self._download_serve_logs(serve_environment)
                    except TimeoutError:
                        logging.getLogger(__name__).warning("serve log collection exceeded cleanup budget")
                    try:
                        await serve_environment.stop(delete=True)
                    except Exception as exc:  # noqa: BLE001 - 收尾异常不能遮蔽主结果
                        logging.getLogger(__name__).error(
                            "停止 patched Serve Job 失败：%s", exc
                        )

            await asyncio.shield(_teardown())

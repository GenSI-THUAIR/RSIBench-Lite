"""instruction 单模版编译:上下文构造 + Jinja2 渲染。

`instruction.md = render(schemas/templates/instruction.md.tmpl, task.yaml)`。
渲染引擎使用 Jinja2，严格性配置如下:

* ``StrictUndefined``:模版里引用不存在的路径直接报错,不产出半成品题面;
* 可选展示字段(如 display.variant)在上下文侧补 ``None`` 默认;
* 过滤器只加一个 ``round2``,其余用 Jinja 内建。

题目展示信息来自 ``metadata.display/contract/resources``;
一切评测事实从 verifier_config /
environment 渲染,derived 表负责派生短语与形态布尔。
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
    autoescape=False,
)
_ENV.filters["round2"] = lambda value: f"{float(value):.2f}"

TEMPLATE_NAME = "instruction.md.tmpl"
# 生成题面统一英文。
# 展示字段使用英文，模型输入、路径与协议值保持原样。
_DISPLAY_OPTIONAL = ("variant",)
_CONTRACT_OPTIONAL = ("deploy_note", "max_context", "context_at_least", "endpoints")

# 平台卡型标签 → 题面展示名。新卡型接入时在此登记。
_ACCELERATOR_DISPLAY = {
    "H200": ("NVIDIA H200", "GPU"),
    "ASCEND910B2": ("Ascend 910B2", "NPU"),
    "NVIDIA_H20": ("NVIDIA H20", "GPU"),
}

# 引擎仓库目录名 → 题面展示名。
_ENGINE_DISPLAY = {
    "vllm": "vLLM",
    "vllm-ascend": "vLLM-Ascend",
    "sglang": "SGLang",
    "vllm-omni": "vLLM-Omni",
    "tensorrt_llm": "TensorRT-LLM",
    "SkyRL": "SkyRL",
    # verl 仓库以 verl-recipe 的 verl_tinker 网关(python -m verl_tinker.start)对外服务。
    "verl": "verl-tinker",
}

# 与 verify/ 各 domain 评测脚本的模块常量同源(常量收编前的过渡)。
_GSM8K_NUM_QUESTIONS = 200
_GSM8K_PARALLEL = 64
_MMMU_PARALLEL = 16
_GPQA_NUM_QUESTIONS = 198
# 与 verify/text/longbench_eval.py 同源:冻结子集每档 100 题,并发/预算缺省值。
_LONGBENCH_NUM_QUESTIONS = 100
_LONGBENCH_PARALLEL = 4
_LONGBENCH_MAX_TOKENS = 4096
# 服务端口不再是固定数字:环境层把 head pod 平台分配的 PORT0 注入为该变量
# (HARBOR_SERVICE_PORT),题面只约定变量名。
_SERVICE_PORT_ENV = "HARBOR_SERVICE_PORT"


def render_from_config(cfg: dict[str, Any]) -> str | None:
    """按共用模版渲染题面;无 metadata.display 的 task(未迁移/冒烟题)返回 None。

    ``instruction.template: verbatim`` 只保留给 tasks/ 之外的历史副本
    (legacy/、docs/evidence/)做逐字节内联;tasks/ 下的推理题与训练题都从
    config 渲染(test_schemas 断言 tasks/ 里没有 verbatim 题面)。
    """
    if "prose" in cfg:
        raise ValueError("prose is no longer supported; remove the top-level prose field")
    ins = cfg.get("instruction")
    if isinstance(ins, dict) and ins.get("template") == "verbatim":
        body = ins.get("body")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("instruction.template=verbatim 需要非空的 body 字符串")
        if not body.endswith("\n"):
            raise ValueError("instruction.body 必须以换行结尾(题面文件的约定)")
        return body
    if not cfg.get("metadata", {}).get("display"):
        return None
    context = build_context(cfg)
    return _ENV.get_template(TEMPLATE_NAME).render(**context)


def build_context(cfg: dict[str, Any]) -> dict[str, Any]:
    vc = cfg["verifier"]["scoring"]
    env = cfg["environment"]
    # 拓扑与卡型来自平台无关的 environment 声明(每节点口径),不再读 roles。
    nodes = int(env.get("nodes", 1))
    gpu = int(env["gpus"])
    gpuv = env["gpu_types"][0]
    accelerator, unit = _ACCELERATOR_DISPLAY[gpuv]
    engine = _ENGINE_DISPLAY[Path(vc["engine_repo"]).name]
    ray = nodes > 1

    display = {
        **{key: None for key in _DISPLAY_OPTIONAL},
        **cfg["metadata"]["display"],
    }
    contract = {
        **{key: None for key in _CONTRACT_OPTIONAL},
        **cfg["metadata"].get("contract", {}),
    }
    metadata = {
        **cfg["metadata"],
        "display": display,
        "contract": contract,
        "resources": cfg["metadata"].get("resources", []),
    }

    gate = vc["gate"]
    gate_kind = gate["kind"]
    has_rollout = vc.get("agentic_rollout") is not None
    has_video = vc.get("video_benchmark") is not None
    bench = vc.get("bench_serving") or {}
    perf_is_window = "window_duration_sec" in bench
    arrival_mode = bench.get("arrival_mode", "trace")
    if not isinstance(arrival_mode, str) or arrival_mode not in {"trace", "asap"}:
        raise ValueError("bench_serving.arrival_mode must be trace or asap")
    if "arrival_mode" in bench and bench.get("dataset_name") != "mooncake":
        raise ValueError("bench_serving.arrival_mode is only valid for mooncake")
    reserved = {"--mooncake-arrival-mode"}
    if "arrival_mode" in bench:
        reserved.update({"--request-rate", "--use-trace-timestamps"})
    if any(str(arg).partition("=")[0] in reserved for arg in bench.get("extra_args", [])):
        raise ValueError("arrival scheduling must be set by bench_serving.arrival_mode, not extra_args")
    bench_extra = _pairs(bench.get("extra_args", []))
    perf_is_replay = (
        not has_rollout and not has_video
        and bench.get("dataset_name") == "mooncake"
    )

    derived: dict[str, Any] = {
        # ---- 身份与拓扑 ----
        "engine": engine,
        "accelerator": accelerator,
        "title": f"Optimize {engine} inference performance for {display['model']}",
        "tp": nodes * gpu,
        "env_phrase": _env_phrase(nodes, gpu, accelerator, unit),
        "node_phrase": _node_phrase(nodes, gpu, accelerator, ray),
        "location_phrase": "the head node" if nodes > 1 else "the node",
        "debug_phrase": _debug_phrase(nodes, gpu, unit),
        "apply_phrase": f"{nodes} clean nodes" if nodes > 1 else "a clean node",
        # OCI layer 内的 launch.sh 条目路径(相对 rootfs 根)与题目追加的禁止路径。
        "layer_launch_member": vc["engine_repo"].lstrip("/") + "/launch.sh",
        "forbidden_extra": "".join(f", `{p}`" for p in vc.get("forbidden_paths", [])),
        "deploy_target": _deploy_target(nodes, gpu),
        "deploy_scale": _deploy_scale(nodes, gpu, accelerator),
        "serve_phrase": f"{nodes} nodes" if nodes > 1 else "a single node",
        "exec_clause": _exec_clause(nodes),
        # profiling 授权节的按卡型工具链(六题通用节,硬件差异在此收敛)。
        "debug_scope_phrase": (
            f"the entire {nodes}-node cluster"
            if nodes > 1
            else (f"all {gpu} accelerators" if gpu > 1 else "the accelerator")
        ),
        "profiler_tools": (
            "torch_npu profiler, msprof, `npu-smi`, and the CANN toolchain"
            if gpuv.startswith("ASCEND")
            else "torch profiler, Nsight Systems, and `nvidia-smi`"
        ),
        "kernel_stack": (
            "Triton-on-Ascend / AscendC / custom torch_npu operators"
            if gpuv.startswith("ASCEND")
            else "Triton / custom CUDA operators"
        ),
        "no_restart_clause": (
            "Do not restart or reschedule the service cluster."
            if ray
            else "Do not restart or reschedule the sandbox itself."
        ),
        "serve_apply_para": _serve_apply_para(nodes),
        # ---- 服务契约 ----
        "port_var": _SERVICE_PORT_ENV,
        "endpoint_local": f"http://127.0.0.1:${{{_SERVICE_PORT_ENV}}}",
        "endpoints_inline": _endpoints_inline(contract, vc),
        "context_lead": "at least" if contract.get("context_at_least") else "exactly",
        # ---- gate ----
        "gate_name": _gate_name(gate_kind),
        "gate_name_sp": _gate_name(gate_kind),
        "gate_data_label": _gate_data_label(gate_kind),
        "gate_band": _gate_band(gate),
        "gate_stop_tokens_json": json.dumps(gate["stop_tokens"], ensure_ascii=False),
        # 可信 gate 自测脚本(gsm8k / gpqa 共用渲染);其余 gate 形态不渲染自测命令。
        "gsm8k_script_path": None,
        "gate_script_path": None,
        "gsm8k_num_questions": _GSM8K_NUM_QUESTIONS,
        "gsm8k_parallel": _GSM8K_PARALLEL,
        "mmmu_parallel": _MMMU_PARALLEL,
        "gpqa_num_questions": _GPQA_NUM_QUESTIONS,
        "longbench_num_questions": gate.get("num_questions", _LONGBENCH_NUM_QUESTIONS),
        "longbench_parallel": gate.get("max_concurrency", _LONGBENCH_PARALLEL),
        "longbench_max_tokens": gate.get("max_tokens", _LONGBENCH_MAX_TOKENS),
        # ---- 评测形态(互斥) ----
        "has_rollout": has_rollout,
        "perf_is_video": has_video,
        "perf_is_replay": perf_is_replay,
        "perf_is_window": perf_is_window,
        "bench_arrival_mode": arrival_mode,
        "perf_is_bench": not has_rollout
        and not has_video
        and not perf_is_replay,
        # ---- 资产寻址 ----
        "mounts": {
            Path(mount).name: mount
            for mount in env.get("mounts", [])
        },
    }

    # 题面/自测只透传显式gate选项；旧题保持原预算、JSON和HDFS脚本。
    derived["gate_selftest_extra_json"] = "".join(
        f", {json.dumps(key)}: {json.dumps(gate[key], ensure_ascii=False)}"
        for key in ("max_tokens", "strict_final_answer", "reasoning_effort")
        if key in gate
    )
    if gate_kind.startswith("gsm8k"):
        derived["gsm8k_chat_max_tokens"] = gate.get("max_tokens", 1024)
        if "agent_script_path" in gate:
            path = gate["agent_script_path"]
            if not isinstance(path, str) or not path.startswith("/"):
                raise ValueError("gate.agent_script_path 必须是非空绝对路径")
            derived["gsm8k_script_path"] = path
        elif has_rollout or perf_is_replay:
            derived["gsm8k_script_path"] = f"{derived['mounts']['rsibench']}/gsm8k_eval.py"
        derived["gate_script_path"] = derived["gsm8k_script_path"]
    elif gate_kind.startswith("gpqa") and (has_rollout or perf_is_replay):
        # gpqa 与 gsm8k 共用自测渲染:脚本与数据同为可信副本,口径从 gate 配置出。
        derived["gate_script_path"] = f"{derived['mounts']['rsibench']}/gpqa_eval.py"

    derived["quick_check"] = None
    if gate_kind == "longbench-v2-openai-chat":
        # 自测脚本与评测方可信副本同源(HDFS rsibench/longbench_eval.py);
        # 题面只透传显式 gate 选项,缺省沿用脚本常量。
        derived["longbench_script_path"] = f"{derived['mounts']['rsibench']}/longbench_eval.py"
        derived["gate_selftest_extra_json"] = "".join(
            f", {json.dumps(key)}: {json.dumps(gate[key], ensure_ascii=False)}"
            for key in ("max_tokens", "max_concurrency", "reasoning_effort")
            if key in gate
        )
        # 原 GSM8K/GPQA 门保留为参考门(gate_*_reference,不参与判定):题面把它渲染成
        # 分钟级正确性快检,脚本/数据与评测方可信副本同源;LongBench 只作交付前最终确认。
        reference = vc.get("gate_gsm8k_reference") or vc.get("gate_gpqa_reference")
        if reference:
            derived["quick_check"] = _quick_check(reference, derived["mounts"])

    if bench:
        agent_script = bench.get("agent_script_path")
        if agent_script is not None and (
            not isinstance(agent_script, str) or not agent_script.startswith("/")
        ):
            raise ValueError("bench_serving.agent_script_path 必须是非空绝对路径")
        pending = bench.get("agent_script_pending", False)
        if not isinstance(pending, bool):
            raise TypeError("bench_serving.agent_script_pending 必须是布尔值")
        derived["agent_script_pending"] = pending
        derived["agent_script_path"] = agent_script
        if agent_script is None and (perf_is_replay or has_rollout):
            derived["agent_script_path"] = (
                f"{derived['mounts']['rsibench']}/vendored_bench_serving.py"
            )
        if agent_script is not None and not perf_is_window:
            derived["bench_selftest_command"] = _agent_benchmark_command(
                vc, bench, derived["endpoint_local"]
            )
        derived["bench_context_length"] = bench_extra.get("--mooncake-context-length")
        derived["bench_prompt_protocol"] = (
            "token-ids-v1" if derived["bench_context_length"] is not None
            else "legacy-text-v0"
        )
        derived["bench_dataset_path"] = bench_extra.get("--dataset-path")
        derived["bench_dataset_file"] = (
            Path(bench_extra["--dataset-path"]).name
            if "--dataset-path" in bench_extra
            else None
        )
        derived["bench_slowdown"] = bench_extra.get("--mooncake-slowdown-factor")
        derived["bench_is_text"] = bench.get("dataset_name") != "image"
        derived["dataset_display"] = {
            "random-ids": "`random-ids` (no external dataset download required)",
            "random": "`random`",
            "image": "**multimodal** `image` (randomly synthesized images)",
            "mooncake": "mooncake replay",
        }[bench["dataset_name"]]
        if bench.get("dataset_name") == "image":
            derived["image"] = {
                "count": bench_extra["--image-count"],
                "resolution": bench_extra["--image-resolution"],
                "format": bench_extra["--image-format"].upper(),
            }
    if has_rollout:
        subset = list(vc["agentic_rollout"]["subset"])
        derived["rollout"] = {
            "n_tasks": len(subset),
            "subset_inline": ", ".join(subset),
        }
    if has_video:
        video = vc["video_benchmark"]
        request = video["request"]
        quality = video["quality"]
        if video.get("mode", "audio-video") == "video-only":
            derived["video"] = {
                "video_only": True,
                "case_count": len(video["case_ids"]),
                "video_codec": _codec(quality["expected_video_codec"]),
            }
        else:
            derived["video"] = {
                "task_display": str(request["extra_params"]["task"]).upper(),
                "audio_khz": int(quality["expected_audio_sample_rate"]) // 1000,
                "video_codec": _codec(quality["expected_video_codec"]),
                "audio_codec": _codec(quality["expected_audio_codec"]),
                "audio_flow_shift": _trim_float(request["extra_params"]["audio_flow_shift"]),
            }


    return {**cfg, "metadata": metadata, "derived": derived}


def _agent_benchmark_command(vc: dict[str, Any], bench: dict[str, Any], endpoint: str) -> str:
    """从评分argv生成Agent自测命令；仅脚本路径/声明环境不同，完整保留extra_args。"""
    script_env = bench.get("agent_script_env", {})
    if not isinstance(script_env, dict) or not all(
        isinstance(key, str) and key and "=" not in key
        and isinstance(value, str)
        for key, value in script_env.items()
    ):
        raise TypeError("bench_serving.agent_script_env 必须是字符串环境变量表")
    environment = {"NO_PROXY": "*", "no_proxy": "*", "OMP_NUM_THREADS": "1", **script_env}
    command = [
        "env", *[f"{key}={value}" for key, value in environment.items()],
        str(vc["python"]), str(bench["agent_script_path"]),
        "--backend", str(bench["backend"]), "--base-url", endpoint,
        "--model", str(vc["model_path"]),
    ]
    if bench["backend"] != "sglang":
        command.extend(("--served-model-name", str(vc["served_model_name"])))
    command.extend(("--tokenizer", str(vc["model_path"])))
    for flag, key in (
        ("--dataset-name", "dataset_name"),
        ("--random-input-len", "random_input_len"),
        ("--random-output-len", "random_output_len"),
        ("--random-range-ratio", "random_range_ratio"),
        ("--max-concurrency", "max_concurrency"),
        ("--num-prompts", "num_prompts"),
        ("--seed", "seed"),
    ):
        command.extend((flag, str(bench[key])))
    command.extend(("--request-rate", "inf"))
    if "arrival_mode" in bench:
        command.extend(("--mooncake-arrival-mode", bench["arrival_mode"]))
    command.extend(str(arg) for arg in bench["extra_args"])
    # 服务端口只在pod内解析；其余参数全部shell转义，避免路径/参数含空格时分裂。
    return " ".join(
        f'"{value}"' if value == endpoint else shlex.quote(value)
        for value in command
    )


# ---------------------------------------------------------------------------
# 派生短语(与 verifier 的分派维度同构:拓扑 / ray / gate.kind / 评测形态)
# ---------------------------------------------------------------------------


def _env_phrase(nodes: int, gpu: int, accelerator: str, unit: str) -> str:
    if nodes > 1:
        return f"{nodes}-node environment with {nodes * gpu} {accelerator} {unit}s in total"
    if gpu > 1:
        return f"single-node environment with {gpu} {accelerator} {unit}s"
    return f"single-node environment with one {accelerator} {unit}"


def _node_phrase(nodes: int, gpu: int, accelerator: str, ray: bool) -> str:
    if nodes > 1:
        return (
            f"The cluster consists of 1 head and {nodes - 1} worker node(s), "
            f"with {gpu} {accelerator} accelerators per node."
        )
    tail = "" if ray else " Deployment is local to this node, without Ray orchestration."
    if gpu > 1:
        return f"The single node has {gpu} {accelerator} accelerators.{tail}"
    return f"The single node has one {accelerator} accelerator.{tail}"


def _debug_phrase(nodes: int, gpu: int, unit: str) -> str:
    if nodes > 1:
        return "the full cluster"
    return f"all {gpu} {unit}s" if gpu > 1 else f"the {unit}"


def _deploy_target(nodes: int, gpu: int) -> str:
    if nodes > 1:
        return f"a {nodes}-node cluster"
    return f"a single node with {gpu} accelerators" if gpu > 1 else "a single accelerator"


def _deploy_scale(nodes: int, gpu: int, accelerator: str) -> str:
    if nodes > 1:
        return f"all {nodes} nodes ({gpu} accelerators per node)"
    return f"all {gpu} {accelerator} accelerators" if gpu > 1 else f"one {accelerator} accelerator"


def _exec_clause(nodes: int) -> str:
    if nodes > 1:
        workers = ", ".join(f"worker-{i}" for i in range(nodes - 1))
        return (
            f"The same script must run on head and {workers}, determining "
            "`node-rank` automatically from `MY_POD_NAME`."
        )
    return "Run on a single node without relying on orchestration information other than `MY_POD_NAME`."


def _serve_apply_para(nodes: int) -> str:
    if nodes > 1:
        return (
            f"The evaluator applies the same OCI layer to all {nodes} nodes of a fresh "
            "Serve Job, then executes this `launch.sh` on each node. Deployment fails "
            "if any node has a baseline mismatch, the layer cannot be applied, "
            "the script is missing, or the service fails to start."
        )
    return (
        "The evaluator applies the OCI layer to a fresh single-node Serve Job, then "
        "executes this `launch.sh`. Deployment fails if the baseline does not match, "
        "the layer cannot be applied, the script is missing, or the service fails to start."
    )


def _endpoints_inline(contract: dict[str, Any], vc: dict[str, Any]) -> str:
    endpoints = contract.get("endpoints")
    if endpoints:
        return ", ".join(str(endpoint) for endpoint in endpoints)
    video = vc.get("video_benchmark")
    if video:
        return f"`GET /health` and `POST {video['endpoint_path']}`"
    raise ValueError("metadata.contract.endpoints 缺失且无法从 video_benchmark 派生")


def _quick_check(reference: dict[str, Any], mounts: dict[str, str]) -> dict[str, Any]:
    """gate_gsm8k_reference / gate_gpqa_reference → 题面快检段的事实(全部来自参考门配置)。"""
    kind = str(reference["kind"])
    if kind.startswith("gsm8k"):
        script = reference.get("agent_script_path") or f"{mounts['rsibench']}/gsm8k_eval.py"
        if kind == "gsm8k-openai-chat":
            protocol = (f"0-shot chat, first {_GSM8K_NUM_QUESTIONS} questions, concurrency {_GSM8K_PARALLEL}, "
                        f"temperature 0, `max_tokens={reference.get('max_tokens', 1024)}`")
        else:
            protocol = f"5-shot generate, first {_GSM8K_NUM_QUESTIONS} questions, concurrency {_GSM8K_PARALLEL}, temperature 0"
    elif kind.startswith("gpqa"):
        script = f"{mounts['rsibench']}/gpqa_eval.py"
        protocol = f"{_GPQA_NUM_QUESTIONS} questions from official simple-evals, frozen option order"
    else:
        raise ValueError(f"参考门 kind {kind!r} 不支持渲染为快检")
    if not isinstance(script, str) or not script.startswith("/"):
        raise ValueError("参考门 agent_script_path 必须是非空绝对路径")
    return {
        "kind": kind,
        "name": _gate_name(kind),
        "data_label": _gate_data_label(kind),
        "data": reference["data"],
        "script_path": script,
        "protocol": protocol,
        "stop_tokens_json": json.dumps(reference.get("stop_tokens", []), ensure_ascii=False),
        "extra_json": "".join(
            f", {json.dumps(key)}: {json.dumps(reference[key], ensure_ascii=False)}"
            for key in ("max_tokens", "strict_final_answer", "reasoning_effort")
            if key in reference
        ),
        "band": _gate_band(reference),
    }


def _gate_band(gate: dict[str, Any]) -> str:
    comparison = gate.get("comparison", "two_sided")
    if not isinstance(comparison, str) or comparison not in ("two_sided", "lower_bound"):
        raise ValueError("gate.comparison 必须是 two_sided 或 lower_bound")
    if comparison == "lower_bound":
        lower = float(gate["center"]) - float(gate["tolerance"])
        return (
            f"`accuracy ≥ {lower:.12g}` (inclusive lower bound; "
            f"center={gate['center']}, tolerance={gate['tolerance']}, no upper bound)"
        )
    return f"`{gate['center']} ± {gate['tolerance']}`"


def _gate_name(kind: str) -> str:
    return {
        "gsm8k-openai-chat": "GSM8K",
        "gsm8k-sglang-generate": "GSM8K",
        "mmmu-openai-chat": "MMMU",
        "gpqa-diamond-openai-chat": "GPQA",
        "longbench-v2-openai-chat": "LongBench v2",
        "video-reference": "reference-video",
    }[kind]


def _gate_data_label(kind: str) -> str:
    return {
        "gsm8k-openai-chat": "GSM8K data",
        "gsm8k-sglang-generate": "GSM8K data",
        "mmmu-openai-chat": "MMMU data",
        "gpqa-diamond-openai-chat": "GPQA diamond data",
        "longbench-v2-openai-chat": "LongBench v2 data",
        "video-reference": "BF16 reference video",
    }[kind]


def _pairs(args: list[Any]) -> dict[str, str]:
    return {str(args[i]): str(args[i + 1]) for i in range(0, len(args) - 1, 2)}


def _trim_float(value: Any) -> str:
    text = repr(value)
    return text[:-2] if text.endswith(".0") else text


def _codec(codec: str) -> str:
    return {"h264": "H.264"}.get(codec, codec.upper())

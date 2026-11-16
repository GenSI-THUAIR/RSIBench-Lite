"""按测量条件判断推理基线是否适用；源码指纹仅用于溯源。

参数：运行配置与已解析的评分配置。示例：baseline_conditions(runtime, scoring)。
修改测量语义时应更新协议版本或相应配置，不以仓库文件字节代替协议身份。
"""
from copy import deepcopy


def baseline_conditions(runtime: dict, scoring: dict) -> dict:
    """只记录硬件、输入、基础镜像及实际 gate/性能参数，排除代码路径和说明文字。"""
    benchmark_key = 'video_benchmark' if scoring['gate']['kind'] == 'video-reference' else 'bench_serving'
    benchmark = {key: value for key, value in scoring[benchmark_key].items()
                 if key not in {"script_path", "agent_script_path", "agent_script_pending",
                                "bench_timeout_sec"}}
    gate = {key: value for key, value in scoring["gate"].items() if key != "source"}
    resources = {key: runtime[key] for key in (
        "backend", "gpu", "gpus", "serve_cpus", "serve_memory_mb",
        "benchmark_cpus", "benchmark_memory_mb")}
    # 历史 v1 基线的 Benchmark 均为 CPU；新增 GPU 分配需进入测量条件。
    if runtime.get('benchmark_gpus', 0):
        resources.update(benchmark_gpus=runtime['benchmark_gpus'],
                         benchmark_gpu_types=runtime['benchmark_gpu_types'])
    return deepcopy({
        "protocol": "inference-baseline/1",
        "resources": resources,
        "image": runtime["image"],
        "inputs": sorted([
            {key: group[key] for key in ("kind", "manifest_sha256")}
            for group in runtime["inputs"]
        ], key=lambda group: group["kind"]),
        "driver_env": runtime.get("driver_env", {}),
        "gate": gate,
        benchmark_key: benchmark,
    })


def require_matching_conditions(baseline: dict, conditions: dict) -> None:
    """缺失条件或条件变化时拒绝复用，避免把旧基线静默认作当前配置。"""
    if baseline.get("baseline_conditions") != conditions:
        raise ValueError("Baseline measurement conditions differ; calibrate for these conditions")


def check_baseline_policy(baseline: dict, conditions: dict, policy: str = 'strict') -> None:
    """显式允许复用资源条件不同的旧分母；协议、镜像和资产仍必须一致。"""
    if policy == 'strict':
        require_matching_conditions(baseline, conditions)
    elif policy == 'reuse_resources':
        recorded = baseline.get('baseline_conditions')
        if not isinstance(recorded, dict) or not isinstance(recorded.get('resources'), dict):
            raise ValueError('Baseline has no recorded resource conditions')
        if recorded['resources']['backend'] != conditions['resources']['backend']:
            raise ValueError('Baseline backend differs; resource reuse cannot cross platforms')
        comparable = {**conditions, 'resources': recorded['resources']}
        require_matching_conditions(baseline, comparable)
    else:
        raise ValueError(f'Unknown baseline_policy: {policy}')

"""验证基线随测量条件变化失效，源码、说明和工具路径变化不触发重新标定。

参数：无。示例：python3 -m pytest tests/test_baseline_conditions.py。
使用真实两题配置及生成器；不申请远端资源、不改写任务包。
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from schemas import run_config as builder
import yaml
from verify.inference.baseline import baseline_conditions, require_matching_conditions, check_baseline_policy


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/infer-qwen3p8-27b"


def generated():
    job=yaml.safe_load((ROOT/'configs/harbor-modal.yaml').read_text())
    # 这些测试验证历史标定本身的严格匹配，显式重建当时的资源条件。
    recorded=json.loads((TASK/'runtimes/modal-baseline.json').read_text())['performance_baseline']['baseline_conditions']['resources']
    for phase in ('agent','benchmark','serve'):
        target=job['runtime']['resources'][phase]
        prefix='serve' if phase=='serve' else 'benchmark'
        target.update(cpus=recorded[prefix+'_cpus'],memory_mb=recorded[prefix+'_memory_mb'],
                      gpus=recorded['gpus'] if phase=='serve' else 0,
                      gpu_types=[recorded['gpu']] if phase=='serve' else [])
        if phase == 'benchmark':
            target.update(gpus=recorded.get('benchmark_gpus', 0),
                          gpu_types=recorded.get('benchmark_gpu_types', []))
    return builder.modal_files(TASK,builder.load_yaml(TASK/'task.yaml'),job['runtime']['resources'],'score')


def configuration():
    files=generated()
    runtime=json.loads(files[Path('tests/runtime.json')])
    scoring=json.loads(files[Path('tests/verifier-config.json')])
    scoring.update(json.loads(files[Path('tests/modal-scoring.json')]))
    return runtime, scoring


@pytest.mark.parametrize("field", [
    "gpu", "gpus", "serve_cpus", "serve_memory_mb", "benchmark_cpus",
    "benchmark_memory_mb", "image",
])
def test_changed_runtime_conditions_reject_baseline(field):
    runtime, scoring = configuration()
    runtime[field] = "changed"
    with pytest.raises(ValueError, match="measurement conditions"):
        require_matching_conditions(scoring["performance_baseline"],
                                    baseline_conditions(runtime, scoring))


@pytest.mark.parametrize("group,field", [
    ("bench_serving", "max_concurrency"), ("bench_serving", "window_duration_sec"),
    ("bench_serving", "trace_sha256"), ("bench_serving", "seed"),
    ("gate", "max_tokens"), ("gate", "center"),
])
def test_changed_workload_rejects_baseline(group, field):
    runtime, scoring = configuration()
    scoring[group][field] = "changed"
    with pytest.raises(ValueError, match="measurement conditions"):
        require_matching_conditions(scoring["performance_baseline"],
                                    baseline_conditions(runtime, scoring))


@pytest.mark.parametrize("kind", ["model", "data"])
def test_changed_input_content_rejects_baseline(kind):
    runtime, scoring = configuration()
    next(group for group in runtime["inputs"] if group["kind"] == kind)["manifest_sha256"] = "changed"
    with pytest.raises(ValueError, match="measurement conditions"):
        require_matching_conditions(scoring["performance_baseline"],
                                    baseline_conditions(runtime, scoring))


def test_code_and_documentation_are_provenance_only():
    runtime, scoring = configuration()
    for key in ("source_task_sha256", "source_launch_sha256", "evaluator_sha256",
                "bootstrap_sha256", "prepare_sha256", "dockerfile_sha256"):
        runtime[key] = "new-code"
    scoring["gate"]["source"] = "revised explanation"
    scoring["bench_serving"]["script_path"] = "/new/location/benchmark.py"
    scoring["bench_serving"]["agent_script_path"] = "/new/location/agent-benchmark.py"
    require_matching_conditions(scoring["performance_baseline"],
                                baseline_conditions(runtime, scoring))


def test_missing_conditions_are_not_silently_accepted():
    runtime, scoring = configuration()
    with pytest.raises(ValueError, match="measurement conditions"):
        require_matching_conditions({}, baseline_conditions(runtime, scoring))


def test_generator_accepts_new_source_and_preserves_measured_baselines(monkeypatch):
    old_runtime, scoring = configuration()
    read_bytes = Path.read_bytes
    launch = ROOT / "tasks/infer-qwen3p8-27b/environment/launch.sh"

    def revised_source(path):
        data = read_bytes(path)
        return data + b"\n# source revision\n" if path == launch else data

    monkeypatch.setattr(Path, "read_bytes", revised_source)
    monkeypatch.setattr(builder, "evaluator_sha256", lambda: "new-evaluator")
    files = generated()
    runtime = json.loads(files[Path("tests/runtime.json")])
    assert runtime["sha256"] != old_runtime["sha256"]
    assert runtime["evaluator_sha256"] == "new-evaluator"
    assert runtime["source_launch_sha256"] != old_runtime["source_launch_sha256"]
    assert json.loads(files[Path("tests/modal-scoring.json")])["performance_baseline"] == scoring["performance_baseline"]


def test_generator_rejects_changed_workload(monkeypatch):
    load_yaml = builder.load_yaml

    def changed_workload(path):
        config = deepcopy(load_yaml(path))
        if path == ROOT / "tasks/infer-qwen3p8-27b/task.yaml":
            config["verifier"]["scoring"]["bench_serving"]["max_concurrency"] = 99
        return config

    monkeypatch.setattr(builder, "load_yaml", changed_workload)
    with pytest.raises(ValueError, match="measurement conditions"):
        generated()


def test_explicit_resource_reuse_preserves_old_baseline_and_checks_workload():
    runtime, scoring = configuration()
    original=deepcopy(scoring['performance_baseline'])
    runtime['serve_cpus'] += 1
    current=baseline_conditions(runtime,scoring)
    check_baseline_policy(original,current,'reuse_resources')
    assert scoring['performance_baseline']==original
    with pytest.raises(ValueError,match='measurement conditions'):
        check_baseline_policy(original,current,'strict')
    for key,value in [('image','another-image'),('gate',{}),('inputs',[])]:
        changed={**current,key:value}
        with pytest.raises(ValueError,match='measurement conditions'):
            check_baseline_policy(original,changed,'reuse_resources')
    with pytest.raises(ValueError,match='Unknown baseline_policy'):
        check_baseline_policy(original,current,'ignore-everything')
    changed=deepcopy(current)
    changed['resources']['backend']='merlin'
    with pytest.raises(ValueError,match='cannot cross platforms'):
        check_baseline_policy(original,changed,'reuse_resources')


def test_benchmark_gpu_allocation_is_part_of_measurement_conditions():
    runtime,scoring=configuration()
    before=baseline_conditions(runtime,scoring)
    runtime.update(benchmark_gpus=0 if runtime.get('benchmark_gpus') else 1,
                   benchmark_gpu_types=[] if runtime.get('benchmark_gpus') else ['H200'])
    assert baseline_conditions(runtime,scoring)!=before

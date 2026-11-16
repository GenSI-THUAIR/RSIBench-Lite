"""验证五题选集、真实任务编译及文本/视频标定边界，不申请云资源。"""
from copy import deepcopy
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor_envs.abstract_mlplat.mlplatform_base import MLPlatformEnvironment
from harbor_envs.modal.asset_verify import validate_manifest
from schemas.run_config import prepare_run
from scripts.calibration.record_harbor_baseline import validate_full_workload
from scripts.prepare.prepare_modal_data import checked_buffer
from verify.inference.baseline import baseline_conditions
from verify.inference.runtime_infer_verifier import RuntimeInferVerifier

ROOT = Path(__file__).resolve().parents[1]
TASKS = ['infer-qwen3p8-27b', 'infer-museglimmer-30b', 'infer-qwen3vl-32b',
         'infer-llada2p1mini-16b', 'infer-wan2p2ti2v-5b']


@pytest.mark.parametrize('name', ['modal', 'modal-smoke', 'modal-cpu', 'modal-calibrate'])
def test_five_task_selection(name):
    config = yaml.safe_load((ROOT / f'configs/harbor-{name}.yaml').read_text())
    assert [Path(item['path']).name for item in config['tasks']] == TASKS


@pytest.mark.parametrize('name', ['modal-smoke', 'modal-cpu', 'modal-calibrate'])
def test_all_five_compile_with_original_protocols(tmp_path, name):
    config = yaml.safe_load((ROOT / f'configs/harbor-{name}.yaml').read_text())
    compiled = prepare_run(config, directory=tmp_path / name)
    assert len(compiled['tasks']) == 5
    for index, entry in enumerate(compiled['tasks']):
        task = Task(entry['path'])
        scoring = json.loads((task.paths.tests_dir / 'verifier-config.json').read_text())
        assert scoring['serve_gpus_per_node'] == 1
        if name.startswith('modal'):
            runtime = json.loads((task.paths.tests_dir / 'runtime.json').read_text())
            assert runtime['gpu'] == 'H200' and runtime['gpus'] == 1
            for group in runtime['inputs']:
                validate_manifest({'files': group['files'],
                                   'total_bytes': sum(item['bytes'] for item in group['files'])})
            verifier = object.__new__(RuntimeInferVerifier)
            verifier.task = task
            verifier.mode = 'smoke' if name == 'modal-smoke' else 'calibrate'
            verifier.environment = Mock(spec=MLPlatformEnvironment)
            verifier.environment.platform_name = 'modal'
            verifier.trial_paths = TrialPaths(trial_dir=tmp_path / f'trial-{index}')
            resolved = verifier._resolve_config()
            if index >= 2:
                source = ROOT / 'tasks' / TASKS[index]
                from schemas.build import load_yaml
                baseline_path = source / load_yaml(source/'task.yaml')['platforms']['modal']['baseline']
                if baseline_path.exists() and name != 'modal-cpu':
                    registered = json.loads(baseline_path.read_text())['performance_baseline']
                    assert resolved['performance_baseline'] == registered
                else:
                    assert resolved['performance_baseline']['status'] == 'UNCALIBRATED'
            if index in (2, 3):
                assert 'window_duration_sec' not in resolved['bench_serving']
            if index == 4:
                assert 'bench_serving' not in resolved
                assert resolved['performance_baseline']['unit'] == 'videos/hour'
                assert len(resolved['video_benchmark']['case_ids']) == 4
                conditions = baseline_conditions(runtime, resolved)
                changed = deepcopy(resolved)
                changed['video_benchmark']['request']['num_frames'] = 1
                assert conditions != baseline_conditions(runtime, changed)


@pytest.mark.parametrize('task', TASKS[2:])
def test_no_h20_baseline_can_be_used_for_h200_score(tmp_path, task):
    source = tmp_path / task
    shutil.copytree(ROOT / 'tasks' / task, source, symlinks=True)
    from schemas.build import load_yaml
    (source / load_yaml(source/'task.yaml')['platforms']['modal']['baseline']).unlink(missing_ok=True)
    config = yaml.safe_load((ROOT / 'configs/harbor-modal.yaml').read_text())
    config['tasks'] = [{'path': str(source)}]
    with pytest.raises(ValueError, match='No calibrated Modal baseline'):
        prepare_run(config, directory=tmp_path / 'compiled')


def test_video_smoke_uses_complete_reference_workset():
    verifier = object.__new__(RuntimeInferVerifier)
    verifier.mode = 'smoke'
    summary = {'gate': {'accuracy': 1.0, 'passed': True, 'cases': [{}] * 4},
               'performance': {'output_throughput': 8.0, 'request_success_rate': 1.0,
                               'completed': 4, 'wall_time_sec': 1800}}
    config = SimpleNamespace(gate_kind='video-reference')
    assert verifier._score_summary(summary, config)['smoke_complete'] == 1
    summary['performance']['completed'] = 2
    assert verifier._score_summary(summary, config)['smoke_complete'] == 0
    summary['performance']['completed'] = 4
    summary['gate']['passed'] = False
    assert verifier._score_summary(summary, config)['smoke_complete'] == 0


@pytest.mark.parametrize('questions,window,requests', [(100, 3600, None), (847, None, 32)])
def test_calibration_requires_task_specific_full_workload(questions, window, requests):
    scoring = {'gate': {'kind': 'longbench-v2-openai-chat', 'data': '/gate.jsonl',
                        'num_questions': questions},
               'bench_serving': {'dataset_name':'image', 'window_full_duration': bool(window),
                                 'window_duration_sec': window, 'num_prompts': requests}}
    inputs = {'files': [{'path': '/gate.jsonl', 'records': questions}]}
    summary = {'gate': {'num_questions': questions},
               'performance': {'wall_time_sec': window, 'completed': requests}}
    validate_full_workload(summary, scoring, inputs)
    summary['gate']['num_questions'] = 3
    with pytest.raises(ValueError, match='full correctness workload'):
        validate_full_workload(summary, scoring, inputs)
    summary['gate']['num_questions'] = questions
    summary['performance'].update(wall_time_sec=120, completed=2)
    with pytest.raises(ValueError, match='full .*workload'):
        validate_full_workload(summary, scoring, inputs)


def test_full_mooncake_replay_uses_frozen_trace_length_not_request_cap():
    scoring = {'gate': {'kind':'longbench-v2-openai-chat', 'data':'/gate.jsonl', 'num_questions':36},
               'bench_serving': {'dataset_name':'mooncake', 'num_prompts':10000,
                                 'extra_args':['--dataset-path','/trace.jsonl']}}
    inputs = {'files':[{'path':'/gate.jsonl','records':36}, {'path':'/trace.jsonl','records':140}]}
    summary = {'gate':{'num_questions':36}, 'performance':{'completed':140}}
    validate_full_workload(summary, scoring, inputs)
    summary['performance']['completed']=2
    with pytest.raises(ValueError, match='full request workload'):
        validate_full_workload(summary, scoring, inputs)


def test_video_calibration_rejects_missing_cases_and_wrong_timing():
    scoring = {'gate': {'kind': 'video-reference'}, 'video_benchmark': {'case_ids': ['a', 'b']}}
    summary = {'gate': {'passed': True, 'cases': [{'id': 'a', 'passed': True}, {'id': 'b', 'passed': True}]},
               'performance': {'completed': 2, 'num_prompts': 2,
                               'output_throughput': 12.0, 'wall_time_sec': 600}}
    validate_full_workload(summary, scoring, {})
    summary['performance']['output_throughput'] = 24
    with pytest.raises(ValueError, match='timing'):
        validate_full_workload(summary, scoring, {})
    summary['gate']['cases'].pop()
    with pytest.raises(ValueError, match='full video workload'):
        validate_full_workload(summary, scoring, {})


def test_data_publishing_checks_size_and_hash_before_upload():
    import hashlib
    data = b'original frozen input'
    item = {'path': 'gate.jsonl', 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    with checked_buffer(io.BytesIO(data), item) as buffer:
        assert buffer.read() == data
    for changed in (data + b'x', data[:-1], b'x' * len(data)):
        with pytest.raises(ValueError):
            checked_buffer(io.BytesIO(changed), item)

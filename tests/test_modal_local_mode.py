"""Single-H200 budget through real compilation, launcher and environment factories."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml
from harbor.models.task.task import Task
from schemas.run_config import prepare_run

ROOT = Path(__file__).resolve().parents[1]


def job():
    config = yaml.safe_load((ROOT / 'configs/harbor-modal.yaml').read_text())
    config['runtime']['evaluation_mode'] = 'local'
    return config


def test_all_five_keep_complete_protocol_and_single_gpu(tmp_path):
    local = job()
    separate = deepcopy(local)
    separate['runtime'].pop('evaluation_mode')
    normal = prepare_run(separate, directory=tmp_path / 'separate')
    compiled = prepare_run(local, directory=tmp_path / 'local')
    assert compiled['n_concurrent_trials'] == 1
    assert compiled['environment']['kwargs']['local_mode'] is True
    assert len(compiled['tasks']) == 5
    for item, original in zip(compiled['tasks'], normal['tasks'], strict=True):
        task = Task(item['path'])
        assert task.config.environment.gpus == 0
        assert task.config.verifier.environment.gpus == 0
        assert task.config.verifier.environment_mode.value == 'separate'
        runtime = json.loads((task.paths.tests_dir / 'runtime.json').read_text())
        assert (runtime['gpu'], runtime['gpus'], runtime['benchmark_gpus']) == ('H200', 1, 0)
        assert runtime['evaluation_mode'] == 'local'
        for filename in ('verifier-config.json', 'modal-scoring.json'):
            assert (task.paths.tests_dir / filename).read_bytes() == (
                Path(original['path']) / 'tests' / filename).read_bytes()
        # Resource provenance must distinguish CPU drivers from the historical GPU driver.
        assert 'benchmark_gpus' not in runtime['baseline_conditions']['resources']


@pytest.mark.parametrize('change,match', [
    ({'n_concurrent_trials': 2}, 'n_concurrent_trials'),
    ({'environment': {'override_gpus': 1}}, 'GPU overrides'),
    ({'environment': {'kwargs': {'local_mode': False}}}, 'resource guard'),
])
def test_rejects_budget_bypasses(tmp_path, change, match):
    config = job()
    if 'environment' in change:
        config['environment'].update(change['environment'])
    else:
        config.update(change)
    with pytest.raises(ValueError, match=match):
        prepare_run(config, directory=tmp_path)


@pytest.mark.parametrize('extra,success', [
    ([], True), (['--n-concurrent', '2'], False),
    (['--override-gpus', '1'], False), (['--environment-kwarg', 'local_mode=false'], False),
])
def test_launcher_local_mode_and_native_overrides(extra, success):
    result = subprocess.run([
        sys.executable, 'scripts/run/run_modal_lite.py', '--local-mode', '--mode', 'score',
        '--', '--path', 'tasks/infer-qwen3vl-32b', '--print-config', *extra,
    ], cwd=ROOT, capture_output=True, text=True)
    assert (result.returncode == 0) is success, result.stderr
    if success:
        config = json.loads(result.stdout)
        task = Task(config['tasks'][0]['path'])
        assert config['environment']['kwargs']['local_mode'] is True
        assert config['verifier']['kwargs']['mode'] == 'score'
        assert task.config.environment.gpus == task.config.verifier.environment.gpus == 0
    else:
        assert 'local-mode' in result.stderr


def test_strict_baseline_does_not_hide_changed_driver_resources(tmp_path):
    config = job()
    config['tasks'] = config['tasks'][-1:]
    config['runtime']['baseline_policy'] = 'strict'
    with pytest.raises(ValueError, match='measurement conditions'):
        prepare_run(config, directory=tmp_path)

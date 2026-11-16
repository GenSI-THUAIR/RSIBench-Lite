"""Paper anchors, frozen Modal inputs, post-GRM gates, and persisted outputs."""
import asyncio
import json
import math
from pathlib import Path
import shutil
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths

from harbor_envs.abstract_mlplat.mlplatform_base import MLPlatformEnvironment
from schemas.run_config import prepare_run
from verify.grading import grm
from verify.grading.grm_verifier import VerifyOutcome
from verify.inference.runtime_infer_verifier import RuntimeInferVerifier
from verify.inference.sol_score import normalized_sol_score, reference_digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('ratio', [1.01, 1.4643, 1.9501, 2.5330, 3, 10])
def test_paper_anchors_and_above_sol(ratio):
    assert normalized_sol_score(0, ratio) == 0
    assert normalized_sol_score(1, ratio) == pytest.approx(.5)
    assert normalized_sol_score(ratio, ratio) == pytest.approx(1)
    assert normalized_sol_score(ratio * 1.001, ratio) > 1
    for speedup in [.01, .5, 1.2]:
        expected = 1 / (1 + (1 / speedup - 1 / ratio) / (1 - 1 / ratio))
        if expected < 0:
            assert normalized_sol_score(speedup, ratio) is None
        else:
            assert normalized_sol_score(speedup, ratio) == pytest.approx(expected)


def test_pole_is_explicitly_undefined_and_extreme_inputs_are_finite():
    assert normalized_sol_score(2, 4/3) is None
    assert normalized_sol_score(4, 1.5) is None
    assert math.isfinite(normalized_sol_score(1e-320, 3))
    assert math.isfinite(normalized_sol_score(1e308, 3))


@pytest.mark.parametrize('speedup,ratio', [(-1, 2), (float('nan'), 2), (1, 1), (1, float('inf'))])
def test_invalid_formula_inputs_rejected(speedup, ratio):
    with pytest.raises(ValueError):
        normalized_sol_score(speedup, ratio)


def compile_score(tmp_path, *, local=False):
    config = yaml.safe_load((ROOT / 'configs/harbor-modal.yaml').read_text())
    if local:
        config['runtime']['evaluation_mode'] = 'local'
    return prepare_run(config, directory=tmp_path)


@pytest.mark.parametrize('local', [False, True])
def test_all_five_score_snapshots_bind_h200_reference(tmp_path, local):
    compiled = compile_score(tmp_path, local=local)
    for entry in compiled['tasks']:
        task = Path(entry['path'])
        runtime = json.loads((task/'tests/runtime.json').read_text())
        scoring = json.loads((task/'tests/modal-scoring.json').read_text())
        ref = scoring['sol_reference']
        assert ref['hardware'] == 'H200' and ref['gpu_count'] == runtime['gpus'] == 1
        assert ref['baseline_throughput'] == scoring['performance_baseline']['output_throughput']
        assert ref['unit'] == scoring['performance_baseline']['unit']
        assert runtime['sol_reference_sha256'] == reference_digest(ref)


def verifier_for(task_path, trial_dir):
    verifier = object.__new__(RuntimeInferVerifier)
    verifier.mode = 'score'
    verifier.step_name = None
    verifier.task = Task(task_path)
    verifier.environment = Mock(spec=MLPlatformEnvironment)
    verifier.environment.platform_name = 'modal'
    verifier.trial_paths = TrialPaths(trial_dir=trial_dir)
    verifier.trial_paths.mkdir()
    return verifier


@pytest.fixture
def verifier(tmp_path):
    compiled = compile_score(tmp_path/'compiled')
    return verifier_for(compiled['tasks'][0]['path'], tmp_path/'trial')


@pytest.mark.parametrize('kind', ['pass', 'quality_failure', 'request_failure', 'grm_veto', 'deployment_failure', 'grm_unavailable', 'above_sol', 'undefined'])
def test_actual_modal_result_write_after_all_gates(verifier, kind):
    raw = json.loads((verifier.task.paths.tests_dir/'modal-scoring.json').read_text())
    ratio = raw['sol_reference']['sol_throughput_estimate'] / raw['performance_baseline']['output_throughput']
    speedup = ratio * 1.001 if kind == 'above_sol' else 1.0
    if kind == 'undefined':
        speedup = ratio / (2 - ratio) * 2  # Qwen H200 ratio is between 1 and 2.
    reward = 0.0 if kind in {'quality_failure', 'request_failure', 'deployment_failure'} else speedup
    rewards = {'performance_score': 0.0 if kind == 'request_failure' else speedup,
               'reward': reward, 'deploy_success': 1}
    if kind == 'deployment_failure':
        rewards = {'reward': 0.0, 'deploy_success': 0}
    grade = {'grm_transferable_infra_optimization': 0.0} if kind == 'grm_veto' else {}
    if kind == 'grm_unavailable':
        grade = {'grm_error': 'judge unavailable'}
    verifier._verify_impl = AsyncMock(return_value=VerifyOutcome(
        rewards, Path('/fixture/layer'), Path('/fixture/metadata'), failed=kind=='deployment_failure'))
    with patch.object(grm, 'grade_best_effort', new=AsyncMock(return_value=grade)):
        result = asyncio.run(verifier.verify())
    written = json.loads((verifier.trial_paths.verifier_dir/'reward.json').read_text())
    if kind == 'undefined':
        assert written['sol_score'] is None and written['sol_score_valid'] == 0
        assert 'sol_score' not in result.rewards  # Harbor only accepts finite numeric rewards.
        assert result.rewards['performance_score'] == speedup
    else:
        expected = (0 if kind in {'quality_failure', 'request_failure', 'grm_veto', 'deployment_failure'}
                    else normalized_sol_score(speedup, ratio))
        assert result.rewards['sol_score'] == written['sol_score'] == pytest.approx(expected)
        assert result.rewards['sol_score_valid'] == 1
    assert written['sol_reference']['hardware'] == 'H200'
    assert 'sol_reference' not in result.rewards
    assert json.loads(result.model_dump_json())['rewards']['performance_score'] == written['performance_score']
    if kind == 'grm_veto':
        assert result.rewards['performance_score'] == speedup
        assert result.rewards['reward'] == 0


def test_snapshot_never_rereads_live_presets_or_baselines(verifier):
    with patch('schemas.sol_presets.sol_reference', side_effect=AssertionError('live read')):
        raw = verifier._resolve_config()
    assert verifier._sol_reference['baseline_throughput'] == raw['performance_baseline']['output_throughput']


@pytest.mark.parametrize('change', ['missing', 'hash', 'baseline', 'hardware'])
def test_invalid_snapshot_rejected_before_execution(verifier, change):
    path = verifier.task.paths.tests_dir/'modal-scoring.json'
    raw = json.loads(path.read_text())
    if change == 'missing':
        raw.pop('sol_reference')
    elif change == 'baseline':
        raw['performance_baseline']['output_throughput'] *= .5
    elif change == 'hardware':
        raw['sol_reference']['hardware'] = 'H20'
        runtime_path = verifier.task.paths.tests_dir/'runtime.json'
        runtime = json.loads(runtime_path.read_text())
        runtime['sol_reference_sha256'] = reference_digest(raw['sol_reference'])
        runtime_path.write_text(json.dumps(runtime))
    else:
        raw['sol_reference']['sol_throughput_estimate'] *= 2
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        verifier._resolve_config()


def test_copied_task_uses_selected_baseline_not_checkout_baseline(tmp_path):
    source = tmp_path/'infer-qwen3p8-27b'
    shutil.copytree(ROOT/'tasks'/source.name, source)
    cfg = yaml.safe_load((source/'task.yaml').read_text())
    path = source/cfg['platforms']['modal']['baseline']
    baseline = json.loads(path.read_text())
    baseline['performance_baseline']['output_throughput'] *= .9
    path.write_text(json.dumps(baseline))
    job = yaml.safe_load((ROOT/'configs/harbor-modal.yaml').read_text())
    job['tasks'] = [{'path': str(source)}]
    compiled = prepare_run(job, directory=tmp_path/'compiled')
    scoring = json.loads((Path(compiled['tasks'][0]['path'])/'tests/modal-scoring.json').read_text())
    assert scoring['sol_reference']['baseline_throughput'] == baseline['performance_baseline']['output_throughput']

    baseline['performance_baseline']['output_throughput'] = scoring['sol_reference']['sol_throughput_estimate']
    path.write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match='above baseline'):
        prepare_run(job, directory=tmp_path/'invalid-baseline')


def test_infrastructure_error_does_not_produce_scores(verifier):
    verifier._verify_impl = AsyncMock(side_effect=RuntimeError('infrastructure unavailable'))
    with pytest.raises(RuntimeError, match='infrastructure unavailable'):
        asyncio.run(verifier.verify())
    assert not (verifier.trial_paths.verifier_dir/'reward.json').exists()


@pytest.mark.parametrize('mode', ['smoke', 'calibrate'])
def test_non_score_modes_have_no_sol_binding(tmp_path, mode):
    job = yaml.safe_load((ROOT/'configs/harbor-modal.yaml').read_text())
    job['verifier']['kwargs']['mode'] = mode
    compiled = prepare_run(job, directory=tmp_path/mode)
    for entry in compiled['tasks']:
        scoring = json.loads((Path(entry['path'])/'tests/modal-scoring.json').read_text())
        assert 'sol_reference' not in scoring

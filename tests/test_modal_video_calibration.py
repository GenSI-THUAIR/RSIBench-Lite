"""Hardware-specific references must not silently change the frozen workload."""
from copy import deepcopy
from pathlib import Path

import pytest

from harbor_envs.modal.video_calibration import apply_video_calibration, validate_hardware
from schemas.build import load_yaml
from schemas.run_config import modal_files

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tasks/infer-wan2p2ti2v-5b'


def test_modal_reference_override_preserves_h20_source_and_all_requests():
    config = load_yaml(SOURCE / 'task.yaml')
    original = deepcopy(config['verifier']['scoring'])
    identity = deepcopy(original['video_benchmark']['identity'])
    identity['hardware'] = {'model': 'NVIDIA H200', 'count': 1, 'driver': 'test-driver'}
    override = {'gate': {'data': '/h200/reference-manifest.json'},
                'video_benchmark': {'identity': identity, 'reference_manifest_sha256': 'a' * 64}}
    config['platforms']['modal']['video_calibration'] = override
    # A test reference intentionally has no admitted performance baseline.
    config['platforms']['modal'].pop('baseline', None)
    resources = {phase: {'gpu_types': ['H200'], 'storage_mb': None}
                 for phase in ('agent', 'benchmark', 'serve')}
    files = modal_files(SOURCE, config, resources, 'calibrate')
    import json
    raw = json.loads(files[Path('tests/verifier-config.json')])
    overlay = json.loads(files[Path('tests/modal-scoring.json')])
    runtime = json.loads(files[Path('tests/runtime.json')])
    assert raw == {**original, 'serve_node_count': 1, 'serve_gpus_per_node': 1}
    assert config['verifier']['scoring'] == original
    assert overlay['gate']['data'] == '/h200/reference-manifest.json'
    assert overlay['video_benchmark']['request'] == original['video_benchmark']['request']
    assert overlay['video_benchmark']['case_ids'] == original['video_benchmark']['case_ids']
    assert runtime['baseline_conditions']['video_benchmark']['identity'] == identity
    assert runtime['video_calibration_hardware'] == identity['hardware']


@pytest.mark.parametrize('override', [
    {'gate': {'tolerance': 0.1}},
    {'video_benchmark': {'request': {'num_inference_steps': 25}}},
    {'video_benchmark': {'case_ids': ['boxing-cats-42']}},
    {'video_benchmark': {'quality': {'expected_frame_count': 61}}},
])
def test_recalibration_cannot_weaken_workload_or_media_contract(override):
    scoring = load_yaml(SOURCE / 'task.yaml')['verifier']['scoring']
    with pytest.raises(ValueError):
        apply_video_calibration(scoring, override)


def test_reference_rejects_gpu_and_driver_mismatch():
    hardware = {'model': 'NVIDIA H200', 'count': 1, 'driver': '610.57.04'}
    validate_hardware(hardware, 'H200', 1, 'NVIDIA H200, GPU-id, 143771 MiB, 610.57.04\n')
    with pytest.raises(ValueError, match='requested'):
        validate_hardware(hardware, 'H100', 1)
    with pytest.raises(ValueError, match='actual'):
        validate_hardware(hardware, 'H200', 1, 'NVIDIA H200, GPU-id, 143771 MiB, 535.0\n')

"""SOL selection must not silently mix hardware, baselines, or evidence."""
import json
import shutil
from pathlib import Path

import pytest

from schemas.sol_presets import ROOT, sol_reference


@pytest.mark.parametrize('hardware', ['H20', 'H200'])
def test_five_presets_match_cited_evidence_and_units(hardware):
    catalog = json.loads((ROOT / 'assets/sol-presets.json').read_text())
    tasks = catalog['presets'][hardware]['tasks']
    assert set(tasks) == {p.parent.name for p in (ROOT / 'tasks').glob('*/task.yaml')}
    for task in tasks:
        result = sol_reference(task, hardware)
        assert result['sol_baseline_ratio_estimate'] == pytest.approx(
            result['sol_throughput_estimate'] / result['baseline_throughput'])
        assert result['official_reward'] is None
        if hardware == 'H200' and result['protocol'] == 'fixed_window_3600':
            assert result['evidence_status'] == 'historical_prefix_rate_only'
            assert result['record_coverage_seconds'] < result['required_window_seconds']


@pytest.fixture
def isolated_catalog(tmp_path):
    task = 'infer-qwen3p8-27b'
    catalog_path = ROOT / 'assets/sol-presets.json'
    catalog = json.loads(catalog_path.read_text())
    preset = catalog['presets']['H200']
    paths = ['assets/sol-presets.json', preset['evidence']['path'],
             preset['tasks'][task]['baseline']['path']]
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path, task, tmp_path / paths[-1]


def test_baseline_updates_change_ratio_without_changing_sol(isolated_catalog):
    root, task, path = isolated_catalog
    before = sol_reference(task, 'H200', root=root)
    document = json.loads(path.read_text())
    document['performance_baseline']['output_throughput'] *= 2
    path.write_text(json.dumps(document))
    after = sol_reference(task, 'H200', root=root)
    assert after['sol_throughput_estimate'] == before['sol_throughput_estimate']
    assert after['sol_baseline_ratio_estimate'] == before['sol_baseline_ratio_estimate'] / 2
    assert after['baseline_source_sha256'] != before['baseline_source_sha256']


@pytest.mark.parametrize('change', ['unit', 'gpu', 'gpus', 'missing_resources', 'nan'])
def test_incompatible_baseline_rejected(isolated_catalog, change):
    root, task, path = isolated_catalog
    document = json.loads(path.read_text())
    baseline = document['performance_baseline']
    if change == 'unit':
        baseline['unit'] = 'videos/hour'
    elif change == 'nan':
        baseline['output_throughput'] = float('nan')
    elif change == 'missing_resources':
        baseline.pop('baseline_conditions')
    else:
        baseline['baseline_conditions']['resources'][change] = 'H20' if change == 'gpu' else 2
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        sol_reference(task, 'H200', root=root)


def test_modified_sol_requires_matching_evidence(isolated_catalog):
    root, task, _ = isolated_catalog
    path = root / 'assets/sol-presets.json'
    data = json.loads(path.read_text())
    data['presets']['H200']['tasks'][task]['throughput'] *= 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='differs from its cited evidence'):
        sol_reference(task, 'H200', root=root)


def test_unknown_hardware_and_task_rejected():
    with pytest.raises(ValueError, match='Unknown SOL hardware'):
        sol_reference('infer-qwen3p8-27b', 'H200 NVL')
    with pytest.raises(ValueError, match='No H20 SOL reference'):
        sol_reference('missing-task', 'H20')

"""Read hardware-specific analytical SOL references without changing rewards.

Example: python -m schemas.sol_presets --hardware H200 --json
Absolute throughput belongs to a hardware/task pair. SOL/baseline is derived
from the referenced baseline file each time, never treated as a fixed score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def _source(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('SOL reference paths must stay inside the repository')
    return path


def sol_reference(task_id: str, hardware: str, *, root: Path = ROOT,
                  baseline_path: Path | None = None) -> dict:
    """Return an analytical reference and a ratio using the current baseline.

    Hardware must be explicit. Unknown hardware/tasks, changed SOL evidence,
    incompatible units, or mismatched baseline GPU identity are rejected.
    This function does not normalize an agent score or modify verifier output.
    """
    catalog_path = root / 'assets/sol-presets.json'
    catalog = json.loads(catalog_path.read_text())
    if catalog.get('schema_version') != 1:
        raise ValueError('Unsupported SOL preset schema')
    presets = catalog['presets']
    if hardware not in presets:
        raise ValueError(f'Unknown SOL hardware {hardware!r}; choose {sorted(presets)}')
    preset = presets[hardware]
    if task_id not in preset['tasks']:
        raise ValueError(f'No {hardware} SOL reference for {task_id!r}')
    entry = preset['tasks'][task_id]
    sol = _positive(entry['throughput'], 'SOL throughput')
    source = preset['evidence']
    evidence = _source(root, source['path']).read_bytes()
    if hashlib.sha256(evidence).hexdigest() != source['sha256']:
        raise ValueError('SOL evidence hash differs; review and update the preset')
    matching = [row for row in json.loads(evidence)['results'] if row['task'] == task_id]
    if (len(matching) != 1 or matching[0][source['throughput_field']] != sol
            or matching[0]['unit'] != entry['unit']):
        raise ValueError('SOL preset differs from its cited evidence')

    baseline_ref = entry['baseline']
    path = _source(root, baseline_ref['path'])
    if baseline_path is not None:
        # Compilation may use a copied task or a separately calibrated baseline.
        # Bind the selected file, not the current baseline in the source checkout.
        path = baseline_path.resolve()
        baseline_ref = {'path': str(path), 'keys': ['performance_baseline']}
    raw = path.read_bytes()
    baseline = yaml.safe_load(raw) if path.suffix in {'.yaml', '.yml'} else json.loads(raw)
    for key in baseline_ref['keys']:
        baseline = baseline[key]
    value = _positive(baseline['output_throughput'], 'Baseline throughput')
    if baseline.get('unit') != entry['unit'] or baseline.get('higher_is_better') is not True:
        raise ValueError('Baseline unit or optimization direction differs from SOL')
    resources = baseline.get('baseline_conditions', {}).get('resources', {})
    if (resources or preset.get('require_baseline_hardware')) and (
            resources.get('gpu') != hardware or resources.get('gpus') != preset['gpu_count']):
        raise ValueError('Baseline GPU identity differs from SOL hardware')
    return {
        'hardware': hardware, 'variant': preset['variant'], 'gpu_count': preset['gpu_count'],
        'task': task_id, 'unit': entry['unit'],
        'sol_throughput_estimate': sol, 'baseline_throughput': value,
        'sol_baseline_ratio_estimate': sol / value,
        'evidence_status': entry['evidence_status'], 'protocol': entry['protocol'],
        'limitations': entry['limitations'], 'assumption': preset.get('assumption'),
        'record_coverage_seconds': entry.get('record_coverage_seconds'),
        'required_window_seconds': entry.get('required_window_seconds'),
        'official_reward': None,
        'baseline_source': baseline_ref,
        'baseline_source_sha256': hashlib.sha256(raw).hexdigest(),
        'sol_evidence': source,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', choices=['H20', 'H200'],
                        help='Default: show both reference presets')
    parser.add_argument('--task', help='Optional task ID')
    parser.add_argument('--json', action='store_true', help='Include full provenance and limitations')
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'assets/sol-presets.json').read_text())
    rows = []
    for hardware in [args.hardware] if args.hardware else catalog['presets']:
        for task in [args.task] if args.task else catalog['presets'][hardware]['tasks']:
            rows.append(sol_reference(task, hardware))
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print('Analytical references only; ratios use the current baseline files.')
        print('Hardware\tTask\tBaseline\tSOL estimate\tSOL/baseline\tUnit\tEvidence status')
        for row in rows:
            print(f"{row['hardware']}\t{row['task']}\t{row['baseline_throughput']:.4f}\t"
                  f"{row['sol_throughput_estimate']:.4f}\t{row['sol_baseline_ratio_estimate']:.4f}\t"
                  f"{row['unit']}\t{row['evidence_status']}")
        print('H200 assumes SXM. Fixed-window prefix estimates are incomplete; '
              'LLaDA uses native single-round routing repriced for each hardware. '
              'Wan uses an analytical graph. Use --json for details.')


if __name__ == '__main__':
    main()

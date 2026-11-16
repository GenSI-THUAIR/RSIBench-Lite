"""Reprice retained numerical Lite workloads with an external AISimulate build.

No GPU job is launched. Hardware recosting preserves captured workload shapes;
it does not synthesize H200 scheduling, cache hits, or missing window events.
See docs/sol.md for the method and evidence limits.
LLaDA uses reprice_llada_routing_sol.py and its native-routing input bundle.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reprice(workload, system, evidence, model_class):
    config = dict(workload['config'], system=system,
                  model=str((evidence / workload['config']['model']).resolve()),
                  systems_paths=[str((evidence / 'systems').resolve())])
    model = model_class.best_available(config)
    try:
        vision = 0.0
        operations = []
        if workload.get('vision_shape'):
            operations = model.vision_operations([workload['vision_shape']])
            vision = math.fsum(row[1] for row in operations)
        cache = {}

        def cost(shape):
            key = tuple(shape)
            if key not in cache:
                np, new, past, nd, kv, images = shape
                value = model.estimate_forward_pass_time_ms({
                    'version': 1, 'worker_id': '0', 'dp_rank': 0,
                    'counter_id': 0, 'wall_time': 0,
                    'scheduled_requests': {
                        'num_prefill_requests': np, 'sum_prefill_tokens': new,
                        'sum_prefill_kv_tokens': past, 'var_prefill_length': 0,
                        'num_decode_requests': nd, 'sum_decode_kv_tokens': kv,
                        'var_decode_kv_tokens': 0,
                    },
                    'queued_requests': {},
                }) + images * vision
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f'Invalid forward cost: {value}')
                cache[key] = value
            return cache[key]

        rows = workload['queries']
        selected = math.fsum(cost(row[:6]) * row[6] for row in rows)
        available = math.fsum(cost(row[:6]) * row[7] for row in rows)
        if workload['protocol'] == 'fixed_video_graph':
            selected = available = vision
        first_round = None
        if workload.get('first_round_queries'):
            first_round = math.fsum(cost(row[:6]) * row[6]
                                    for row in workload['first_round_queries'])
        return {
            'selected_ms': selected, 'available_ms': available,
            'selected_prefill_ms': math.fsum((cost(row[:6]) - row[5] * vision) * row[6]
                                            for row in rows if row[0]),
            'selected_decode_ms': math.fsum(cost(row[:6]) * row[6]
                                           for row in rows if row[3]),
            'selected_vision_ms': math.fsum(row[5] * vision * row[6] for row in rows),
            'first_round_ms': first_round,
            'vision_ms_per_call': vision,
            'vision_operations': [list(row) for row in operations],
            'estimator': model.diagnostics()['provenance']['selected_estimation_mode'],
        }
    finally:
        model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aisimulate-source', type=Path, required=True,
                        help='External AISimulate python/aisimulate/src directory')
    parser.add_argument('--native-package-source', type=Path,
                        help='Optional source tree containing the matching built native module')
    parser.add_argument('--evidence', type=Path,
                        default=Path('docs/evidence/sol'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', action='append', help='Optional task ID filter')
    args = parser.parse_args()
    # Retain the captured engine versions. SOL uses analytical operators, not
    # maintained empirical timing tables for current/next engine versions.
    os.environ['AIC_ALLOW_UNLISTED_VERSIONS'] = '1'
    sys.path.insert(0, str(args.aisimulate_source.resolve()))
    import aisimulate
    if args.native_package_source:
        aisimulate.__path__.append(str(args.native_package_source.resolve() / 'aisimulate'))
    from aisimulate_core.sdk import RustForwardPassPerfModel
    from aisimulate import _runtime

    def provenance():
        source = args.aisimulate_source.resolve()
        loaded = {}
        for name, module in list(sys.modules.items()):
            if not name.startswith('aisimulate_core'):
                continue
            file = getattr(module, '__file__', None)
            if file and Path(file).is_file() and Path(file).is_relative_to(source):
                loaded[str(Path(file).relative_to(source))] = sha256(Path(file))
        commit = subprocess.check_output(
            ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        return {
            'source_commit': commit,
            'loaded_source_sha256': dict(sorted(loaded.items())),
            'native_module_sha256': sha256(Path(_runtime.__file__)),
            'allow_unlisted_engine_versions': True,
            'correction_enabled': False,
            'hardware_sha256': {p.name: sha256(p) for p in sorted((args.evidence / 'systems').glob('*.yaml'))},
            'calculator_sha256': sha256(Path(__file__)),
            'baselines_sha256': sha256(args.evidence / 'baselines.json'),
        }

    baselines = json.loads((args.evidence / 'baselines.json').read_text())
    results = []
    for path in sorted(args.evidence.glob('infer-*-workload.json.gz')):
        with gzip.open(path, 'rt') as stream:
            workload = json.load(stream)
        task = workload['task']
        if args.task and task not in args.task:
            continue
        print(f'Repricing {task}', flush=True)
        h20 = reprice(workload, workload['h20_system'], args.evidence,
                      RustForwardPassPerfModel)
        h200 = reprice(workload, 'h200_sxm', args.evidence,
                       RustForwardPassPerfModel)
        if not math.isclose(h20['selected_ms'], workload['h20_selected_ms'],
                            rel_tol=1e-9, abs_tol=1e-5):
            raise ValueError(f'{task}: H20 reproduction mismatch: '
                             f"{h20['selected_ms']} != {workload['h20_selected_ms']}")
        for value in (h20, h200):
            if value['estimator'] != 'op_level':
                raise ValueError('Unexpected estimator fallback')
        video = workload['protocol'] == 'fixed_video_graph'
        throughput = ((3_600_000 if video else workload['output_tokens'] * 1000)
                      / h200['selected_ms'])
        fixed_window = workload['protocol'] == 'fixed_window_3600'
        result = {
            'task': task, 'unit': 'videos/hour' if video else 'tokens/s',
            'hardware': 'H200', 'protocol': workload['protocol'],
            'status': ('historical_prefix_rate_only' if fixed_window else
                       'historical_repeated_trace_rate' if 'repeated' in workload['protocol']
                       else 'fixed_workload_analytical'),
            'sol_throughput_estimate': throughput,
            'h200_baseline_throughput': baselines[task]['output_throughput'],
            'sol_baseline_ratio_estimate': throughput / baselines[task]['output_throughput'],
            'exact_h200_fixed_window_throughput': None,
            'official_reward': None,
            'h20_published_throughput': workload['h20_published_throughput'],
            'h20_reproduction_error_ms': h20['selected_ms'] - workload['h20_selected_ms'],
            'hardware_speedup_same_work': h20['selected_ms'] / h200['selected_ms'],
            'selected_output_tokens': workload['output_tokens'],
            'h20': h20, 'h200': h200,
            'workload_file': path.name, 'workload_sha256': sha256(path),
            'limitation': workload['limitation'],
        }
        if fixed_window:
            result['h200_record_coverage_seconds'] = h200['available_ms'] / 1000
            result['missing_window_seconds'] = max(0.0, 3600 - h200['available_ms'] / 1000)
        if h200['first_round_ms'] is not None:
            result['first_round_output_tokens'] = workload['first_round_output_tokens']
            result['first_round_sol_throughput'] = workload['first_round_output_tokens'] * 1000 / h200['first_round_ms']
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'provenance': provenance(), 'results': results}, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k not in ('h20', 'h200')}), flush=True)


if __name__ == '__main__':
    main()

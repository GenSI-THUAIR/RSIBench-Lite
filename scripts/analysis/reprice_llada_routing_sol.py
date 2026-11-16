"""Reprice audited single-round LLaDA work with native expert counts.

Uses retained AISimulate per-operator compute/memory budgets plus the explicit
full-logits, bidirectional-attention, strict-FP32-router and sampling formulas.
No GPU or external simulator is needed after numerical workload extraction.
Example: python scripts/analysis/reprice_llada_routing_sol.py --output /tmp/llada.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'docs/evidence/sol/llada-routing-workload.json.gz'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(source, output):
    """Export anonymous numerical aggregates, checking the original audit hashes."""
    names = ['capture-audit.json', 'routing-audit.json', 'sol-result.json',
             'cost-table.json', 'audited-forwards.jsonl', 'routing-summary.jsonl',
             'audited-sampling.json']
    blobs = {n: (source / n).read_bytes() for n in names}
    capture, routing, result = [json.loads(blobs[n]) for n in names[:3]]
    assert capture['passed'] and routing['passed'] and routing['all_cuda_graph']
    # Only check inputs used here; other provenance entries may live in archives.
    for record in (capture, routing, result):
        for path, digest in record['source_sha256'].items():
            if Path(path).parent == source and Path(path).name in blobs:
                assert hashlib.sha256(blobs[Path(path).name]).hexdigest() == digest, path
    forwards = [json.loads(line) for line in blobs['audited-forwards.jsonl'].splitlines()]
    routes = [json.loads(line) for line in blobs['routing-summary.jsonl'].splitlines()]
    assert len(forwards) == len(routes) == capture['forward_count'] == 40458
    assert {r['index'] for r in routes} == {f['index'] for f in forwards}
    contexts = Counter()
    for f in forwards:
        assert f['input_shape'] == [32] and f['logits_shape'] == [32, 157184]
        assert f['logits_dtype'] == 'torch.float32'
        contexts[f['seq_lens'][0] - 32] += 1
    histograms = [Counter() for _ in range(19)]
    for row in routes:
        assert len(row['unique_experts']) == 19
        for layer, count in enumerate(row['unique_experts']):
            assert 8 <= count <= 256
            histograms[layer][count] += 1
    table = json.loads(blobs['cost-table.json'])
    excluded = {'context_attention', 'context_logits_gemm', 'context_router_gemm', 'context_moe'}
    budgets = {}
    for past, count in sorted(contexts.items()):
        ops = []
        for op in table[str(past)]['old_operations']:
            if op['name'] in excluded:
                continue
            sol = op['details']['sol']
            assert not op['details']['fallbacks']
            assert math.isclose(op['latency_ms'], max(sol['math_ms'], sol['memory_ms']), abs_tol=1e-12)
            assert op['source'] == 'sol' or op['latency_ms'] == 0
            # Retained ops use BF16 GEMMs or ideal logical byte traffic.
            ops.append([op['name'], sol['math_ms'] * 148e12 / 1000,
                        sol['memory_ms'] * 4022784000000 / 1000])
        budgets[str(past)] = {'count': count, 'operations': ops}
    samples = json.loads(blobs['audited-sampling.json'])
    assert len(samples) == capture['sampling_count'] == 31840
    assert all(s['shape'] == [32, 157184] and s['input_dtype'] == s['output_dtype'] == 'torch.float32' for s in samples)
    workload = {
        'task': result['task'], 'output_tokens': capture['output_tokens'],
        'completed_requests': capture['requests'], 'forward_count': len(forwards),
        'sampling_count': len(samples), 'contexts': budgets,
        'layer_unique_expert_histograms': [dict(sorted(h.items())) for h in histograms],
        'h20_reference_ms': result['analytical_total_ms'],
        'h20_reference_phase_ms': result['phase_ms'],
        'source_sha256': {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()},
        'method': 'Retained AISimulate BF16 per-operator FLOPs/bytes; explicit corrected graph and native expert-count histograms. Aggregation preserves additive cost, not execution order.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress((json.dumps(workload, separators=(',', ':')) + '\n').encode(), mtime=0))


def calculate(workload, hardware):
    bw, peak, fp32 = (hardware[k] for k in ('mem_bw', 'bfloat16_tc_flops', 'fp32_flops'))
    parts = {}

    def add(name, value, count):
        parts.setdefault(name, []).append(value * count)

    def roof(flops, memory, compute=peak):
        return max(flops / compute, memory / bw) * 1000

    q, h, vocab = 32, 2048, 157184
    assert sum(r['count'] for r in workload['contexts'].values()) == workload['forward_count']
    for past, row in workload['contexts'].items():
        count, full = row['count'], int(past) + q
        for name, flops, memory in row['operations']:
            add(name, roof(flops, memory), count)
        add('block_bidirectional_attention', 20 * roof(4*16*128*q*full, 2*(2*16*q*128+2*4*full*128)), count)
        add('qk_norm_partial_rope_kv_write', 20*q*(8*(2048+512)+4*0.5*(2048+512)+2*2*512)/bw*1000, count)
        add('full_logits_projection', roof(2*q*vocab*h, 2*(q*h+h*vocab+q*vocab)), count)
        add('final_norm', (8*q*h+2*h)/bw*1000, count)
        add('logits_fp32_cast', q*vocab*6/bw*1000, count)
        add('router_fp32_gemm', 19*roof(2*q*256*h, 4*(q*h+h*256+q*256), fp32), count)
        add('router_precision_casts', 19*6*q*(2048+256)/bw*1000, count)
    per_layer = []
    for histogram in workload['layer_unique_expert_histograms']:
        assert sum(histogram.values()) == workload['forward_count']
        per_layer.append(math.fsum(roof(2*3*q*8*h*512,
            2*(2*q*8*h+3*q*8*512)+int(unique)*3*h*512*2)*count
            for unique, count in histogram.items()))
    assert len(per_layer) == 19
    add('context_moe', math.fsum(per_layer), 1)
    add('sampling_argmax', q*vocab*4/bw*1000, workload['sampling_count'])
    add('sampling_softmax', q*vocab*8/bw*1000, workload['sampling_count'])
    add('sampling_gather_threshold_edit', q*(4+8+4+128)/bw*1000, workload['sampling_count'])
    phase = {name: math.fsum(values) for name, values in parts.items()}
    total = math.fsum(phase.values())
    return {'phase_ms': phase, 'per_layer_moe_ms': per_layer, 'total_ms': total,
            'sol_throughput_estimate': workload['output_tokens']*1000/total}


def check_operators(workload, source, native_source):
    """Independently evaluate retained operators with an external AISimulate build."""
    os.environ['AIC_ALLOW_UNLISTED_VERSIONS'] = '1'
    sys.path.insert(0, str(source.resolve()))
    import aisimulate
    if native_source:
        aisimulate.__path__.append(str(native_source.resolve() / 'aisimulate'))
    from aisimulate_core.sdk import RustForwardPassPerfModel
    from aisimulate import _runtime
    evidence = ROOT / 'docs/evidence/sol'
    config = dict(model=str(evidence / 'models/infer-llada2p1mini-16b'),
                  system='h200_sxm', backend='sglang', backend_version='0.5.11',
                  worker_type='aggregated', tp=1, moe_tp_size=1, moe_ep_size=1,
                  gemm_quant_mode='bfloat16', moe_quant_mode='bfloat16',
                  kvcache_quant_mode='bfloat16', fmha_quant_mode='bfloat16',
                  estimation_mode='op_level', database_mode='SOL', fallback_policy='deny',
                  systems_paths=[str(evidence / 'systems')],
                  estimator_config={'correction': {'enabled': False}})
    model = RustForwardPassPerfModel.best_available(config)
    count, error = 0, 0.0
    try:
        for past, row in workload['contexts'].items():
            details = {op['name']: op for op in model.static_phase_diagnostics(
                batch_size=1, context_length=int(past)+32, prefix=int(past), prefill=True)}
            for name, flops, memory in row['operations']:
                expected = max(flops/989.5e12, memory/4.8e12)*1000
                actual = details[name]['latency_ms']
                assert not details[name]['details']['fallbacks']
                assert details[name]['source'] == 'sol' or actual == 0
                assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
                count += 1
                error = max(error, abs(actual-expected))
    finally:
        model.close()
    loaded = {str(Path(module.__file__).relative_to(source.resolve())): sha(Path(module.__file__))
              for name, module in sys.modules.items() if name.startswith('aisimulate_core')
              and getattr(module, '__file__', None) and Path(module.__file__).is_file()
              and Path(module.__file__).is_relative_to(source.resolve())}
    return dict(passed=True, contexts=len(workload['contexts']), operator_checks=count,
                max_error_ms=error, loaded_source_sha256=loaded,
                native_module_sha256=sha(Path(_runtime.__file__)),
                model_sha256=sha(evidence / 'models/infer-llada2p1mini-16b/config.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-evidence', type=Path, help='Optional original audited evidence directory to extract')
    parser.add_argument('--workload', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--aisimulate-source', type=Path, help='Optional external source for independent operator validation')
    parser.add_argument('--native-package-source', type=Path)
    args = parser.parse_args()
    if args.native_evidence:
        extract(args.native_evidence.resolve(), args.workload)
    workload = json.loads(gzip.decompress(args.workload.read_bytes()))
    import yaml
    hardware_path = ROOT / 'docs/evidence/sol/systems/h200_sxm.yaml'
    h200_hw = yaml.safe_load(hardware_path.read_text())['gpu']
    h20_hw = {'mem_bw': 4022784000000, 'bfloat16_tc_flops': 148e12, 'fp32_flops': 44e12}
    h20, h200 = calculate(workload, h20_hw), calculate(workload, h200_hw)
    assert math.isclose(h20['total_ms'], workload['h20_reference_ms'], rel_tol=1e-10)
    assert h20['phase_ms'].keys() == workload['h20_reference_phase_ms'].keys()
    for key, value in workload['h20_reference_phase_ms'].items():
        assert math.isclose(h20['phase_ms'][key], value, rel_tol=1e-10, abs_tol=1e-9), key
    result = {'task': workload['task'], 'hardware': 'H200', 'unit': 'tokens/s',
              'protocol': 'single_round_140_requests', 'status': 'native_single_round_routing_analytical',
              'sol_throughput_estimate': h200['sol_throughput_estimate'],
              'h20_reproduction_error_ms': h20['total_ms']-workload['h20_reference_ms'],
              'h20': h20, 'h200': h200, 'official_reward': None,
              'output_tokens': workload['output_tokens'], 'completed_requests': workload['completed_requests'],
              'native_forward_count': workload['forward_count'], 'moe_layer_histograms': 19*workload['forward_count'],
              'moe_routing_validated': True, 'physical_hbm_traffic_validated': False,
              'native_h200_capture': False,
              'hardware_parameters': {k: h200_hw[k] for k in h20_hw},
              'source_sha256': {'workload': sha(args.workload), 'calculator': sha(Path(__file__)), 'h200_system': sha(hardware_path)},
              'limitation': 'Fixed H20 native scheduling/cache/JointThreshold/routing repriced for H200 SXM, not native H200 execution. Each accessed expert weight assumed read once per forward; physical HBM traffic, kernel padding, L2 hits, host/network time and auxiliary implementation overheads are not measured.'}
    if args.aisimulate_source:
        result['independent_operator_check'] = check_operators(workload, args.aisimulate_source, args.native_package_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('sol_throughput_estimate', 'h20_reproduction_error_ms')}))


if __name__ == '__main__':
    main()

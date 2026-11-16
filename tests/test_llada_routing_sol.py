"""Check the analytical recost against the accepted native H20 workload."""
import gzip
import hashlib
import json

import pytest
import yaml

from schemas.sol_presets import ROOT, sol_reference
from scripts.analysis.reprice_llada_routing_sol import DEFAULT, calculate


def test_native_h20_control_and_h200_component_costs():
    workload = json.loads(gzip.decompress(DEFAULT.read_bytes()))
    h20 = calculate(workload, dict(mem_bw=4022784000000,
                                  bfloat16_tc_flops=148e12, fp32_flops=44e12))
    assert h20['sol_throughput_estimate'] == pytest.approx(642.2888515233753, rel=1e-10)
    assert h20['phase_ms'] == pytest.approx(workload['h20_reference_phase_ms'], rel=1e-10)
    hardware_path = ROOT / 'docs/evidence/sol/systems/h200_sxm.yaml'
    h200 = calculate(workload, yaml.safe_load(hardware_path.read_text())['gpu'])
    # All observed expert counts are bandwidth-bound on H200. Independently
    # total bytes over the 768,702 layer calls, rather than summing per-op costs.
    histograms = workload['layer_unique_expert_histograms']
    calls = sum(sum(h.values()) for h in histograms)
    experts = sum(int(u)*n for h in histograms for u, n in h.items())
    assert calls == 768702
    assert h200['phase_ms']['context_moe'] == pytest.approx(
        (experts*6291456 + calls*2883584) / 4.8e12 * 1000)
    assert h200['total_ms'] == pytest.approx(sum(h200['phase_ms'].values()))
    assert h200['sol_throughput_estimate'] == pytest.approx(57329000/h200['total_ms'])
    preset = sol_reference(workload['task'], 'H200')
    assert preset['protocol'] == 'single_round_140_requests'
    assert preset['sol_throughput_estimate'] == h200['sol_throughput_estimate']


def test_recalculation_receipts_and_inputs_are_bound():
    directory = ROOT / 'docs/evidence/sol'
    receipt = json.loads((directory / 'h200.json').read_text())
    for name, digest in receipt['calculation_receipts'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    llada = json.loads((directory / 'llada.json').read_text())
    assert hashlib.sha256(DEFAULT.read_bytes()).hexdigest() == llada['source_sha256']['workload']
    script = ROOT / 'scripts/analysis/reprice_llada_routing_sol.py'
    assert hashlib.sha256(script.read_bytes()).hexdigest() == llada['source_sha256']['calculator']
    assert llada['independent_operator_check']['passed']
    assert llada['independent_operator_check']['operator_checks'] == 12987
    assert llada['native_h200_capture'] is False

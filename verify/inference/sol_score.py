"""Paper SOL normalization of gated reward using a frozen hardware reference."""
from __future__ import annotations

import hashlib
import json
import math

FORMULA = '1 / (1 + (1 / speedup - 1 / sol_ratio) / (1 - 1 / sol_ratio))'


def reference_digest(reference: dict) -> str:
    return hashlib.sha256(json.dumps(reference, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_reference(reference: dict, baseline: dict, hardware: str, gpus: int) -> None:
    if reference.get('hardware') != hardware or reference.get('gpu_count') != gpus:
        raise ValueError('SOL reference hardware differs from the runtime')
    if reference.get('unit') != baseline.get('unit'):
        raise ValueError('SOL reference unit differs from the runtime baseline')
    base, sol = baseline.get('output_throughput'), reference.get('sol_throughput_estimate')
    if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in (base, sol)):
        raise ValueError('SOL and baseline throughput must be finite and positive')
    if reference.get('baseline_throughput') != base:
        raise ValueError('SOL reference baseline differs from the run snapshot')
    if sol <= base:
        raise ValueError('Paper SOL normalization requires SOL throughput above baseline')


def normalized_sol_score(speedup: float, sol_ratio: float) -> float | None:
    """Oracle=0.5, SOL=1; retain >1, and represent a pole/out-of-domain as None.

    Zero is the paper convention for failed/gated runs and zero throughput.
    The algebraic forms avoid overflow for very small or very large speedups.
    """
    if not math.isfinite(speedup) or speedup < 0:
        raise ValueError('Speedup must be finite and nonnegative')
    if not math.isfinite(sol_ratio) or sol_ratio <= 1:
        raise ValueError('SOL/baseline must be finite and greater than one')
    if speedup == 0:
        return 0.0
    inverse_sol = 1 / sol_ratio
    if speedup <= 1:
        denominator = 1 + speedup * (1 - 2 * inverse_sol)
        numerator = speedup * (1 - inverse_sol)
    else:
        denominator = 1 / speedup + 1 - 2 * inverse_sol
        numerator = 1 - inverse_sol
    if denominator <= 0:
        return None
    score = numerator / denominator
    return score if math.isfinite(score) else None


def add_sol_score(rewards: dict, reference: dict) -> dict:
    """Run after GRM so failed correctness/compliance cannot earn SOL credit."""
    result = dict(rewards)
    if result.get('deploy_success') == 0:
        result.setdefault('performance_score', 0.0)
    if 'performance_score' not in result or 'reward' not in result:
        raise ValueError('SOL scoring requires a completed performance or deployment-failure outcome')
    ratio = reference['sol_throughput_estimate'] / reference['baseline_throughput']
    score = normalized_sol_score(result['reward'], ratio)
    result.update(
        sol_score=score,
        sol_score_valid=int(score is not None),
        sol_score_status='computed' if score is not None else 'undefined_nonpositive_or_nonfinite_denominator',
        sol_score_formula=FORMULA,
        sol_score_input='reward_after_quality_and_grm_gates',
        sol_reference=reference,
    )
    return result

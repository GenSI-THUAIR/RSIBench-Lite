# Evaluation and scoring

The agent submits `/logs/submission/submit_oci.tar.gz`. The verifier applies the
OCI layer to a clean engine environment, starts an isolated service, checks
correctness and compliance, and measures the complete task workload.
`task.yaml` is the authoritative task contract.

## Outputs

Harbor writes run results, logs, and verifier artifacts under `jobs/`.
The verifier's `reward.json` records:

| Field | Meaning |
| --- | --- |
| `performance_score` | Measured throughput divided by the selected baseline, subject to request-success checks |
| `reward` | Performance score after correctness and compliance gates |
| `sol_score` | SOL-normalized reward: Oracle = 0.5, analytical SOL = 1 |
| `sol_score_valid` | Whether the SOL-normalization formula is defined |

For a valid run exactly matching its baseline:

```json
{"performance_score": 1.0, "reward": 1.0, "sol_score": 0.5, "sol_score_valid": 1}
```

With `p = reward` and `r = SOL throughput / baseline throughput`:

```text
sol_score = 1 / (1 + (1/p - 1/r) / (1 - 1/r))
```

Failed correctness/compliance checks or zero performance produce `sol_score = 0`.
A failed gate can leave a positive raw `performance_score` while `reward` and
`sol_score` are zero. Infrastructure failures remain errors. Missing or
unconfigured compliance judging is not a pass.

Finite positive SOL scores above 1 are retained. At the formula's pole or a
nonpositive denominator, `reward.json` records `sol_score: null`,
`sol_score_valid: 0`, and a reason; Harbor omits the undefined numeric reward.
A SOL reference at or below baseline is rejected before dispatch.

Compilation freezes the selected hardware, baseline, SOL reference, and hashes
in each run's task snapshot. Resumed runs use that snapshot. Smoke and
calibration runs do not emit formal score pairs.

## Baselines

The supplied H200 baselines are arithmetic means of three complete Oracle
throughput measurements. For video, this means `mean(3600 × 4 / seconds_i)`.
Each receipt records individual measurements, spread, resource conditions,
correctness outcomes, and evidence hashes. Correctness passed for these
measurements; GRM judging was not configured.

The task's `platforms.modal.baseline` selects the baseline file. The supplied
configurations use `runtime.baseline_policy: reuse_resources`, allowing changes
to CPU/memory allocations while preserving model, image, data, and protocol
identities. Set `baseline_policy: strict` to require matching resource conditions.
Single-GPU local mode records its CPU-driver conditions in `baseline-usage.json`.
Reusing a denominator is not a new calibration.

## Register a baseline

Run the complete Oracle workload under the intended measurement conditions:

```bash
bash scripts/run/run_task.sh --config configs/harbor-modal-calibrate.yaml \
  --path "tasks/$TASK" --agent oracle
python scripts/calibration/record_harbor_baseline.py jobs/CALIBRATION_JOB
```

Registration requires a complete run, passing quality checks, and cleanup
evidence. It rejects overwriting an existing baseline. Validate a newly
registered denominator with a fresh complete Oracle score run.

Video references are hardware-specific. `platforms.modal.video_calibration`
binds the reference, quality limits, and calibration receipt. Calibration uses
three independent baseline runs, one graph-mode positive control, and eight
predeclared negative families. The fixed 25% separation rule must accept all
positive cases and reject every negative run; a missing gap prevents admission.
The requested and observed GPU/driver must match the reference identity.

For a new video reference, the registration tool also supports a median-of-three
baseline at a new destination:

```bash
bash scripts/run/run_task.sh --config configs/harbor-modal-calibrate.yaml \
  --path tasks/infer-wan2p2ti2v-5b --agent oracle --n-attempts 3 --n-concurrent 1
python scripts/calibration/record_harbor_baseline.py jobs/CALIBRATION_JOB --video-median
```

That option computes `3600 × 4 / median(full-workset seconds)`, including media
delivery. It is a different aggregation from the supplied arithmetic means.

# RSIBench Lite

**A benchmark for evaluating coding agents on inference optimization.**

RSIBench Lite evaluates changes to SGLang, vLLM, and vLLM-Omni while preserving
model precision, correctness, and fixed evaluation workloads. Agents submit an
OCI filesystem layer, which a trusted verifier evaluates in isolated environments
on [Modal](https://modal.com/docs). The framework builds on
[Harbor](https://github.com/laude-institute/harbor).

[Project page](https://rsibench-infra.github.io/) ·
[Run the benchmark](docs/running.md) · [Evaluation](docs/evaluation.md) ·
[SOL references](docs/sol.md) · [Development](docs/development.md)

## Tasks and reference results

Each task runs on one H200 model-service GPU. Baselines are the arithmetic mean
of three complete Oracle throughput measurements with passing correctness gates.
SOL is an analytical device-throughput reference, not measured performance.

| Task | Engine | Performance workload | H200 baseline | H200 SOL | Unit |
| --- | --- | --- | ---: | ---: | --- |
| [Qwen3.8 27B FP8](tasks/infer-qwen3p8-27b) | SGLang | 256k trace, concurrency 1, 3600 s | 202.769074 | 395.415134 | tokens/s |
| [Muse Glimmer 30B BF16](tasks/infer-museglimmer-30b) | vLLM | 128k trace, concurrency 1, 3600 s | 60.472963 | 88.550921 | tokens/s |
| [Qwen3 VL 32B FP8](tasks/infer-qwen3vl-32b) | SGLang | 32 image requests, concurrency 8, 1080p | 243.573318 | 729.139081 | tokens/s |
| [LLaDA2.1 mini 16B BF16](tasks/infer-llada2p1mini-16b) | SGLang | 140-request 32k trace, concurrency 4 | 90.492705 | 1054.792824 | tokens/s |
| [Wan2.2 TI2V 5B BF16](tasks/infer-wan2p2ti2v-5b) | vLLM-Omni | Four cases, concurrency 1, 121 frames, 50 steps | 26.037343 | 65.953724 | videos/hour |

The Qwen3.8 and Muse SOL values are prefix-rate estimates; their available
records do not cover a full H200 analytical hour. LLaDA uses single-round native
expert routing, and Wan uses an analytical generation graph. See
[SOL references](docs/sol.md) for the H20 values, hardware assumptions, and scope.

Task definitions pin the workload, model revisions, images, correctness gates,
and input manifests. All baseline measurements passed correctness checks;
compliance judging was not configured for those measurements. Formal agent
runs require the configured correctness and compliance checks.

## Quick start

Prerequisites: Python 3.12 or 3.13, [uv](https://docs.astral.sh/uv/), Bash,
and a Modal workspace with H200 access.

```bash
uv sync --frozen --extra modal
source .venv/bin/activate
export PYTHONPATH="$PWD"

# Inspect the resolved configuration without starting cloud resources.
TASK=infer-qwen3p8-27b
bash scripts/run/run_task.sh --config configs/harbor-modal-smoke.yaml \
  --path "tasks/$TASK" --print-config
```

For cloud execution, authenticate with `modal setup`, configure your coding
agent and compliance judge, and prepare the task's model and evaluation inputs.
The [running guide](docs/running.md) covers each step.

**Evaluation inputs:** the repository includes frozen manifests and small
fixtures, but the complete evaluation data bundle is not publicly distributed.
You must supply matching data before running the full benchmark. The preparation
tools validate and upload those files; model weights are downloaded separately.

Once inputs are prepared, run the reference solution:

```bash
bash scripts/run/run_task.sh --config configs/harbor-modal.yaml \
  --path "tasks/$TASK" --agent oracle
```

To evaluate a coding agent, add `--agent "$AGENT_NAME" --model "$AGENT_MODEL"`.
Add `--local-mode` for one H200 total per job, with the Agent and Benchmark
running on CPU. Normal configurations allocate separate Agent, Benchmark, and
Serve resources. Both modes execute on Modal.

## Scores

- `performance_score`: measured throughput divided by the task baseline.
- `reward`: performance score after correctness and compliance gates.
- `sol_score`: SOL-normalized reward, with Oracle = 0.5 and analytical SOL = 1.

A successful deployment alone does not establish a valid score. Smoke and
calibration runs do not produce formal score pairs. See the
[evaluation guide](docs/evaluation.md) for formulas and result interpretation.

## Repository

| Directory | Contents |
| --- | --- |
| `tasks/` | Task contracts, launch scripts, reference solutions, and input manifests |
| `configs/` | Modal execution presets |
| `schemas/` | Task and run-configuration compilation |
| `harbor_envs/` | Modal execution and lifecycle helpers |
| `verify/` | Trusted deployment, quality checks, measurement, and grading |
| `scripts/` | Execution, preparation, calibration, analysis, and release tools |
| `docs/evidence/sol/` | Numerical SOL results and reproducible calculation inputs |
| `tests/` | Protocol, execution, scoring, and release checks |

See [Development](docs/development.md) for validation and source export.

## License

Original project code is licensed under the [MIT License](LICENSE).
Third-party code, models, and datasets retain their respective licenses.

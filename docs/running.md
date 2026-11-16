# Running the benchmark

## Getting started

### 1. Install

Prerequisites: Python **3.12 or 3.13**, [uv](https://docs.astral.sh/uv/), Bash, and a Modal
workspace with access to the resources required by the selected configuration.
Run the following commands from the repository root:

```bash
uv sync --frozen --extra modal
source .venv/bin/activate
export PYTHONPATH="$PWD"
```

### 2. Configure authentication

For cloud execution, authenticate to your Modal workspace:

```bash
modal setup
```

Configure the judge and coding agent using your own credentials:

| Setting | Purpose |
| --- | --- |
| `GRM_JUDGE_BASE_URL`, `GRM_JUDGE_API_KEY` | Endpoint and authentication for compliance grading |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY` | Alternative judge settings when the corresponding `GRM_JUDGE_*` value is unset |
| `REWARDKIT_JUDGE` | Optional override of the rubric's judge model |
| `GRM_JUDGE_MODEL_METADATA` | JSON model metadata for judge models without public metadata; requires a verified `max_input_tokens` and compatible `litellm_provider` |

Coding agents use the authentication settings required by their upstream Harbor adapters.
Keep credentials in your shell environment or local configuration; no keys are bundled with this repository.

### 3. Inspect a task offline

The examples below use Qwen3.8. Keep `TASK` set in the same shell, or replace it with another
Task ID from the `tasks/` directory.

```bash
TASK=infer-qwen3p8-27b

bash scripts/run/run_task.sh \
  --config configs/harbor-modal-smoke.yaml \
  --path "tasks/$TASK" \
  --print-config
```

This compiles the task and prints the resolved configuration without creating cloud resources.

### 4. Prepare model and evaluation inputs

**The following preparation and execution commands use your Modal workspace.** Model preparation
runs in a CPU sandbox and writes weights directly to a Modal Volume, avoiding model downloads on
the control host.

```bash
python scripts/prepare/prepare_modal_model.py "tasks/$TASK/task.yaml"

python scripts/prepare/prepare_modal_data.py "tasks/$TASK/task.yaml" \
  --source /path/to/frozen-data
```

Replace `/path/to/frozen-data` with the dataset directory for the selected task. Its relative
paths, file sizes, and SHA-256 values must match `tasks/$TASK/runtimes/open-lite-data.json`.
Repeat preparation for each task you intend to run, using the same Modal workspace for preparation
and execution. The publishers verify inputs before recording readiness.

The data publisher reads the explicit local `--source` and checks every file against the
frozen manifest before uploading it to your Modal workspace.

## Usage

### Run the reference solution

The `oracle` agent submits the supplied reference solution. Start with delivery and protocol checks,
then run formal scoring for a task with an accepted baseline:

```bash
# CPU delivery check: verification and scoring are disabled.
bash scripts/run/run_task.sh \
  --config configs/harbor-modal-cpu.yaml --path "tasks/$TASK" --agent oracle

# GPU smoke: verify the deployment and benchmark wiring.
bash scripts/run/run_task.sh \
  --config configs/harbor-modal-smoke.yaml --path "tasks/$TASK" --agent oracle

# Formal scoring: run the full workload and correctness gate.
bash scripts/run/run_task.sh \
  --config configs/harbor-modal.yaml --path "tasks/$TASK" --agent oracle
```

Smoke results are not accepted performance baselines or formal scores. Video smoke retains all
four cases and the original quality requirements.

### Evaluate a coding agent

Set `AGENT_NAME` to a supported Harbor agent and `AGENT_MODEL` to its model identifier, then run:

```bash
bash scripts/run/run_task.sh \
  --config configs/harbor-modal.yaml \
  --path "tasks/$TASK" \
  --agent "$AGENT_NAME" \
  --model "$AGENT_MODEL"
```

The launcher forwards Harbor's agent and model options. Use `harbor run --help` for supported CLI
options, or edit a copy of a run configuration for repeated experiments.

### Choose a run mode

| Configuration | Purpose |
| --- | --- |
| [harbor-modal-cpu.yaml](../configs/harbor-modal-cpu.yaml) | CPU agent delivery check without verification |
| [harbor-modal-smoke.yaml](../configs/harbor-modal-smoke.yaml) | Deployment and protocol smoke test |
| [harbor-modal-calibrate.yaml](../configs/harbor-modal-calibrate.yaml) | Full Oracle quality/performance calibration |
| [harbor-modal.yaml](../configs/harbor-modal.yaml) | Formal scoring against a registered baseline |

`--path` selects one task; omitting it selects all five configured tasks. Agent, Benchmark, and
Serve are separate allocations. Normal GPU configurations assign one H200 to each role, so a
single-task run is not a one-GPU total resource budget.

### Single-GPU local-mode

Add `--local-mode` to run with **one H200 total per job**. Agent editing and the trusted
Benchmark driver run in CPU-only Modal sandboxes; the isolated model service gets one H200.
All five tasks run serially. The name describes the single-GPU resource mode; execution still
uses Modal, so authentication and prepared model/data Volumes are required.

```bash
# Full evaluation, all five tasks, one H200 at a time.
python scripts/run/run_modal_lite.py --local-mode --mode score

# Select one task, or append --print-config for an offline resource check.
bash scripts/run/run_task.sh --local-mode \
  --config configs/harbor-modal.yaml --path "tasks/$TASK" --agent oracle
```

For saved configurations, set `runtime.evaluation_mode: local`. The mode overrides GPU
allocations to Agent=0, Benchmark=0, Serve=1 H200, retaining CPU and memory settings.
Concurrency above one and GPU overrides for Agent/Benchmark are rejected before dispatch.
Each submission still gets a fresh isolated service and an unpatched trusted Benchmark.
Agents can edit and submit on CPU; GPU development self-tests are unavailable in this mode.

Quality thresholds, model precision, full workloads, and scoring formulas are unchanged.
`--mode smoke` remains a smoke test; `--mode calibrate` runs full calibration. With the supplied
`reuse_resources` policy, existing denominators are reused and the CPU driver conditions are
recorded in `baseline-usage.json`; these are not newly measured local-mode baselines. Use
`baseline_policy: strict` to require an exact resource match. Separate concurrently launched
jobs have independent GPU budgets.


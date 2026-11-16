# Development

Run commands from the repository root:

```bash
uv sync --frozen --extra modal
source .venv/bin/activate
python -m pytest tests -q
python -m schemas.build tasks/*/
python -m schemas.build --check tasks/*/
```

`task.yaml` and the shared instruction template are authoritative. Do not edit
generated `task.toml`, `instruction.md`, or `tests/verifier-config.json` files
directly. The launcher compiles isolated task snapshots for execution.

Contributions should preserve workload identity, model precision, correctness
thresholds, and baseline provenance. Changes to bound inputs require updated
hashes and validation of the affected task compilation and scoring paths.

## Source release

Stage the intended public files, scan them, and export a source snapshot:

```bash
python scripts/release/export_source.py
python scripts/release/export_source.py --output /tmp/rsibench-lite.tar.gz
```

The exporter reads indexed paths from the working tree, rejects credentials,
private runtime files, and symlinks, and scans supported nested archives.
New files must be added to the index to enter the export. Git history and
untracked runtime files are not exported. Review the archive before publishing.

The public documentation presents current results and methods. Keep private
storage locations, resource identifiers, dated experiment narratives, and
local workspace paths out of published documentation and evidence metadata.
Numerical inputs, model specifications, hardware assumptions, and SHA-256
identities support reproducibility without exposing those details.

## Analysis

[SOL references](sol.md) documents the analytical inputs and calculation tools.
AISimulate is an optional external analysis dependency, separate from the
benchmark runtime. Reproducing an analytical reference does not run a GPU
benchmark or validate native H200 scheduling.

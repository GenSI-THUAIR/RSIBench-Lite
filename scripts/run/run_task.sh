#!/usr/bin/env bash
# 功能：编译 lite set 配置并启动原生 Harbor Modal 任务。
# 参数：--local-mode 限制单 H200；其余 Harbor run 参数透传，默认 Oracle。
# 示例：bash scripts/run/run_task.sh --config configs/harbor-modal-smoke.yaml --path tasks/infer-qwen3p8-27b
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
HARBOR_BIN="${HARBOR_BIN:-$ROOT/.venv/bin/harbor}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python3}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$ROOT/scripts/run/run_task_model.py" "$HARBOR_BIN" \
  --config "$ROOT/configs/harbor-modal.yaml" "$@"

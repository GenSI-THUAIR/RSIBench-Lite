#!/usr/bin/env bash
# 功能：Muse Glimmer 官方 BF16 单 H20 文本基线，vLLM 固定源码运行；尚待实机准入。
# 参数：无；HARBOR_SERVICE_PORT 为平台注入服务端口。
# 示例：HARBOR_SERVICE_PORT="$PORT0" bash /vllm-workspace/vllm/launch.sh
set -euo pipefail

: "${HARBOR_SERVICE_PORT:?平台必须注入 HARBOR_SERVICE_PORT}"
REPO=/vllm-workspace/vllm
MODEL=/mnt/rsibench/muse-glimmer-30b-bf16-a4e59da52a7b
export PYTHONNOUSERSITE=1
# H20宿主R535：使用vLLM官方镜像已有的用户态CUDA兼容库，不升级宿主驱动。
export VLLM_ENABLE_CUDA_COMPATIBILITY="${HARBOR_CUDA_COMPATIBILITY:?环境后端必须注入CUDA兼容策略}"
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH="$REPO"
cd "$REPO"
test -s "$MODEL/model.safetensors.index.json"
# 使用最终 Python 的实际 import 与官方 EOS，不能仅验证镜像 label。
python3 -c 'import json,pathlib,sys,vllm; assert pathlib.Path(vllm.__file__).resolve() == pathlib.Path(sys.argv[1])/"vllm/__init__.py"; assert json.load(open(sys.argv[2]))["eos_token_id"] == [200001,200008]; print("vllm",vllm.__version__,vllm.__file__)' "$REPO" "$MODEL/generation_config.json"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name muse-glimmer-30b \
  --host "${HARBOR_SERVICE_HOST:?HARBOR_SERVICE_HOST 未由环境后端注入}" \
  --port "$HARBOR_SERVICE_PORT" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":0}' \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --generation-config auto

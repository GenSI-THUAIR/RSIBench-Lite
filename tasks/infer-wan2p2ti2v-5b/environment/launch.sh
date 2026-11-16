#!/usr/bin/env bash
# 功能：按原始 BF16 Wan2.2 TI2V-5B recipe 启动单卡 T2V 服务；禁止自行后台化。
# 参数：无；HARBOR_SERVICE_PORT 为平台注入的动态服务端口。
# 示例：HARBOR_SERVICE_PORT="$PORT0" bash launch.sh
# 状态：未实机验收；镜像/完整 scheduler 与 VAE 配置核验完成前不准入。
set -euo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
MODEL=/mnt/rsibench/wan22-model
test -s "$MODEL/model_index.json"
exec vllm serve "$MODEL" \
  --omni \
  --host "${HARBOR_SERVICE_HOST:-::}" \
  --port "${HARBOR_SERVICE_PORT:?HARBOR_SERVICE_PORT 未注入}" \
  --dtype bfloat16 \
  --served-model-name Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --vae-use-tiling \
  --enforce-eager

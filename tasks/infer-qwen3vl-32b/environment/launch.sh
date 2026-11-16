#!/bin/bash
# 单节点单卡 H20 启动 Qwen3-VL-32B FP8 多模态服务。
# 参数：无。
# 示例：bash /sgl-workspace/sglang/launch.sh

set -eo pipefail

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY

# 基线取模型卡/官方 cookbook 的保守配置:fa3 注意力 + 0.8 静态显存;
# 多模态特有旋钮(keep-mm-feature-on-device、SGLANG_USE_CUDA_IPC_TRANSPORT、
# SGLANG_VLM_CACHE_SIZE_MB、--mm-attention-backend、chunked prefill、
# max-running-requests)刻意不启用,留作优化空间。
exec python3 -m sglang.launch_server \
  --model-path /mnt/rsibench/qwen3vl_32b_fp8 \
  --tp-size 1 \
  --context-length 131072 \
  --trust-remote-code \
  --host "${HARBOR_SERVICE_HOST:-::}" \
  --port "${HARBOR_SERVICE_PORT:?HARBOR_SERVICE_PORT 未注入}" \
  --served-model-name qwen3-vl-32b \
  --mem-fraction-static 0.8 \
  --attention-backend fa3 \
  --enable-multimodal \
  --enable-metrics

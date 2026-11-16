#!/bin/bash
# 单节点单卡 H20 启动 LLaDA2.1-mini BF16 服务。

set -eo pipefail

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY

# inclusionAI 模型卡和 SGLang cookbook 的 JointThreshold 默认 Speed 基线；
# 无 YAML 时 SGLang v0.5.11 使用 threshold=0.5/edit_threshold=0。
# LLaDA2.1-mini 约占 47GB 显存，TP1 可放入单张 96GB H20。
exec python3 -m sglang.launch_server \
  --model-path /mnt/rsibench/llada21_16b_bf16 \
  --tp-size 1 \
  --dtype bfloat16 \
  --context-length 32768 \
  --trust-remote-code \
  --dllm-algorithm JointThreshold \
  --attention-backend flashinfer \
  --mem-fraction-static 0.8 \
  --max-running-requests 1 \
  --host "${HARBOR_SERVICE_HOST:-::}" \
  --port "${HARBOR_SERVICE_PORT:?HARBOR_SERVICE_PORT 未注入}" \
  --served-model-name llada2.1-mini \
  --enable-metrics

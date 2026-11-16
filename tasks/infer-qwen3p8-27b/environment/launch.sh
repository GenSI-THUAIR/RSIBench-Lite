#!/bin/bash
# 单节点单卡 H20 启动 Qwen3.8-27B FP8 服务。
# 参数：无。
# 示例：bash /sgl-workspace/sglang/launch.sh

set -eo pipefail

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY

# 官方 SGLang cookbook / day-0 博客对 Qwen3.8 系列的推荐配置：
# in-checkpoint MTP 走 EAGLE 入口，3 steps / topk 1 / 4 draft tokens。
# 显式给定 --max-running-requests：开 speculative 时 SGLang 会把默认值
# 重置为 48，与内存计算脱钩。
exec python3 -m sglang.launch_server \
  --model-path /mnt/rsibench/qwen3p8_27b_fp8 \
  --tp-size 1 \
  --context-length 262144 \
  --trust-remote-code \
  --host "${HARBOR_SERVICE_HOST:?HARBOR_SERVICE_HOST 未由环境后端注入}" \
  --port "${HARBOR_SERVICE_PORT:?HARBOR_SERVICE_PORT 未注入}" \
  --served-model-name qwen3.8-27b \
  --mem-fraction-static 0.85 \
  --max-running-requests 64 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-metrics

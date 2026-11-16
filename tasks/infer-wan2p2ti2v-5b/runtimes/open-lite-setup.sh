#!/usr/bin/env bash
# 功能：在 Modal CPU 镜像构建阶段安装与原题一致的 Wan 引擎和质量指标依赖。
# 参数：无。示例：bash open-lite-setup.sh；仅在独占 Sandbox 镜像中运行。
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ffmpeg
rm -rf /var/lib/apt/lists/*
mkdir -p /vllm-workspace
git init /vllm-workspace/vllm-omni
cd /vllm-workspace/vllm-omni
git remote add origin https://github.com/vllm-project/vllm-omni.git
timeout 300 git fetch --depth 1 origin ff6e906a25de1ca3122c0d0d5250fde5a0ddbc7b
git checkout --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = ff6e906a25de1ca3122c0d0d5250fde5a0ddbc7b
timeout 1800 python3 -m pip install --no-cache-dir -e . transformers==5.14.1 diffusers==0.40.0 lpips==0.1.4 av==16.0.1
python3 -m pip check
python3 -c 'import vllm, torch; assert vllm.__version__.startswith("0.28.0"), vllm.__version__; assert torch.version.cuda == "12.9", torch.version.cuda'
timeout 600 python3 -c 'import lpips; lpips.LPIPS(net="alex", verbose=False)'
git diff --exit-code
git diff --cached --exit-code

#!/usr/bin/env bash
# 镜像内安装步骤，由通用构建器执行，无命令行参数。
set -euo pipefail
cd /sgl-workspace/sglang && if [ ! -d .git ]; then git init -q -b main && git config user.email image-baseline@example.invalid && git config user.name image-baseline && git add -A && git commit -qm "image baseline"; fi && git config --system --add safe.directory "*"

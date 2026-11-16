#!/usr/bin/env bash
# 功能：在官方 vLLM 镜像内恢复同 commit 的可编辑源码，复用镜像已有 CUDA 二进制。
# 参数：无。调用：bash open-lite-setup.sh；仅在独有远端 Sandbox 内执行。
set -euo pipefail
test "${VLLM_BUILD_COMMIT:?镜像必须声明构建 commit}" = 2cf0a6915ce544dc493a0990f2ea38d81601128a
apt-get update
apt-get install -y --no-install-recommends git ca-certificates
PACKAGE_PARENT=$(python3 -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent.parent)')
mkdir -p /vllm-workspace/vllm
cd /vllm-workspace/vllm
git init
git remote add origin https://github.com/vllm-project/vllm.git
git fetch --depth=1 origin refs/tags/v0.28.0:refs/tags/v0.28.0
git checkout --detach v0.28.0
test "$(git rev-parse HEAD)" = 2cf0a6915ce544dc493a0990f2ea38d81601128a
# CPU准备只复用同commit已编译的扩展，避免把整个安装包重打成巨型临时wheel。
python3 - "$PACKAGE_PARENT/vllm" <<'PY'
from pathlib import Path
import sys
import shutil
source=Path(sys.argv[1])
target=Path('/vllm-workspace/vllm/vllm')
extensions=list(source.rglob('*.so'))
version=source/'_version.py'
assert extensions and version.is_file(), 'Missing official precompiled extension/version files'
for path in source.rglob('*'):
    if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':
        continue
    destination=target/path.relative_to(source)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        continue
    if path.suffix=='.so' or path==version:
        destination.symlink_to(path)
    else:
        # wheel还包含构建时收集的第三方Python包/数据，不能只搬.so。
        shutil.copy2(path,destination)
assert (target/'vllm_flash_attn/layers/rotary.py').is_file(), 'Missing wheel-provided FlashAttention Python package'
PY
export PYTHONPATH=/vllm-workspace/vllm
cd /tmp
python3 -c 'import pathlib,vllm; assert pathlib.Path(vllm.__file__).resolve() == pathlib.Path("/vllm-workspace/vllm/vllm/__init__.py"); assert vllm.__version__.split("+")[0] == "0.28.0"; print(vllm.__version__,vllm.__file__)'
git -C /vllm-workspace/vllm diff --exit-code

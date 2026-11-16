"""Agent Job 收尾时校验 agent 交付的 OCI image layer 并生成元数据(不再生成 Git diff)。

本文件由 schemas/build.py 原样嵌进每道 layer 型 task 的 [[verifier.collect]] heredoc,
在 Agent Job 的 pod 内以 root 执行;只依赖标准库。agent 自己负责生成并维护
`<output_dir>/<layer_artifact>`(标准 OCI layer tar+gzip,路径相对 rootfs 根);
这里只做:文件存在、体积上限、gzip 可解、是 tar、含约定 launch.sh 条目,并计算
压缩 blob sha256 与未压缩 diff_id,连同引擎仓库的 git 基线/终态 tree 一起写入
`<output_dir>/patch-metadata.json`(文件名沿用,verifier 按 layer_* 字段分派)。

参数:
  argv[1] engine_repo          引擎源码树(须是 git 仓库)
  argv[2] output_dir           交付目录(/logs/submission)
  argv[3] baseline_tree_file   环境层在 agent 起跑前写下的基线 worktree tree
  argv[4] layer_artifact       约定的层文件名(submit_oci.tar.gz)

示例:
  python3 collect_oci_layer.py /sgl-workspace/sglang /logs/submission \
      /tmp/.harbor-git-baseline-tree submit_oci.tar.gz
"""
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

MAX_LAYER_BYTES = 100 * 1024 * 1024
MAX_UNPACKED_BYTES = 1024 * 1024 * 1024

repo = Path(sys.argv[1])
output = Path(sys.argv[2])
baseline_tree_file = Path(sys.argv[3])
layer_artifact = sys.argv[4]
layer_path = output / layer_artifact
if not (repo / ".git").is_dir():
    raise RuntimeError(f"引擎 Git 仓库不存在：{repo}")
if not baseline_tree_file.is_file():
    raise RuntimeError(f"缺少 Agent 启动前的 worktree snapshot：{baseline_tree_file}")
if not layer_path.is_file():
    raise RuntimeError(f"agent 未交付 OCI layer：{layer_path}")

size = layer_path.stat().st_size
if size > MAX_LAYER_BYTES:
    raise RuntimeError(f"OCI layer 压缩体积 {size} 超过上限 {MAX_LAYER_BYTES}")
blob = hashlib.sha256(layer_path.read_bytes()).hexdigest()
diff = hashlib.sha256()
unpacked = 0
with gzip.open(layer_path, "rb") as stream:
    while True:
        chunk = stream.read(1 << 20)
        if not chunk:
            break
        diff.update(chunk)
        unpacked += len(chunk)
        if unpacked > MAX_UNPACKED_BYTES:
            raise RuntimeError(f"OCI layer 解压体积超过上限 {MAX_UNPACKED_BYTES}")
launch_member = str(Path(*repo.parts[1:]) / "launch.sh")
with tarfile.open(layer_path, "r:gz") as tar:
    names = [name.removeprefix("./") for name in tar.getnames()]
if launch_member not in names:
    raise RuntimeError(f"OCI layer 未包含约定启动脚本条目：{launch_member}")

base_commit = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
baseline_tree = baseline_tree_file.read_text().strip()
subprocess.run(
    ["git", "-C", str(repo), "cat-file", "-e", f"{baseline_tree}^{{tree}}"], check=True
)
final_index = Path("/tmp/.harbor-git-final.index")
final_index.unlink(missing_ok=True)
env = {**os.environ, "GIT_INDEX_FILE": str(final_index)}
try:
    subprocess.run(["git", "-C", str(repo), "read-tree", "HEAD"], env=env, check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A", "--", "."], env=env, check=True)
    final_tree = subprocess.check_output(
        ["git", "-C", str(repo), "write-tree"], env=env, text=True
    ).strip()
finally:
    final_index.unlink(missing_ok=True)
changed = subprocess.check_output(
    ["git", "-C", str(repo), "diff", "--name-only", "-z", baseline_tree, final_tree, "--"]
).decode(errors="surrogateescape").rstrip("\0").split("\0")

metadata = {
    "layer_artifact": layer_artifact,
    "layer_sha256": f"sha256:{blob}",
    "diff_id": f"sha256:{diff.hexdigest()}",
    "layer_size_bytes": size,
    "layer_unpacked_bytes": unpacked,
    "layer_entries": len(names),
    "base_commit": base_commit,
    "baseline_tree": baseline_tree,
    "final_tree": final_tree,
    "changed_files": [name for name in changed if name],
    "launch_script": "launch.sh",
}
(output / "patch-metadata.json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
)
print(json.dumps(metadata, ensure_ascii=False))

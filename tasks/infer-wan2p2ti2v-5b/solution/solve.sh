#!/bin/bash
# 功能：检查环境预置的 Wan2.2 基线，并显式交付 OCI filesystem layer。
# 参数：无。示例：bash /solution/solve.sh
# Collector 只校验；不交 git patch、不自行启动服务。
set -euo pipefail
REPO=/vllm-workspace/vllm-omni
test -d "$REPO/.git"
test -s "$REPO/launch.sh"
bash -n "$REPO/launch.sh"
git -C "$REPO" status --short

# ---- 交付:把相对基线的仓库改动打包成 OCI image layer(与 agent 同一协议)----
# 基线 tree 由环境层在 agent 起跑前写入;A/M/T 复制终态文件,D 写 .wh. whiteout,
# 再 `tar -C upper .` 生成 /logs/submission/submit_oci.tar.gz,collect 只校验不生成。
OUT="${HARBOR_PATCH_OUTPUT_DIR:-/logs/submission}"
BASE_TREE=$(cat "${HARBOR_BASELINE_TREE_FILE:-/tmp/.harbor-git-baseline-tree}")
UPPER=$(mktemp -d)
INDEX=$(mktemp -u)
# 与 verify_clean_baseline 同款:solve.sh 默认以 tiger 执行,而 add 要往 .git/objects 写。
sudo -n env GIT_INDEX_FILE="$INDEX" git -C "$REPO" read-tree HEAD
sudo -n env GIT_INDEX_FILE="$INDEX" git -C "$REPO" add -A -- .
FINAL_TREE=$(sudo -n env GIT_INDEX_FILE="$INDEX" git -C "$REPO" write-tree)
sudo -n rm -f "$INDEX"
while IFS= read -r -d '' status && IFS= read -r -d '' path; do
  mkdir -p "$UPPER$REPO/$(dirname "$path")"
  if [[ $status == D ]]; then
    : > "$UPPER$REPO/$(dirname "$path")/.wh.$(basename "$path")"
  else
    cp -a "$REPO/$path" "$UPPER$REPO/$path"
  fi
done < <(git -C "$REPO" diff-tree -r -z --name-status --no-renames "$BASE_TREE" "$FINAL_TREE")
test -e "$UPPER$REPO/launch.sh"
# 所有复制完成后还原目录属性,避免临时 mkdir 属性改写基座祖先目录。
while IFS= read -r -d '' dir; do
  source_dir="${dir#"$UPPER"}"
  test -d "$source_dir"
  sudo -n chown --reference="$source_dir" "$dir"
  sudo -n chmod --reference="$source_dir" "$dir"
  sudo -n touch -r "$source_dir" "$dir"
done < <(find "$UPPER" -depth -mindepth 1 -type d -print0)
sudo -n tar -czf "$OUT/submit_oci.tar.gz" -C "$UPPER" .
sudo -n chmod 0644 "$OUT/submit_oci.tar.gz"
sudo -n rm -rf "$UPPER"
ls -l "$OUT/submit_oci.tar.gz"

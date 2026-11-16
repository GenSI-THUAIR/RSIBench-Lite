"""在 sandbox 内冻结引擎基线，再安装任务启动脚本。

参数：引擎仓库绝对路径；可选 --baseline 指定待注入的任务 environment/。
示例：python3 sandbox_bootstrap.py /workspace/engine --baseline /tmp/task-environment。
只使用标准库；Agent 与独立 Serve 必须从相同镜像执行此准备步骤。
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

BASELINE_TREE_FILE = Path("/tmp/.harbor-git-baseline-tree")
PREPARED_DIRS = ("/logs/submission", "/logs/verifier", "/workspace/deploy")


def prepare(repo: Path, baseline: Path | None = None, *, trust_repository: bool = False) -> str:
    if not repo.is_absolute() or not repo.is_dir():
        raise ValueError("Engine repository must be an existing absolute directory")
    if trust_repository:
        # 官方镜像的源码目录可能属于构建用户；只信任本题引擎路径，不开放 '*'。
        subprocess.run(["git", "config", "--system", "--add", "safe.directory", str(repo)], check=True)
    env = {**os.environ, "GIT_AUTHOR_DATE": "2000-01-01T00:00:00 +0000",
           "GIT_COMMITTER_DATE": "2000-01-01T00:00:00 +0000"}

    def git(*args, git_env=env):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
            env=git_env, text=True,
        ).strip()

    if not (repo / ".git").exists():
        git("init", "-q", "-b", "main")
        git("config", "user.name", "harbor")
        git("config", "user.email", "harbor@example.invalid")
        git("add", "-A", "--", ".")
        git("commit", "--allow-empty", "-qm", "image baseline")
    with tempfile.TemporaryDirectory(prefix="rsibench-index-") as temp:
        index_env = {**env, "GIT_INDEX_FILE": str(Path(temp) / "index")}
        git("read-tree", "HEAD", git_env=index_env)
        git("add", "-A", "--", ".", git_env=index_env)
        tree = git("write-tree", git_env=index_env)
    git("update-ref", "refs/harbor/baseline-tree", tree)
    git("config", "gc.auto", "0")
    BASELINE_TREE_FILE.write_text(tree + "\n")
    for directory in PREPARED_DIRS:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o777)
    if baseline is not None:
        if not baseline.is_dir() or not any(baseline.iterdir()):
            raise ValueError("Baseline directory must exist and contain task files")
        shutil.copytree(baseline, repo, dirs_exist_ok=True)
    return tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    print(prepare(args.repo, args.baseline, trust_repository=True))


if __name__ == "__main__":
    main()

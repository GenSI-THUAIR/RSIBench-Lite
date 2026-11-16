"""oracle solve.sh 的 OCI layer 打包块 → 共享 collector → apply_oci_layer 的本地闭环测试。

参数:无。从每题 solve.sh 抽出「交付」打包块,在临时 Git 仓库(修改/新增/删除/可执行位)
上执行;再用 verify/delivery/collect_oci_layer.py 校验生成元数据;最后把层叠到一棵
只含基线的 rootfs 上,断言终态与 oracle 工作树逐字节一致(含 whiteout 删除)。

示例:
  PYTHONPATH=. .venv/bin/python3 -m unittest tests.test_solve_oci_packing
"""

from __future__ import annotations

from verify.resources import payload_file
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from schemas.build import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = REPO_ROOT / "verify/delivery/collect_oci_layer.py"
OCI_LAYER = payload_file('oci_layer.py')
MARKER = "# ---- 交付:把相对基线的仓库改动打包成 OCI image layer"


def layer_task_dirs() -> list[Path]:
    """发现所有声明 OCI layer 交付协议的 authored task。"""
    found: list[Path] = []
    for task_yaml in sorted((REPO_ROOT / "tasks").glob("*/task.yaml")):
        scoring = load_yaml(task_yaml)["verifier"]["scoring"]
        if "layer_artifact" in scoring:
            found.append(task_yaml.parent)
    return found


def packing_block(solve: Path) -> str:
    text = solve.read_text()
    assert MARKER in text, solve
    return text[text.index(MARKER):]


def git(repo: Path, *args: str, env: dict | None = None) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, env=env).strip()


def load_oci_layer():
    spec = importlib.util.spec_from_file_location("oci_layer", OCI_LAYER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SolvePackingTest(unittest.TestCase):
    def test_every_layer_task_has_identical_oracle_packer(self) -> None:
        """Oracle 全量数据集里的每个 layer task 都必须自行打包同一协议。"""
        task_dirs = layer_task_dirs()
        self.assertTrue(task_dirs)
        solve_scripts = [task_dir / "solution/solve.sh" for task_dir in task_dirs]
        self.assertTrue(all(path.is_file() for path in solve_scripts))
        blocks = {packing_block(path) for path in solve_scripts}
        self.assertEqual(len(blocks), 1, "所有 layer task 的 solve.sh 打包块必须逐字相同")

        import yaml
        config = yaml.safe_load((REPO_ROOT / 'configs/harbor-modal.yaml').read_text())
        dataset_tasks = {(REPO_ROOT / entry['path']).resolve() for entry in config['tasks']}
        self.assertEqual(dataset_tasks, set(task_dirs))

    def test_pack_collect_apply_roundtrip(self) -> None:
        if shutil.which("sudo") is None or subprocess.run(
            ["sudo", "-n", "true"], capture_output=True, check=False
        ).returncode != 0:
            self.skipTest("真实 Oracle 打包与 collect 测试需要免密 sudo")
        block = packing_block(layer_task_dirs()[0] / "solution/solve.sh")
        with tempfile.TemporaryDirectory() as tmp:
            try:
                root = Path(tmp)
                repo = root / "agent-rootfs/sgl-workspace/sglang"
                repo.mkdir(parents=True)
                git(repo, "init", "-q", "-b", "main")
                git(repo, "config", "user.email", "t@example.com")
                git(repo, "config", "user.name", "t")
                (repo / "python").mkdir()
                (repo / "python/server.py").write_text("VALUE = 1\n")
                (repo / "README.md").write_text("readme\n")
                (repo / "keep.txt").write_text("keep\n")
                git(repo, "add", "-A")
                git(repo, "commit", "-qm", "baseline")
                index = root / "baseline.index"
                env = {**os.environ, "GIT_INDEX_FILE": str(index)}
                git(repo, "read-tree", "HEAD", env=env)
                git(repo, "add", "-A", "--", ".", env=env)
                baseline_tree = git(repo, "write-tree", env=env)
                tree_file = root / "baseline-tree"
                tree_file.write_text(baseline_tree + "\n")
                # 干净基座 rootfs:与 agent 仓库同一起点(不含 .git 以外的任何后续改动)
                lower = root / "serve-rootfs" / repo.relative_to("/")
                shutil.copytree(repo, lower, symlinks=True)

                # oracle/agent 的改动:修改、新增可执行 launch.sh、新增子目录二进制、删除
                (repo / "python/server.py").write_text("VALUE = 2\n")
                (repo / "launch.sh").write_text("#!/bin/bash\necho launch\n")
                (repo / "launch.sh").chmod(0o755)
                (repo / "kernels").mkdir()
                (repo / "kernels/fast.so").write_bytes(b"\x7fELF\x00")
                (repo / "README.md").unlink()
                ancestor = repo.parent
                ancestor.chmod(0o751)
                os.utime(ancestor, (1690000000, 1690000000))

                out = root / "submission"
                out.mkdir()
                script = f"set -euo pipefail\nREPO={repo}\n" + block
                run_env = {**os.environ, "HARBOR_PATCH_OUTPUT_DIR": str(out),
                           "HARBOR_BASELINE_TREE_FILE": str(tree_file)}
                packed = subprocess.run(["bash", "-c", script], env=run_env, capture_output=True, text=True, check=False)
                self.assertEqual(packed.returncode, 0, packed.stderr)
                layer = out / "submit_oci.tar.gz"
                self.assertTrue(layer.is_file())
                self.assertEqual(layer.stat().st_mode & 0o777, 0o644)
                with tarfile.open(layer, "r:gz") as archive:
                    ancestor_member = next(
                        item for item in archive.getmembers()
                        if item.name.removeprefix("./") == str(ancestor).lstrip("/")
                    )
                    self.assertEqual(ancestor_member.mode, 0o751)
                    self.assertEqual(ancestor_member.mtime, 1690000000)
                    self.assertEqual(ancestor_member.uid, ancestor.stat().st_uid)
                    self.assertEqual(ancestor_member.gid, ancestor.stat().st_gid)

                # 打包块以 sudo 写了 .git/objects;生产里 collect hook 同样以 root 运行。
                collected = subprocess.run(
                    ["sudo", "-n", sys.executable, str(COLLECTOR), str(repo), str(out), str(tree_file), "submit_oci.tar.gz"],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(collected.returncode, 0, collected.stderr)
                metadata = json.loads((out / "patch-metadata.json").read_text())
                self.assertEqual(metadata["baseline_tree"], baseline_tree)
                self.assertEqual(set(metadata["changed_files"]),
                                 {"python/server.py", "launch.sh", "kernels/fast.so", "README.md"})

                summary = load_oci_layer().apply_oci_layer(layer, root / "serve-rootfs")
                self.assertEqual(summary["layer_sha256"], metadata["layer_sha256"])
                self.assertEqual(summary["diff_id"], metadata["diff_id"])
                self.assertEqual(summary["whiteouts"], [str(repo.relative_to("/") / "README.md")])
                for rel in ("python/server.py", "launch.sh", "kernels/fast.so", "keep.txt"):
                    self.assertEqual((lower / rel).read_bytes(), (repo / rel).read_bytes(), rel)
                self.assertFalse((lower / "README.md").exists())
                self.assertTrue(os.access(lower / "launch.sh", os.X_OK))
                # 终态 worktree tree 一致:oracle 交付的层完整表达了它的仓库改动
                index2 = root / "final.index"
                env2 = {**os.environ, "GIT_INDEX_FILE": str(index2)}
                git(lower, "read-tree", "HEAD", env=env2)
                git(lower, "add", "-A", "--", ".", env=env2)
                self.assertEqual(git(lower, "write-tree", env=env2), metadata["final_tree"])
            finally:
                # 打包块经 sudo 写了 root 属主的 git 对象,收回属主才能清理临时目录。
                subprocess.run(["sudo", "-n", "chown", "-R", str(os.getuid()), tmp], check=True)


if __name__ == "__main__":
    unittest.main()

"""verify/delivery/serve_controller.py 的 --oracle 模式测试。

参数：无。用假 repo 与假 launch.sh 验证 argv 解析、分派与 oracle 启动路径,
不触碰真实 Merlin/Ray。

示例：
  PYTHONPATH=. .venv/bin/python3 -m unittest tests.test_serve_controller_oracle
"""

from __future__ import annotations

from verify.resources import payload_file
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

SCRIPT = payload_file('serve_controller.py')


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("serve_controller", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_config(directory: Path, node_count: int) -> Path:
    config_path = directory / "verifier-config.json"
    config_path.write_text(
        json.dumps(
            {
                "engine_repo": "/sgl-workspace/sglang",
                "serve_node_count": node_count,
                "serve_gpus_per_node": 16,
            }
        )
    )
    return config_path


class OracleArgvTest(unittest.TestCase):
    """--oracle 只需 config 路径,不读 patch/metadata。"""

    def test_multi_node_dispatch_passes_script(self) -> None:
        module = load_module()
        seen: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = write_config(root, 4)
            script = root / "solve.sh"
            script.write_text("#!/bin/bash\ntrue\n")
            with (
                patch.object(
                    module, "run_multi_node",
                    lambda *args, **kwargs: seen.update(
                        {"mode": "multi", "kwargs": kwargs, "argc": len(args)}
                    ),
                ),
                patch.object(sys, "argv",
                             ["sc", "--oracle", str(script), str(config_path)]),
            ):
                module.main()
        self.assertEqual(seen["mode"], "multi")
        self.assertEqual(seen["kwargs"], {"oracle_script": str(script), "forbidden_paths": []})

    def test_single_node_dispatch_passes_script(self) -> None:
        module = load_module()
        seen: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = write_config(root, 1)
            script = root / "solve.sh"
            script.write_text("#!/bin/bash\ntrue\n")
            with (
                patch.object(
                    module, "run_single_node",
                    lambda *args, **kwargs: seen.update({"kwargs": kwargs}),
                ),
                patch.object(sys, "argv",
                             ["sc", "--oracle", str(script), str(config_path)]),
            ):
                module.main()
        self.assertEqual(
            seen["kwargs"], {"relaunch": False, "oracle_script": str(script), "forbidden_paths": []}
        )

    def test_oracle_requires_script_argument(self) -> None:
        module = load_module()
        with patch.object(sys, "argv", ["sc", "--oracle"]):
            # argparse 对缺值的 --oracle 直接以用法错误退出(SystemExit)。
            with self.assertRaises(SystemExit):
                module.main()

    def test_patch_mode_still_reads_metadata(self) -> None:
        """不带 --oracle 时仍按 patch/metadata/config 三参解析(既有行为)。"""
        module = load_module()
        seen: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = write_config(root, 1)
            patch_file = root / "engine.patch"
            patch_file.write_bytes(b"diff --git a/x b/x\n")
            metadata = root / "patch-metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "patch_sha256": "deadbeef",
                        "base_commit": "c0ffee",
                        "baseline_tree": "7ree",
                    }
                )
            )
            with (
                patch.object(
                    module, "run_single_node",
                    lambda p, m, repo, **kwargs: seen.update(
                        {"patch": p, "meta": m, "kwargs": kwargs}
                    ),
                ),
                patch.object(sys, "argv", [
                    "sc", str(patch_file), str(metadata), str(config_path)
                ]),
            ):
                module.main()
        self.assertEqual(seen["patch"], b"diff --git a/x b/x\n")
        self.assertEqual(seen["meta"]["base_commit"], "c0ffee")  # type: ignore[index]
        self.assertEqual(
            seen["kwargs"], {"relaunch": False, "oracle_script": "", "forbidden_paths": []}
        )


class OracleRunnerTest(unittest.TestCase):
    """oracle 模式跳过三项校验,执行脚本后直接启动。"""

    def test_runs_script_then_launches(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / "repo"
            repo.mkdir()
            (repo / "launch.sh").write_text("#!/bin/bash\ntrue\n")
            script = root / "solve.sh"
            script.write_text("#!/bin/bash\ntrue\n")

            runner = module.ServiceRunner(0, str(repo))
            ran: list[list[str]] = []

            def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
                ran.append(list(cmd))
                return subprocess.CompletedProcess(cmd, 0)

            with (
                patch.object(module.subprocess, "run", side_effect=fake_run),
                patch.object(runner, "_launch", return_value={"running": True}),
                patch.object(module, "worktree_tree",
                             side_effect=AssertionError("oracle 不应校验 tree")),
            ):
                result = runner.apply_and_start(
                    b"", "unused-sha", "unused-commit", "unused-tree",
                    "fd00::1", oracle_script=str(script),
                )

            self.assertEqual(result, {"running": True})
            joined = [" ".join(cmd) for cmd in ran]
            self.assertTrue(any(str(script) in text for text in joined),
                            f"solve.sh 未被执行: {joined}")
            self.assertTrue(any("chown" in text for text in joined),
                            f"缺少 chown: {joined}")

    def test_multi_node_tolerates_empty_metadata(self) -> None:
        """oracle 模式 metadata 为空 dict:run_multi_node 不得 KeyError。

        r11 实测:此前 metadata["patch_sha256"] 直接下标,oracle 走到这里即崩;
        而 argv 测试 mock 掉了 run_multi_node 本身,恰好漏掉这一层。
        """
        module = load_module()
        sent: list[dict[str, object]] = []
        actor_options: list[dict[str, object]] = []

        class FakeActorHandle:
            class _Method:
                def remote(self, *args, **kwargs):  # noqa: ANN002, ANN003
                    sent.append({"args": args, "kwargs": kwargs})
                    return "ref"

            apply_and_start = _Method()

        class FakeRemoteRunner:
            def options(self, **kwargs):  # noqa: ANN003
                actor_options.append(kwargs)
                return self

            def remote(self, *args, **kwargs):  # noqa: ANN002, ANN003
                return FakeActorHandle()

        fake_nodes = [
            {"NodeID": f"n{i}", "Alive": True,
             "NodeManagerAddress": f"[fd00::{i}]",
             "Resources": {"head": 100.0} if i == 0 else {"worker": 100.0}}
            for i in range(2)
        ]
        fake_ray = type("FakeRay", (), {
            "remote": staticmethod(lambda **kwargs: (lambda cls: FakeRemoteRunner())),
            "init": staticmethod(lambda **kwargs: None),
            "nodes": staticmethod(lambda: fake_nodes),
            "get": staticmethod(lambda refs: [{"running": True} for _ in refs]),
        })()
        fake_strategies = type("S", (), {
            "NodeAffinitySchedulingStrategy": staticmethod(
                lambda **kwargs: None
            )
        })()

        with (
            patch.dict(sys.modules, {
                "ray": fake_ray,
                "ray.util": type("U", (), {})(),
                "ray.util.scheduling_strategies": fake_strategies,
            }),
            patch.object(module, "wait_for_ray_gcs", return_value="[fd00::1]:9502"),
            patch.dict("os.environ", {
                "BYTED_RAY_POD_IP": "fd00::1", "HARBOR_SERVICE_PORT": "23456",
                "HARBOR_DIST_PORT": "23457", "HARBOR_SERVICE_HOST": "::",
            }, clear=False),
        ):
            module.run_multi_node(
                b"", {}, "/sgl-workspace/sglang", 2, 16,
                oracle_script="/tmp/solve.sh",
            )

        self.assertEqual(len(sent), 2, "应向两个节点各派一次")
        for options in actor_options:
            self.assertEqual(options["runtime_env"]["env_vars"]["HARBOR_SERVICE_HOST"], "::")
        for call in sent:
            self.assertEqual(call["kwargs"], {"oracle_script": "/tmp/solve.sh", "oci": None})

    def test_multi_node_patch_mode_requires_metadata(self) -> None:
        """patch 模式 metadata 缺字段应在入口 fail loud,而非到 actor 才崩。"""
        module = load_module()
        with self.assertRaises(ValueError) as ctx:
            module.run_multi_node(b"x", {}, "/sgl-workspace/sglang", 2, 16)
        self.assertIn("patch_sha256", str(ctx.exception))

    def test_missing_script_fails_loud(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "repo"
            repo.mkdir()
            runner = module.ServiceRunner(0, str(repo))
            with self.assertRaises(FileNotFoundError):
                runner.apply_and_start(
                    b"", "s", "c", "t", "fd00::1",
                    oracle_script=str(Path(temp_dir) / "nope.sh"),
                )


if __name__ == "__main__":
    unittest.main()

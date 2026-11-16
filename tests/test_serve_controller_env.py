"""验证 controller 的可信 head 端口跨 Ray worker 传递。

参数：无。模拟 Ray actor 的独立环境（不继承 driver env），但实际执行
ServiceRunner、Popen 和临时 launch.sh；Git/sudo 与 Ray 调度使用 stub，
不访问 Merlin、不启动模型、不读写 task 编译产物。
示例：python3 -B tests/test_serve_controller_env.py -v
"""

from verify.resources import payload_file
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_serve_controller_oracle import load_module


class ControllerEnvironmentTest(unittest.TestCase):
    """通过真实子 shell 的输出验证端口、head IP 与 rank，而非只检查命令字符串。"""

    def setUp(self) -> None:
        """构造隔离源码树与 Ray worker 环境，所有子进程均登记后按进程组清理。"""
        temp = tempfile.TemporaryDirectory(prefix="controller-env-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "launch.sh").write_text(
            '#!/bin/bash\nset -eu\n'
            'printf "%s\\n" "${HARBOR_SERVICE_PORT:?HARBOR_SERVICE_PORT 未注入}" '
            '"$RAY_IP" "$HARBOR_SERVE_RANK" "$PORT0" > "$HARBOR_CAPTURE"\n'
            'exec sleep 30\n'
        )
        self.oracle = self.root / "oracle.sh"
        self.oracle.write_text("#!/bin/bash\ntrue\n")
        self.module = load_module()
        self.runner_class = self.module.ServiceRunner
        self.runners = []
        self.addCleanup(self._stop_runners)
        self.metadata = {
            "patch_sha256": hashlib.sha256(b"").hexdigest(),
            "base_commit": "fixture-commit",
            "baseline_tree": "fixture-tree",
        }
        self.enterContext(patch.object(self.module, "WORKDIR", self.root / "work"))
        # 只 stub 外部准备动作；_launch / Popen / launch wrapper 均真实执行。
        self.prepares = self.enterContext(patch.object(
            self.module.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0),
        ))
        self.enterContext(patch.object(
            self.module.subprocess, "check_output", return_value="fixture-commit\n"
        ))
        self.enterContext(patch.object(
            self.module, "worktree_tree", return_value="fixture-tree"
        ))
        self.enterContext(patch.object(
            self.module, "wait_for_ray_gcs", return_value="[fd00::1]:15000"
        ))
        owner = self

        class FakeRemoteRunner:
            def options(self, **options):
                """保存 Ray actor options；环境只从 runtime_env 显式装入。"""
                self.actor_options = options
                return self

            def remote(self, rank, repo):
                """建立缺失或陈旧端口的 worker，不从 driver 拷贝环境。"""
                runner = owner._new_runner(rank, repo)
                worker_env = owner._worker_env(rank)
                if rank % 2:
                    worker_env["HARBOR_SERVICE_PORT"] = "65000"
                worker_env.update(
                    self.actor_options.get("runtime_env", {}).get("env_vars", {})
                )

                def invoke(*args, **kwargs):
                    """在 worker 环境执行真实 apply_and_start 与子 shell。"""
                    with patch.dict(os.environ, worker_env, clear=True):
                        return runner.apply_and_start(*args, **kwargs)

                return SimpleNamespace(apply_and_start=SimpleNamespace(remote=invoke))

        nodes = [
            {"NodeID": f"node-{rank}", "Alive": True,
             "NodeManagerAddress": f"fd00::{rank + 1}",
             "Resources": {"head": 1} if rank == 0 else {"worker": 1}}
            for rank in (2, 0, 3, 1)
        ]
        self.ray_init = Mock()
        fake_ray = SimpleNamespace(
            remote=lambda **kwargs: lambda cls: FakeRemoteRunner(),
            init=self.ray_init, nodes=lambda: nodes, get=lambda results: results,
        )
        self.enterContext(patch.dict(sys.modules, {
            "ray": fake_ray,
            "ray.util": SimpleNamespace(),
            "ray.util.scheduling_strategies": SimpleNamespace(
                NodeAffinitySchedulingStrategy=lambda **kwargs: kwargs
            ),
        }))

    def _new_runner(self, rank, repo):
        """创建真实 runner 并登记，测试失败时也能回收已启动的子进程。"""
        runner = self.runner_class(rank, repo)
        self.runners.append(runner)
        return runner

    def _worker_env(self, rank):
        """每个 worker 有自己的 PORT0、旧 head IP 与旧 rank，不能拿来猜端口。"""
        return {
            "PATH": os.defpath, "PORT0": str(41000 + rank),
            "RAY_IP": "fd00::dead", "HARBOR_SERVE_RANK": "99",
            "HARBOR_CAPTURE": str(self.root / f"capture-{rank}"),
        }

    def _stop_runners(self) -> None:
        """只停止本测试登记的进程组，并关闭对应日志句柄。"""
        for runner in self.runners:
            runner.stop()
            if runner.log_file is not None:
                runner.log_file.close()
        self.runners.clear()

    def _assert_captures(self, count, port) -> None:
        """读取真实 shell 输出，验证所有 rank 使用同一可信服务端口。"""
        for rank in range(count):
            self.assertEqual(
                (self.root / f"capture-{rank}").read_text().splitlines(),
                [port, "fd00::1", str(rank), str(41000 + rank)],
            )

    def test_multinode_patch_and_oracle_use_trusted_head_port(self) -> None:
        """四个 worker（含 rank 0）从缺失/陈旧环境启动，patch 与 oracle 都需成功。"""
        for oracle in (False, True):
            with self.subTest(oracle=oracle), patch.dict(os.environ, {
                "PATH": os.defpath, "RAY_IP": "fd00::1",
                "HARBOR_SERVICE_PORT": "23456", "HARBOR_DIST_PORT": "23457",
                "PORT0": "12000",
            }, clear=True):
                try:
                    self.module.run_multi_node(
                        b"", {} if oracle else self.metadata, str(self.repo), 4, 16,
                        oracle_script=str(self.oracle) if oracle else "",
                    )
                    self._assert_captures(4, "23456")
                finally:
                    self._stop_runners()

    def test_multinode_oci_layer_ships_helper_to_every_worker(self) -> None:
        """OCI 模式的 Ray 多节点:worker 没有 /logs/verifier 上传件,应用器源码随 actor 参数下发并逐节点落盘。"""
        import gzip
        import json

        layer = gzip.compress(b"layer-bytes")
        metadata = {
            "layer_sha256": "sha256:" + hashlib.sha256(layer).hexdigest(),
            "diff_id": "sha256:" + hashlib.sha256(b"layer-bytes").hexdigest(),
            "base_commit": "fixture-commit", "baseline_tree": "fixture-tree",
        }
        real_helper = payload_file('oci_layer.py')
        helper_calls = []

        def check_output(argv, **kwargs):
            if "rev-parse" in argv:
                return "fixture-commit\n"
            helper_calls.append(argv)
            return json.dumps({"layer_sha256": metadata["layer_sha256"], "diff_id": metadata["diff_id"]})

        with patch.dict(os.environ, {
            "PATH": os.defpath, "RAY_IP": "fd00::1",
            "HARBOR_SERVICE_PORT": "23456", "HARBOR_DIST_PORT": "23457",
            "PORT0": "12000",
        }, clear=True), patch.object(self.module, "OCI_LAYER_SCRIPT", str(real_helper)), patch.object(
            self.module, "OCI_LAYER_TMP", self.root / "submit_oci.tar.gz",
        ), patch.object(self.module.subprocess, "check_output", side_effect=check_output):
            try:
                self.module.run_multi_node(layer, metadata, str(self.repo), 4, 16)
                self._assert_captures(4, "23456")
            finally:
                self._stop_runners()
        local_helper = self.root / "work/oci_layer.py"
        self.assertEqual(local_helper.read_text(), real_helper.read_text())
        self.assertEqual(len(helper_calls), 4)
        for argv in helper_calls:
            self.assertEqual(argv[:4], ["sudo", "-n", sys.executable, str(local_helper)])
        for rank in range(4):
            self.assertTrue((self.root / f"work/oci-layer-rank-{rank}.json").is_file())
        self.assertFalse(any("apply" in call.args[0] for call in self.prepares.call_args_list))

    def test_missing_head_port_fails_before_ray_even_with_port0(self) -> None:
        """可信端口缺失或为空时拒绝部署，不能从 PORT0 或默认 8000 兜底。"""
        for oracle in (False, True):
            for port in (None, ""):
                env = {"RAY_IP": "fd00::1", "PORT0": "12000", "HARBOR_DIST_PORT": "23457"}
                if port is not None:
                    env["HARBOR_SERVICE_PORT"] = port
                with self.subTest(oracle=oracle, port=port), patch.dict(
                    os.environ, env, clear=True
                ), patch.object(self.module, "wait_for_ray_gcs", side_effect=AssertionError(
                    "缺少可信端口时不得连接 Ray"
                )):
                    with self.assertRaisesRegex(RuntimeError, "HARBOR_SERVICE_PORT"):
                        self.module.run_multi_node(
                            b"", {} if oracle else self.metadata, str(self.repo), 4, 16,
                            oracle_script=str(self.oracle) if oracle else "",
                        )
        self.ray_init.assert_not_called()
        self.assertEqual(self.runners, [])

    def test_missing_dist_port_fails_before_ray(self) -> None:
        """缺少 HARBOR_DIST_PORT 时拒绝多节点部署，不能回退 5000。"""
        with patch.dict(os.environ, {
            "RAY_IP": "fd00::1", "PORT0": "12000", "HARBOR_SERVICE_PORT": "23456",
        }, clear=True), patch.object(self.module, "wait_for_ray_gcs", side_effect=AssertionError(
            "缺少分布式端口时不得连接 Ray"
        )):
            with self.assertRaisesRegex(RuntimeError, "HARBOR_DIST_PORT"):
                self.module.run_multi_node(b"", self.metadata, str(self.repo), 4, 16)
        self.ray_init.assert_not_called()
        self.assertEqual(self.runners, [])

    def test_single_node_patch_and_oracle_keep_inherited_port(self) -> None:
        """单节点仍使用本进程注入的服务端口，不经过 Ray 或改写为 PORT0。"""
        for oracle in (False, True):
            env = self._worker_env(0)
            env.update(HARBOR_SERVICE_PORT="34567", RAY_IP="fd00::1")
            with self.subTest(oracle=oracle), patch.dict(os.environ, env, clear=True), \
                    patch.object(self.module, "ServiceRunner", self._new_runner):
                try:
                    self.module.run_single_node(
                        b"", {} if oracle else self.metadata, str(self.repo),
                        oracle_script=str(self.oracle) if oracle else "",
                    )
                    self._assert_captures(1, "34567")
                finally:
                    self._stop_runners()
        self.ray_init.assert_not_called()


if __name__ == "__main__":
    unittest.main()

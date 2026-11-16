"""infer_verifier 对已退出 launch 进程的 fail-fast 与 serve_controller rc 契约。

参数：无。rc 探测部分用假环境驱动 `_raise_if_launch_exited`;包装命令部分
在临时目录真实执行 bash,验证 launch 退出后 rc 文件语义。

示例：
    python3 -m unittest tests.test_serve_failfast
"""

from __future__ import annotations

from verify.resources import payload_dir
import importlib.util
import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from verify.inference.infer_verifier import InferVerifier

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_serve_controller():
    """按 verifier 上传方式以独立模块加载 serve_controller;输出模块对象。"""
    path = payload_dir() / "serve_controller.py"
    spec = importlib.util.spec_from_file_location("serve_controller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verifier_with_probe(*outputs: object) -> tuple[InferVerifier, MagicMock]:
    """构造仅含 rc 探测依赖的 verifier;输入 exec 依次返回值,输出实例与环境。"""
    verifier = InferVerifier.__new__(InferVerifier)
    verifier.logger = MagicMock()
    environment = MagicMock()
    environment.exec = AsyncMock(
        side_effect=[
            value
            if isinstance(value, (Exception, MagicMock))
            else MagicMock(return_code=0, stdout=value, stderr="")
            for value in outputs
        ]
    )
    return verifier, environment


class LaunchExitFailFastTest(unittest.IsolatedAsyncioTestCase):
    """rc 文件出现且非零时立即抛根因,而不是等满 health 窗口。"""

    async def test_nonzero_rc_raises_with_log_tail(self) -> None:
        verifier, environment = _verifier_with_probe(
            "1", "RuntimeError: aclnnAddRmsNormBias not in libopapi.so"
        )
        with self.assertRaises(RuntimeError) as raised:
            await verifier._raise_if_launch_exited(environment)
        message = str(raised.exception)
        self.assertIn("rc=1", message)
        self.assertIn("aclnnAddRmsNormBias", message)

    async def test_running_marker_keeps_waiting(self) -> None:
        verifier, environment = _verifier_with_probe("RUNNING")
        await verifier._raise_if_launch_exited(environment)
        environment.exec.assert_awaited_once()

    async def test_zero_rc_defers_to_health_poll(self) -> None:
        # launch.sh 合法地允许 daemonize 后退出 0;此时服务可用性仍由
        # health poll 判定,不能误判为部署失败。
        verifier, environment = _verifier_with_probe("0")
        await verifier._raise_if_launch_exited(environment)
        environment.exec.assert_awaited_once()

    async def test_transport_error_skips_this_round(self) -> None:
        verifier, environment = _verifier_with_probe(RuntimeError("ssh flake"))
        await verifier._raise_if_launch_exited(environment)


class LaunchWrapperCommandTest(unittest.TestCase):
    """serve_controller 包装命令的真实 bash 语义。"""

    def test_failing_launch_writes_rc_and_log_survives(self) -> None:
        serve_controller = _load_serve_controller()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            launch = work / "launch.sh"
            launch.write_text("#!/bin/bash\necho boom-log\nexit 7\n")
            rc_path = work / "rank-0.rc"
            log_path = work / "rank-0.log"
            command = serve_controller.launch_wrapper_command(launch, rc_path)
            with log_path.open("wb") as log_file:
                subprocess.run(
                    ["bash", "-c", command],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=30,
                )
            self.assertEqual(rc_path.read_text().strip(), "7")
            self.assertIn("boom-log", log_path.read_text())

    def test_healthy_exec_style_launch_writes_no_rc_while_running(self) -> None:
        serve_controller = _load_serve_controller()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            launch = work / "launch.sh"
            # 模拟 oracle 姿势:exec 常驻服务进程,launch.sh 不返回。
            launch.write_text("#!/bin/bash\nexec sleep 5\n")
            rc_path = work / "rank-0.rc"
            command = serve_controller.launch_wrapper_command(launch, rc_path)
            process = subprocess.Popen(
                ["bash", "-c", command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                # 服务存活期间 rc 不出现——健康路径不会被误判为退出。
                self.assertIsNone(process.poll())
                self.assertFalse(rc_path.exists())
            finally:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()

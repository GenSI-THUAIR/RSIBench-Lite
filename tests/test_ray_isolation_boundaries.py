"""本地覆盖 Ray 启动竞态、硬截止时间和禁止自动建群的地址边界。

参数：无。真实 Popen/loopback 配合模拟 Ray 进程身份；Ray 模块仅用 stub。
示例：python -m unittest discover -s tests -p test_ray_isolation_boundaries.py -v
"""

import asyncio
import json
import os
import select
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from verify.delivery import patch_eval
from verify.resources import payload_module
guard = payload_module("ray_isolation.py")

PROBE = Path(guard.__file__)


@contextmanager
def listener_child():
    """启动本测试拥有的 Python socket 子进程，通过 stdin 控制何时 listen。"""
    code = (
        "import socket,sys\n"
        "s=socket.socket(); s.bind(('127.0.0.1',0))\n"
        "print(s.getsockname()[1],flush=True)\n"
        "sys.stdin.readline(); s.listen(); print('LISTEN',flush=True)\n"
        "sys.stdin.readline()\n"
    )
    with subprocess.Popen([sys.executable, "-u", "-c", code], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as child:
        try:
            if not select.select([child.stdout], [], [], 5)[0]:
                raise AssertionError("local socket child did not bind")
            yield child, int(child.stdout.readline())
        finally:
            if child.poll() is None:
                child.terminate()
            child.communicate(timeout=5)


def listen(child):
    """只通知本测试拥有的子进程进入监听阶段。"""
    child.stdin.write("listen\n")
    child.stdin.flush()
    if not select.select([child.stdout], [], [], 5)[0]:
        raise AssertionError("local socket child did not listen")
    assert child.stdout.readline().strip() == "LISTEN"


class StartupBoundaryTest(unittest.TestCase):
    """实际 socket 由子进程拥有；仅替换 Ray 进程枚举，不启动真实 Ray。"""

    def setUp(self):
        """本组只测 Ray/socket 阶段；controller 的真实进程契约由 launch 回归覆盖。"""
        self.enterContext(patch.object(guard, "_controller_launch_running", return_value=True))

    def test_gcs_visible_before_bind_then_raylet_ready(self):
        """缺进程、GCS 尚未监听、缺 Raylet 均明确 not-ready，最终才允许 ready。"""
        with listener_child() as (child, port):
            address = f"127.0.0.1:{port}"
            processes = []
            with patch.object(guard, "_ray_processes", return_value=processes):
                self.assertEqual(guard.local_member(address, 0),
                                 {"ready": False, "reason": "gcs_process_missing"})
                processes.append((str(child.pid), ["gcs_server", f"--gcs_server_port={port}"]))
                self.assertIsNone(guard._owns_listener(str(child.pid), port))
                self.assertEqual(guard.local_member(address, 0),
                                 {"ready": False, "reason": "gcs_listener_missing"})
                listen(child)
                self.assertTrue(guard._owns_listener(str(child.pid), port))
                self.assertEqual(guard.local_member(address, 0),
                                 {"ready": False, "reason": "raylet_process_missing"})
                processes.append(("999999", ["raylet", f"--gcs-address={address}",
                                             "--node_id=owned-node",
                                             f"--python_worker_command={sys.executable} worker.py"]))
                self.assertEqual(guard.local_member(address, 0)["node_id"], "owned-node")

    def test_foreign_listener_and_multiple_gcs_are_not_startup_not_ready(self):
        """监听属于子进程而非声明的 GCS PID 时硬拒绝，缺 GCS 也不能放过旧监听。"""
        with listener_child() as (child, port):
            listen(child)
            address = f"127.0.0.1:{port}"
            for processes in ([], [(str(os.getpid()), ["gcs_server", f"--gcs_server_port={port}"])]):
                with self.subTest(processes=processes), patch.object(guard, "_ray_processes", return_value=processes):
                    with self.assertRaisesRegex(RuntimeError, "not owned"):
                        guard.local_member(address, 0)
            processes = [(str(pid), ["gcs_server", f"--gcs_server_port={port}"])
                         for pid in (child.pid, os.getpid())]
            with patch.object(guard, "_ray_processes", return_value=processes):
                with self.assertRaisesRegex(RuntimeError, "Multiple GCS"):
                    guard.local_member(address, 0)
            with patch.object(guard.os, "readlink", side_effect=PermissionError("denied")):
                with self.assertRaises(PermissionError):
                    guard._owns_listener(str(child.pid), port)

    def test_foreign_raylet_and_proc_permission_errors_propagate(self):
        """外来 GCS 地址和 /proc 权限错误不能通过重试掩盖。"""
        with patch.object(guard, "_ray_processes", return_value=[
            ("123", ["raylet", "--gcs-address=127.0.0.2:12345"]),
        ]):
            with self.assertRaisesRegex(RuntimeError, "Foreign Raylet"):
                guard.local_member("127.0.0.1:12345", 0)
        with patch.object(Path, "iterdir", return_value=iter([Path("/proc/123")])), \
                patch.object(Path, "read_bytes", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                list(guard._ray_processes())


class AddressBoundaryTest(unittest.TestCase):
    """真实 CLI 子进程只会导入测试 stub，绝不启动 Ray。"""

    def test_invalid_targets_fail_before_importing_ray(self):
        """local/auto/URI/IPv6/非法端口在导入 Ray 前拒绝，不能触发建群。"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / "ray-imported"
            (root / "ray.py").write_text(
                "import os\nfrom pathlib import Path\n"
                "Path(os.environ['RAY_TEST_MARKER']).write_text('IMPORTED')\n"
                "raise AssertionError('Ray must not be imported for invalid target')\n"
            )
            env = dict(os.environ, PYTHONPATH=temp, RAY_TEST_MARKER=str(marker))
            for target in ("local", "auto", "", "127.0.0.1", "localhost:12345",
                           "ray://127.0.0.1:12345", "[::1]:12345", "127.0.0.1:0",
                           "127.0.0.1:65536", "127.0.0.1:-1", "127.0.0.1:１２３４５",
                           "0.0.0.0:12345", "224.0.0.1:12345"):
                with self.subTest(target=target), subprocess.Popen(
                    [sys.executable, str(PROBE), "cluster", target], env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                ) as child:
                    stdout, stderr = child.communicate(timeout=5)
                    self.assertNotEqual(child.returncode, 0)
                    self.assertIn("IPv4", stderr)
                    self.assertNotIn("HARBOR_RAY_ISOLATION=", stdout)
                    self.assertFalse(marker.exists())

    def test_explicit_target_reaches_stub_and_overrides_stale_environment(self):
        """合法地址原样传给只读 stub；旧 RAY_ADDRESS=local 也不能覆盖目标。"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / "ray-init"
            (root / "ray.py").write_text(
                "import json,os\nfrom pathlib import Path\n"
                "def init(**kwargs):\n"
                "    Path(os.environ['RAY_TEST_MARKER']).write_text(json.dumps([kwargs['address'],os.environ['RAY_ADDRESS']]))\n"
                "def nodes(): return []\n"
                "def shutdown(): pass\n"
            )
            env = dict(os.environ, PYTHONPATH=temp, RAY_ADDRESS="local", RAY_TEST_MARKER=str(marker))
            with subprocess.Popen([sys.executable, str(PROBE), "cluster", "127.0.0.1:12345"],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as child:
                stdout, stderr = child.communicate(timeout=5)
                self.assertEqual(child.returncode, 0, stderr)
                self.assertIn("HARBOR_RAY_ISOLATION=", stdout)
            self.assertEqual(json.loads(marker.read_text()), ["127.0.0.1:12345"] * 2)


class DeadlineBoundaryTest(unittest.IsolatedAsyncioTestCase):
    """可控业务时钟与真实 asyncio 计时分别验证截止时间和可取消等待。"""

    def setUp(self):
        """业务时钟单独替换，避免改动 asyncio 自己的 monotonic 时钟。"""
        self.clock = SimpleNamespace(now=0.0)
        self.fake_time = SimpleNamespace(monotonic=lambda: self.clock.now)
        self.calls = []
        self.nodes = [{"NodeID": pod, "Alive": True, "Resources": {"GPU": 8}}
                      for pod in ("head", "worker")]

    def probe(self, durations):
        """模拟每次远端等待，记录分配的剩余预算后推进业务时钟。"""
        steps = iter(durations)

        async def invoke(environment, pod, operation, address, **kwargs):
            self.calls.append((operation, kwargs["timeout_sec"]))
            self.clock.now += next(steps)
            if operation == "cluster":
                return {"nodes": self.nodes}
            return {"ready": True, "node_id": pod, "python": sys.executable}
        return invoke

    async def wait(self, budget=10):
        """只执行隔离等待，不启动 controller 或任何服务。"""
        await patch_eval._wait_for_ray_membership(object(), ["head", "worker"],
                                                   "127.0.0.1:12345", 8, budget)

    async def test_every_probe_receives_remaining_budget(self):
        """十秒总预算按 10/7/4 递减，而非每个探针重新获得三十秒。"""
        with patch.object(patch_eval, "time", self.fake_time), \
                patch.object(patch_eval, "_ray_probe", side_effect=self.probe([3, 3, 3])):
            await self.wait()
        self.assertEqual(self.calls, [("member", 10), ("member", 7), ("cluster", 4)])
        self.assertEqual(self.clock.now, 9)

    async def test_late_member_or_cluster_success_is_rejected(self):
        """即使异步替身忽略预算并返回成功，也不能在截止时间之后放行。"""
        for durations, count in (([11], 1), ([3, 3, 5], 3)):
            self.clock.now = 0
            self.calls.clear()
            with self.subTest(durations=durations), patch.object(patch_eval, "time", self.fake_time), \
                    patch.object(patch_eval, "_ray_probe", side_effect=self.probe(durations)):
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    await self.wait()
            self.assertEqual(len(self.calls), count)

    async def test_deadline_is_checked_after_membership_validation(self):
        """成功判定本身消耗完预算，也不得在末尾直接 return。"""
        def slow_validation(*args):
            self.clock.now += 10
            return True
        with patch.object(patch_eval, "time", self.fake_time), \
                patch.object(patch_eval, "_ray_probe", side_effect=self.probe([1, 1, 1])), \
                patch.object(guard, "validate_membership", side_effect=slow_validation):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                await self.wait()

    async def test_true_async_timeout_cancels_ignored_exec_timeout(self):
        """平台替身忽略 timeout_sec 时，总 asyncio 上限仍取消等待。"""
        cancelled = asyncio.Event()
        captured = []

        async def hung_exec(command, **kwargs):
            captured.append(kwargs["timeout_sec"])
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.set()

        started = time.perf_counter()
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            await patch_eval._wait_for_ray_membership(SimpleNamespace(exec=hung_exec),
                                                       ["head"], "127.0.0.1:12345", 8, 0.05)
        self.assertLess(time.perf_counter() - started, 1)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(captured, [1])

    async def test_only_explicit_startup_reasons_and_missing_nodes_retry(self):
        """三个明确启动原因及缺少预期节点可继续等，其余错误不转 not-ready。"""
        for reason in ("gcs_process_missing", "gcs_listener_missing", "raylet_process_missing"):
            rounds, clusters, sleeps = 0, 0, []
            self.clock.now = 0

            async def probe(environment, pod, operation, address, **kwargs):
                nonlocal rounds, clusters
                self.clock.now += 1
                if operation == "cluster":
                    clusters += 1
                    return {"nodes": self.nodes[:1] if clusters == 1 else self.nodes}
                if pod == "head":
                    rounds += 1
                    if rounds == 1:
                        return {"ready": False, "reason": reason}
                return {"ready": True, "node_id": pod, "python": sys.executable}

            async def sleep(delay):
                sleeps.append(delay)
                self.clock.now += delay

            with self.subTest(reason=reason), patch.object(patch_eval, "time", self.fake_time), \
                    patch.object(patch_eval, "_ray_probe", side_effect=probe), \
                    patch.object(patch_eval.asyncio, "sleep", side_effect=sleep):
                await self.wait(100)
            self.assertEqual(len(sleeps), 2)

    async def test_errors_and_unrecognized_not_ready_never_retry(self):
        """权限、外来节点和非 owned 监听错误立即向上传播，不 sleep。"""
        for error in (PermissionError("denied"), RuntimeError("Foreign NodeID"), RuntimeError("not owned")):
            probe = AsyncMock(side_effect=error)
            with self.subTest(error=error), patch.object(patch_eval, "_ray_probe", probe), \
                    patch.object(patch_eval.asyncio, "sleep", new_callable=AsyncMock) as sleep:
                with self.assertRaises(type(error)):
                    await self.wait()
                self.assertEqual(probe.await_count, 1)
                sleep.assert_not_awaited()
        with patch.object(patch_eval, "_ray_probe", new=AsyncMock(return_value={"ready": False})), \
                patch.object(patch_eval.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            with self.assertRaisesRegex(RuntimeError, "Unrecognized"):
                await self.wait()
            sleep.assert_not_awaited()

    async def test_sleep_is_capped_by_remaining_budget(self):
        """节点未齐时末尾 sleep 也只允许使用剩余的一秒。"""
        sleeps = []

        async def probe(*args, **kwargs):
            self.clock.now += 1
            return {"ready": False, "reason": "raylet_process_missing"}

        async def sleep(delay):
            sleeps.append(delay)
            self.clock.now += delay

        with patch.object(patch_eval, "time", self.fake_time), \
                patch.object(patch_eval, "_ray_probe", side_effect=probe), \
                patch.object(patch_eval.asyncio, "sleep", side_effect=sleep):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                await self.wait(3)
        self.assertEqual(sleeps, [1])
        self.assertEqual(self.clock.now, 3)


if __name__ == "__main__":
    unittest.main()

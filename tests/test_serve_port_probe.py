"""服务端口归属校验：探针脚本双层判定 + verifier 接线。

参数：无。探针测试在本进程真实监听随机端口后以子进程运行脚本；接线测试用
假 environment，不连接真实 Merlin。

为什么存在：Merlin GPU pod 用宿主机网络，health 200 可能来自同宿主机其它
job 的服务（0902 llada 实录：自家实例 EADDRINUSE 秒退，gate/benchmark 给
别人的 Qwen3 模型打了分且 deploy_success=1）。另外共享 netns 下
/proc/net 是全机表、单次读取可能漏行（0902 深夜首版探针据此误杀健康部署），
所以 nonce 端到端仲裁必须能在 socket 表未命中时救回 OWNED。

示例：
  PYTHONPATH=. python3 -m unittest tests.test_serve_port_probe
"""

from __future__ import annotations

from verify.resources import payload_dir
import inspect
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from harbor.environments.base import ExecResult

from verify.delivery import patch_eval
from verify.inference.infer_verifier import InferVerifier
from verify.delivery.patch_eval import ForeignPortOwner, ServiceNotListening, PortProbeError, assert_port_owned

PROBE = payload_dir() / "serve_port_probe.py"
# 单测不需要对抗真实宿主机的表遍历竞态,把重试压到毫秒级。
FAST_ENV = {
    **os.environ,
    "PORT_PROBE_TABLE_ATTEMPTS": "2",
    "PORT_PROBE_TABLE_RETRY_SEC": "0.05",
    "PORT_PROBE_NONCE_WAIT_SEC": "0.3",
}


def _run_probe(
    port: int, pid_file: Path, *extra: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PROBE), str(port), str(pid_file), *extra],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
        env=FAST_ENV,
    )


class ServePortProbeScriptTest(unittest.TestCase):
    """脚本对真实 LISTEN socket 的双层判定。"""

    def _pid_file(self, pid: int) -> Path:
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "rank-0.pid"
        path.write_text(f"{pid}\n")
        return path

    def _sleep_child(self) -> subprocess.Popen[bytes]:
        child = subprocess.Popen(["sleep", "30"])
        self.addCleanup(child.wait)
        self.addCleanup(child.kill)
        return child

    def test_owned_listener_ipv4(self) -> None:
        """监听者在 pid 树内（本进程自己）→ OWNED、exit 0。"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = sock.getsockname()[1]
            proc = _run_probe(port, self._pid_file(os.getpid()))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("OWNED", proc.stdout)

    def test_owned_listener_ipv6(self) -> None:
        """tcp6 表也要能解析（launch.sh 绑 [::]:8000 走这条）。"""
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
            sock.listen(1)
            port = sock.getsockname()[1]
            proc = _run_probe(port, self._pid_file(os.getpid()))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_foreign_listener_fails_with_diag(self) -> None:
        """监听者不在 pid 树内且无 nonce 仲裁 → FOREIGN、exit 3、附诊断。"""
        child = self._sleep_child()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = sock.getsockname()[1]
            # pid 树根指向 sleep 子进程；监听者（本进程)不在其树内。
            proc = _run_probe(port, self._pid_file(child.pid))
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        self.assertIn("FOREIGN", proc.stdout)
        self.assertIn("DIAG", proc.stdout)

    def test_no_listener(self) -> None:
        """端口无监听（health 后服务消失）且无仲裁 → exit 2。"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        proc = _run_probe(port, self._pid_file(os.getpid()))
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)

    def test_missing_pid_file(self) -> None:
        """pid 文件缺失（controller 没启动过服务）→ exit 4。"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = sock.getsockname()[1]
            proc = _run_probe(port, Path("/nonexistent/rank-0.pid"))
        self.assertEqual(proc.returncode, 4, proc.stdout + proc.stderr)

    def _http_server(self, log_path: Path, *, route_only: bool = False) -> tuple[subprocess.Popen[bytes], int]:
        """起一个把 access log 写进 *log_path* 的本地 HTTP 服务。

        route_only=True 模拟 Ray Serve 形态:access log 只记 `<request-id> -- GET / 404`,不记原始路径。
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_sock:
            probe_sock.bind(("127.0.0.1", 0))
            port = probe_sock.getsockname()[1]
        log_file = open(log_path, "wb")  # noqa: SIM115 - 交给 addCleanup
        self.addCleanup(log_file.close)
        if route_only:
            command = [sys.executable, "-c", _ROUTE_ONLY_SERVER, str(port)]
        else:
            command = [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"]
        server = subprocess.Popen(command, stdout=log_file, stderr=log_file)
        self.addCleanup(server.wait)
        self.addCleanup(server.kill)
        for _ in range(50):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("本地 HTTP 服务未就绪")
        return server, port

    def test_nonce_arbitration_rescues_owned(self) -> None:
        """socket 表未命中(pid 树根错位模拟漏行)时,nonce 落进服务日志应判 OWNED。

        这是 0902 深夜误杀事故的回归测试:共享宿主机 netns 下 /proc/net
        单次读取漏掉自家 LISTEN 行,不能只凭表就定罪。
        """
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        log_path = tmp / "rank-0.log"
        _, port = self._http_server(log_path)
        child = self._sleep_child()
        proc = _run_probe(
            port,
            self._pid_file(child.pid),
            "--endpoint",
            f"http://127.0.0.1:{port}",
            "--log",
            str(log_path),
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("via=nonce", proc.stdout)

    def test_nonce_in_request_id_header_rescues_route_only_access_log(self) -> None:
        """access log 只记路由 + 请求 ID(Ray Serve 形态)时,X-Request-ID 里的 nonce 应判 OWNED。

        0910 verl-tinker 实录:Ray Serve 把探针请求记成 `GET / 404`,路径里的 nonce 永远不落日志。
        """
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        log_path = tmp / "rank-0.log"
        _, port = self._http_server(log_path, route_only=True)
        child = self._sleep_child()
        proc = _run_probe(
            port,
            self._pid_file(child.pid),
            "--endpoint",
            f"http://127.0.0.1:{port}",
            "--log",
            str(log_path),
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("via=nonce", proc.stdout)
        self.assertNotIn("harbor-port-probe", log_path.read_text().split("\n")[0].split(" -- ")[-1])

    def test_nonce_miss_confirms_foreign(self) -> None:
        """应答方的 access log 不是我们的日志 → nonce 仲裁不过、FOREIGN。"""
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        server_log = tmp / "other.log"
        our_log = tmp / "rank-0.log"
        our_log.write_text("")
        _, port = self._http_server(server_log)
        child = self._sleep_child()
        proc = _run_probe(
            port,
            self._pid_file(child.pid),
            "--endpoint",
            f"http://127.0.0.1:{port}",
            "--log",
            str(our_log),
        )
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        self.assertIn("FOREIGN", proc.stdout)


# 模拟 Ray Serve 的 access log:只记请求 ID 与匹配路由(固定 "/"),不记原始路径。
_ROUTE_ONLY_SERVER = """
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(404); self.end_headers()
        rid = self.headers.get("X-Request-ID", "generated-id")
        print(f"default_TinkerServer replica {rid} -- GET / 404 1.0ms", flush=True)
    def log_message(self, *args): pass
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


class _FakeServeEnvironment:
    """回放 exec 结果序列；记录上传与命令供断言。"""

    def __init__(self, exec_results: list[ExecResult | Exception]) -> None:
        self.exec_results = list(exec_results)
        self.commands: list[str] = []
        self.uploads: list[str] = []

    async def upload_file(self, source: object, destination: str) -> None:
        self.uploads.append(destination)

    async def exec(self, command: str, **kwargs: object) -> ExecResult:
        self.commands.append(command)
        outcome = self.exec_results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


_OWNED = ExecResult(
    stdout="OWNED port=8000 via=socket-table attempt=1", stderr="", return_code=0
)
_FOREIGN = ExecResult(
    stdout="FOREIGN port=8000 nonce仲裁=nonce 请求失败（疑同宿主机其它 job）",
    stderr="",
    return_code=3,
)
_SSH_FLAKE = ExecResult(stdout="", stderr="Error: ssh probe failed", return_code=1)

_LOGGER = logging.getLogger(__name__)


class AssertPortOwnedTest(unittest.IsolatedAsyncioTestCase):
    """assert_port_owned 的通过/失败/瞬时抖动边界。"""

    async def test_owned_passes_and_uploads_probe(self) -> None:
        environment = _FakeServeEnvironment([_OWNED])
        await assert_port_owned(
            cast(Any, environment), _LOGGER, endpoint="http://[::1]:8000", port=8000
        )
        self.assertEqual(
            environment.uploads, ["/logs/verifier/serve_port_probe.py"]
        )
        # nonce 仲裁参数必须透传到探针命令。
        self.assertIn("serve_port_probe.py 8000", environment.commands[0])
        self.assertIn("--endpoint", environment.commands[0])
        self.assertIn("--log", environment.commands[0])

    async def test_foreign_raises(self) -> None:
        environment = _FakeServeEnvironment([_FOREIGN])
        with self.assertRaisesRegex(ForeignPortOwner, "同宿主机"):
            await assert_port_owned(cast(Any, environment), _LOGGER, port=8000)

    async def test_transient_ssh_failure_retries(self) -> None:
        environment = _FakeServeEnvironment([_SSH_FLAKE, _OWNED])
        with patch("harbor_envs.abstract_mlplat.retry.asyncio.sleep", new=AsyncMock()):
            await assert_port_owned(cast(Any, environment), _LOGGER, port=8000)
        self.assertEqual(len(environment.commands), 2)

    async def test_no_listener_and_missing_pid_are_not_foreign_owner(self) -> None:
        for code, error in ((2, ServiceNotListening), (4, PortProbeError), (1, PortProbeError)):
            environment = _FakeServeEnvironment([ExecResult(stdout="probe evidence", stderr="", return_code=code)])
            with self.subTest(code=code), self.assertRaises(error):
                await assert_port_owned(cast(Any, environment), _LOGGER, port=8000)


class VerifierWiringTest(unittest.TestCase):
    """结构性锁定：归属校验必须接在就绪判定之后、打分之前,且带 endpoint。"""

    def test_infer_probe_runs_after_health(self) -> None:
        source = inspect.getsource(InferVerifier._start_patched_service)
        self.assertIn("assert_port_owned", source)
        self.assertGreater(
            source.index("assert_port_owned"),
            source.rindex("_wait_for_health"),
        )
        # nonce 仲裁依赖 endpoint;丢了它,表遍历漏行会重演 0902 误杀。
        self.assertIn("endpoint=endpoint", source)
        # 归属未知保留基础设施异常，不转成候选 reward=0。
        self.assertIn("ServiceEndpointError", source)


    def test_probe_script_is_shipped(self) -> None:
        """探针必须存在于可信 payload 清单(verifier_script 缺失即 fail loud)。"""
        self.assertTrue(
            patch_eval.verifier_script(patch_eval.PORT_PROBE_NAME).is_file()
        )


if __name__ == "__main__":
    unittest.main()

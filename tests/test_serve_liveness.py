"""服务存活探测的契约：必须按 **pid** 判活，不能按进程名。

这条路径 2026-08-26 修过一次，两个方向的错都踩到了，所以固化下来：

1. `launch.sh` 结尾是 `exec`（rank 0 换成 server 进程，其余 rank 换成常驻
   等待），exec 之后进程的 cmdline 不再含 `bash .../launch.sh` —— 按名字
   pgrep **必然找不到**，看门狗会把健康的服务误判成已死。
2. 反过来，远端执行的是 `bash -c '… pgrep -f "bash .../launch.sh" …'`，
   模式串就在执行命令自己的 cmdline 里，pgrep 会 **self-match** 从而永远
   报「活着」—— 看门狗变成彻底的空操作，于是服务早就死了却还要空等
   `readiness.timeout_sec`（本题是 5400 秒）才报「未就绪」。

exec 不改 pid，所以 controller 落一个 pid 文件、宿主机 `kill -0` 判活，
对两种 rank 都成立，也不会 self-match。
"""

from __future__ import annotations

from verify.resources import payload_dir
import asyncio
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (
    REPO_ROOT,
    payload_dir(),
):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import serve_controller


@dataclass
class FakeExecResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


@dataclass
class FakeEnvironment:
    """记录探测命令，并按脚本化的答案回应。"""

    verdict: str = "alive"
    execs: list[tuple[str, str | None]] = field(default_factory=list)

    def role_pod_names(self) -> list[str]:
        return ["head-0", "worker-0"]

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        user: str | None = None,
        pod_name: str | None = None,
    ) -> FakeExecResult:
        self.execs.append((command, pod_name))
        if "tail -c" in command:
            return FakeExecResult(stdout="…最后 3000 字节日志…")
        return FakeExecResult(stdout=f"{self.verdict}\n")


CONFIG = {
    "serve_node_count": 2,
    "service_label": "SkyRL Tinker API",
    "readiness": {"path": "/api/v1/healthz", "timeout_sec": 5400},
}

# 看门狗每 30 秒探一次;测试里把等待压成 0。必须先抓住**原始** asyncio.sleep，
# 否则替身会调用被自己替换掉的那个 sleep，直接递归爆栈。
_REAL_SLEEP = asyncio.sleep


def _no_wait(_delay: float) -> object:
    return _REAL_SLEEP(0)


class ControllerPidFileTest(unittest.TestCase):
    def test_pid_path_sits_next_to_rank_log(self) -> None:
        runner = serve_controller.ServiceRunner(1, "/skyrl-workspace/SkyRL")
        self.assertEqual(runner.pid_path, serve_controller.WORKDIR / "rank-1.pid")
        self.assertEqual(
            runner.log_path.parent,
            runner.pid_path.parent,
            "pid 与 log 必须同目录，收尾时按 rank 一起抓",
        )

    def test_stop_removes_pid_file(self) -> None:
        runner = serve_controller.ServiceRunner(0, "/skyrl-workspace/SkyRL")
        # 没起过进程时 stop() 也不该因为 pid 文件不存在而抛。
        runner.pid_path = Path("/tmp/.harbor-test-rank-0.pid")
        runner.pid_path.write_text("4242\n")
        runner.stop()
        self.assertFalse(
            runner.pid_path.exists(), "stop() 必须清掉 pid 文件，否则下一轮误判活着"
        )
        runner.stop()  # 幂等：文件已不在，不能抛




if __name__ == "__main__":
    unittest.main()

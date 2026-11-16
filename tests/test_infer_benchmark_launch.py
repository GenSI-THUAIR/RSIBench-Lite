"""验证评测启动命令真正脱离前台，成功/失败均保存实际退出码。
参数：无。示例：.venv/bin/python -m unittest tests.test_infer_benchmark_launch
使用真实本地shell和子进程，只写临时目录，不连接Merlin。
"""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest

from verify.inference.infer_verifier import _detached_benchmark_launch


class DetachedBenchmarkLaunchTest(unittest.TestCase):
    def test_launcher_returns_before_benchmark_and_records_real_exit(self):
        for expected_rc in (0, 7):
            with self.subTest(rc=expected_rc), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                release = root / "release"
                prefix = root / "test benchmark"
                rc_path = Path(str(prefix) + ".rc")
                pid_path = Path(str(prefix) + ".pid")
                body = (
                    "from pathlib import Path\nimport time,sys\n"
                    f"release=Path({str(release)!r})\n"
                    "while not release.exists(): time.sleep(0.02)\n"
                    "print('benchmark completed', flush=True)\n"
                    f"sys.exit({expected_rc})\n"
                )
                command = shlex.join([sys.executable, "-c", body])
                rc_path.write_text("99\n")
                try:
                    launched = subprocess.run(
                        ["bash", "-c", _detached_benchmark_launch(command, str(prefix))],
                        capture_output=True, text=True, timeout=3,
                    )
                    self.assertEqual(launched.returncode, 0, launched.stderr)
                    self.assertFalse(rc_path.exists(), "不能把启动rc当成评测rc")
                    pid = int(pid_path.read_text())
                    os.kill(pid, 0)
                finally:
                    release.touch()
                    deadline = time.monotonic() + 5
                    while not rc_path.exists() and time.monotonic() < deadline:
                        time.sleep(0.02)
                self.assertEqual(rc_path.read_text().strip(), str(expected_rc))
                self.assertIn("benchmark completed", Path(str(prefix) + ".log").read_text())


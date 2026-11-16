"""阶段预算和服务退出的真实本地进程回归；同套用例可在独立云端 CPU Job 运行。

参数：无。示例：python -m unittest tests.test_verifier_phases
不加载模型，不生成 reward，不接触其他服务。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import shlex
import tempfile
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request
from unittest.mock import AsyncMock

from verify.grading.phase_budget import phase
from verify.grading.service_liveness import check_launches, run_with_service_watch, ServiceExitedError
from verify.grading.service_liveness import check_ray_launches, ServiceProbeError


class PhaseTest(unittest.IsolatedAsyncioTestCase):
    async def test_connection_refused_reports_service_root_cause(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = root / 'server.py'
            server.write_text("import socket, pathlib\n"
                "s=socket.socket(); s.bind(('127.0.0.1', 0)); s.listen()\n"
                f"pathlib.Path({str(root / 'port')!r}).write_text(str(s.getsockname()[1]))\n"
                "c,_=s.accept(); c.recv(4096); c.sendall(b'HTTP/1.0 200 OK\\r\\nContent-Length: 2\\r\\n\\r\\nok'); c.close(); s.close()\n"
                "print('fixture engine terminated', flush=True); raise SystemExit(7)\n")
            command = (f'echo $$ > {root}/rank-0.pid; {shlex.quote(sys.executable)} {server}; '
                       f'echo $? > {root}/rank-0.rc')
            with (root/'rank-0.log').open('wb') as log:
                process = await asyncio.create_subprocess_exec('bash', '-c', command, stdout=log)
            async def execute(command, **kwargs):
                proc = await asyncio.create_subprocess_exec('bash', '-c', command.replace('/workspace/deploy', str(root)),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                out, err = await proc.communicate()
                return SimpleNamespace(return_code=proc.returncode, stdout=out.decode(), stderr=err.decode())
            try:
                async with asyncio.timeout(5):
                    while not (root/'port').exists():
                        await asyncio.sleep(.01)
                endpoint = 'http://127.0.0.1:' + (root/'port').read_text()
                def request():
                    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(endpoint, timeout=2) as response:
                        return response.status
                async def driver():
                    self.assertEqual(await asyncio.to_thread(request), 200)
                    await process.wait()
                    return await asyncio.to_thread(request)
                with self.assertRaises(ServiceExitedError) as caught:
                    await run_with_service_watch(driver,
                        lambda: check_launches(SimpleNamespace(exec=execute), [None], label='http-fault'),
                        evidence_path=root/'failure.json', interval_sec=10)
                self.assertIsInstance(caught.exception.__cause__, urllib.error.URLError)
                self.assertIn('rc=7', str(caught.exception))
                self.assertIn('fixture engine terminated', str(caught.exception))
            finally:
                if process.returncode is None:
                    process.terminate()
                await process.wait()

    async def test_ray_worker_failure_not_hidden_by_healthy_head(self):
        rows = [{'rank': 0, 'running': True, 'launch_rc': None},
                {'rank': 1, 'running': False, 'launch_rc': 17, 'tail': 'worker failed'}]
        env = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=0,
            stdout='HARBOR_RAY_STATUS='+json.dumps(rows), stderr='')))
        with self.assertRaisesRegex(ServiceExitedError, 'rank-1.*rc=17.*worker failed'):
            await check_ray_launches(env, 'python3', 2)
        env.exec.return_value.stdout = 'HARBOR_RAY_STATUS=' + json.dumps(rows[:1])
        with self.assertRaises(ServiceProbeError):
            await check_ray_launches(env, 'python3', 2)

    async def test_child_cannot_extend_parent_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(TimeoutError):
                async with phase(root, "total", .1):
                    async with phase(root, "driver", 5):
                        await asyncio.sleep(10)
            rows = {row["phase"]: row for row in map(lambda p: json.loads(p.read_text()), root.glob("*.json"))}
            self.assertEqual(rows["total"]["status"], "TIMED_OUT")
            self.assertLessEqual(rows["driver"]["effective_budget_sec"], .1)
            self.assertLess(rows["total"]["elapsed_sec"], 1)

    async def test_parallel_profiles_preserve_every_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            async def work(name):
                async with phase(root, name, 1):
                    await asyncio.sleep(.01)
            await asyncio.gather(work("gate"), work("performance"))
            records = [json.loads(p.read_text()) for p in root.glob("*.json")]
            self.assertEqual({r["phase"] for r in records}, {"gate", "performance"})
            self.assertTrue(all(r["status"] == "COMPLETED" for r in records))

    async def test_cleanup_requires_and_enforces_its_own_limit(self):
        work = Path(__file__).resolve().parents[1] / '.agents/workdir'
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'finite budget'):
                async with phase(root, 'invalid-cleanup', cleanup=True):
                    self.fail('unbounded cleanup entered')
            with self.assertRaises(TimeoutError):
                async with phase(root, 'cleanup', .02, cleanup=True):
                    await asyncio.Event().wait()
            row = json.loads(next(root.glob('*.json')).read_text())
            self.assertEqual(row['status'], 'TIMED_OUT')
            self.assertTrue(row['cleanup'])

    async def test_cancellation_records_current_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, ready = Path(tmp), asyncio.Event()
            async def work():
                async with phase(root, "artifact", 10):
                    ready.set()
                    await asyncio.sleep(10)
            job = asyncio.create_task(work())
            await ready.wait()
            job.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await job
            self.assertEqual(json.loads(next(root.glob("*.json")).read_text())["status"], "CANCELLED")

    async def test_service_exit_reports_real_rc_and_cancels_driver(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Bash 模拟真实 launch wrapper，rc 与日志来自独立进程而非固定回执。
            command = f"echo $$ > {root}/rank-0.pid; sleep .3; echo EADDRINUSE-fixture; echo 7 > {root}/rank-0.rc"
            with (root / "rank-0.log").open("wb") as log:
                process = await asyncio.create_subprocess_exec("bash", "-c", command, stdout=log)
            async def execute(command, **kwargs):
                command = command.replace("/workspace/deploy", str(root))
                proc = await asyncio.create_subprocess_exec("bash", "-c", command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                out, err = await proc.communicate()
                return SimpleNamespace(return_code=proc.returncode, stdout=out.decode(), stderr=err.decode())
            env = SimpleNamespace(exec=execute)
            cancelled = asyncio.Event()
            async def work():
                try:
                    await asyncio.sleep(10)
                finally:
                    cancelled.set()
            try:
                while not (root / "rank-0.pid").exists():
                    await asyncio.sleep(.01)
                with self.assertRaisesRegex(ServiceExitedError, "rc=7.*EADDRINUSE-fixture"):
                    await run_with_service_watch(work, lambda: check_launches(env, [None], label="cloud-fixture"), evidence_path=root / "failure.json", interval_sec=.05)
                self.assertTrue(cancelled.is_set())
                record = json.loads((root / "failure.json").read_text())
                self.assertEqual(record["status"], "INFRA_ERROR")
                self.assertEqual(record["responsibility"], "undetermined")
                self.assertFalse((root / "reward.json").exists())
            finally:
                if process.returncode is None:
                    process.terminate()
                await process.wait()


if __name__ == "__main__":
    unittest.main()

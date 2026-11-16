"""真实本地解包/执行与篡改拒绝测试；不申请远端资源。

示例：python -m pytest tests/test_trusted_bundle.py。
"""
import asyncio
import io
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

from harbor.environments.base import ExecResult
from verify.delivery.patch_eval import upload_trusted_bundle


class TrustedBundleTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.agents/workdir')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.remote = self.root/'remote'
        self.remote.mkdir()
        self.source = self.root/'driver.py'
        self.source.write_text("print('trusted-source-v1')\n")

    async def execute(self, command, **kwargs):
        proc = await asyncio.create_subprocess_exec('bash','-c',command,
            stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        out,err = await proc.communicate()
        return ExecResult(return_code=proc.returncode,stdout=out.decode(),stderr=err.decode())

    async def test_delivered_driver_executes_exact_source(self):
        uploads = []
        async def upload(source, destination):
            uploads.append(str(destination))
            shutil.copyfile(source,destination)
        await upload_trusted_bundle(SimpleNamespace(upload_file=upload,exec=self.execute),
            [self.source],self.root/'bundle',str(self.remote))
        self.assertEqual(len(uploads),1)
        self.assertEqual((self.remote/'driver.py').read_bytes(),self.source.read_bytes())
        proc = await asyncio.create_subprocess_exec(sys.executable,str(self.remote/'driver.py'),stdout=asyncio.subprocess.PIPE)
        stdout,_ = await proc.communicate()
        self.assertEqual((proc.returncode,stdout),(0,b'trusted-source-v1\n'))

    async def test_valid_archive_with_changed_source_is_rejected(self):
        async def corrupt(source,destination):
            with tarfile.open(source,'r:gz') as original,tarfile.open(destination,'w:gz') as changed:
                for member in original.getmembers():
                    data = original.extractfile(member).read()
                    if member.name=='driver.py':
                        data=b"print('corrupted-driver')\n"
                    member.size=len(data)
                    changed.addfile(member,io.BytesIO(data))
        with self.assertRaisesRegex(RuntimeError,'哈希校验失败'):
            await upload_trusted_bundle(SimpleNamespace(upload_file=corrupt,exec=self.execute),
                [self.source],self.root/'bundle',str(self.remote))

    async def test_duplicate_names_fail_before_upload(self):
        async def upload(*args):
            self.fail('invalid source list must not be uploaded')
        with self.assertRaises(ValueError):
            await upload_trusted_bundle(SimpleNamespace(upload_file=upload),
                [self.source,self.source],self.root/'bundle',str(self.remote))

"""验证显式基线先于候选 OCI 层应用，包含真实文件写入、覆写和 whiteout。

调用：python -m unittest tests.test_evaluation_baseline；无远端任务。
"""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace

from verify.delivery.evaluation_baseline import build, install, validate_hashes
from verify.delivery.oci_layer import apply_oci_layer


class EvaluationBaselineTest(unittest.TestCase):



    @staticmethod
    def write_manifest(root, manifest):
        target = root / 'manifest.json'
        target.write_text(json.dumps(manifest))
        return target

    def test_candidate_can_replace_or_whiteout_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, repo = root / 'source', root / 'rootfs/engine'
            source.mkdir()
            repo.mkdir(parents=True)
            (source / 'nested').mkdir()
            for name in ('nested/replace.py', 'nested/delete.py'):
                (source / name).write_bytes(b'baseline')
            hashes = {n: hashlib.sha256(b'baseline').hexdigest() for n in ('nested/replace.py', 'nested/delete.py')}
            install(build(source, hashes), repo)
            layer = root / 'candidate.tar.gz'
            with tarfile.open(layer, 'w:gz') as archive:
                for name, content in [('engine/nested/replace.py', b'candidate'), ('engine/nested/.wh.delete.py', b'')]:
                    entry = tarfile.TarInfo(name)
                    entry.size = len(content)
                    archive.addfile(entry, io.BytesIO(content))
            apply_oci_layer(layer, root / 'rootfs')
            self.assertEqual((repo / 'nested/replace.py').read_bytes(), b'candidate')
            self.assertFalse((repo / 'nested/delete.py').exists())

    def test_reject_hash_mismatch_paths_and_symlinks_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'file.py').write_bytes(b'original')
            digest = hashlib.sha256(b'original').hexdigest()
            for name in ('../x', '/x', '.git', 'dir/../x', 'dir//x', 'dir/.git/x', 'dir/', 'dir\\x'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    validate_hashes({name: digest})
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                build(root, {'file.py': '0' * 64})
            manifest = build(root, {'file.py': digest})
            manifest['files']['file.py']['data'] = 'Y2hhbmdlZA=='
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                install(manifest, root)
            self.assertEqual((root / 'file.py').read_bytes(), b'original')
            (root / 'link.py').symlink_to(root / 'file.py')
            with self.assertRaises(ValueError):
                build(root, {'link.py': digest})

    def test_parent_symlink_conflicts_and_bad_mode_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, repo = root / 'source', root / 'repo'
            (source / 'nested').mkdir(parents=True)
            repo.mkdir()
            (source / 'first.py').write_bytes(b'new')
            (repo / 'first.py').write_bytes(b'old')
            (source / 'nested/second.py').write_bytes(b'second')
            hashes = {n: hashlib.sha256((source / n).read_bytes()).hexdigest()
                      for n in ('first.py', 'nested/second.py')}
            manifest = build(source, hashes)
            (repo / 'nested').symlink_to(source / 'nested', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlinks'):
                install(manifest, repo)
            with self.assertRaisesRegex(ValueError, 'symlinks'):
                build(repo, {'nested/second.py': hashes['nested/second.py']})
            self.assertEqual((repo / 'first.py').read_bytes(), b'old')
            (repo / 'nested').unlink()
            (repo / 'nested').write_bytes(b'file')
            with self.assertRaisesRegex(ValueError, 'parent'):
                install(manifest, repo)
            (repo / 'nested').unlink()
            manifest['files']['nested/second.py']['mode'] = 0o4755
            with self.assertRaisesRegex(ValueError, 'mode'):
                install(manifest, repo)
            self.assertEqual((repo / 'first.py').read_bytes(), b'old')
            self.assertFalse((repo / 'nested').exists())
            with self.assertRaisesRegex(ValueError, 'conflict'):
                validate_hashes({'nested': '0' * 64, 'nested/file': '0' * 64})


if __name__ == '__main__':
    unittest.main()

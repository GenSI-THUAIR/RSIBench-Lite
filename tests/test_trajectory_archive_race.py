"""验证轨迹采集跨await持有旧账本时不会回退其他源的哈希。
参数：无。示例：.venv/bin/python -m unittest tests.test_trajectory_archive_race
只写临时夹具，不改任何运行记录。
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from harbor_envs.abstract_mlplat.trajectory import _RawArchive, reconcile_trial_trajectory


class ArchiveInterleavingTest(unittest.TestCase):
    def test_stale_collector_preserves_other_source_update(self):
        with tempfile.TemporaryDirectory() as directory:
            agent_dir = Path(directory)
            first = _RawArchive(agent_dir, "trial")
            first.store("opencode", "opencode.txt", b"first\n", "native")
            first.store("atif", "trajectory.json", b"{}", "native")
            second = _RawArchive(agent_dir, "trial")
            first.store("opencode", "opencode.txt", b"first\nsecond\n", "native")
            # 模拟second在远端await之后返回；它此前加载的是first更新前的账本。
            second.store("atif", "trajectory.json", b'{"steps":[]}', "native")
            manifest = json.loads(first.path.read_bytes())
            for entry in manifest["sources"].values():
                self.assertEqual(
                    hashlib.sha256((first.root / entry["raw_path"]).read_bytes()).hexdigest(),
                    entry["sha256"],
                )

    def test_stale_error_update_preserves_new_source(self):
        with tempfile.TemporaryDirectory() as directory:
            first = _RawArchive(Path(directory), "trial")
            first.store("opencode", "opencode.txt", b"first\n", "native")
            second = _RawArchive(Path(directory), "trial")
            first.store("atif", "trajectory.json", b"{}", "native")
            second.collection_error("remote", RuntimeError("SSH failed"))
            manifest = json.loads(first.path.read_bytes())
            self.assertIn("atif:trajectory.json", manifest["sources"])
            self.assertIn("remote", manifest["collection_errors"])

    def test_hash_corruption_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _RawArchive(root, "trial")
            archive.store("atif", "trajectory.json", b"{}", "native")
            (archive.root / "atif/trajectory.json").write_bytes(b'{"changed":true}')
            with self.assertRaisesRegex(ValueError, "raw archive hash mismatch"):
                reconcile_trial_trajectory(root)


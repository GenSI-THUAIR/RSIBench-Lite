"""核验锁定的官方发行包未被本仓修改；python3 -m unittest tests.test_upstream_dependencies。"""
import base64
import hashlib
import inspect
from importlib.metadata import distribution
from pathlib import Path
import unittest


class OfficialDependencyTest(unittest.TestCase):
    def test_official_packages_match_installed_wheel_record(self):
        for name, version in (("harbor", "0.21.0"), ("harbor-rewardkit", "0.1.7")):
            with self.subTest(name=name):
                dist = distribution(name)
                self.assertEqual(dist.version, version)
                self.assertIsNone(dist.read_text("direct_url.json"))
                count = 0
                for entry in dist.files:
                    if entry.suffix != ".py" or entry.hash is None:
                        continue
                    self.assertEqual(entry.hash.mode, "sha256")
                    actual = base64.urlsafe_b64encode(hashlib.sha256(Path(dist.locate_file(entry)).read_bytes()).digest()).rstrip(b"=").decode()
                    self.assertEqual(actual, entry.hash.value, str(entry))
                    count += 1
                self.assertGreater(count, 30)

"""验证 domain 源码发布后仍能独立运行，保留协议身份、数据和内容完整性。"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from verify import resources


@pytest.fixture(autouse=True)
def fresh_bundle(monkeypatch):
    monkeypatch.setattr(resources, '_PINNED', None)


def test_payload_inventory_and_publication():
    assert len(resources.PAYLOADS) == 17
    bundle = resources.payload_dir()
    for name in resources.PAYLOADS:
        assert (bundle / name).read_bytes() == resources.source_file(name).read_bytes()
    assert resources.source_file("video_eval.py").parent.name == "video"
    assert resources.source_file("oci_layer.py").parent.name == "delivery"


def test_process_pins_files_modules_and_fingerprint_across_source_updates(tmp_path, monkeypatch):
    script = tmp_path / "example.py"
    script.write_text("value = 1\n")
    monkeypatch.setattr(resources, "SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(resources, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(resources, "PAYLOADS", {"example.py": "example.py"})
    before = resources.evaluator_sha256()
    module = resources.payload_module('example.py')
    assert resources.evaluator_sha256() == before
    script.write_text("value = 2\n")
    assert resources.evaluator_sha256() == before
    assert resources.payload_module('example.py') is module
    assert resources.payload_file('example.py').read_text() == 'value = 1\n'
    # 模拟新的 worker 进程；新进程应看到新版，而不是全局永远缓存旧版。
    monkeypatch.delitem(sys.modules, 'example')
    monkeypatch.setattr(resources, '_PINNED', None)
    assert resources.evaluator_sha256() != before
    assert resources.payload_module('example.py').value == 2
    monkeypatch.delitem(sys.modules, 'example')


def test_atomic_publication_and_corruption_fail_fast(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "REPO_ROOT", tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: resources.payload_dir(), range(8)))
    assert len(set(paths)) == 1
    (paths[0] / "oci_layer.py").write_text("corrupt")
    with pytest.raises(ValueError, match="内容不匹配"):
        resources.payload_dir()


@pytest.mark.parametrize("name", ["../README.md", "/etc/passwd", "__init__.py", "missing.py"])
def test_unknown_resource_is_rejected(name):
    with pytest.raises(KeyError):
        resources.payload_file(name)


def test_cross_domain_sibling_imports_and_adjacent_data_in_fresh_process(tmp_path):
    bundle = resources.payload_dir()
    code = """
import json
from pathlib import Path
import clean_benchmark
import gsm8k_eval
assert clean_benchmark.gsm8k_eval is gsm8k_eval
root = Path(clean_benchmark.__file__).parent
assert (root / 'gsm8k-official-test.jsonl').is_file()
print(json.dumps({'root': str(root)}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(bundle)},
        check=True, capture_output=True, text=True,
    )
    assert Path(json.loads(result.stdout)["root"]) == bundle


def test_host_protocols_share_flat_module_identity():
    bench = resources.payload_module("clean_benchmark.py")
    gate = resources.payload_module("gsm8k_eval.py")
    assert bench.gsm8k_eval is gate
    assert resources.payload_module("clean_benchmark.py") is bench


def test_conflicting_module_fails_before_import(monkeypatch):
    import types
    fake = types.ModuleType("oci_layer")
    fake.__file__ = "/untrusted/oci_layer.py"
    monkeypatch.setitem(sys.modules, "oci_layer", fake)
    with pytest.raises(ImportError, match="名称冲突"):
        resources.payload_module("serve_controller.py")


def test_existing_bundle_path_cannot_be_shadowed(tmp_path):
    (tmp_path / "oci_layer.py").write_text("raise RuntimeError('untrusted module executed')")
    code = """
import sys
from pathlib import Path
from verify.resources import payload_dir, payload_module
bundle = payload_dir()
sys.path.insert(0, str(bundle))
sys.path.insert(0, sys.argv[1])
module = payload_module('oci_layer.py')
assert Path(module.__file__).parent == bundle
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                   cwd=resources.REPO_ROOT, check=True, capture_output=True, text=True)

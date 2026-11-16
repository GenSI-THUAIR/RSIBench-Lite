"""公开源码扫描、无历史导出及完整 Modal 题面的真实本地验证。"""
import io
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import pytest
import yaml

from schemas.run_config import prepare_run
from schemas.build import load_yaml
from scripts.release.export_source import export, findings

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_lite_protocol_and_asset_identities():
    records = json.loads((ROOT / 'assets/lite-protocol-sha256.json').read_text())
    assert set(records) == {p.parent.name for p in (ROOT / 'tasks').glob('*/task.yaml')}
    for name, record in records.items():
        source = ROOT / 'tasks' / name
        scoring = load_yaml(source / 'task.yaml')['verifier']['scoring']
        encoded = json.dumps(scoring, sort_keys=True, separators=(',', ':')).encode()
        assert hashlib.sha256(encoded).hexdigest() == record['scoring_sha256']
        for filename, digest in record['files'].items():
            assert hashlib.sha256((source / filename).read_bytes()).hexdigest() == digest


def test_scanner_checks_nested_archives_without_echoing_secret():
    secret = b'sk-' + b'0123456789abcdefghijklmnopqrstuvwxyz'
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        entry = tarfile.TarInfo('config.txt')
        entry.size = len(secret)
        archive.addfile(entry, io.BytesIO(secret))
    result = findings('fixture.tar.gz', stream.getvalue())
    assert result == ['fixture.tar.gz!config.txt:1: credential token']
    assert secret.decode() not in str(result)
    assert findings('source.py', b'from package.hf_transformers_monkey_patch import apply') == []
    assert findings('fixture.json.gz', gzip.compress(secret)) == ['fixture.json.gz!decompressed:1: credential token']


def test_public_docs_reject_private_paths_and_dated_narratives():
    assert findings('README.md', b'Result: 642.288852 tokens/s.') == []
    assert findings('README.md', b'Updated 20' + b'26-11-16')
    assert findings('README.md', b'See evidence/sol-20' + b'261116/results.json')
    assert findings('README.md', b'/home/' + b'researcher/results.json')
    assert findings('receipt.json', b'/opt/' + b'tiger/private/result.json')
    # Protocol timestamp fixtures and public container paths remain valid code.
    assert findings('test_clock.py', b'timestamp = "20' + b'26-11-16"') == []
    assert findings('README.md', b'Submit /logs/submission/submit_oci.tar.gz') == []


def test_export_uses_index_without_git_history_or_untracked_credentials(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    (repo / 'README.md').write_text('public\n')
    (repo / '.env').write_text('private local authentication\n')
    subprocess.run(['git', 'add', 'README.md'], cwd=repo, check=True)
    output = tmp_path / 'release.tar.gz'
    assert export(output, root=repo) == 1
    with tarfile.open(output) as archive:
        assert archive.getnames() == ['rsibench-lite/README.md']
        assert archive.extractfile(archive.getmembers()[0]).read() == b'public\n'
    subprocess.run(['git', 'add', '.env'], cwd=repo, check=True)
    with pytest.raises(ValueError, match='Private/runtime path'):
        export(root=repo)


def test_modal_compilation_keeps_full_task_protocol(tmp_path):
    config = yaml.safe_load((ROOT / 'configs/harbor-modal-cpu.yaml').read_text())
    compiled = prepare_run(config, directory=tmp_path)
    assert len(compiled['tasks']) == 5
    for task in compiled['tasks']:
        text = (Path(task['path']) / 'instruction.md').read_text()
        assert len(text) > 5000
        assert 'NVIDIA H200' in text
        assert 'submit_oci.tar.gz' in text
        assert 'Merlin' not in text
        assert 'http://[::1]' not in text

"""扫描公开源码并导出不含 Git 历史、凭据和运行产物的源码包。

参数：可选 --output 指定 tar.gz；省略时只扫描。
示例：python scripts/release/export_source.py --output .agents/workdir/rsibench-lite.tar.gz
只包含 Git 索引中的现存普通文件。新增公开文件须先 git add；不跟随符号链接。
扫描是发布前检查，不能替代对新文件及二进制内容的人工审阅。
"""
from __future__ import annotations

import argparse
import gzip
import io
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {'.git', '.agents', '.venv', '.env', '.modal.toml', '__pycache__',
            '.pytest_cache', 'jobs', 'logs', 'runs'}
PATTERNS = {
    'credential token': re.compile(rb'\b(?:sk-|plat_|ghp_|github_pat_|AKIA|as-|ak-)[A-Za-z0-9_-]{20,}|\bhf_[A-Za-z0-9]{30,}\b'),
    'private key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'credential URL': re.compile(rb'https?://[^\s/@:]+:[^\s/@]+@'),
    'literal credential': re.compile(rb'(?i)(?:api[_-]?key|token_secret|password)\s*[\"\x27]?\s*[:=]\s*[\"\x27]([A-Za-z0-9_+./~=!-]{24,})[\"\x27]'),
    'internal endpoint': re.compile(rb'(?:hub|bytedpypi|super-relay)\.byted\.org|suwen\.chatfold\.pro'),
    'removed platform integration': re.compile(rb'harbor_envs\.byted_merlin|HARBOR_MERLIN[_]|merlin[-]cli'),
    'private workspace or storage path': re.compile(rb'/(?:opt)/(?:tiger)/|/(?:remote_rootfs)/|hdfs[:]/[/]'),
}
DOC_PATTERNS = {
    'dated experiment documentation': re.compile(rb'\b20\d{2}[-/]\d{2}[-/]\d{2}\b|(?<![A-Za-z0-9])20\d{6}(?![A-Za-z0-9])'),
    'personal home path in documentation': re.compile(rb'/home/[^/\s`]+/|/Users/[^/\s`]+/'),
}


def findings(name: str, data: bytes) -> list[str]:
    """Return locations and categories, never matched credential values."""
    result = []
    patterns = dict(PATTERNS)
    if name.endswith(('.md', '.rst', '.md.tmpl')):
        patterns.update(DOC_PATTERNS)
    for category, pattern in patterns.items():
        for match in pattern.finditer(data):
            line = data[:match.start()].count(b'\n') + 1
            result.append(f'{name}:{line}: {category}')
    if name.endswith(('.json.gz', '.jsonl.gz', '.txt.gz')):
        try:
            result.extend(findings(f'{name}!decompressed', gzip.decompress(data)))
        except (OSError, EOFError):
            result.append(f'{name}: invalid gzip stream')
    elif name.endswith(('.tar.gz', '.tgz', '.tar')):
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive.getmembers():
                if member.isfile():
                    if member.size > 16 * 1024 * 1024:
                        result.append(f'{name}: oversized nested archive member requires review')
                    else:
                        result.extend(findings(f'{name}!{member.name}', archive.extractfile(member).read()))
    return result


def source_files(root: Path = ROOT) -> list[Path]:
    entries = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    files = []
    for name in sorted(set(entries) - {''}):
        path = root / name
        if path.is_symlink():
            raise ValueError(f'Source export refuses symlink: {name}')
        if not path.exists():
            continue
        if set(PurePosixPath(name).parts) & EXCLUDED or Path(name).name.startswith('.env.'):
            raise ValueError(f'Private/runtime path tracked: {name}')
        if not path.is_file():
            raise ValueError(f'Source export requires ordinary files: {name}')
        files.append(path)
    if not files:
        raise ValueError('No indexed source files')
    return files


def export(output: Path | None = None, *, root: Path = ROOT) -> int:
    files = source_files(root)
    problems = [item for path in files for item in findings(str(path.relative_to(root)), path.read_bytes())]
    if problems:
        raise ValueError('Release scan failed:\n' + '\n'.join(problems))
    if output is not None:
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, 'w:gz') as archive:
            for path in files:
                entry = archive.gettarinfo(str(path), arcname=f'rsibench-lite/{path.relative_to(root)}')
                entry.uid = entry.gid = 0
                entry.uname = entry.gname = ''
                entry.mtime = 0
                with path.open('rb') as stream:
                    archive.addfile(entry, stream)
    return len(files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(f'PASS: scanned {export(args.output)} source files; no detected release blockers')

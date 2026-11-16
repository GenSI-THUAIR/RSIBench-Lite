"""准备与安装显式声明的评测基线文件；只在 OCI 交付层之前使用。

宿主调用 build(directory, hashes) 生成受哈希约束的包；远端以 root 调用
python evaluation_baseline.py manifest.json /engine/repo，输出逐文件回执。
支持仓库内相对路径的普通文件，拒绝路径穿越、链接和文件/目录冲突。
不隐式扫描整个 environment/；文件清单及 SHA256 由任务声明。
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
import stat
import sys


def validate_hashes(hashes):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('evaluation_baseline must be a nonempty file/SHA256 mapping')
    for name, digest in hashes.items():
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)*', name)
                or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)):
            raise ValueError('Invalid evaluation baseline filename/SHA256')
        if any(str(parent) in hashes for parent in Path(name).parents if str(parent) != '.'):
            raise ValueError(f'Baseline file/directory conflict: {name}')


def checked_path(root: Path, name: str) -> Path:
    """逐级拒绝链接；尚不存在的父目录仅在整包校验通过后创建。"""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f'Baseline root must be a directory: {root}')
    path = root
    for part in Path(name).parts:
        if path.exists() and not path.is_dir():
            raise ValueError(f'Baseline parent is not a directory: {path}')
        path = path / part
        if path.is_symlink():
            raise ValueError(f'Baseline path must not contain symlinks: {path}')
    return path


def build(directory: Path, hashes: dict) -> dict:
    validate_hashes(hashes)
    files = {}
    for name, digest in hashes.items():
        path = checked_path(directory, name)
        if not path.is_file():
            raise ValueError(f'Baseline must be a regular file: {path}')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'Baseline source SHA256 mismatch: {name}')
        files[name] = {'sha256': digest, 'data': base64.b64encode(data).decode('ascii'),
                       'mode': stat.S_IMODE(path.stat().st_mode) & 0o777}
    return {'schema': 'evaluation-baseline/2', 'files': files}


def install(manifest: dict, repo: Path) -> dict:
    if manifest.get('schema') != 'evaluation-baseline/2':
        raise ValueError('Unknown evaluation baseline schema')
    files = manifest['files']
    validate_hashes({name: entry['sha256'] for name, entry in files.items()})
    decoded = {}
    # 全包验证成功之后才写入；不跟随现有文件链接。
    for name, entry in files.items():
        target = checked_path(repo, name)
        data = base64.b64decode(entry['data'], validate=True)
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'Baseline payload SHA256 mismatch: {name}')
        if target.exists() and not target.is_file():
            raise ValueError(f'Baseline target is not a regular file: {name}')
        if type(entry['mode']) is not int or not 0 <= entry['mode'] <= 0o777:
            raise ValueError(f'Invalid baseline mode: {name}')
        decoded[name] = data
    receipt = {'schema': 'evaluation-baseline-receipt/1', 'files': {}}
    for name, data in decoded.items():
        path = repo / name
        before = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(files[name]['mode'])
        receipt['files'][name] = {'before_sha256': before, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'mode': stat.S_IMODE(path.stat().st_mode)}
    return receipt


if __name__ == '__main__':
    print(json.dumps(install(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))))

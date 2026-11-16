"""冻结公开 Hugging Face 模型的文件身份，不下载权重。

参数：模型仓库、revision、输出 JSON。revision 可用 main，输出始终固定为 commit。
示例：python3 scripts/prepare/freeze_hf_manifest.py inclusionAI/LLaDA2.1-mini main manifest.json。
大文件取 Hub 的 LFS SHA256；只读取不超过 32 MiB 的 Git 文件并核验 blob SHA1。
"""
import argparse
import hashlib
import json
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_url


def freeze(repository, revision):
    info = HfApi().model_info(repository, revision=revision, files_metadata=True)
    files = []
    for entry in sorted(info.siblings, key=lambda item: item.rfilename):
        if entry.lfs is not None:
            digest = entry.lfs.sha256
        else:
            if entry.size > 32 * 1024 * 1024:
                raise ValueError(f'Git file too large for metadata preparation: {entry.rfilename}')
            response = requests.get(hf_hub_url(repository, entry.rfilename, revision=info.sha), timeout=90)
            response.raise_for_status()
            data = response.content
            blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            if len(data) != entry.size or blob != entry.blob_id:
                raise ValueError(f'Git blob identity mismatch: {entry.rfilename}')
            digest = hashlib.sha256(data).hexdigest()
        files.append({'path': entry.rfilename, 'bytes': entry.size, 'sha256': digest})
    if not files:
        raise ValueError('Empty model repository')
    return {'repository': repository, 'revision': info.sha, 'files': files,
            'total_bytes': sum(item['bytes'] for item in files)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository')
    parser.add_argument('revision')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    manifest = freeze(args.repository, args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'revision': manifest['revision'], 'files': len(manifest['files']),
                      'manifest_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))

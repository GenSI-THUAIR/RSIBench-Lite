"""把冻结评测数据发布到 Modal Volume，校验后才写就绪标记，不申请 GPU。

参数：task.yaml；可选 --source 指向与清单相同布局的本地数据目录。
示例：python3 scripts/prepare/prepare_modal_data.py tasks/infer-qwen3vl-32b/task.yaml。
必须用 --source 指定与冻结清单一致的数据目录；不会访问任何内部数据服务。
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import modal
from harbor_envs.modal.asset_verify import validate_manifest
from schemas.build import load_yaml


def checked_buffer(stream, item):
    output = io.BytesIO()
    digest = hashlib.sha256()
    while chunk := stream.read(4 * 1024 * 1024):
        if output.tell() + len(chunk) > item['bytes']:
            raise ValueError(f'Input exceeds frozen size: {item["path"]}')
        output.write(chunk)
        digest.update(chunk)
    if output.tell() != item['bytes'] or digest.hexdigest() != item['sha256']:
        raise ValueError(f'Input identity mismatch: {item["path"]}')
    output.seek(0)
    return output


def prepare(task, source):
    descriptor = load_yaml(task)['platforms']['modal']['data']
    raw = (task.parent / descriptor['manifest']).read_bytes()
    manifest = json.loads(raw)
    validate_manifest(manifest)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != descriptor['manifest_sha256']:
        raise ValueError('Task and data manifest disagree')
    if manifest['total_bytes'] > 128 * 1024 * 1024:
        raise ValueError('Data exceeds the bounded 128 MiB transfer allowance')
    volume = modal.Volume.from_name(descriptor['volume'], create_if_missing=True)
    try:
        marker = json.loads(b''.join(volume.read_file('.rsibench-ready.json')))
    except FileNotFoundError:
        marker = None
    if marker is not None:
        if marker.get('verified') is not True or marker.get('manifest_sha256') != digest:
            raise ValueError('Volume contains a different or invalid manifest')
        print(json.dumps({'volume': descriptor['volume'], 'status': 'already prepared'}))
        return
    for item in manifest['files']:
        path = (source / item['path']).resolve(strict=True)
        if not path.is_relative_to(source.resolve(strict=True)):
            raise ValueError(f'Input escapes source directory: {item["path"]}')
        with path.open('rb') as stream:
            data = checked_buffer(stream, item)
        with data, volume.batch_upload() as batch:
            batch.put_file(data, item['path'])
        remote_digest = hashlib.sha256()
        size = 0
        for chunk in volume.read_file(item['path']):
            size += len(chunk)
            remote_digest.update(chunk)
        if size != item['bytes'] or remote_digest.hexdigest() != item['sha256']:
            raise ValueError(f'Published data identity mismatch: {item["path"]}')
        print('verified', item['path'], flush=True)
    marker = {'verified': True, 'manifest_sha256': digest,
              'file_count': len(manifest['files']), 'total_bytes': manifest['total_bytes']}
    with volume.batch_upload() as batch:
        batch.put_file(io.BytesIO((json.dumps(marker) + '\n').encode()), '.rsibench-ready.json')
    if json.loads(b''.join(volume.read_file('.rsibench-ready.json'))) != marker:
        raise ValueError('Published readiness marker differs')
    print(json.dumps({'volume': descriptor['volume'], **marker}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', type=Path, metavar='task.yaml')
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    with modal.enable_output():
        prepare(args.task, args.source)

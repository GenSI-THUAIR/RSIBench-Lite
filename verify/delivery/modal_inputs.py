"""在可信 CPU Benchmark 中读取冻结输入；只下载 tokenizer/评测数据，不下载模型权重。

参数：runtime.json、输出收据。示例：python3 modal_inputs.py runtime.json /logs/verifier/inputs.json。
Modal 凭据仅在可信 Benchmark 的这一次执行中注入，不传给 Agent。
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import modal


async def prepare(runtime, output):
    records=[]
    for group in runtime['inputs']:
        volume=modal.Volume.from_name(group['volume'])
        async with asyncio.timeout(120):
            marker=json.loads(b''.join([part async for part in volume.read_file.aio('.rsibench-ready.json')]))
        if marker.get('verified') is not True or marker['manifest_sha256'] != group['manifest_sha256']:
            raise ValueError('Input Volume readiness does not match frozen manifest')
        files=[item for item in group['files'] if group['kind']=='data' or
               not item['path'].endswith(('.safetensors','.bin','.pt','.pth'))]
        if sum(item['bytes'] for item in files)>128*1024*1024:
            raise ValueError('CPU input transfer exceeds the bounded metadata/data allowance')
        for item in files:
            destination=Path(group['mount'])/item['path']
            destination.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256()
            size=0
            async with asyncio.timeout(300):
                with destination.open('wb') as stream:
                    async for chunk in volume.read_file.aio(item['path']):
                        stream.write(chunk)
                        size+=len(chunk)
                        digest.update(chunk)
            if size!=item['bytes'] or digest.hexdigest()!=item['sha256']:
                raise ValueError(f"Input content mismatch: {item['path']}")
            record={'path':str(destination),'bytes':size,'sha256':digest.hexdigest()}
            if destination.suffix == '.jsonl':
                with destination.open() as stream:
                    record['records']=sum(1 for line in stream if line.strip())
            records.append(record)
            print('verified',destination,flush=True)
    output.write_text(json.dumps({'verified':True,'files':records},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    asyncio.run(prepare(json.loads(args.runtime.read_text()),args.output))

"""在 CPU sandbox 从固定公开 HF revision 准备模型，只读使用前逐文件验 hash。

参数：task.yaml；--output 指定小型本地回执目录。
示例：python3 scripts/prepare/prepare_modal_model.py tasks/infer-qwen3p8-27b/task.yaml。
权重只写入 Modal Volume，不落控制机磁盘；准备阶段始终不传 GPU 参数。
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import modal
from schemas.build import load_yaml
from harbor_envs.modal.asset_verify import validate_manifest

READY_FILE = ".rsibench-ready.json"
PREPARE_CODE = '''import hashlib, json, os, sys
from pathlib import Path
from huggingface_hub import snapshot_download
from asset_verify import verify_assets
raw = Path("/tmp/model-manifest.json").read_bytes()
manifest = json.loads(raw)
print("Downloading frozen revision", manifest["revision"], flush=True)
snapshot_download(repo_id=manifest["repository"], revision=manifest["revision"], local_dir="/model", allow_patterns=[f["path"] for f in manifest["files"]], max_workers=4)
result = verify_assets(manifest, Path("/model"), progress=lambda n,p:print(f"verified {n}/{len(manifest['files'])}: {p}",flush=True))
result["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
Path("/model/.rsibench-ready.json").write_text(json.dumps(result,sort_keys=True)+"\\n")
print(json.dumps(result),flush=True)
'''


async def prepare(args):
    runtime = load_yaml(args.runtime)['platforms']['modal']
    model = runtime['model']
    args.manifest = args.runtime.parent / model['manifest']
    args.volume = model['volume']
    if args.output is None:
        args.output = ROOT / '.agents/workdir/modal-model-preparation' / args.volume
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    validate_manifest(manifest)
    if (hashlib.sha256(raw).hexdigest() != model['manifest_sha256']
            or manifest['repository'] != model['repository']
            or manifest['revision'] != model['revision']):
        raise ValueError('Runtime and frozen model manifest disagree')
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("Hugging Face revision must be an immutable 40-digit commit")
    manifest_sha = hashlib.sha256(raw).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"started_at": time.time(), "stage": "prepare", "gpu": None,
               "volume": args.volume, "manifest_sha256": manifest_sha}
    volume = modal.Volume.from_name(args.volume, create_if_missing=True)
    try:
        async with asyncio.timeout(90):
            existing = b"".join([chunk async for chunk in volume.read_file.aio(READY_FILE)])
    except FileNotFoundError:
        existing = None
    if existing is not None:
        data = json.loads(existing)
        if data.get("manifest_sha256") != manifest_sha or data.get("verified") is not True:
            raise ValueError("Volume already has a different or invalid asset manifest")
        receipt.update(stage="already prepared", readiness=data)
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
        return
    finished = asyncio.Event()
    sandbox = None

    async def monitor():
        while not finished.is_set():
            (args.output / "status.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(f"stage={receipt['stage']} elapsed={time.time()-receipt['started_at']:.0f}s", flush=True)
            try:
                await asyncio.wait_for(finished.wait(), 30)
            except TimeoutError:
                pass

    monitor_task = asyncio.create_task(monitor())
    try:
        async with asyncio.timeout(7200):
            app = await modal.App.lookup.aio("rsibench-modal-preparation", create_if_missing=True)
            image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==1.27.0")
            sandbox = await modal.Sandbox.create.aio(
                "sleep", "infinity", app=app, image=image, cpu=2, memory=4096,
                timeout=7500, volumes={"/model": volume},
                name="rsibench-model-prepare-" + uuid.uuid4().hex[:10],
            )
            receipt.update(stage="download and verify", sandbox_id=sandbox.object_id)
            await sandbox.filesystem.copy_from_local.aio(args.manifest, "/tmp/model-manifest.json")
            await sandbox.filesystem.copy_from_local.aio(ROOT / "harbor_envs/modal/asset_verify.py", "/tmp/asset_verify.py")
            process = await sandbox.exec.aio("python3", "-u", "-c", PREPARE_CODE, workdir="/tmp")

            async def copy_log(stream, filename):
                with (args.output / filename).open("w") as log:
                    async for line in stream:
                        log.write(line)
                        log.flush()

            await asyncio.gather(copy_log(process.stdout, "prepare.stdout"), copy_log(process.stderr, "prepare.stderr"))
            receipt["returncode"] = await process.wait.aio()
            if receipt["returncode"] != 0:
                raise RuntimeError(f"Model preparation failed; see {args.output / 'prepare.stderr'}")
    finally:
        receipt["stage"] = "cleanup"
        try:
            if sandbox is not None:
                await asyncio.shield(sandbox.terminate.aio())
                await asyncio.shield(sandbox.wait.aio(raise_on_termination=False))
                receipt["terminal_status"] = await sandbox.poll.aio()
        finally:
            finished.set()
            await monitor_task
            receipt["finished_at"] = time.time()
            (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    async with asyncio.timeout(90):
        readiness = json.loads(b"".join([chunk async for chunk in volume.read_file.aio(READY_FILE)]))
    if readiness.get("verified") is not True or readiness.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("Published volume readiness did not match the frozen manifest")
    receipt.update(stage="verified", readiness=readiness)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path, metavar="task.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with modal.enable_output():
        asyncio.run(prepare(args))


if __name__ == "__main__":
    main()

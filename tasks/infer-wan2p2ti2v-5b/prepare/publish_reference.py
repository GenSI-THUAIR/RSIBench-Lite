#!/usr/bin/env python3
"""校验原始reference及每份视频hash后原子发布唯一只读HDFS参考目录，不包含warmup。

参数：已完成的本地reference目录、HDFS挂载内目标目录。
示例：python publish_reference.py /path/to/reference /mnt/rsibench/wan22-artifacts/h20-reference
"""
import json
import os
import shutil
import sys
from pathlib import Path

from video_eval import _sha256

source, target = map(Path, sys.argv[1:])
manifest_path = source / "reference-manifest.json"
manifest = json.loads(manifest_path.read_text())
measurement = json.loads((source / "measurement.json").read_text())
if len(manifest["cases"]) != 4 or not all(case["contract"]["passed"] for case in measurement["cases"]):
    raise ValueError("reference完整四case媒体合约未通过")
if target.exists():
    raise FileExistsError("禁止覆盖已发布reference")
staging = target.with_name(target.name + ".publishing")
staging.mkdir(parents=True, exist_ok=False)
for case in manifest["cases"]:
    relative = Path(case["reference"])
    if relative.name != str(relative):
        raise ValueError("reference只接受已冻结纯文件名")
    path = source / relative
    if _sha256(path) != case["sha256"]:
        raise ValueError("reference本地hash不匹配")
    destination = staging / relative
    shutil.copyfile(path, destination)
    if _sha256(destination) != case["sha256"]:
        raise ValueError("reference上传hash不匹配")
    destination.chmod(0o444)
shutil.copyfile(manifest_path, staging / manifest_path.name)
(staging / manifest_path.name).chmod(0o444)
staging.chmod(0o555)
os.replace(staging, target)
print(json.dumps({"reference_manifest_sha256": _sha256(target / manifest_path.name),
                  "measurement_sha256": _sha256(source / "measurement.json"),
                  "target": str(target), "cases": manifest["cases"]}), flush=True)

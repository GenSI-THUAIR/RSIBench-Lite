"""按冻结清单流式校验资产，不加载模型、不申请 GPU。

参数：manifest JSON、资产根目录。示例：python3 asset_verify.py manifest.json /model。
只依赖标准库；文件必须是根目录内的普通文件，内容以 SHA-256 和字节数核对。
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def validate_manifest(manifest):
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("Asset manifest must contain a nonempty files list")
    paths = set()
    for item in files:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) != item["path"] or not relative.parts:
            raise ValueError(f"Invalid asset path: {item['path']!r}")
        if item["path"] in paths:
            raise ValueError(f"Duplicate asset: {item['path']}")
        paths.add(item["path"])
        if type(item["bytes"]) is not int or item["bytes"] < 0:
            raise ValueError("Asset byte counts must be nonnegative integers")
        if not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Asset SHA-256 must contain 64 lowercase hex digits")
    if sum(item["bytes"] for item in files) != manifest["total_bytes"]:
        raise ValueError("Asset total_bytes does not match the file list")


def verify_assets(manifest, root: Path, progress=None):
    validate_manifest(manifest)
    root = root.resolve(strict=True)
    for index, item in enumerate(manifest["files"], 1):
        path = root / item["path"]
        if path.is_symlink() or not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"Asset is not a regular file inside root: {item['path']}")
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"Asset size mismatch: {item['path']}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Asset SHA-256 mismatch: {item['path']}")
        if progress:
            progress(index, item["path"])
    return {"verified": True, "file_count": len(manifest["files"]), "total_bytes": manifest["total_bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    print(json.dumps(verify_assets(manifest, args.root)))


if __name__ == "__main__":
    main()

"""定位可信评测资源，并生成保留远端文件名的平铺运行目录。

源码按 domain 维护；运行目录仅包含清单中的脚本和数据，供本地子进程与远端上传共用。
示例：payload_file("clean_benchmark.py")；payload_module("video_eval.py")。
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(__file__).resolve().parent
_GROUPS = {
    "delivery": "serve_controller oci_layer serve_port_probe ray_isolation modal_inputs ray_service_probe evaluation_baseline",
    "inference": "clean_benchmark fixed_window_benchmark run_official_bench vendored_bench_serving",
    "text": "gsm8k_eval gpqa_eval longbench_eval",
    "vision": "mmmu_eval",
    "video": "video_eval",
}
PAYLOADS = {
    f"{name}.py": f"{domain}/{name}.py"
    for domain, names in _GROUPS.items() for name in names.split()
}
PAYLOADS.update({
    "gsm8k-official-test.jsonl": "text/gsm8k-official-test.jsonl",
})
_PINNED: tuple[Path, dict[str, bytes]] | None = None
_PIN_LOCK = threading.RLock()


def source_file(name: str) -> Path:
    """只接受清单中的平铺文件名，未知名称直接报错。"""
    path = SOURCE_ROOT / PAYLOADS[name]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def evaluator_sha256() -> str:
    """记录当前进程实际使用的 Python payload 指纹。"""
    payload_dir()
    assert _PINNED is not None
    return hashlib.sha256(b"".join(
        name.encode() + b"\0" + data
        for name, data in _PINNED[1].items() if name.endswith(".py")
    )).hexdigest()


def payload_dir() -> Path:
    """首次使用固定整个资源包；长进程不混用随后更新的源码和已加载模块。

    每次仍校验已发布内容；磁盘损坏不能由缓存或重新发布掩盖。
    新进程会选择新版本，旧 worker 的本地导入、上传和指纹保持一致。
    """
    global _PINNED
    with _PIN_LOCK:
        if _PINNED is None:
            files = {name: source_file(name).read_bytes() for name in sorted(PAYLOADS)}
            _PINNED = (_publish(files), files)
        destination, files = _PINNED
        for name, data in files.items():
            if (destination / name).read_bytes() != data:
                raise ValueError(f"可信评测 bundle 内容不匹配：{destination / name}")
        return destination


def _publish(files: dict[str, bytes]) -> Path:
    digest = hashlib.sha256(b"".join(
        name.encode() + b"\0" + data for name, data in files.items()
    )).hexdigest()
    cache = REPO_ROOT / ".agents/workdir/verify-payloads"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / digest
    with (cache / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            with tempfile.TemporaryDirectory(dir=cache, prefix=".building-") as temporary:
                bundle = Path(temporary) / "payload"
                bundle.mkdir()
                for name, data in files.items():
                    (bundle / name).write_bytes(data)
                bundle.rename(destination)
        for name, data in files.items():
            if (destination / name).read_bytes() != data:
                raise ValueError(f"可信评测 bundle 内容不匹配：{destination / name}")
    return destination


def payload_file(name: str) -> Path:
    if name not in PAYLOADS:
        raise KeyError(name)
    return payload_dir() / name


def payload_module(name: str) -> ModuleType:
    """以远端相同的模块名加载，保证跨 domain 的同目录导入及类型身份一致。"""
    if name not in PAYLOADS:
        raise KeyError(name)
    if not name.endswith(".py"):
        raise ValueError(f"不是 Python payload：{name}")
    bundle = payload_dir()
    # 提前检查所有已加载的 payload，避免依赖模块被同名第三方模块替代。
    for filename in PAYLOADS:
        module = sys.modules.get(Path(filename).stem) if filename.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != bundle / filename:
            raise ImportError(f"评测模块名称冲突：{filename}: {module.__file__}")
    if str(bundle) in sys.path:
        sys.path.remove(str(bundle))
    sys.path.insert(0, str(bundle))
    return importlib.import_module(Path(name).stem)

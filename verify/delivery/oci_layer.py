"""把 agent 交付的 OCI image layer 应用到一棵 rootfs 上(替代 git apply)。

交付物约定为一个 OCI filesystem layer(`submit_oci.tar.gz`,media type
`application/vnd.oci.image.layer.v1.tar+gzip`):tar 内路径相对 rootfs 根,
删除用 `.wh.<name>` whiteout 表示,清空目录用 `.wh..wh..opq` opaque 标记。
我们没有能力构建新镜像,所以评测端不组 candidate image,而是在同一基座镜像
起的干净 pod 上,用本模块把这一层原地"叠"到 `/` 上,再启动 launch.sh。

语义按 OCI image-spec layer.md:先处理本层全部 whiteout(只作用于 lower,
即 pod 当前 rootfs),再按 tar 顺序写入普通条目;普通文件/目录/符号链接/硬链接
保留 mode、uid/gid(以 root 运行时)与 mtime，已有目录也应用本层声明的属性，
目录属性在子项写完后恢复；拒绝设备、FIFO 等其它类型。先检查归档，再只读模拟
所有路径操作（含删除、类型替换和链接），预检失败不写 rootfs；实际写入的 I/O
错误直接抛出，不做回滚。目标 rootfs 必须静止且由调用方独占，归档在调用期间不可修改。

参数(apply_oci_layer):
  layer_path  : 压缩 layer 文件路径
  rootfs      : 应用目标根目录(生产为 `/`,单测为临时目录)
  forbidden   : 追加的禁止路径前缀(相对 rootfs);公共禁止集合 FORBIDDEN 不可放宽
返回:dict,含 layer_sha256(压缩 blob)、diff_id(未压缩 tar)、size_bytes、
      unpacked_bytes、entries、whiteouts、opaque_dirs 等审计信息。

示例:
  python3 verify/delivery/oci_layer.py submit_oci.tar.gz /tmp/rootfs usr/lib/cuda
  (生产由 serve_controller 以 root 调用 apply_oci_layer(layer, Path("/")))
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

LAYER_ARTIFACT = "submit_oci.tar.gz"
LAYER_MEDIA_TYPE = "application/vnd.oci.image.layer.v1.tar+gzip"
MAX_LAYER_BYTES = 100 * 1024 * 1024        # 压缩 blob 上限
MAX_UNPACKED_BYTES = 1024 * 1024 * 1024    # 解压后 tar 流上限(防 gzip bomb)
WHITEOUT_PREFIX = ".wh."
OPAQUE_MARKER = ".wh..wh..opq"
# 公共禁止前缀(相对 rootfs 根):内核/设备/运行时状态、Harbor 日志与交付目录。
FORBIDDEN = ("proc", "sys", "dev", "run", "logs")
_ALLOWED_TYPES = {
    tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE, tarfile.SYMTYPE, tarfile.LNKTYPE,
}


class LayerError(ValueError):
    """layer 内容不符合交付协议;校验阶段抛出,此时 rootfs 未被写入。"""


@dataclass
class LayerPlan:
    """校验通过后的应用计划:whiteout 先于普通条目执行。"""

    layer_sha256: str
    diff_id: str
    size_bytes: int
    unpacked_bytes: int
    whiteouts: list[str] = field(default_factory=list)      # 要删除的相对路径
    opaque_dirs: list[str] = field(default_factory=list)    # 要清空的相对目录
    entries: list[tarfile.TarInfo] = field(default_factory=list)


def _normalize(name: str) -> str:
    """把 tar 成员名规范成相对 rootfs 的干净路径;非法形式一律 LayerError。"""
    if "\0" in name:
        raise LayerError("layer 成员路径含 NUL")
    if name.startswith("/"):
        raise LayerError(f"layer 成员不能是绝对路径：{name!r}")
    parts = [p for p in PurePosixPath(name).parts if p != "."]
    if not parts:
        raise LayerError(f"layer 成员路径为空：{name!r}")
    if any(p == ".." or p == "" for p in parts):
        raise LayerError(f"layer 成员路径含 '..'：{name!r}")
    if any(p.startswith(WHITEOUT_PREFIX) for p in parts[:-1]):
        raise LayerError(f"whiteout 标记只能出现在最后一个路径分量：{name!r}")
    return "/".join(parts)


def _is_forbidden(relative: str, forbidden: tuple[str, ...]) -> bool:
    """相对路径是否落在禁止前缀内(前缀本身或其子路径)。"""
    return any(relative == p or relative.startswith(p + "/") for p in forbidden)


def _hash_and_measure(layer_path: Path) -> tuple[str, str, int, int]:
    """流式计算压缩 blob sha256、未压缩 diff_id 与两者字节数,不落临时文件。"""
    blob = hashlib.sha256()
    diff = hashlib.sha256()
    size = 0
    unpacked = 0
    with layer_path.open("rb") as raw:
        while chunk := raw.read(1 << 20):
            blob.update(chunk)
            size += len(chunk)
            if size > MAX_LAYER_BYTES:
                raise LayerError(f"layer 压缩体积 {size} 超过上限 {MAX_LAYER_BYTES}")
    with gzip.open(layer_path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            diff.update(chunk)
            unpacked += len(chunk)
            if unpacked > MAX_UNPACKED_BYTES:
                raise LayerError(f"layer 解压体积超过上限 {MAX_UNPACKED_BYTES}")
    return f"sha256:{blob.hexdigest()}", f"sha256:{diff.hexdigest()}", size, unpacked


def inspect_layer(layer_path: Path, forbidden: tuple[str, ...] = ()) -> LayerPlan:
    """只读校验整层并生成应用计划;任何一条不合法即 LayerError,不写 rootfs。"""
    layer_sha256, diff_id, size, unpacked = _hash_and_measure(layer_path)
    banned = FORBIDDEN + tuple(_normalize(p) for p in forbidden)
    plan = LayerPlan(layer_sha256, diff_id, size, unpacked)
    seen: dict[str, tarfile.TarInfo] = {}
    members: set[str] = set()
    with tarfile.open(layer_path, "r:gz") as tar:
        for member in tar:
            # `tar -C upper .` 会带一个 `./` 根目录条目:合法且无事可做(不改 / 的属性)。
            if member.isdir() and all(p == "." for p in PurePosixPath(member.name).parts):
                continue
            relative = _normalize(member.name)
            if relative in members:
                raise LayerError(f"layer 成员路径重复：{relative}")
            members.add(relative)
            if (member.size < 0 or not 0 <= member.uid < 2**32 - 1
                    or not 0 <= member.gid < 2**32 - 1 or not math.isfinite(member.mtime)):
                raise LayerError(f"layer 成员元数据非法：{relative}")
            if member.issparse():
                raise LayerError(f"不支持 sparse tar 成员：{relative}")
            name = PurePosixPath(relative).name
            if name.startswith(WHITEOUT_PREFIX) and (not member.isfile() or member.size != 0):
                raise LayerError(f"whiteout 必须是空普通文件：{relative}")
            parent = str(PurePosixPath(relative).parent)
            parent = "" if parent == "." else parent
            # whiteout/opaque 的禁止检查落在它们作用的目标路径上,而不是标记本身。
            if name == OPAQUE_MARKER:
                if not parent:
                    raise LayerError("禁止对 rootfs 根目录打 opaque 标记(会清空整个 /)")
                if _is_forbidden(parent, banned):
                    raise LayerError(f"opaque 标记落在禁止路径：{parent}")
                plan.opaque_dirs.append(parent)
                continue
            if name.startswith(WHITEOUT_PREFIX):
                target = name[len(WHITEOUT_PREFIX):]
                if not target or target in (".", "..") or target.startswith(WHITEOUT_PREFIX):
                    raise LayerError(f"非法 whiteout 名称：{relative}")
                target = f"{parent}/{target}" if parent else target
                if _is_forbidden(target, banned):
                    raise LayerError(f"whiteout 指向禁止路径：{target}")
                plan.whiteouts.append(target)
                continue
            if _is_forbidden(relative, banned):
                raise LayerError(f"layer 触碰禁止路径：{relative}")
            if member.type not in _ALLOWED_TYPES:
                raise LayerError(f"layer 成员类型不允许({member.type!r})：{relative}")
            if member.islnk():
                link = _normalize(member.linkname)
                if link not in seen or not (seen[link].isfile() or seen[link].islnk()):
                    raise LayerError(f"硬链接目标必须先于本条目出现在本层且是普通文件：{relative} -> {link}")
            if member.issym() and (member.linkname == "" or "\0" in member.linkname):
                raise LayerError(f"符号链接目标为空：{relative}")
            member.name = relative
            seen[relative] = member
            plan.entries.append(member)
    return plan


class _PlannedRoot:
    """只读模拟 layer 操作后的路径类型；不复制基座文件，不修改 rootfs。"""

    def __init__(self, root: Path, banned: tuple[str, ...]):
        self.root = root.resolve()
        self.banned = banned
        self.nodes: dict[str, tuple[str, str] | None] = {}
        self.hidden: set[str] = set()

    def node(self, path: str) -> tuple[str, str] | None:
        """返回模拟节点类型/链接目标，未被层覆盖的节点读取基座 lstat。"""
        if not path:
            return ("dir", "")
        if path in self.nodes:
            return self.nodes[path]
        if any(path == p or path.startswith(p + "/") for p in self.hidden):
            return None
        target = self.root / path
        try:
            mode = target.lstat().st_mode
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(mode):
            return ("symlink", os.readlink(target))
        if stat.S_ISDIR(mode):
            return ("dir", "")
        return ("file" if stat.S_ISREG(mode) else "special", "")

    def resolve(self, relative: str, depth: int = 0) -> str:
        """按模拟树解析父目录，链接不能越出目标根或进入禁止目录。"""
        if depth > 40:
            raise LayerError(f"父目录符号链接循环：{relative}")
        parts = PurePosixPath(relative).parts
        current = ""
        for part in parts[:-1]:
            current = f"{current}/{part}" if current else part
            value = self.node(current)
            if value and value[0] == "symlink":
                # 链接的相对目标以链接所在目录为准；绝对目标仍要求位于目标 rootfs 内。
                target = Path(os.path.normpath((self.root / current).parent / value[1]))
                if not target.is_relative_to(self.root):
                    raise LayerError(f"路径经符号链接逃逸 rootfs：{relative}")
                link = target.relative_to(self.root).as_posix()
                current = self.resolve(link + "/.probe", depth + 1).removesuffix("/.probe")
                if current == ".probe":
                    current = ""
                value = self.node(current)
            if _is_forbidden(current, self.banned):
                raise LayerError(f"路径经符号链接落入禁止目录：{relative}")
            if value is None or value[0] != "dir":
                raise LayerError(f"父目录在 rootfs 中不存在或不是目录：{relative}")
        result = f"{current}/{parts[-1]}" if current else parts[-1]
        if _is_forbidden(result, self.banned):
            raise LayerError(f"layer 触碰禁止路径：{relative}")
        return result

    def replace(self, path: str, value: tuple[str, str] | None, destructive: bool) -> None:
        """更新模拟节点；删除/类型替换必须同时保护禁止路径的祖先目录。"""
        if destructive:
            if any(p == path or p.startswith(path + "/") for p in self.banned):
                raise LayerError(f"不能替换或清空禁止路径的祖先：{path}")
            target = self.root / path
            if target.is_dir() and not target.is_symlink():
                for base, dirs, _ in os.walk(target, followlinks=False):
                    if os.path.ismount(base) or any(os.path.ismount(Path(base) / d) for d in dirs):
                        raise LayerError(f"不能删除挂载目录：{path}")
            self.hidden.add(path)
            self.nodes = {p: v for p, v in self.nodes.items() if not p.startswith(path + "/")}
        self.nodes[path] = value


def _preflight(plan: LayerPlan, root: Path, banned: tuple[str, ...]):
    """整层模拟后返回已解析的删除、opaque、写入路径；校验失败时 rootfs 未改变。"""
    tree = _PlannedRoot(root, banned)
    whiteouts: list[str] = []
    opaque: list[str] = []
    writes: list[tuple[tarfile.TarInfo, str]] = []
    for relative in plan.whiteouts:
        target = tree.resolve(relative)
        if tree.node(target) is None:
            raise LayerError(f"whiteout 目标在 rootfs 中不存在(基座不一致?)：{relative}")
        tree.replace(target, None, destructive=True)
        whiteouts.append(target)
    for relative in plan.opaque_dirs:
        target = tree.resolve(relative)
        if tree.node(target) != ("dir", ""):
            raise LayerError(f"opaque 标记所在路径不是目录：{relative}")
        tree.replace(target, ("dir", ""), destructive=True)
        opaque.append(target)
    for member in plan.entries:
        target = tree.resolve(member.name)
        old = tree.node(target)
        if member.islnk():
            link = tree.resolve(_normalize(member.linkname))
            if tree.node(link) != ("file", "") or link == target:
                raise LayerError(f"硬链接目标不是当前存在的普通文件：{member.name}")
            member.linkname = link
        kind = "dir" if member.isdir() else "symlink" if member.issym() else "file"
        tree.replace(target, (kind, member.linkname if member.issym() else ""),
                     destructive=old is not None and not (kind == "dir" and old[0] == "dir"))
        # 新目录也不能看到同路径已删除的 lower 子项。
        if old is None and kind == "dir":
            tree.hidden.add(target)
        writes.append((member, target))
    return whiteouts, opaque, writes


def _remove(path: Path) -> None:
    """删除文件、链接或整棵目录(不跟随符号链接)。"""
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def _write_entry(tar: tarfile.TarFile, member: tarfile.TarInfo, target: Path,
                 rootfs: Path, chown: bool) -> None:
    """按 OCI 语义落一条普通条目:同名旧条目(不论类型)先被本层条目覆盖。"""
    keep_dir = member.isdir() and target.is_dir() and not target.is_symlink()
    if (target.is_symlink() or target.exists()) and not keep_dir:
        _remove(target)
    if member.isdir():
        # 全部子项写完再恢复目录权限和时间，避免只读目录拦住后续子项。
        target.mkdir(exist_ok=True)
        return
    elif member.issym():
        os.symlink(member.linkname, target)
    elif member.islnk():
        os.link(rootfs / _normalize(member.linkname), target)
    else:
        source = tar.extractfile(member)
        if source is None:
            raise LayerError(f"无法读取 layer 成员内容：{member.name}")
        with target.open("wb") as sink:
            shutil.copyfileobj(source, sink)
    _attributes(member, target, chown)


def _attributes(member: tarfile.TarInfo, target: Path, chown: bool) -> None:
    """恢复条目属主、权限和时间；先 chown 再 chmod，避免清掉 setid 位。"""
    if chown:
        os.lchown(target, member.uid, member.gid)
    if not member.issym():
        os.chmod(target, member.mode & 0o7777)
    os.utime(target, (member.mtime, member.mtime), follow_symlinks=False)


def apply_oci_layer(layer_path: Path, rootfs: Path,
                    forbidden: tuple[str, ...] = ()) -> dict[str, object]:
    """校验后把 layer 叠到 rootfs:先 whiteout/opaque,再写入普通条目;返回审计摘要。"""
    if not rootfs.is_dir():
        raise LayerError(f"rootfs 不是目录：{rootfs}")
    plan = inspect_layer(layer_path, forbidden)
    banned = FORBIDDEN + tuple(_normalize(p) for p in forbidden)
    rootfs = rootfs.resolve()
    archive = layer_path.resolve()
    if archive.is_relative_to(rootfs):
        banned += (archive.relative_to(rootfs).as_posix(),)
    whiteouts, opaque, writes = _preflight(plan, rootfs, banned)
    chown = os.geteuid() == 0
    for relative in whiteouts:
        _remove(rootfs / relative)
    for relative in opaque:
        for child in (rootfs / relative).iterdir():
            _remove(child)
    directories: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(layer_path, "r:gz") as tar:
        for member, relative in writes:
            _write_entry(tar, member, rootfs / relative, rootfs, chown)
            # 同层后续类型替换后，不得再把旧目录属性写到新文件或符号链接上。
            if not member.isdir():
                directories = {p: m for p, m in directories.items()
                               if p != relative and not p.startswith(relative + "/")}
            else:
                directories[relative] = member
    for relative in sorted(directories, key=lambda p: p.count("/"), reverse=True):
        _attributes(directories[relative], rootfs / relative, chown)
    return {
        "media_type": LAYER_MEDIA_TYPE,
        "layer_sha256": plan.layer_sha256,
        "diff_id": plan.diff_id,
        "size_bytes": plan.size_bytes,
        "unpacked_bytes": plan.unpacked_bytes,
        "entries": len(plan.entries),
        "whiteouts": plan.whiteouts,
        "opaque_dirs": plan.opaque_dirs,
        "owner_preserved": chown,
    }


def main(argv: list[str]) -> None:
    """CLI:`oci_layer.py <layer.tar.gz> <rootfs> [追加禁止路径,逗号分隔]`,应用后打印 JSON 摘要。"""
    if len(argv) not in (3, 4):
        raise SystemExit("用法：oci_layer.py <layer.tar.gz> <rootfs> [forbidden,paths]")
    forbidden = tuple(p for p in (argv[3].split(",") if len(argv) == 4 else []) if p)
    summary = apply_oci_layer(Path(argv[1]), Path(argv[2]), forbidden)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv)

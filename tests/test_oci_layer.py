"""verify/delivery/oci_layer.py 的 dummy OCI layer 往返测试。

参数：无。用 tarfile 生成符合 OCI layer.md 的 dummy 层(普通文件、目录、可执行位、
符号链接、硬链接、whiteout、opaque),叠到临时 rootfs 上核对字节与元数据;并覆盖
校验阶段的各类拒绝路径(禁止目录、绝对路径、`..`、设备、符号链接逃逸、根 opaque),
且断言拒绝时 rootfs 未被写入。

示例：
  python3 -m unittest tests.test_oci_layer
"""

from __future__ import annotations

from verify.resources import payload_file
import importlib.util
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

SCRIPT = payload_file('oci_layer.py')


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("oci_layer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass 解析注解需要模块已注册
    spec.loader.exec_module(module)
    return module


def build_layer(path: Path, entries: list[tuple], mode: str = "w:gz") -> Path:
    """按 (kind, name, payload) 序列生成 dummy layer。

    kind: file(payload=bytes 或 (bytes, mode))、dir、symlink(payload=目标)、
          hardlink(payload=同层已出现的路径)、whiteout(name 为被删路径)、
          opaque(name 为目录)、char(设备,用于负例)。
    """
    with tarfile.open(path, mode) as tar:
        for kind, name, *rest in entries:
            payload = rest[0] if rest else None
            if kind == "file":
                data, fmode = (payload, 0o644) if isinstance(payload, bytes) else payload
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime, info.uid, info.gid = len(data), fmode, 1700000000, 0, 0
                tar.addfile(info, io.BytesIO(data))
            elif kind == "dir":
                info = tarfile.TarInfo(name)
                info.type, info.mode, info.mtime = tarfile.DIRTYPE, 0o755, 1700000000
                tar.addfile(info)
            elif kind == "symlink":
                info = tarfile.TarInfo(name)
                info.type, info.linkname = tarfile.SYMTYPE, payload
                tar.addfile(info)
            elif kind == "hardlink":
                info = tarfile.TarInfo(name)
                info.type, info.linkname = tarfile.LNKTYPE, payload
                tar.addfile(info)
            elif kind == "whiteout":
                parent, base = os.path.split(name)
                info = tarfile.TarInfo(os.path.join(parent, ".wh." + base))
                tar.addfile(info)
            elif kind == "opaque":
                info = tarfile.TarInfo(os.path.join(name, ".wh..wh..opq"))
                tar.addfile(info)
            elif kind == "char":
                info = tarfile.TarInfo(name)
                info.type = tarfile.CHRTYPE
                tar.addfile(info)
            else:
                raise AssertionError(kind)
    return path


def snapshot(root: Path) -> dict[str, str]:
    """rootfs 的可比较快照:路径 -> 类型/内容摘要。"""
    result = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_symlink():
            result[rel] = f"symlink:{os.readlink(path)}"
        elif path.is_dir():
            result[rel] = "dir"
        else:
            result[rel] = f"file:{path.read_bytes()!r}:{oct(path.stat().st_mode & 0o777)}"
    return result


class LowerRootfs(unittest.TestCase):
    """每个用例一棵模拟基座 rootfs(等价于 serve pod 启动时的 /)。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "rootfs"
        self.work = Path(self.tmp.name)
        (self.root / "sgl-workspace/sglang/python/sglang").mkdir(parents=True)
        (self.root / "sgl-workspace/sglang/python/sglang/server.py").write_text("old server\n")
        (self.root / "sgl-workspace/sglang/README.md").write_text("readme\n")
        (self.root / "sgl-workspace/sglang/build").mkdir()
        (self.root / "sgl-workspace/sglang/build/a.o").write_bytes(b"\x00a")
        (self.root / "sgl-workspace/sglang/build/b.o").write_bytes(b"\x00b")
        (self.root / "usr/lib/python3/site-packages").mkdir(parents=True)
        (self.root / "usr/lib/python3/site-packages/dep.py").write_text("dep\n")
        (self.root / "proc").mkdir()
        (self.root / "proc/cpuinfo").write_text("cpu\n")
        (self.root / "logs").mkdir()
        (self.root / "opt").mkdir()
        os.symlink("../proc", self.root / "opt/escape")
        (self.root / "etc").mkdir()
        os.symlink(str(self.work / "outside"), self.root / "etc/abs")
        (self.work / "outside").mkdir()
        self.mod = load_module()

    def tearDown(self) -> None:
        self.tmp.cleanup()


class ApplyDummyLayer(LowerRootfs):
    def test_roundtrip_applies_every_entry_kind(self) -> None:
        layer = build_layer(self.work / "submit_oci.tar.gz", [
            ("dir", "sgl-workspace/sglang/python/sglang"),
            ("file", "sgl-workspace/sglang/python/sglang/server.py", b"new server\n"),
            ("file", "sgl-workspace/sglang/launch.sh", (b"#!/bin/sh\necho hi\n", 0o755)),
            ("dir", "sgl-workspace/sglang/kernels"),
            ("file", "sgl-workspace/sglang/kernels/fast.so", b"\x7fELF"),
            ("hardlink", "sgl-workspace/sglang/kernels/fast_link.so", "sgl-workspace/sglang/kernels/fast.so"),
            ("symlink", "sgl-workspace/sglang/latest.so", "kernels/fast.so"),
            ("whiteout", "sgl-workspace/sglang/README.md"),
            ("opaque", "sgl-workspace/sglang/build"),
            ("file", "sgl-workspace/sglang/build/c.o", b"\x00c"),
            ("file", "usr/lib/python3/site-packages/dep.py", b"dep v2\n"),
        ])
        summary = self.mod.apply_oci_layer(layer, self.root)
        got = snapshot(self.root / "sgl-workspace/sglang")
        self.assertEqual(got, {
            "build": "dir",
            "build/c.o": "file:b'\\x00c':0o644",
            "kernels": "dir",
            "kernels/fast.so": "file:b'\\x7fELF':0o644",
            "kernels/fast_link.so": "file:b'\\x7fELF':0o644",
            "latest.so": "symlink:kernels/fast.so",
            "launch.sh": "file:b'#!/bin/sh\\necho hi\\n':0o755",
            "python": "dir",
            "python/sglang": "dir",
            "python/sglang/server.py": "file:b'new server\\n':0o644",
        })
        self.assertEqual((self.root / "sgl-workspace/sglang/kernels/fast.so").stat().st_nlink, 2)
        self.assertEqual((self.root / "usr/lib/python3/site-packages/dep.py").read_text(), "dep v2\n")
        self.assertEqual((self.root / "proc/cpuinfo").read_text(), "cpu\n")
        self.assertEqual(summary["entries"], 9)
        self.assertEqual(summary["whiteouts"], ["sgl-workspace/sglang/README.md"])
        self.assertEqual(summary["opaque_dirs"], ["sgl-workspace/sglang/build"])
        self.assertEqual(summary["media_type"], "application/vnd.oci.image.layer.v1.tar+gzip")
        self.assertTrue(summary["layer_sha256"].startswith("sha256:"))
        self.assertTrue(summary["diff_id"].startswith("sha256:"))
        self.assertNotEqual(summary["layer_sha256"], summary["diff_id"])
        self.assertEqual(summary["size_bytes"], layer.stat().st_size)
        self.assertEqual(summary["owner_preserved"], os.geteuid() == 0)

    def test_diff_id_is_sha256_of_uncompressed_tar(self) -> None:
        import gzip
        import hashlib
        layer = build_layer(self.work / "l.tar.gz", [("file", "opt/x", b"x")])
        plan = self.mod.inspect_layer(layer)
        raw = gzip.decompress(layer.read_bytes())
        self.assertEqual(plan.diff_id, "sha256:" + hashlib.sha256(raw).hexdigest())
        self.assertEqual(plan.layer_sha256, "sha256:" + hashlib.sha256(layer.read_bytes()).hexdigest())
        self.assertEqual(plan.unpacked_bytes, len(raw))

    def test_file_replaces_directory_and_directory_replaces_file(self) -> None:
        layer = build_layer(self.work / "l.tar.gz", [
            ("file", "sgl-workspace/sglang/build", b"now a file"),
            ("dir", "sgl-workspace/sglang/README.md"),
        ])
        self.mod.apply_oci_layer(layer, self.root)
        self.assertEqual((self.root / "sgl-workspace/sglang/build").read_bytes(), b"now a file")
        self.assertTrue((self.root / "sgl-workspace/sglang/README.md").is_dir())

    def test_existing_directories_receive_layer_attributes_after_children(self) -> None:
        usr = self.root / "usr"
        usr.chmod(0o700)
        layer = build_layer(self.work / "l.tar.gz", [
            ("dir", "usr"), ("dir", "usr/lib"), ("dir", "usr/newdir"),
            ("file", "usr/newdir/x", b"x"),
        ])
        self.mod.apply_oci_layer(layer, self.root)
        self.assertEqual(usr.stat().st_mode & 0o777, 0o755)
        self.assertEqual(usr.stat().st_mtime, 1700000000)
        self.assertEqual((self.root / "usr/newdir").stat().st_mtime, 1700000000)
        self.assertEqual((self.root / "usr/newdir/x").read_bytes(), b"x")

    def test_same_layer_directory_and_relative_link_parent(self) -> None:
        layer = build_layer(self.work / "l.tar.gz", [
            ("dir", "opt/target"),
            ("symlink", "opt/alias", "target"),
            ("file", "opt/alias/child", b"data"),
        ])
        self.mod.apply_oci_layer(layer, self.root)
        self.assertEqual((self.root / "opt/target/child").read_bytes(), b"data")
        self.assertEqual(os.readlink(self.root / "opt/alias"), "target")

    def test_directory_attributes_do_not_follow_later_type_replacement(self) -> None:
        layer = build_layer(self.work / "l.tar.gz", [
            ("dir", "opt/a"), ("dir", "opt/a/b"),
            ("file", "opt/a/b/child", b"child"),
            ("symlink", "opt/alias", "a"),
            ("file", "opt/alias/b", b"replacement"),
        ])
        self.mod.apply_oci_layer(layer, self.root)
        self.assertEqual((self.root / "opt/a/b").read_bytes(), b"replacement")

    def test_dot_slash_prefix_is_normalized(self) -> None:
        layer = build_layer(self.work / "l.tar.gz", [("file", "./opt/./x", b"x")])
        self.mod.apply_oci_layer(layer, self.root)
        self.assertEqual((self.root / "opt/x").read_bytes(), b"x")

    def test_gnu_tar_root_dir_entry_is_noop(self) -> None:
        # agent 最自然的打包姿势 `tar -czf submit_oci.tar.gz -C upper .` 会带 `./` 根条目。
        upper = self.work / "upper"
        (upper / "sgl-workspace/sglang").mkdir(parents=True)
        (upper / "sgl-workspace/sglang/launch.sh").write_text("#!/bin/sh\n")
        (upper / "sgl-workspace/sglang/.wh.README.md").touch()
        layer = self.work / "submit_oci.tar.gz"
        subprocess.run(["tar", "-czf", str(layer), "-C", str(upper), "."], check=True)
        root_mode = self.root.stat().st_mode
        summary = self.mod.apply_oci_layer(layer, self.root)
        self.assertEqual((self.root / "sgl-workspace/sglang/launch.sh").read_text(), "#!/bin/sh\n")
        self.assertFalse((self.root / "sgl-workspace/sglang/README.md").exists())
        self.assertEqual(self.root.stat().st_mode, root_mode)
        self.assertEqual(summary["whiteouts"], ["sgl-workspace/sglang/README.md"])

    def test_cli_prints_summary(self) -> None:
        layer = build_layer(self.work / "submit_oci.tar.gz", [("file", "opt/x", b"cli")])
        out = subprocess.run([sys.executable, str(SCRIPT), str(layer), str(self.root)],
                             capture_output=True, text=True, check=True).stdout
        self.assertIn('"entries": 1', out)
        self.assertEqual((self.root / "opt/x").read_bytes(), b"cli")


class RejectedLayers(LowerRootfs):
    def assert_rejected(self, entries: list[tuple], message: str) -> None:
        before = snapshot(self.root)
        layer = build_layer(self.work / "bad.tar.gz", entries)
        with self.assertRaisesRegex(self.mod.LayerError, message):
            self.mod.apply_oci_layer(layer, self.root)
        self.assertEqual(snapshot(self.root), before, "拒绝的 layer 不得改动 rootfs")

    def test_forbidden_prefix(self) -> None:
        self.assert_rejected([("file", "opt/ok", b"1"), ("file", "proc/evil", b"x")], "禁止路径")
        self.assert_rejected([("file", "logs/verifier/serve_controller.py", b"x")], "禁止路径")
        self.assert_rejected([("whiteout", "proc/cpuinfo")], "whiteout 指向禁止路径")
        self.assert_rejected([("opaque", "proc")], "opaque 标记落在禁止路径")

    def test_task_can_extend_but_not_relax_forbidden(self) -> None:
        layer = build_layer(self.work / "l.tar.gz", [("file", "usr/lib/python3/site-packages/dep.py", b"x")])
        with self.assertRaisesRegex(self.mod.LayerError, "禁止路径"):
            self.mod.apply_oci_layer(layer, self.root, forbidden=("usr/lib",))
        with self.assertRaisesRegex(self.mod.LayerError, "禁止路径"):
            self.mod.apply_oci_layer(build_layer(self.work / "p.tar.gz", [("file", "proc/x", b"x")]),
                                     self.root, forbidden=())

    def test_root_opaque_rejected(self) -> None:
        self.assert_rejected([("opaque", "")], "根目录打 opaque")

    def test_path_traversal_and_absolute(self) -> None:
        self.assert_rejected([("file", "opt/../../x", b"x")], "含 '\\.\\.'")
        self.assert_rejected([("file", "/opt/x", b"x")], "绝对路径")

    def test_symlink_escape_via_lower(self) -> None:
        self.assert_rejected([("file", "opt/escape/x", b"x")], "落入禁止目录")
        self.assert_rejected([("file", "etc/abs/x", b"x")], "逃逸 rootfs")

    def test_device_and_bad_links(self) -> None:
        self.assert_rejected([("char", "dev2/null")], "类型不允许")
        self.assert_rejected([("hardlink", "opt/x", "opt/notyet")], "硬链接目标必须先于")
        self.assert_rejected([("file", "opt/.wh.evil/x", b"x")], "最后一个路径分量")

    def test_whiteout_of_missing_lower_path_means_base_drift(self) -> None:
        self.assert_rejected([("whiteout", "sgl-workspace/sglang/nope")], "基座不一致")

    def test_missing_parent_directory(self) -> None:
        self.assert_rejected([("file", "sgl-workspace/sglang/newdir/x.py", b"x")], "父目录在 rootfs 中不存在")

    def test_late_missing_parent_does_not_modify_earlier_entries(self) -> None:
        self.assert_rejected([
            ("file", "sgl-workspace/sglang/README.md", b"changed"),
            ("file", "missing/child", b"x"),
        ], "父目录")

    def test_late_invalid_whiteout_does_not_delete_earlier_target(self) -> None:
        self.assert_rejected([
            ("whiteout", "sgl-workspace/sglang/README.md"),
            ("whiteout", "sgl-workspace/sglang/nope"),
        ], "基座不一致")

    def test_opaque_then_missing_parent_does_not_clear_directory(self) -> None:
        self.assert_rejected([
            ("opaque", "sgl-workspace/sglang/build"),
            ("file", "missing/child", b"x"),
        ], "父目录")

    def test_forbidden_descendant_survives_ancestor_replacement(self) -> None:
        layer = build_layer(self.work / "l.tar.gz", [("opaque", "usr")])
        before = snapshot(self.root)
        with self.assertRaisesRegex(self.mod.LayerError, "禁止路径的祖先"):
            self.mod.apply_oci_layer(layer, self.root, forbidden=("usr/lib",))
        self.assertEqual(snapshot(self.root), before)

    def test_duplicate_entries_and_nonempty_whiteouts_rejected(self) -> None:
        self.assert_rejected([("file", "opt/x", b"a"), ("file", "./opt/x", b"b")], "重复")
        self.assert_rejected([("file", "opt/.wh.x", b"not-empty")], "空普通文件")
        self.assert_rejected([("whiteout", "opt/..")], "非法 whiteout")

    def test_same_layer_link_parent_is_checked_before_any_writes(self) -> None:
        self.assert_rejected([
            ("file", "opt/first", b"x"),
            ("symlink", "opt/newlink", "../proc"),
            ("file", "opt/newlink/test", b"x"),
        ], "禁止")

    def test_hardlink_cannot_target_directory(self) -> None:
        self.assert_rejected([("dir", "opt/dir"), ("hardlink", "opt/link", "opt/dir")], "普通文件")

    def test_size_limits(self) -> None:
        self.mod.MAX_LAYER_BYTES = 64
        self.assert_rejected([("file", "opt/big", os.urandom(4096))], "压缩体积")

    def test_plain_tar_without_gzip_rejected(self) -> None:
        layer = build_layer(self.work / "plain.tar.gz", [("file", "opt/x", b"x")], mode="w")
        import gzip
        with self.assertRaises(gzip.BadGzipFile):
            self.mod.apply_oci_layer(layer, self.root)


if __name__ == "__main__":
    unittest.main()


class RealBuildahLayer(LowerRootfs):
    """用 buildah 真实产出的层(tests/fixtures/oci)对拍:格式、whiteout、digest 口径。"""

    FIXTURE = Path(__file__).resolve().parent / "fixtures/oci/buildah-whiteout-layer"

    def test_buildah_layer_applies_and_digests_match_oci_metadata(self) -> None:
        import json
        expected = json.loads(self.FIXTURE.with_suffix(".json").read_text())
        summary = self.mod.apply_oci_layer(self.FIXTURE.with_suffix(".tar.gz"), self.root)
        self.assertEqual(summary["layer_sha256"], expected["descriptor_digest"])
        self.assertEqual(summary["diff_id"], expected["diff_id"])
        self.assertEqual(summary["size_bytes"], expected["size"])
        self.assertEqual(summary["whiteouts"], ["sgl-workspace/sglang/README.md"])
        self.assertFalse((self.root / "sgl-workspace/sglang/README.md").exists())
        self.assertEqual((self.root / "sgl-workspace/sglang/python/sglang/server.py").read_text(), "old server\n")
        self.assertEqual((self.root / "sgl-workspace/sglang/python/server.py").read_text(), "new\n")
        self.assertTrue(os.access(self.root / "sgl-workspace/sglang/launch.sh", os.X_OK))


class RegistryPublishedLayer(unittest.TestCase):
    """a container registry 上真实推送/拉回的 sglang-pi05 示例层(含 whiteout + opaque + 符号链接 + 可执行位)。

    lower 按该层作用到的基座路径合成;E2E 对拍(与候选镜像 rootfs 逐字节一致)记录在
    the original validation notes,这里固化 digest 口径与 whiteout/opaque 语义。
    """

    FIXTURE = Path(__file__).resolve().parent / "fixtures/oci/hub-sglang-pi05-demo-layer"

    def test_registry_layer_applies_with_matching_digests(self) -> None:
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "sgl-workspace/sglang"
            (repo / "python/sglang").mkdir(parents=True)
            (repo / "README.md").write_text("base readme\n")
            (repo / "docs_new").mkdir()
            (repo / "docs_new/old.md").write_text("old doc\n")
            (repo / "python/sglang/__init__.py").write_text("base init\n")
            (root / "usr/local/lib/python3.12/dist-packages").mkdir(parents=True)
            mod = load_module()
            expected = json.loads(self.FIXTURE.with_suffix(".json").read_text())
            summary = mod.apply_oci_layer(self.FIXTURE.with_suffix(".tar.gz"), root)
            self.assertEqual(summary["layer_sha256"], expected["descriptor_digest"])
            self.assertEqual(summary["diff_id"], expected["diff_id"])
            self.assertEqual(summary["size_bytes"], expected["size"])
            self.assertEqual(summary["whiteouts"], ["sgl-workspace/sglang/README.md"])
            self.assertEqual(summary["opaque_dirs"], ["sgl-workspace/sglang/docs_new"])
            self.assertFalse((repo / "README.md").exists())
            self.assertEqual(sorted(p.name for p in (repo / "docs_new").iterdir()), ["new.md"])
            self.assertTrue((repo / "python/sglang/__init__.py").read_text().endswith("# oci-layer-demo: modified\n"))
            self.assertEqual(os.readlink(repo / "latest.so"), "kernels/demo.so")
            self.assertEqual((repo / "kernels/demo.so").stat().st_size, 65536)
            self.assertTrue(os.access(repo / "launch.sh", os.X_OK))
            self.assertEqual((root / "usr/local/lib/python3.12/dist-packages/oci_layer_demo.pth").read_text(), "oci_layer_demo\n")

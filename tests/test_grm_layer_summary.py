"""verify.grading.grm 对 OCI layer 交付物的 judge 摘要测试。

参数:无。生成一个含 whiteout 与 launch.sh 的 dummy layer,验证 _summarize_patch
按 gzip 魔数分派到 layer 摘要:成员清单、whiteout 名、元数据与文本源码头部齐全。

示例:
  PYTHONPATH=. .venv/bin/python3 -m unittest tests.test_grm_layer_summary
"""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from verify.grading.grm import _summarize_patch


class LayerSummaryTest(unittest.TestCase):
    def test_layer_summary_lists_entries_whiteouts_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layer = Path(tmp) / "submit_oci.tar.gz"
            with tarfile.open(layer, "w:gz") as tar:
                info = tarfile.TarInfo("sgl-workspace/sglang/launch.sh")
                data = b"#!/bin/bash\necho launch\n"
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                tar.addfile(tarfile.TarInfo("sgl-workspace/sglang/.wh.README.md"))
                blob = tarfile.TarInfo("sgl-workspace/sglang/kernels/demo.so")
                blob.size = 4
                tar.addfile(blob, io.BytesIO(b"\x7fELF"))
            metadata = Path(tmp) / "patch-metadata.json"
            metadata.write_text(json.dumps({"base_commit": "abc", "diff_id": "sha256:def"}))
            text = _summarize_patch(layer, metadata)
        self.assertIn("# OCI image layer produced by the agent", text)
        self.assertIn("diff_id sha256:def", text)
        self.assertIn("# base_commit: abc", text)
        self.assertIn("sgl-workspace/sglang/.wh.README.md (file, 0 bytes)", text)
        self.assertIn("sgl-workspace/sglang/kernels/demo.so (file, 4 bytes)", text)
        self.assertIn("### sgl-workspace/sglang/launch.sh\n#!/bin/bash\necho launch", text)
        self.assertNotIn("\x7fELF", text)

    def test_text_patch_still_uses_diff_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            patch = Path(tmp) / "sglang.patch"
            patch.write_bytes(b"diff --git a/x.py b/x.py\n+1\n")
            metadata = Path(tmp) / "patch-metadata.json"
            metadata.write_text(json.dumps({"base_commit": "abc"}))
            text = _summarize_patch(patch, metadata)
        self.assertIn("# Git patch produced by the agent", text)
        self.assertIn("  x.py", text)

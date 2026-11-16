"""verify/delivery/serve_controller.py 的 OCI layer 交付路径测试。

参数:无。用假引擎仓库验证 metadata 分派(layer_sha256 为默认协议、patch_sha256 为
legacy)、apply_and_start 在 OCI 模式下把层交给 oci_layer.py(以 root、rootfs=/、
携带 task 追加的禁止路径)而不再 git apply,以及 sha256/diff_id 不一致时 fail-fast。
git/sudo/launch 均为 stub,不触碰真实 Merlin。

示例:
  PYTHONPATH=. .venv/bin/python3 -m unittest tests.test_serve_controller_oci
"""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_serve_controller_oracle import load_module

LAYER = gzip.compress(b"not-a-real-tar-but-bytes-suffice-for-controller")
LAYER_SHA = "sha256:" + hashlib.sha256(LAYER).hexdigest()
DIFF_ID = "sha256:" + hashlib.sha256(gzip.decompress(LAYER)).hexdigest()


class SubmissionChecksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def test_layer_metadata_is_default_protocol(self) -> None:
        checks = self.mod.submission_checks(
            {"layer_sha256": LAYER_SHA, "diff_id": DIFF_ID, "base_commit": "c", "baseline_tree": "t"},
            ["usr/lib/x"],
        )
        self.assertEqual(checks, {
            "expected_sha256": LAYER_SHA,
            "oci": {"diff_id": DIFF_ID, "forbidden_paths": ["usr/lib/x"]},
        })

    def test_patch_metadata_is_legacy(self) -> None:
        checks = self.mod.submission_checks(
            {"patch_sha256": "ab", "base_commit": "c", "baseline_tree": "t"}, []
        )
        self.assertEqual(checks, {"expected_sha256": "ab", "oci": None})

    def test_missing_fields_fail_loud(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer_sha256\\|patch_sha256"):
            self.mod.submission_checks({"base_commit": "c", "baseline_tree": "t"}, [])
        with self.assertRaisesRegex(ValueError, "baseline_tree"):
            self.mod.submission_checks({"layer_sha256": LAYER_SHA, "diff_id": DIFF_ID, "base_commit": "c"}, [])
        with self.assertRaisesRegex(ValueError, "diff_id"):
            self.mod.submission_checks({"layer_sha256": LAYER_SHA, "base_commit": "c", "baseline_tree": "t"}, [])


class ApplyOciLayerTest(unittest.TestCase):
    """apply_and_start 的 OCI 分支:sha 校验 → sudo oci_layer.py → digest 核对 → launch。"""

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="controller-oci-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "launch.sh").write_text("#!/bin/bash\ntrue\n")
        self.mod = load_module()
        self.enterContext(patch.object(self.mod, "OCI_LAYER_TMP", self.root / "submit_oci.tar.gz"))
        self.enterContext(patch.object(self.mod, "WORKDIR", self.root / "deploy"))
        self.runs = self.enterContext(patch.object(
            self.mod.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
        ))
        self.enterContext(patch.object(self.mod, "worktree_tree", return_value="fixture-tree"))
        self.enterContext(patch.object(
            self.mod.ServiceRunner, "_launch", return_value={"running": True, "rank": 0}
        ))
        self.oci_calls: list[list[str]] = []
        self.summary = {"layer_sha256": LAYER_SHA, "diff_id": DIFF_ID, "entries": 1}

        def check_output(argv, **kwargs):
            if "rev-parse" in argv:
                return "fixture-commit\n"
            if argv[3] == self.mod.OCI_LAYER_SCRIPT or argv[3].endswith("/deploy/oci_layer.py"):
                self.oci_calls.append(list(argv))
                # 应用器读到的必须是 controller 刚落盘的那份层
                self.assertEqual(Path(argv[4]).read_bytes(), LAYER)
                return json.dumps(self.summary)
            raise AssertionError(f"unexpected check_output: {argv}")

        self.enterContext(patch.object(self.mod.subprocess, "check_output", side_effect=check_output))

    def _apply(self, expected_sha: str = LAYER_SHA, diff_id: str = DIFF_ID, forbidden=("usr/lib/x", "opt/y")):
        runner = self.mod.ServiceRunner(0, str(self.repo))
        return runner.apply_and_start(
            LAYER, expected_sha, "fixture-commit", "fixture-tree", "fd00::1",
            oci={"diff_id": diff_id, "forbidden_paths": list(forbidden)},
        )

    def test_layer_is_applied_by_root_helper_not_git_apply(self) -> None:
        result = self._apply()
        self.assertEqual(result["running"], True)
        self.assertEqual(self.oci_calls, [[
            "sudo", "-n", sys.executable, "/logs/verifier/oci_layer.py",
            str(self.root / "submit_oci.tar.gz"), "/", "usr/lib/x,opt/y",
        ]])
        for call in self.runs.call_args_list:
            self.assertNotIn("apply", call.args[0], "OCI 模式不得 git apply")
        # 仍然 chown 整树以便 launch 以当前用户执行
        self.assertTrue(any("chown" in call.args[0] for call in self.runs.call_args_list))
        # 应用证据落盘,verifier 下载后由 oracle_ci_layer.py 核对
        evidence = json.loads((self.root / "deploy/oci-layer-rank-0.json").read_text())
        self.assertEqual((evidence["rank"], evidence["layer_sha256"], evidence["diff_id"]), (0, LAYER_SHA, DIFF_ID))

    def test_baseline_runs_after_clean_check_before_candidate(self):
        from verify.delivery import evaluation_baseline
        source = self.root / 'source'
        source.mkdir()
        (source / 'recipe.py').write_bytes(b'baseline')
        hashes = {'recipe.py': hashlib.sha256(b'baseline').hexdigest()}
        manifest = self.root / 'baseline.json'
        manifest.write_text(json.dumps(evaluation_baseline.build(source, hashes)))
        self.enterContext(patch.object(self.mod, 'EVALUATION_BASELINE_PATH', manifest))
        self.enterContext(patch.dict(sys.modules, {'evaluation_baseline': evaluation_baseline}))
        stages = []
        def run(argv, **kwargs):
            if '/logs/verifier/evaluation_baseline.py' in argv:
                stages.append('baseline')
                receipt = evaluation_baseline.install(json.loads(manifest.read_text()), self.repo)
                return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(receipt))
            return subprocess.CompletedProcess(argv, 0)
        self.runs.side_effect = run
        runner = self.mod.ServiceRunner(0, str(self.repo))
        def candidate(*args):
            self.assertEqual((self.repo / 'recipe.py').read_bytes(), b'baseline')
            stages.append('candidate')
            (self.repo / 'recipe.py').write_bytes(b'candidate')
        with patch.object(runner, '_apply_oci_layer', side_effect=candidate):
            runner.apply_and_start(LAYER, LAYER_SHA, 'fixture-commit', 'fixture-tree', 'fd00::1',
                                   oci={'diff_id': DIFF_ID}, evaluation_baseline=hashes)
        self.assertEqual(stages, ['baseline', 'candidate'])
        self.assertEqual((self.repo / 'recipe.py').read_bytes(), b'candidate')
        stages.clear()
        with self.assertRaisesRegex(RuntimeError, 'worktree'):
            runner.apply_and_start(LAYER, LAYER_SHA, 'fixture-commit', 'wrong-tree', '',
                                   oci={'diff_id': DIFF_ID}, evaluation_baseline=hashes)
        self.assertEqual(stages, [])

    def test_helper_source_is_materialized_locally_for_ray_workers(self) -> None:
        # 多节点 Ray actor 路径:worker 节点没有 /logs/verifier/oci_layer.py,源码随参数下发。
        runner = self.mod.ServiceRunner(1, str(self.repo))
        runner.apply_and_start(
            LAYER, LAYER_SHA, "fixture-commit", "fixture-tree", "fd00::1",
            oci={"diff_id": DIFF_ID, "forbidden_paths": [], "helper_source": "print('helper')\n"},
        )
        local = self.root / "deploy/oci_layer.py"
        self.assertEqual(local.read_text(), "print('helper')\n")
        self.assertEqual(self.oci_calls[0][3], str(local))
        self.assertTrue((self.root / "deploy/oci-layer-rank-1.json").is_file())

    def test_no_forbidden_paths_passes_empty_arg(self) -> None:
        self._apply(forbidden=())
        self.assertEqual(self.oci_calls[0][-1], "")

    def test_sha_mismatch_fails_before_touching_rootfs(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "SHA256 不一致"):
            self._apply(expected_sha="sha256:" + "0" * 64)
        self.assertEqual(self.oci_calls, [])

    def test_helper_digest_mismatch_fails(self) -> None:
        self.summary["diff_id"] = "sha256:" + "1" * 64
        with self.assertRaisesRegex(RuntimeError, "diff_id 不一致"):
            self._apply()
        self.summary["diff_id"] = DIFF_ID
        self.summary["layer_sha256"] = "sha256:" + "2" * 64
        with self.assertRaisesRegex(RuntimeError, "layer_sha256"):
            self._apply()

    def test_legacy_patch_path_still_git_applies(self) -> None:
        runner = self.mod.ServiceRunner(0, str(self.repo))
        patch_bytes = b"diff --git a/x b/x\n"
        runner.apply_and_start(
            patch_bytes, hashlib.sha256(patch_bytes).hexdigest(), "fixture-commit", "fixture-tree", "fd00::1",
        )
        self.assertEqual(self.oci_calls, [])
        self.assertTrue(any("apply" in call.args[0] for call in self.runs.call_args_list))


if __name__ == "__main__":
    unittest.main()

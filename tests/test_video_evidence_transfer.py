"""验证视频摘要模式不回传 MP4、指标一致，以及默认模式的不可变文件续传。

参数：无。示例：PYTHONPATH=. .venv/bin/python -m unittest tests.test_video_evidence_transfer
不创建远端资源，不启动模型。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from verify.inference.infer_verifier import InferVerifier, _load_verifier_config


class VideoEvidenceTransferTest(unittest.IsolatedAsyncioTestCase):
    async def test_summary_only_never_downloads_mp4_and_preserves_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cases=[{'id':'case-1','file':'video-outputs/case-1.mp4','sha256':'a'*64,'passed':True}]
            gate={'accuracy':1.0,'passed':True,'cases':cases}
            summary={'gate':gate,'performance':{'output_throughput':6.63,'wall_time_sec':2171.9}}
            files={'gate-result.json':gate,'video-evidence.json':{'cases':cases},'benchmark-summary.json':summary}
            calls=[]
            async def download(source,target):
                calls.append(source)
                self.assertFalse(source.endswith('.mp4'))
                Path(target).write_text(json.dumps(files.get(Path(source).name,{})))
            environment=SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(stdout='done',stderr='')),
                download_file=download,download_completed_file=download)
            verifier=SimpleNamespace(environment=environment,trial_paths=SimpleNamespace(verifier_dir=root))
            async def collect(ids,**kwargs):return await InferVerifier._download_video_evidence(verifier,ids,**kwargs)
            verifier._download_video_evidence=collect
            result=await InferVerifier._collect_clean_benchmark(verifier,0,{'video_benchmark':{'mode':'video-only','case_ids':['case-1']}},
                SimpleNamespace(gate_kind='video-reference',collect_video_outputs=False))
            self.assertEqual(result,summary)
            self.assertFalse((root/'video-outputs').exists())
            self.assertFalse(json.loads((root/'video-collection.json').read_text())['media_collected'])

    async def test_summary_only_rejects_inconsistent_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            async def download(source,target):
                name=Path(source).name
                data={'cases':[{'id':'case-1','file':'video-outputs/case-1.mp4','sha256':'a'*64}]} if name=='video-evidence.json' else {}
                if name=='benchmark-summary.json':data={'gate':{'accuracy':1.0}}
                Path(target).write_text(json.dumps(data))
            environment=SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(stdout='done',stderr='')),
                download_file=download,download_completed_file=download)
            verifier=SimpleNamespace(environment=environment,trial_paths=SimpleNamespace(verifier_dir=root))
            async def collect(ids,**kwargs):return await InferVerifier._download_video_evidence(verifier,ids,**kwargs)
            verifier._download_video_evidence=collect
            with self.assertRaisesRegex(ValueError,'不一致'):
                await InferVerifier._collect_clean_benchmark(verifier,0,{'video_benchmark':{'mode':'video-only','case_ids':['case-1']}},
                    SimpleNamespace(gate_kind='video-reference',collect_video_outputs=False))

    def test_wan_compiled_transport_policy_does_not_change_video_protocol(self):
        root=Path(__file__).resolve().parents[1]
        tests=root/'tasks/infer-wan2p2ti2v-5b/tests'
        self.assertFalse(_load_verifier_config(tests).collect_video_outputs)
        raw=json.loads((tests/'verifier-config.json').read_text())
        self.assertNotIn('collect_video_outputs',raw['video_benchmark'])
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            raw['collect_video_outputs']='false'
            (folder/'verifier-config.json').write_text(json.dumps(raw))
            with self.assertRaisesRegex(TypeError,'布尔'):_load_verifier_config(folder)

    async def test_video_files_use_completed_download_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = b"video-evidence"
            manifest = {
                "cases": [{
                    "id": "case-1",
                    "file": "video-outputs/case-1.mp4",
                    "sha256": hashlib.sha256(video).hexdigest(),
                }]
            }
            calls: list[str] = []

            async def download_completed(source: str, target: Path) -> None:
                calls.append(source)
                target = Path(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.endswith("video-evidence.json"):
                    target.write_text(json.dumps(manifest))
                else:
                    target.write_bytes(video)

            environment = SimpleNamespace(
                download_file=AsyncMock(side_effect=AssertionError("legacy download used")),
                download_completed_file=download_completed,
            )
            verifier = SimpleNamespace(
                environment=environment,
                trial_paths=SimpleNamespace(verifier_dir=root),
            )

            await InferVerifier._download_video_evidence(verifier, ["case-1"])

            self.assertEqual(calls, [
                "/logs/verifier/video-evidence.json",
                "/logs/verifier/video-outputs/case-1.mp4",
            ])
            self.assertEqual(
                (root / "video-outputs/case-1.mp4").read_bytes(), video
            )

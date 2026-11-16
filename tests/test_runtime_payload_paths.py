"""通过本地异步替身执行 Modal 资源准备和部署接线，不创建云资源。

示例：python3 -m pytest tests/test_runtime_payload_paths.py。
"""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from harbor.models.trial.paths import TrialPaths
from harbor_envs.abstract_mlplat.mlplatform_base import MLPlatformEnvironment
from verify.inference.infer_verifier import InferVerifier, CONFIG_NAME, METADATA_NAME
from verify.inference.runtime_infer_verifier import RuntimeInferVerifier
from verify.resources import payload_file, source_file


class RuntimePayloadPathsTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / '.agents/workdir'
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.verifier = object.__new__(RuntimeInferVerifier)
        self.verifier.trial_paths = TrialPaths(trial_dir=self.root / 'trial')
        self.verifier.trial_paths.verifier_dir.mkdir(parents=True)
        self.verifier._verifier_config_path.write_text('{}')
        self.verifier.environment = Mock(spec=MLPlatformEnvironment)
        self.verifier.environment.platform_name = 'modal'
        self.verifier.environment.prepare_inputs = AsyncMock()
        self.verifier._wait_for_health = AsyncMock()
        self.metadata = self.root / METADATA_NAME
        self.metadata.write_text('{}')
        self.layer = self.root / 'submit_oci.tar.gz'
        self.layer.write_bytes(b'fixture layer')

    async def test_modal_preflight_resolves_manifest_payload_before_parent_verification(self):
        runtime = {'gpus': 1, 'gpu': 'H200'}
        runtime['sha256'] = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
        runtime_path = self.root / 'runtime.json'
        runtime_path.write_text(json.dumps(runtime))
        self.verifier.task = SimpleNamespace(
            paths=SimpleNamespace(tests_dir=self.root),
            config=SimpleNamespace(metadata={'runtime_spec_sha256': runtime['sha256']}),
        )
        self.verifier._artifact = Mock(return_value=self.metadata)
        outcome = object()
        with (
            patch('verify.inference.runtime_infer_verifier._load_verifier_config',
                  return_value=SimpleNamespace(engine_repo='/engine')),
            patch('verify.delivery.patch_eval.verify_clean_baseline', new_callable=AsyncMock) as baseline,
            patch.object(InferVerifier, '_verify_impl', new_callable=AsyncMock, return_value=outcome) as parent,
        ):
            self.assertIs(await self.verifier._verify_impl(), outcome)
        baseline.assert_awaited_once()
        parent.assert_awaited_once()
        self.verifier.environment.prepare_inputs.assert_awaited_once_with(
            runtime_path, payload_file('modal_inputs.py'))
        self.assertEqual(self.verifier._runtime, runtime)

    async def test_modal_deployment_uploads_trusted_controller_and_oci_payloads(self):
        self.verifier._runtime = {'gpus': 1, 'gpu': 'H200'}
        serve = SimpleNamespace(
            exec=AsyncMock(side_effect=[
                SimpleNamespace(return_code=0, stdout='NVIDIA H200, gpu-fixture, 140000, driver\n'),
                SimpleNamespace(return_code=0, stdout='started'),
            ]),
            lifecycle={}, save_lifecycle=Mock(), upload_file=AsyncMock(),
            service_endpoint=AsyncMock(return_value='https://fixture.invalid'),
        )
        config = SimpleNamespace(python='python3', patch_artifact=self.layer.name)
        endpoint = await self.verifier._start_patched_service(serve, self.layer, self.metadata, config)
        self.assertEqual(endpoint, 'https://fixture.invalid')
        uploaded = {remote: local for local, remote in
                    (call.args for call in serve.upload_file.await_args_list)}
        self.assertEqual(len(uploaded), 5)
        for name in ('serve_controller.py', 'oci_layer.py'):
            local = uploaded['/logs/verifier/' + name]
            self.assertEqual(local, payload_file(name))
            self.assertEqual(local.read_bytes(), source_file(name).read_bytes())
        self.assertEqual(uploaded['/logs/verifier/' + CONFIG_NAME], self.verifier._verifier_config_path)
        self.verifier._wait_for_health.assert_awaited_once_with(serve, endpoint, config)
        self.assertEqual(serve.lifecycle['status'], 'measuring')

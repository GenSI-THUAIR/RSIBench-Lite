"""Modal 共用生命周期、创建取消与只读挂载的回归；不申请云资源。"""
import asyncio
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from harbor_envs.abstract_mlplat.mlplatform_base import MLPlatformEnvironment

HAS_MODAL=importlib.util.find_spec('modal') is not None
if HAS_MODAL:
    from harbor_envs.modal import ModalEnvironment

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_MODAL,'requires Modal extra')
class ModalLifecycleTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.agents/workdir')
        self.addCleanup(self.temp.cleanup)
        environment_dir=Path(self.temp.name)/'environment'
        environment_dir.mkdir()
        (environment_dir/'Dockerfile').write_text('FROM debian:bookworm-slim\n')
        self.env=ModalEnvironment(
            environment_dir=environment_dir,
            environment_name='test',session_id='test__patched_serve',
            trial_paths=TrialPaths(trial_dir=Path(self.temp.name)),
            task_env_config=EnvironmentConfig(workdir='/app'))

    async def test_shared_start_failures_reclaim_owned_sandbox(self):
        self.env._stop_owned_job=AsyncMock()
        with patch.object(MLPlatformEnvironment,'start',new=AsyncMock(side_effect=RuntimeError('bootstrap failed'))):
            with self.assertRaisesRegex(RuntimeError,'bootstrap failed'):
                await self.env.start(False)
        self.env._stop_owned_job.assert_awaited_once()

    async def test_single_node_service_watch_uses_modal_exec(self):
        from verify.grading.service_liveness import check_launches, ServiceExitedError
        self.env._strategy.exec = AsyncMock(return_value=SimpleNamespace(
            stdout='RUNNING\n', stderr='', return_code=0))
        await check_launches(self.env, [None], label='Modal')
        self.env._strategy.exec.assert_awaited_once()
        self.env._strategy.exec.side_effect = [
            SimpleNamespace(stdout='1\n', stderr='', return_code=0),
            SimpleNamespace(stdout='service failed', stderr='', return_code=0),
        ]
        with self.assertRaisesRegex(ServiceExitedError, 'service failed'):
            await check_launches(self.env, [None], label='Modal')

    async def test_named_pod_is_rejected_before_modal_execution(self):
        self.env._strategy.exec = AsyncMock()
        with self.assertRaisesRegex(ValueError, 'one execution target'):
            await self.env.exec('true', pod_name='head-1')
        self.env._strategy.exec.assert_not_awaited()

    async def test_stop_still_terminates_when_shared_cleanup_fails(self):
        sandbox=SimpleNamespace(object_id='sb-owned',terminate=SimpleNamespace(aio=AsyncMock()),
            wait=SimpleNamespace(aio=AsyncMock()),poll=SimpleNamespace(aio=AsyncMock(return_value=137)))
        self.env._sandbox=sandbox
        self.env.lifecycle={}
        with patch.object(MLPlatformEnvironment,'stop',new=AsyncMock(side_effect=RuntimeError('cleanup failed'))):
            with self.assertRaisesRegex(RuntimeError,'cleanup failed'):
                await self.env.stop(True)
        sandbox.terminate.aio.assert_awaited_once()
        self.assertIsNone(self.env._sandbox)
        self.assertEqual(self.env.lifecycle['serve_terminal_status'],137)

    async def test_terminal_observation_is_read_only_and_timeout_is_unknown(self):
        poll = AsyncMock(side_effect=[None, 137, TimeoutError('temporary observation timeout')])
        sandbox = SimpleNamespace(object_id='sb-test', poll=SimpleNamespace(aio=poll),
                                  terminate=SimpleNamespace(aio=AsyncMock()))
        self.env._sandbox = sandbox
        self.assertIsNone(await self.env.terminal_state_reason())
        self.assertIn('exit=137', await self.env.terminal_state_reason())
        self.assertIsNone(await self.env.terminal_state_reason())
        sandbox.terminate.aio.assert_not_awaited()

    def test_evaluation_factory_disables_agent_trajectory_collection(self):
        self.env._sandbox = SimpleNamespace(object_id='sb-benchmark')
        task = SimpleNamespace(short_name='fixture', paths=SimpleNamespace(environment_dir=self.env.environment_dir),
                               config=SimpleNamespace(verifier=SimpleNamespace(timeout_sec=60)))
        runtime = {'gpus': 0, 'gpu': 'H200', 'serve_cpus': 1, 'serve_memory_mb': 1024,
                   'network': {}, 'inputs': [], 'sha256': 'test'}
        serve = self.env.create_evaluation_environment(task=task, trial_paths=self.env.trial_paths,
            logger=self.env.logger, runtime=runtime, scoring={})
        self.assertFalse(serve._collect_agent_trajectory)

    def test_local_mode_rejects_native_gpu_override_before_dispatch(self):
        with self.assertRaisesRegex(ValueError, 'CPU-only'):
            ModalEnvironment(environment_dir=self.env.environment_dir,
                environment_name='test', session_id='test-local-override',
                trial_paths=self.env.trial_paths,
                task_env_config=EnvironmentConfig(workdir='/app', gpus=0),
                local_mode=True, override_gpus=1)

    def test_local_mode_evaluation_factory_rejects_extra_gpus(self):
        runtime = {'evaluation_mode': 'local', 'gpus': 2, 'gpu': 'H200'}
        with self.assertRaisesRegex(ValueError, 'exactly one H200'):
            self.env.create_evaluation_environment(task=None, trial_paths=self.env.trial_paths,
                logger=self.env.logger, runtime=runtime, scoring={})

    async def test_cancelled_create_is_claimed_and_terminated(self):
        self.env._encrypted_ports=[8000]
        self.env._image=SimpleNamespace(build=SimpleNamespace(aio=AsyncMock()),object_id='im-test')
        self.env._app=Mock()
        started=asyncio.Event()
        release=asyncio.Event()
        sandbox=SimpleNamespace(object_id='sb-owned',terminate=SimpleNamespace(aio=AsyncMock()),
            wait=SimpleNamespace(aio=AsyncMock()),poll=SimpleNamespace(aio=AsyncMock(return_value=137)))
        async def create(*args,**kwargs):
            started.set()
            await release.wait()
            return sandbox
        with patch('harbor_envs.modal.environment.modal.Sandbox.create.aio',side_effect=create), \
             patch.object(self.env,'_secrets_config',return_value=[]):
            pending=asyncio.create_task(self.env._create_sandbox(entrypoint=['sleep','infinity']))
            await started.wait()
            pending.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await pending
        sandbox.terminate.aio.assert_awaited_once()
        self.assertIsNone(self.env._sandbox)

    async def test_rebuilt_image_is_allowed_and_recorded(self):
        self.env._encrypted_ports=[8000]
        self.env._image=SimpleNamespace(build=SimpleNamespace(aio=AsyncMock()),object_id='im-new')
        self.env._app=Mock()
        self.env.lifecycle={}
        sandbox=SimpleNamespace(object_id='sb-rebuilt')
        with patch('harbor_envs.modal.environment.modal.Sandbox.create.aio',new=AsyncMock(return_value=sandbox)) as create, \
             patch.object(self.env,'_secrets_config',return_value=[]):
            self.assertIs(await self.env._create_sandbox(),sandbox)
            create.assert_awaited_once()
        self.assertEqual(self.env.lifecycle['service_image_id'],'im-new')

    def test_model_volume_is_read_only_and_conflicts_fail(self):
        self.env._readonly_volumes={'/model':'frozen-model'}
        with patch('harbor_envs.modal.environment.modal.Volume.from_name') as volume:
            mount=volume.return_value.with_mount_options.return_value
            self.assertIs(self.env._volumes_config()['/model'],mount)
            volume.return_value.with_mount_options.assert_called_once_with(read_only=True)
            self.env._volumes={'/model':'writable-model'}
            with self.assertRaisesRegex(ValueError,'both writable and read-only'):
                self.env._volumes_config()


if __name__=='__main__':
    unittest.main()

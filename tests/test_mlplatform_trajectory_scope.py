"""验证Agent轨迹只由Agent实例采集，独立verifier/Serve不能污染同一trial账本。

参数：无。示例：python3 -m pytest tests/test_mlplatform_trajectory_scope.py。
"""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from harbor.environments.base import ExecResult
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from harbor_envs.modal import ModalEnvironment

ROOT = Path(__file__).resolve().parents[1]


class TrajectoryScopeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.agents/workdir')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = TrialPaths(trial_dir=self.root / 'task__verifier__in-name__abc')

    def environment(self, role, **kwargs):
        return ModalEnvironment(environment_dir=self.root / 'environment',
            environment_name='scope-test', session_id=f'{self.paths.trial_dir.name}__{role}',
            trial_paths=self.paths, task_env_config=EnvironmentConfig(workdir='/tmp/scope-test', docker_image='python:3.12-slim'),
            **kwargs)

    async def test_evaluator_finalization_cannot_overwrite_agent_archive(self):
        path = self.paths.agent_dir / 'trajectory-raw/manifest.json'
        path.parent.mkdir(parents=True)
        original = b'{"version":1,"trial_session_id":"agent","sources":{},"collection_errors":{}}\n'
        path.write_bytes(original)
        for env in (self.environment('verifier__step__nested'),
                    self.environment('patched_serve', collect_agent_trajectory=False)):
            with self.subTest(session_id=env.session_id):
                env._exec_cmd_platform = AsyncMock(return_value=ExecResult(return_code=1, stderr='ssh unavailable'))
                await env.finalize_trajectory_sync(strict=True)
                env._exec_cmd_platform.assert_not_awaited()
                self.assertEqual(path.read_bytes(), original)

    async def test_lifecycle_starts_poller_only_for_agent(self):
        for role, options, expected in (
            ('env', {}, True),
            ('verifier__trial', {}, False),
            ('patched_serve', {'collect_agent_trajectory': False}, False),
        ):
            with self.subTest(role=role):
                env = self.environment(role, **options)
                for method in ('_ensure_job', '_wait_for_exec', '_start_cluster',
                               '_wait_for_cluster_ready', 'empty_dirs', '_install_task_skills',
                               '_on_environment_started'):
                    setattr(env, method, AsyncMock())
                env._requires_workdir_lock = Mock(return_value=False)
                env._exec_cmd_platform = AsyncMock(return_value=ExecResult(return_code=0))
                env._sync_trajectory_loop = AsyncMock()
                await env.start(False)
                await asyncio.sleep(0)
                self.assertEqual(env._sync_trajectory_loop.await_count, int(expected))
                self.assertEqual(env._trajectory_sync_task is not None, expected)
                await env._stop_trajectory_sync()

    def test_collection_option_rejects_non_boolean(self):
        with self.assertRaisesRegex(TypeError, 'boolean'):
            self.environment('env', collect_agent_trajectory='false')

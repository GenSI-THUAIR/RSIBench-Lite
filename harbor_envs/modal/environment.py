"""Modal 的 MLPlatformEnvironment 实现；复用上游传输、公共生命周期。

由原生 Harbor environment.import_path 加载，无启动 wrapper。
kwargs 延用 Harbor Modal 配置；服务额外支持只读 Volume 和 HTTPS 端口。
示例：bash scripts/run/run_task.sh --config configs/harbor-modal-smoke.yaml --agent oracle。
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import modal
from harbor.environments.modal import ModalEnvironment as HarborModalEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor_envs.abstract_mlplat.mlplatform_base import MLPlatformEnvironment


class ModalEnvironment(MLPlatformEnvironment, HarborModalEnvironment):
    """Agent、Benchmark 和 Serve 共用的 Modal 平台适配。"""

    def __init__(self, *args, readonly_volumes=None, encrypted_ports=None,
                 lifecycle=None, local_mode=False, **kwargs):
        self._readonly_volumes = readonly_volumes or {}
        self._encrypted_ports = encrypted_ports or []
        self.lifecycle = lifecycle
        self._force_build = False
        # 外场只使用Harbor显式传入的skills，不隐式读取内场HDFS技能注册表。
        kwargs.setdefault('install_task_skills', False)
        super().__init__(*args, **kwargs)
        if local_mode and self.task_env_config.gpus:
            raise ValueError('local-mode Agent/Benchmark must be CPU-only')
        self._session_tag = f'{self.session_id}:{os.getpid()}'
        if self._encrypted_ports and (self._compose_mode or self._dynamic_network
                                     or self._network_is_allowlist or self._network_disabled):
            raise ValueError('Published model services require a direct public Modal sandbox')

    # 快速SDK文件传输替代公共exec/base64实现，其余初始化与回收走mlplat。
    upload_file = HarborModalEnvironment.upload_file
    upload_dir = HarborModalEnvironment.upload_dir
    download_file = HarborModalEnvironment.download_file
    download_dir = HarborModalEnvironment.download_dir
    attach = HarborModalEnvironment.attach

    def _requires_workdir_lock(self):
        return False  # 每个实例新建独占Sandbox，不复用外部工作目录。

    def create_evaluation_environment(self, *, task, trial_paths, logger,
                                      profile=None, mode='score', runtime=None, scoring=None):
        if profile is not None:
            raise NotImplementedError('Modal training evaluation is not configured')
        if runtime is None or scoring is None:
            raise ValueError('Modal inference requires a verified runtime and scoring configuration')
        if runtime.get('evaluation_mode') == 'local':
            if (runtime['gpus'] != 1 or runtime['gpu'] != 'H200'
                    or runtime.get('benchmark_gpus', 0) != 0 or self.task_env_config.gpus):
                raise ValueError('local-mode requires CPU Benchmark and exactly one H200 Serve')
        config=EnvironmentConfig(os='linux',workdir='/workspace/deploy',
            gpus=runtime['gpus'],gpu_types=[runtime['gpu']],cpus=runtime['serve_cpus'],
            memory_mb=runtime['serve_memory_mb'],env={**runtime['network'],**runtime.get('driver_env',{})})
        return type(self)(environment_dir=task.paths.environment_dir,
            environment_name=f'{task.short_name}-serve',
            session_id=f'{trial_paths.trial_dir.name}__patched_serve',trial_paths=trial_paths,logger=logger,
            collect_agent_trajectory=False,
            task_env_config=config,app_name=self._app_name+'-serve',encrypted_ports=[8000],
            readonly_volumes={group['mount']:group['volume'] for group in runtime['inputs'] if group['kind']=='model'},
            sandbox_timeout_secs=min(86400,int(task.config.verifier.timeout_sec)+600),
            lifecycle={'backend':'modal','runtime_spec_sha256':runtime['sha256'],
                'evaluation_mode':runtime.get('evaluation_mode', 'separate'),
                'agent_runtime':'harbor_envs.modal.ModalEnvironment',
                'benchmark_session_id':self.session_id,'benchmark_sandbox_id':self.sandbox_id})

    async def start(self, force_build):
        self._force_build = force_build
        try:
            await super().start(force_build)
        except BaseException:
            await asyncio.shield(self._stop_owned_job())
            raise

    async def _ensure_job(self):
        await HarborModalEnvironment.start(self, self._force_build)

    async def terminal_state_reason(self):
        if self._sandbox is None:
            return None
        try:
            status = await self._sandbox.poll.aio()
        except modal.exception.NotFoundError:
            return f'Modal sandbox {self._sandbox.object_id} no longer exists'
        except (TimeoutError, OSError, modal.exception.TimeoutError,
                modal.exception.ServiceError, modal.exception.InternalError) as exc:
            self.logger.warning('Modal terminal-state observation failed: %s', exc)
            return None
        if status is not None:
            return f'Modal sandbox {self._sandbox.object_id} entered terminal state: exit={status}'
        return None

    async def _stop_owned_job(self):
        sandbox = self._sandbox
        if sandbox is None:
            return
        await sandbox.terminate.aio()
        await sandbox.wait.aio(raise_on_termination=False)
        terminal_status = await sandbox.poll.aio()
        if terminal_status is None:
            raise RuntimeError(f'Modal sandbox did not terminate: {sandbox.object_id}')
        self._sandbox = None
        if self.lifecycle is not None:
            self.lifecycle.update(serve_terminal_status=terminal_status,
                                  finished_at=time.time())
            self.save_lifecycle()

    async def stop(self, delete):
        try:
            await super().stop(delete)
        finally:
            if self._sandbox is not None:
                await asyncio.shield(self._stop_owned_job())

    async def _exec_cmd_platform(self, command, *, cwd=None, env=None,
                                 timeout_sec=None, user=None, pod_name=None, on_output=None):
        if pod_name is not None:
            raise ValueError('Modal sandbox has one execution target; pod_name is unsupported')
        result = await self.exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)
        if on_output is not None:
            if result.stdout:
                await on_output(result.stdout, 'stdout')
            if result.stderr:
                await on_output(result.stderr, 'stderr')
        return result

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None, *, pod_name=None):
        if pod_name is not None:
            raise ValueError('Modal sandbox has one execution target; pod_name is unsupported')
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)

    def _volumes_config(self):
        mounts = super()._volumes_config()
        if mounts.keys() & self._readonly_volumes.keys():
            raise ValueError('A Volume mount cannot be both writable and read-only')
        mounts.update({path: modal.Volume.from_name(name).with_mount_options(read_only=True)
                       for path, name in self._readonly_volumes.items()})
        return mounts

    async def _create_sandbox(self, *, entrypoint=None, block_network=None, experimental_options=None):
        if not self._encrypted_ports:
            return await super()._create_sandbox(entrypoint=entrypoint,
                block_network=block_network, experimental_options=experimental_options)
        if experimental_options:
            raise ValueError('Model services require the standard GPU-capable Modal runtime')
        await self._image.build.aio(self._app)
        # 不重试创建，避免不确定响应造成重复申请；取消时仍接管并回收已创建实例。
        async def create_and_claim():
            self._sandbox = await modal.Sandbox.create.aio(*(entrypoint or ()),
                app=self._app, image=self._image, name=self.session_id,
                timeout=self._sandbox_timeout, cpu=self._cpu_config(), memory=self._memory_config(),
                gpu=self._gpu_config(), volumes=self._volumes_config(), secrets=self._secrets_config(),
                encrypted_ports=self._encrypted_ports)
            if self.lifecycle is not None:
                self.lifecycle.update(serve_id=self._sandbox.object_id,
                                      service_image_id=self._image.object_id, created_at=time.time())
                self.save_lifecycle()
            return self._sandbox
        pending = asyncio.create_task(create_and_claim())
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            try:
                await pending
            finally:
                await self._stop_owned_job()
            raise

    async def service_endpoint(self, port):
        return (await self._sandbox.tunnels.aio())[port].url

    async def prepare_inputs(self, runtime_path, script_path):
        """可信CPU环境校验数据/tokenizer，完整权重留在服务的只读Volume。"""
        from modal.config import config
        auth={name:str(config.get(key)) for name,key in
              [('MODAL_TOKEN_ID','token_id'),('MODAL_TOKEN_SECRET','token_secret')]}
        if any(value in {'None',''} for value in auth.values()):
            raise ValueError('Modal credentials unavailable for trusted input preparation')
        await self.upload_file(script_path,'/logs/verifier/modal_inputs.py')
        await self.upload_file(runtime_path,'/logs/verifier/runtime.json')
        result=await self.exec(
            'python3 /logs/verifier/modal_inputs.py /logs/verifier/runtime.json /logs/verifier/inputs.json',
            cwd='/tmp',env=auth,timeout_sec=1800)
        if result.return_code:
            raise RuntimeError(f'CPU input preparation failed: {result.stderr or result.stdout}')
        await self.download_file('/logs/verifier/inputs.json',self.trial_paths.verifier_dir/'inputs.json')

    def save_lifecycle(self):
        if self.lifecycle is not None:
            out = self.trial_paths.verifier_dir
            out.mkdir(parents=True, exist_ok=True)
            (out/'runtime-lifecycle.json').write_text(json.dumps(self.lifecycle, indent=2)+'\n')

    @property
    def sandbox_id(self):
        return self._sandbox.object_id

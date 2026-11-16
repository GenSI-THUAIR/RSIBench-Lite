"""原生 Harbor 配置选择的推理验证器：复用评分协议，按环境部署独立模型服务。

参数由 Harbor verifier kwargs 提供，mode=smoke|calibrate|score。
示例：bash scripts/run/run_task.sh --config configs/harbor-modal-smoke.yaml --agent oracle。
Agent/Benchmark/Serve 共用 MLPlatformEnvironment；此处只选择协议与部署控制器。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
import urllib.request

from harbor.models.verifier.result import VerifierResult
from harbor_envs.abstract_mlplat.mlplatform_base import MLPlatformEnvironment
from verify.resources import payload_file
from verify.inference.baseline import baseline_conditions, check_baseline_policy
from verify.inference.sol_score import add_sol_score, reference_digest, validate_reference
from verify.inference.infer_verifier import (
    InferVerifier, CONFIG_NAME, METADATA_NAME,
    DeploymentFailed, _accuracy_passes, _load_verifier_config,
)


class RuntimeInferVerifier(InferVerifier):
    def __init__(self, *args, mode='score', **kwargs):
        super().__init__(*args, **kwargs)
        if mode not in {'smoke','calibrate','score'}:
            raise ValueError(f'Unknown verification mode: {mode}')
        self.mode=mode

    def _finalize_rewards(self, rewards):
        if self.mode == 'score' and self.backend == 'modal':
            return add_sol_score(rewards, self._sol_reference)
        return rewards

    def _score_summary(self, summary, config):
        if self.mode=='score':
            return super()._score_summary(summary,config)
        gate, perf=summary['gate'],summary['performance']
        valid=perf['output_throughput']>0 and perf['request_success_rate']==1
        if self.mode=='calibrate':
            valid=valid and _accuracy_passes(gate['accuracy'],config.gate_center,config.gate_tolerance,config.gate_comparison)
        elif config.gate_kind == 'video-reference':
            # 视频 reference receipt 绑定完整工作集；smoke 也保留全部 case 和原采样参数。
            valid=valid and gate['passed'] and perf['completed']==len(gate['cases'])
        else:
            valid=valid and perf['completed']==2 and gate['num_questions']==3 and gate['invalid_rate']==0
        return {f'{self.mode}_complete':int(valid), 'output_throughput':perf['output_throughput'],
                'gate_accuracy':gate['accuracy'],'request_success_rate':perf['request_success_rate'],
                'benchmark_wall_time_sec':perf['wall_time_sec']}

    def _artifact(self, name):
        if self.step_name is None:
            return super()._artifact(name)
        path=self.trial_paths.step_artifacts_dir(self.step_name)/name
        if not path.is_file():
            raise FileNotFoundError(f'Missing collected step artifact: {path}')
        return path

    def _health_opener(self):
        if self.backend == 'modal':
            # Modal HTTPS tunnel is public; the control host may require an egress proxy.
            return urllib.request.build_opener(urllib.request.ProxyHandler())
        return super()._health_opener()

    async def _poll_benchmark_status(self, command: str | None = None):
        if self.backend != 'modal':
            return await super()._poll_benchmark_status(command)
        from python_socks import ProxyError, ProxyTimeoutError

        # SDK未重试HTTP CONNECT代理的504；这里只重试无副作用的状态查询。
        # 启动、上传、评分与永久鉴权错误仍直接失败，不推断评测已经结束。
        delays = (5, 10, 20)
        for attempt in range(len(delays) + 1):
            try:
                return await super()._poll_benchmark_status(command)
            except (ProxyError, ProxyTimeoutError) as error:
                if isinstance(error, ProxyError) and error.error_code not in {502, 503, 504}:
                    raise
                if attempt == len(delays):
                    raise
                self.logger.warning(
                    'Transient Modal benchmark status connection failure; retrying in %ss: %s',
                    delays[attempt], error,
                )
                await asyncio.sleep(delays[attempt])

    @property
    def backend(self):
        if not isinstance(self.environment, MLPlatformEnvironment):
            raise TypeError('Runtime verifier requires an MLPlatformEnvironment')
        return self.environment.platform_name

    @property
    def _verifier_config_path(self):
        return self.trial_paths.verifier_dir / CONFIG_NAME

    def _resolve_config(self):
        raw=json.loads((self.task.paths.tests_dir/CONFIG_NAME).read_text())
        if self.backend=='modal':
            raw.update(json.loads((self.task.paths.tests_dir/'modal-scoring.json').read_text()))
            runtime=json.loads((self.task.paths.tests_dir/'runtime.json').read_text())
            self._baseline_conditions=baseline_conditions(runtime, raw)
            policy=runtime.get('baseline_policy','strict')
            if self.mode=='score' or (policy=='reuse_resources'
                    and raw['performance_baseline'].get('status')=='CALIBRATED'):
                check_baseline_policy(raw['performance_baseline'], self._baseline_conditions, policy)
            recorded=raw['performance_baseline'].get('baseline_conditions')
            self.trial_paths.verifier_dir.mkdir(parents=True,exist_ok=True)
            (self.trial_paths.verifier_dir/'baseline-usage.json').write_text(json.dumps({
                'policy':policy, 'mode':self.mode, 'used_for_score':self.mode=='score',
                'conditions_match':recorded==self._baseline_conditions,
                'reused_with_different_resources':policy=='reuse_resources' and recorded!=self._baseline_conditions,
                'recorded_conditions':recorded, 'runtime_conditions':self._baseline_conditions,
                'baseline_source':raw['performance_baseline'].get('source'),
                'baseline_output_throughput':raw['performance_baseline'].get('output_throughput'),
            },indent=2)+'\n')
        if self.mode=='smoke' and raw['gate']['kind']!='video-reference':
            raw['gate']['num_questions']=3
            raw['bench_serving']['num_prompts']=2
            if 'window_duration_sec' in raw['bench_serving']:
                raw['bench_serving'].update(window_duration_sec=120,window_full_duration=False)
        if (self.mode=='score' and self.backend=='modal'
                and raw['performance_baseline'].get('status')!='CALIBRATED'):
            raise ValueError('No calibrated baseline for this runtime; run calibration first')
        if self.mode == 'score' and self.backend == 'modal':
            reference = raw.get('sol_reference')
            if not isinstance(reference, dict):
                raise ValueError('Missing frozen SOL reference; compile a new score-mode task snapshot')
            if reference_digest(reference) != runtime.get('sol_reference_sha256'):
                raise ValueError('Frozen SOL reference hash differs from the run snapshot')
            validate_reference(reference, raw['performance_baseline'], runtime['gpu'], runtime['gpus'])
            self._sol_reference = reference
        self.trial_paths.verifier_dir.mkdir(parents=True,exist_ok=True)
        self._verifier_config_path.write_text(json.dumps(raw,indent=2)+'\n')
        self._runtime_scoring=raw
        return raw

    async def verify(self):
        self._resolve_config()
        if self.mode=='score':
            return await super().verify()
        outcome=await self._verify_impl()
        if outcome.failed or outcome.rewards.get(f'{self.mode}_complete')!=1:
            raise RuntimeError(f'{self.mode} did not pass: {outcome.rewards}')
        record={**outcome.rewards, 'mode':self.mode, 'backend':self.backend,
                'runtime_spec_sha256':self.task.config.metadata.get('runtime_spec_sha256') if self.backend=='modal' else None,
                'source_task_sha256':self.task.config.metadata['source_task_sha256'],
                'source_launch_sha256':self.task.config.metadata['source_launch_sha256'],
                'evidence':str(self.trial_paths.trial_dir.resolve())}
        if self.backend=='modal':
            record['baseline_conditions']=self._baseline_conditions
        (self.trial_paths.verifier_dir/f'{self.mode}.json').write_text(json.dumps(record,indent=2)+'\n')
        return VerifierResult(rewards=outcome.rewards)

    async def _verify_impl(self):
        if self.backend=='modal':
            from verify.delivery.patch_eval import verify_clean_baseline
            config=_load_verifier_config(self._verifier_config_path.parent)
            metadata=json.loads(self._artifact(METADATA_NAME).read_text())
            await verify_clean_baseline(self.environment,config.engine_repo,metadata)
            runtime_path=self.task.paths.tests_dir/'runtime.json'
            self._runtime=json.loads(runtime_path.read_text())
            claimed=self._runtime.pop('sha256')
            if hashlib.sha256(json.dumps(self._runtime,sort_keys=True).encode()).hexdigest()!=claimed:
                raise ValueError('Runtime specification has changed')
            self._runtime['sha256']=claimed
            if claimed!=self.task.config.metadata['runtime_spec_sha256']:
                raise ValueError('Task/runtime identity mismatch')
            # 与本次结果一起保存已校验的运行条件，标定登记不依赖工作区生成文件。
            (self.trial_paths.verifier_dir/'runtime.json').write_text(json.dumps(self._runtime,indent=2)+'\n')
            await self.environment.prepare_inputs(runtime_path,payload_file('modal_inputs.py'))
        return await super()._verify_impl()

    async def _start_patched_service(self, serve_environment, patch_path, metadata_path, config):
        if self.backend!='modal':
            return await super()._start_patched_service(serve_environment,patch_path,metadata_path,config)
        hardware=await serve_environment.exec(
            'nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader',cwd='/tmp')
        if hardware.return_code:
            raise RuntimeError(hardware.stderr or hardware.stdout)
        if len(hardware.stdout.strip().splitlines())!=self._runtime['gpus'] or self._runtime['gpu'] not in hardware.stdout:
            raise RuntimeError(f'Unexpected GPU allocation: {hardware.stdout}')
        serve_environment.lifecycle.update(hardware=hardware.stdout,status='deploying submitted layer')
        serve_environment.save_lifecycle()
        if 'video_calibration_hardware' in self._runtime:
            from harbor_envs.modal.video_calibration import validate_hardware
            validate_hardware(self._runtime['video_calibration_hardware'],
                              self._runtime['gpu'], self._runtime['gpus'], hardware.stdout)
        for path,name in [(payload_file('serve_controller.py'),'serve_controller.py'),
                          (payload_file('oci_layer.py'),'oci_layer.py'),
                          (patch_path,config.patch_artifact),(metadata_path,METADATA_NAME),
                          (self._verifier_config_path,CONFIG_NAME)]:
            await serve_environment.upload_file(path,'/logs/verifier/'+name)
        command=shlex.join([config.python,'/logs/verifier/serve_controller.py',
            '/logs/verifier/'+config.patch_artifact,'/logs/verifier/'+METADATA_NAME,
            '/logs/verifier/'+CONFIG_NAME])
        result=await serve_environment.exec(command,cwd='/tmp',timeout_sec=1200)
        if result.return_code:
            raise DeploymentFailed(f'Submitted service failed: {result.stderr or result.stdout}')
        endpoint=await serve_environment.service_endpoint(8000)
        await self._wait_for_health(serve_environment,endpoint,config)
        serve_environment.lifecycle.update(endpoint=endpoint,status='measuring')
        serve_environment.save_lifecycle()
        return endpoint

    async def _run_clean_benchmark(self, endpoint, metadata, config):
        summary=await super()._run_clean_benchmark(endpoint,metadata,config)
        # 生命周期回执由环境层写入；benchmark 完成与资源终止是两个独立状态。
        if self.backend=='modal':
            self._serve_environment.lifecycle.update(measurement_complete=True,status='measurement complete')
            self._serve_environment.save_lifecycle()
        return summary

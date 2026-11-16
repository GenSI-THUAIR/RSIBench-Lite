"""验证两层配置真实编译、启动和 SDK 资源传递；云端创建以本地替身拦截。"""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from harbor.environments.factory import EnvironmentFactory
from harbor.models.job.config import JobConfig
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from schemas.build import load_yaml, _materialize, emit_task_toml
from schemas.run_config import prepare_run
import tomllib

ROOT=Path(__file__).resolve().parents[1]


def config(backend='modal',mode='smoke'):
    result=yaml.safe_load((ROOT/f'configs/harbor-{backend}-smoke.yaml').read_text())
    result['verifier']['kwargs']['mode']=mode
    result['tasks']=result['tasks'][:1]
    return result


def test_inherits_task_resources_unless_explicitly_overridden(tmp_path):
    job=config()
    for phase in ('agent','benchmark','serve'):
        job['runtime']['resources'][phase].pop('cpus',None)
        job['runtime']['resources'][phase].pop('memory_mb',None)
    source=load_yaml(ROOT/job['tasks'][0]['path']/'task.yaml')
    compiled=prepare_run(job,directory=tmp_path)
    task=Task(compiled['tasks'][0]['path'])
    assert task.config.environment.cpus==source['environment']['cpus']
    assert task.config.environment.memory_mb==source['environment']['memory_mb']
    runtime=json.loads((task.paths.tests_dir/'runtime.json').read_text())
    assert runtime['serve_cpus']==source['environment']['cpus']
    assert runtime['serve_memory_mb']==source['environment']['memory_mb']
    assert runtime['benchmark_cpus']==source['verifier']['environment']['cpus']


@pytest.mark.parametrize('invalid', ['storage','typo','top-level','phase','multi-node','zero-memory'])
def test_rejects_unsupported_or_unused_configuration(tmp_path,invalid):
    job=config()
    settings=job['runtime']['resources']
    if invalid=='storage':
        settings['serve'].pop('storage_mb')  # 原题 50 GiB 不能静默丢失。
    elif invalid=='typo':
        settings['agent']['cpu']=2
    elif invalid=='top-level':
        job['concurrent_trials']=2
    elif invalid=='phase':
        settings['editor']={}
    elif invalid=='multi-node':
        settings['serve']['nodes']=2
    else:
        settings['agent']['memory_mb']=0
    with pytest.raises(ValueError):
        prepare_run(job,directory=tmp_path)


def test_changed_resources_require_recalibration_for_score_but_allow_calibrate(tmp_path):
    job=config(mode='score')
    job['runtime']['baseline_policy']='strict'
    job['runtime']['resources']['serve']['cpus']=10
    with pytest.raises(ValueError,match='measurement conditions'):
        prepare_run(job,directory=tmp_path/'score')
    job['verifier']['kwargs']['mode']='calibrate'
    compiled=prepare_run(job,directory=tmp_path/'calibrate')
    task=Path(compiled['tasks'][0]['path'])
    assert json.loads((task/'tests/runtime.json').read_text())['serve_cpus']==10
    assert json.loads((task/'tests/modal-scoring.json').read_text())['performance_baseline']['status']=='UNCALIBRATED'






@pytest.mark.parametrize('local_mode', [False, True])
@pytest.mark.parametrize('override',[False,True])
def test_modal_sdk_receives_resources_for_all_three_phases(tmp_path,override,local_mode):
    job=config()
    if local_mode:
        job['runtime']['evaluation_mode'] = 'local'
    settings=job['runtime']['resources']
    if override:
        for phase,cpu,memory in [('agent',3,12288),('benchmark',5,20480),('serve',10,147456)]:
            settings[phase].update(cpus=cpu,memory_mb=memory)
    compiled=prepare_run(job,directory=tmp_path)
    task=Task(compiled['tasks'][0]['path'])
    job_config=JobConfig.model_validate(compiled)
    paths=TrialPaths(trial_dir=tmp_path/'trial')
    envs={}
    for phase,cfg,envdir in [('agent',task.config.environment,task.paths.environment_dir),
                              ('benchmark',task.config.verifier.environment,task.paths.tests_dir)]:
        envs[phase]=EnvironmentFactory.create_environment_from_config(config=job_config.environment,
            environment_dir=envdir,environment_name='audit',session_id='audit-'+phase,
            trial_paths=paths,task_env_config=cfg)
    benchmark=envs['benchmark']
    benchmark._sandbox=SimpleNamespace(object_id='local-only')
    envs['serve']=benchmark.create_evaluation_environment(task=task,trial_paths=paths,logger=benchmark.logger,
        runtime=json.loads((task.paths.tests_dir/'runtime.json').read_text()),scoring={})
    async def check():
        for phase,env in envs.items():
            env._image=SimpleNamespace(build=SimpleNamespace(aio=AsyncMock()),object_id='local-image')
            env._app=object()
            with patch('modal.Sandbox.create.aio',new=AsyncMock(return_value=SimpleNamespace(object_id='local-only'))) as create, patch.object(env,'_volumes_config',return_value={}):
                await env._create_sandbox()
                kwargs=create.await_args.kwargs
                expected_cpu=settings[phase]['cpus'] if override else 22
                assert kwargs['cpu']==(expected_cpu,expected_cpu)
                assert kwargs['memory']==(settings[phase]['memory_mb'] if override else 232448)
                assert kwargs.get('gpu') == (None if local_mode and phase != 'serve' else 'H200:1')
                assert 'arnold_config' not in kwargs
    asyncio.run(check())


def test_real_launcher_selects_task_and_preserves_native_arguments(tmp_path):
    result=subprocess.run(['bash',str(ROOT/'scripts/run/run_task.sh'),'--config','configs/harbor-modal.yaml',
        '--path','tasks/infer-qwen3p8-27b','--agent','oracle','--verifier-kwarg','mode=smoke',
        '--n-concurrent','2','--print-config'],cwd=tmp_path,capture_output=True,text=True,
        env={**os.environ,'PYTHON_BIN':sys.executable,'HARBOR_BIN':str(ROOT/'.venv/bin/harbor')})
    assert result.returncode==0,result.stderr
    resolved=json.loads(result.stdout)
    assert resolved['n_concurrent_trials']==2
    assert resolved['verifier']['kwargs']['mode']=='smoke'
    assert len(resolved['tasks'])==1
    task=Path(resolved['tasks'][0]['path'])
    assert '.agents/workdir/run-config-' in str(task)
    assert (task/'task.toml').exists()
    assert JobConfig.model_validate(resolved).agents[0].name=='oracle'

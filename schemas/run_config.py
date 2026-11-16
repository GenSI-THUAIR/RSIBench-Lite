"""启动时整合运行 YAML 和 task.yaml，生成本次运行独占的 Harbor 任务包。

无独立 CLI；由 scripts/run/run_task_model.py 调用 prepare_run。
只写小型配置/源码快照，不申请资源，不修改源任务，不删除其他运行的文件。
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import shlex

from schemas.build import _materialize, emit_task_toml, load_yaml
from schemas.modal_recipe import cpu_image_recipe, sha
from schemas.instruction_template import render_from_config
from schemas.sol_presets import sol_reference
from harbor_envs.modal.network import modal_driver_env, modal_network_env
from harbor_envs.modal.video_calibration import apply_video_calibration, validate_hardware
from harbor_envs.modal.local_mode import configure_local_job, local_allocations
from verify.inference.baseline import baseline_conditions, check_baseline_policy
from verify.inference.sol_score import reference_digest, validate_reference
from verify.resources import evaluator_sha256
from harbor.models.job.config import JobConfig
from harbor.models.task.task import Task

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_KEYS = {'nodes', 'gpus', 'gpu_types', 'cpus', 'memory_mb', 'storage_mb'}
BACKENDS = {'harbor_envs.modal:ModalEnvironment': 'modal'}


def resources(base, override):
    unknown = set(override) - RESOURCE_KEYS
    if unknown:
        raise ValueError(f'Unknown resource fields: {sorted(unknown)}')
    result = {key: deepcopy(base[key]) for key in RESOURCE_KEYS if key in base}
    result.update(deepcopy(override))
    for key in ('nodes', 'cpus', 'memory_mb', 'storage_mb', 'gpus'):
        value = result.get(key)
        if value is not None and (type(value) is not int or value < (0 if key == 'gpus' else 1)):
            raise ValueError(f'{key} must be a positive integer (gpus may be zero)')
    if result.get('gpus', 0) > 0 and not result.get('gpu_types'):
        raise ValueError('gpu_types is required for a GPU allocation')
    if 'gpu_types' in result and (not isinstance(result['gpu_types'],list)
            or any(not isinstance(gpu,str) or not gpu for gpu in result['gpu_types'])):
        raise ValueError('gpu_types must be a list of nonempty strings')
    return result


def modal_environment(base, allocation, driver_env):
    if allocation.get('nodes', 1) != 1:
        raise ValueError('Modal currently supports one node per service')
    if allocation.get('storage_mb') is not None:
        raise ValueError('Modal storage_mb is unsupported; explicitly set storage_mb: null to use platform storage')
    # 运行 YAML 的 environment.env 由 Harbor 合并。
    return {'os': base.get('os', 'linux'), 'workdir': base['workdir'],
            'build_timeout_sec': base.get('build_timeout_sec', 3600),
            **{k:v for k,v in allocation.items() if k != 'nodes' and v is not None},
            'env': {**modal_network_env(), **driver_env}}


def step_files(source: Path, cfg: dict, instruction_cfg: dict) -> dict[Path, bytes]:
    """原开源任务的单 step OCI 交付，两后端共用，避免改变基线注入顺序。"""
    files = {}
    def put(name, value):
        files[Path(name)] = value.encode() if isinstance(value, str) else value
    scoring = cfg['verifier']['scoring']
    # 保持现有 OCI 单 step 交付：launch 注入必须晚于镜像 Git 基线快照。
    launch = (source/'environment/launch.sh').read_bytes()
    repo = shlex.quote(scoring['engine_repo'])
    instruction = (
        f"Optimize the engine source in {scoring['engine_repo']} for {cfg['metadata']['display']['model']}.\n"
        "Editing resources are selected by the runtime configuration. GPU deployment and measurement run in isolated verifier-owned services after submission.\n"
        "Submit /logs/submission/submit_oci.tar.gz using the existing OCI filesystem-layer contract. Include launch.sh and preserve the supplied model precision and service contract.\n"
        "The trusted verifier applies the layer to a clean engine snapshot and measures the original frozen workload.\n")
    put('steps/optimize/workdir/launch.sh',launch)
    put('steps/optimize/workdir/setup.sh',f'#!/bin/bash\nset -euo pipefail\ninstall -m 0755 ./launch.sh {repo}/launch.sh\n')
    put('steps/optimize/instruction.md',instruction)
    put('instruction.md',render_from_config(instruction_cfg))
    put('task.toml',emit_task_toml(cfg)+
        '\n[[steps]]\nname = "optimize"\n[steps.agent]\nuser = "root"\n'+
        f'timeout_sec = {cfg["agent"]["timeout_sec"]}\n[steps.verifier]\ntimeout_sec = {cfg["verifier"]["timeout_sec"]}\n')
    return files


def modal_files(source: Path, cfg: dict, overrides: dict, mode: str,
                baseline_policy: str = 'strict', evaluation_mode: str = 'separate') -> dict[Path, bytes]:
    """生成一题 Modal 运行文件；配置只来自调用方与该题 YAML。"""
    cfg = deepcopy(cfg)
    platform = cfg.get('platforms', {}).get('modal')
    if not platform:
        raise ValueError(f'{source.name}: task.yaml has no platforms.modal assets')
    unknown = set(platform) - {'image','entrypoint','model','data','setup','baseline','video_calibration'}
    if unknown:
        raise ValueError(f'Unknown Modal task fields: {sorted(unknown)}')
    files = {}
    def put(name, value):
        files[Path(name)] = value.encode() if isinstance(value, str) else value
    scoring = cfg['verifier']['scoring']
    original_scoring = deepcopy(_materialize(cfg)['verifier']['scoring'])
    if 'video_calibration' in platform:
        scoring = apply_video_calibration(scoring, platform['video_calibration'])
        cfg['verifier']['scoring'] = scoring
    base_agent = cfg['environment']
    base_benchmark = cfg['verifier'].get('environment', base_agent)
    allocations = {phase: resources(base, overrides.get(phase, {})) for phase,base in
                   [('agent',base_agent),('benchmark',base_benchmark),('serve',base_agent)]}
    if evaluation_mode == 'local':
        allocations = local_allocations(allocations)
    driver_env = modal_driver_env('vllm' if scoring['gate']['kind'] == 'video-reference'
                                  else scoring['bench_serving']['backend'])
    envs = {phase:modal_environment(base_benchmark if phase=='benchmark' else base_agent, allocation, driver_env)
            for phase,allocation in allocations.items()}
    serve = allocations['serve']
    if serve.get('gpus', 0) < 1 or len(serve['gpu_types']) != 1:
        raise ValueError('Modal inference Serve requires one GPU type and at least one GPU')
    inputs = []
    for kind, mount in [('model',scoring['model_path']),
                        ('data',platform['data'].get('mount','/mnt/rsibench/rsibench'))]:
        descriptor = platform[kind]
        raw = (source/descriptor['manifest']).read_bytes()
        if sha(raw) != descriptor['manifest_sha256']:
            raise ValueError(f'{source.name}: {kind} manifest SHA mismatch')
        inputs.append({**descriptor,'kind':kind,'mount':mount,'files':json.loads(raw)['files']})
    setup = source/platform['setup'] if platform.get('setup') else None
    dockerfile = cpu_image_recipe(platform['image'],scoring['engine_repo'],setup is not None,platform.get('entrypoint'))
    metadata = cfg.setdefault('metadata', {})
    metadata.update(source_task_sha256=sha((source/'task.yaml').read_bytes()),
                    source_launch_sha256=sha((source/'environment/launch.sh').read_bytes()))
    runtime = {
        'backend':'modal','gpu':serve['gpu_types'][0],'gpus':serve['gpus'],
        'serve_cpus':serve['cpus'],'serve_memory_mb':serve['memory_mb'],
        'benchmark_cpus':allocations['benchmark']['cpus'],
        'benchmark_memory_mb':allocations['benchmark']['memory_mb'],
        'benchmark_gpus':allocations['benchmark']['gpus'],
        'benchmark_gpu_types':allocations['benchmark']['gpu_types'],
        'baseline_policy':baseline_policy,
        'inputs':inputs,'image':platform['image'],'engine_repo':scoring['engine_repo'],
        'source_task_sha256':metadata['source_task_sha256'],
        'source_launch_sha256':metadata['source_launch_sha256'],
        'network':modal_network_env(),
        'bootstrap_sha256':sha((ROOT/'harbor_envs/modal/sandbox_bootstrap.py').read_bytes()),
        'evaluator_sha256':evaluator_sha256(),
        'prepare_sha256':sha(setup.read_bytes()) if setup else None,
        'dockerfile_sha256':sha(dockerfile.encode()),
    }
    if driver_env:
        runtime['driver_env'] = driver_env
    if evaluation_mode == 'local':
        runtime['evaluation_mode'] = 'local'
    if 'video_calibration' in platform:
        hardware = scoring['video_benchmark']['identity']['hardware']
        validate_hardware(hardware, runtime['gpu'], runtime['gpus'])
        runtime['video_calibration_hardware'] = deepcopy(hardware)
    runtime['baseline_conditions'] = baseline_conditions(runtime,scoring)
    runtime['sha256'] = sha(json.dumps(runtime,sort_keys=True).encode())
    metadata['runtime_spec_sha256'] = runtime['sha256']
    cfg['environment'] = envs['agent']
    cfg['verifier']['environment'] = envs['benchmark']
    scoring['serve_node_count'] = 1
    scoring['serve_gpus_per_node'] = serve['gpus']
    scoring['performance_baseline_reference'] = scoring['performance_baseline']
    scoring['performance_baseline'] = {'unit':original_scoring['performance_baseline']['unit'],'higher_is_better':True,
        'output_throughput':1.0,'status':'UNCALIBRATED','source':'Calibration required for these conditions'}
    if platform.get('baseline') and (source/platform['baseline']).exists():
        baseline = json.loads((source/platform['baseline']).read_text())['performance_baseline']
        if baseline_policy == 'reuse_resources' or mode == 'score' or baseline.get('baseline_conditions') == runtime['baseline_conditions']:
            check_baseline_policy(baseline,runtime['baseline_conditions'],baseline_policy)
            scoring['performance_baseline'] = baseline
    if mode == 'score' and scoring['performance_baseline']['status'] != 'CALIBRATED':
        raise ValueError('No calibrated Modal baseline; run calibrate mode first')
    if mode == 'score':
        reference = sol_reference(source.name, runtime['gpu'],
                                  baseline_path=source/platform['baseline'])
        validate_reference(reference, scoring['performance_baseline'], runtime['gpu'], runtime['gpus'])
        scoring['sol_reference'] = reference
        runtime['sol_reference_sha256'] = reference_digest(reference)
        runtime.pop('sha256')
        runtime['sha256'] = sha(json.dumps(runtime, sort_keys=True).encode())
        metadata['runtime_spec_sha256'] = runtime['sha256']
    wire = _materialize(load_yaml(source/'task.yaml'))
    cfg['artifacts'] = wire['artifacts']
    cfg['verifier']['collect'] = wire['verifier']['collect']
    for context in ('environment','tests'):
        put(f'{context}/Dockerfile',dockerfile)
        put(f'{context}/sandbox_bootstrap.py',(ROOT/'harbor_envs/modal/sandbox_bootstrap.py').read_bytes())
        if setup:
            put(f'{context}/runtime-setup.sh',setup.read_bytes())
    instruction_cfg = deepcopy(cfg)
    # Agent may edit on CPU; deployment requirements describe the independently allocated Serve.
    instruction_cfg['environment'] = {**base_agent, **serve, 'nodes': 1}
    files.update(step_files(source,cfg,instruction_cfg))
    put('tests/runtime.json',json.dumps(runtime,indent=2)+'\n')
    put('tests/verifier-config.json',json.dumps(original_scoring,indent=2,ensure_ascii=False)+'\n')
    put('tests/modal-scoring.json',json.dumps({k:v for k,v in scoring.items() if original_scoring.get(k)!=v},indent=2,ensure_ascii=False)+'\n')
    put('tests/test.sh','#!/bin/bash\necho "Configured RSIBench verifier required" >&2\nexit 2\n')
    for path in (source/'solution').rglob('*'):
        if path.is_file():
            put(path.relative_to(source),path.read_bytes())
    return files


def prepare_run(config: dict, *, root: Path = ROOT, directory: Path | None = None) -> dict:
    """生成独占快照并返回原生 JobConfig；快照保留以支持 Harbor resume。"""
    config = deepcopy(config)
    runtime = config.pop('runtime', {})
    unknown = set(config)-set(JobConfig.model_fields)
    if unknown:
        raise ValueError(f'Unknown run config fields: {sorted(unknown)}')
    if set(runtime)-{'resources','baseline_policy','evaluation_mode'}:
        raise ValueError(f'Unknown runtime fields: {sorted(set(runtime)-{"resources","baseline_policy","evaluation_mode"})}')
    evaluation_mode = runtime.get('evaluation_mode', 'separate')
    if evaluation_mode not in {'separate', 'local'}:
        raise ValueError(f'Unknown evaluation_mode: {evaluation_mode}')
    if evaluation_mode == 'local':
        configure_local_job(config)
    baseline_policy = runtime.get('baseline_policy','strict')
    if baseline_policy not in {'strict','reuse_resources'}:
        raise ValueError(f'Unknown baseline_policy: {baseline_policy}')
    overrides = runtime.get('resources', {})
    if set(overrides)-{'agent','benchmark','serve'}:
        raise ValueError('runtime.resources only accepts agent, benchmark, serve')
    backend = BACKENDS.get(config.get('environment',{}).get('import_path'))
    if backend is None:
        raise ValueError('YAML task compilation requires a configured ML platform provider')
    if backend != 'modal' and baseline_policy != 'strict':
        raise ValueError('baseline_policy currently applies only to Modal inference')
    if config.get('datasets'):
        raise ValueError('Use explicit tasks paths in the run YAML; generated datasets are no longer configuration sources')
    if not config.get('tasks'):
        raise ValueError('Run YAML requires at least one task')
    # 所有配置在启动 Harbor 前解析；没有隐式外部 setting 或 runtime YAML。
    work = root/'.agents/workdir'
    work.mkdir(parents=True,exist_ok=True)
    directory = directory or Path(tempfile.mkdtemp(prefix='run-config-',dir=work))
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'run-source.json').write_text(json.dumps({**config,'runtime':runtime},indent=2,default=str)+'\n')
    manifest = []
    names = set()
    for entry in config['tasks']:
        if set(entry)-{'path'}:
            raise ValueError('Compiled YAML tasks accept only a local path')
        source = (root/entry['path']).resolve()
        if source.name in names:
            raise ValueError(f'Duplicate task name: {source.name}')
        names.add(source.name)
        cfg = load_yaml(source/'task.yaml')
        destination = directory/'tasks'/source.name
        destination.mkdir(parents=True,exist_ok=False)
        mode = config.get('verifier',{}).get('kwargs',{}).get('mode','score')
        if mode not in {'score','smoke','calibrate'}:
            raise ValueError(f'Unknown verification mode: {mode}')
        if config.get('verifier',{}).get('disable'):
            mode = 'smoke'
        for relative, data in modal_files(source,cfg,overrides,mode,baseline_policy,evaluation_mode).items():
            target = destination/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
        Task(destination,disable_verification=config.get('verifier',{}).get('disable',False))
        manifest.append({'source':str(source),'task_yaml_sha256':sha((source/'task.yaml').read_bytes()),'compiled':str(destination)})
        entry['path'] = str(destination)
    (directory/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return config

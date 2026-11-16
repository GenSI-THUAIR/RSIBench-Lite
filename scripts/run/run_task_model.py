"""派生运行 JobConfig：保留模型覆写语义，并按最终 env 自动注入网络设置。

参数：Harbor 可执行文件、可选 --local-mode，其后为 `harbor run` 参数（含 --config）。
日志由 Harbor 原生处理，显式传入的插件参数原样透传。
示例：python3 scripts/run/run_task_model.py .venv/bin/harbor \
    --config configs/run.yaml --model openai/example --print-config
"""

import subprocess
import sys
import tempfile
from contextlib import ExitStack
from functools import wraps
from pathlib import Path

import typer
import yaml
from harbor.cli.config_sources import load_config_source
from harbor.cli.jobs import start
from harbor.cli.jobs import parse_kwargs
from harbor_envs.modal.network import configure_job_network
from schemas.run_config import prepare_run
from schemas.build import load_yaml


def main(harbor_bin: str, args: list[str]) -> None:
    """复用 Harbor 参数定义，运行期网络和模型设置由临时配置传入。"""
    local_mode = '--local-mode' in args
    args = [arg for arg in args if arg != '--local-mode']

    @wraps(start)
    def launch(**params) -> None:
        """无显式 agent 时覆写单个模型；临时配置在 Harbor 退出后删除。"""
        models = params["model_names"]
        forwarded = list(args)
        with ExitStack() as stack:
            override = []
            config = load_config_source(params["config_path"]) if params["config_path"] else {}
            if local_mode:
                config.setdefault('runtime', {})['evaluation_mode'] = 'local'
            if config.get('runtime', {}).get('evaluation_mode') == 'local':
                # Apply CLI values before validation; Harbor applies them again downstream.
                if params.get('n_concurrent_trials') is not None:
                    config['n_concurrent_trials'] = params['n_concurrent_trials']
                if params.get('override_gpus') is not None:
                    config.setdefault('environment', {})['override_gpus'] = params['override_gpus']
                if params.get('environment_kwargs'):
                    config.setdefault('environment', {}).setdefault('kwargs', {}).update(
                        parse_kwargs(params['environment_kwargs']))
            root = Path(__file__).resolve().parents[2]
            selected_path = params.get('path')
            if selected_path is not None and (Path(selected_path)/'task.yaml').is_file():
                config['tasks'] = [{'path':str(Path(selected_path).resolve())}]
                config.pop('datasets',None)
                # 已解析的 --path 用快照路径代替，不能让原 CLI 再覆盖回源目录。
                forwarded = []
                iterator = iter(args)
                for value in iterator:
                    if value in ('--path','-p'):
                        next(iterator)
                    elif not value.startswith('--path='):
                        forwarded.append(value)
            selected_environment = params['environment_import_path']
            if selected_environment:
                config.setdefault('environment',{})['import_path'] = selected_environment
            if params['environment'] is not None:
                platform = str(params['environment'])
                platform = getattr(params['environment'],'value',platform)
                if platform == 'modal':
                    provider = 'ModalEnvironment'
                    config.setdefault('environment',{})['import_path'] = f'harbor_envs.{platform}:{provider}'
            yaml_tasks = config.get('tasks',[])
            compile_tasks = 'runtime' in config or any((root/t['path']/'task.yaml').is_file() for t in yaml_tasks)
            if compile_tasks and len(yaml_tasks)==1 and not (params.get('verifier_import_path') or params.get('verifier')):
                source = load_yaml(root/yaml_tasks[0]['path']/'task.yaml')
                task_verifier = source.get('verifier',{}).get('scoring',{}).get('verifier')
                if task_verifier:
                    config.setdefault('verifier',{})['import_path'] = task_verifier
            if params.get('verifier_kwargs'):
                config.setdefault('verifier',{}).setdefault('kwargs',{}).update(parse_kwargs(params['verifier_kwargs']))
            if params.get('disable_verification') or params.get('install_only'):
                config.setdefault('verifier',{})['disable'] = True
            changed = configure_job_network(
                config, params["environment"] or params["environment_import_path"]
            )
            if compile_tasks:
                config = prepare_run(config,root=root)
                changed = True
            if (
                models
                and params["agent_name"] is None
                and params["agent_import_path"] is None
            ):
                if len(models) != 1 or not models[0].strip():
                    raise typer.BadParameter(
                        "model-only override requires one non-empty model; "
                        "use separate runs or an explicit --agent",
                        param_hint="--model",
                    )
                agents = config.get("agents")
                if not isinstance(agents, list) or len(agents) != 1:
                    raise typer.BadParameter(
                        "model-only override requires exactly one configured agent",
                        param_hint="--config",
                    )
                agents[0]["model_name"] = models[0]
                changed = True
            if changed:
                workdir = Path(__file__).resolve().parents[2] / ".agents/workdir"
                workdir.mkdir(parents=True, exist_ok=True)
                stream = stack.enter_context(
                    tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=workdir,
                        prefix="run-task-model-",
                        suffix=".yaml",
                    )
                )
                yaml.safe_dump(config, stream, sort_keys=False)
                stream.flush()
                # 放在最后，沿用原生 CLI 的“最后一个 --config 生效”语义。
                override = ["--config", stream.name]
            raise SystemExit(subprocess.call([harbor_bin, "run", *forwarded, *override]))

    app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})
    app.command()(launch)
    app(args=args, prog_name="harbor run")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

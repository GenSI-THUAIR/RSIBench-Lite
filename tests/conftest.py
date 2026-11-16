"""在收集依赖编译产物的回归之前生成 lite 任务；不申请云资源。"""
from pathlib import Path

from schemas.build import build_task


def pytest_configure(config):
    root = Path(__file__).resolve().parents[1]
    (root / '.agents/workdir').mkdir(parents=True, exist_ok=True)
    for source in sorted((root / 'tasks').glob('*/task.yaml')):
        build_task(source.parent, check=False)

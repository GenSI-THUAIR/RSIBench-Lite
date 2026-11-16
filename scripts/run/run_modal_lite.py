#!/usr/bin/env python3
"""用标准 Modal 凭据启动五题 lite，默认 Oracle smoke。

参数：--mode smoke|score|calibrate|cpu；--local-mode 使用单 H200；-- 后透传 Harbor。
示例：python3 scripts/run/run_modal_lite.py -- --print-config
      python3 scripts/run/run_modal_lite.py
      python3 scripts/run/run_modal_lite.py --mode score -- --path tasks/infer-qwen3vl-32b
smoke 不产生正式分；五题 score 仍受各题质量与基线准入限制。
"""
import argparse
from datetime import datetime
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'score', 'calibrate', 'cpu'), default='smoke')
    parser.add_argument('--local-mode', action='store_true',
                        help='CPU Agent/Benchmark, one H200 Serve, serial trials')
    parser.add_argument('harbor_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env.setdefault('RUN_TASK_JOB_NAME', f'modal-lite-{args.mode}-{datetime.now():%Y%m%d-%H%M%S-%f}')
    suffix = '' if args.mode == 'score' else f'-{args.mode}'
    forwarded = args.harbor_args[1:] if args.harbor_args[:1] == ['--'] else args.harbor_args
    command = ['bash', 'scripts/run/run_task.sh', '--config', f'configs/harbor-modal{suffix}.yaml',
               '--verifier', 'verify.inference.runtime_infer_verifier:RuntimeInferVerifier',
               *(['--local-mode'] if args.local_mode else []), *forwarded]
    os.chdir(root)
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()

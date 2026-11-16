"""把通过完整原生Modal Oracle标定的结果登记为本环境基线；拒绝smoke或失败任务。

参数：Harbor job目录。示例：python3 scripts/calibration/record_harbor_baseline.py jobs/modal-calibration。
先核验全部输入，再写各题基线；下次启动时自动读取，无需手动生成任务包。
"""
import argparse
import hashlib
import json
import math
import statistics
from copy import deepcopy
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from schemas.build import load_yaml
from verify.inference.baseline import baseline_conditions, require_matching_conditions
from verify.inference.infer_verifier import _accuracy_passes


def validate_full_workload(summary, scoring, inputs):
    """用本次冻结配置及可信输入收据验证完整工作量，禁止把 smoke 登记为基线。"""
    gate, performance = summary['gate'], summary['performance']
    if scoring['gate']['kind'] == 'video-reference':
        expected = scoring['video_benchmark']['case_ids']
        cases = gate['cases']
        if ([case['id'] for case in cases] != expected or not gate['passed']
                or not all(case['passed'] for case in cases)
                or performance['completed'] != len(expected)
                or performance['num_prompts'] != len(expected)):
            raise ValueError('The original full video workload was not completed')
        if not math.isclose(performance['output_throughput'],
                            3600 * len(expected) / performance['wall_time_sec']):
            raise ValueError('Video throughput does not match full-workset timing')
        return
    dataset = next(item for item in inputs['files'] if item['path'] == scoring['gate']['data'])
    total = dataset['records']
    expected = scoring['gate'].get('num_questions', total)
    if not 0 < expected <= total or gate['num_questions'] != expected:
        raise ValueError('The original full correctness workload was not completed')
    benchmark = scoring['bench_serving']
    if benchmark.get('window_full_duration'):
        if performance['wall_time_sec'] != benchmark['window_duration_sec']:
            raise ValueError('The original full workload window was not completed')
    else:
        expected_requests=benchmark['num_prompts']
        if benchmark['dataset_name']=='mooncake':
            extra=benchmark['extra_args']
            path=extra[extra.index('--dataset-path')+1]
            trace=next(item for item in inputs['files'] if item['path']==path)
            expected_requests=min(expected_requests,trace['records'])
        if expected_requests<=0 or performance['completed']!=expected_requests:
            raise ValueError('The original full request workload was not completed')


def validated_baselines(job_dir: Path):
    results=sorted(job_dir.glob('*/result.json'))
    if not results:
        raise ValueError('No completed Harbor trial results')
    pending=[]
    for result_path in results:
        result=json.loads(result_path.read_text())
        if not result.get('finished_at') or result.get('exception_info'):
            raise ValueError('Trial has not completed successfully')
        steps=result.get('step_results') or []
        if any(s.get('exception_info') for s in steps):
            raise ValueError('A native Harbor step failed')
        agent=result['config']['agent']
        if agent.get('name')!='oracle' or agent.get('import_path'):
            raise ValueError('Baseline requires the native Oracle agent')
        environment=result['config']['environment']
        provider=environment.get('import_path')
        if not ((environment.get('type')=='modal' and not provider)
                or provider=='harbor_envs.modal:ModalEnvironment'):
            raise ValueError('Baseline requires the configured Modal provider')
        records=list(result_path.parent.rglob('calibrate.json'))
        if len(records)!=1:
            raise ValueError('Exactly one full calibration receipt is required; smoke cannot be registered')
        receipt_path=records[0]
        receipt=json.loads(receipt_path.read_text())
        directory=receipt_path.parent
        summary=json.loads((directory/'benchmark-summary.json').read_text())
        lifecycle=json.loads((directory/'runtime-lifecycle.json').read_text())
        if receipt.get('mode')!='calibrate' or receipt.get('calibrate_complete')!=1:
            raise ValueError('Calibration did not pass')
        scoring_path=directory/'verifier-config.json'
        if not scoring_path.is_file():
            raise ValueError('Missing full workload configuration')
        scoring=json.loads(scoring_path.read_text())
        inputs=json.loads((directory/'inputs.json').read_text())
        validate_full_workload(summary,scoring,inputs)
        if not lifecycle.get('measurement_complete') or lifecycle.get('serve_terminal_status') is None:
            raise ValueError('Missing model measurement or service cleanup evidence')
        throughput=receipt['output_throughput']
        if type(throughput) not in (int,float) or not math.isfinite(throughput) or throughput<=0:
            raise ValueError('Invalid measured throughput')
        if (throughput!=summary['performance']['output_throughput']
                or receipt['gate_accuracy']!=summary['gate']['accuracy']
                or summary['performance']['request_success_rate']!=1.0):
            raise ValueError('Calibration receipt and measured summary disagree')
        name=result['task_name'].split('/')[-1]
        source=ROOT/'tasks'/name
        spec_path=directory/'runtime.json'
        spec=json.loads(spec_path.read_text())
        if spec['sha256'] != receipt['runtime_spec_sha256']:
            raise ValueError('Calibration receipt and saved runtime identity disagree')
        require_matching_conditions(receipt, spec['baseline_conditions'])
        require_matching_conditions(receipt, baseline_conditions(spec, scoring))
        task_config=load_yaml(source/'task.yaml')
        gate=scoring['gate']
        if not _accuracy_passes(summary['gate']['accuracy'],gate['center'],gate['tolerance'],
                               gate.get('comparison','two_sided')):
            raise ValueError('Full original correctness gate failed')
        records_sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [receipt_path,directory/'benchmark-summary.json',directory/'runtime-lifecycle.json',
                               scoring_path,directory/'inputs.json',spec_path]}
        record = {
            'runtime_spec_sha256':receipt['runtime_spec_sha256'], 'evidence_sha256':records_sha,
            'service_image_id':lifecycle['service_image_id'],
            'performance_baseline':{'unit':scoring['performance_baseline']['unit'],'higher_is_better':True,
                'output_throughput':throughput,'status':'CALIBRATED',
                'source':str(receipt_path.resolve()),'runtime_spec_sha256':receipt['runtime_spec_sha256'],
                'baseline_conditions':receipt['baseline_conditions'],
                'service_image_id':lifecycle['service_image_id']},
        }
        if scoring['gate']['kind'] == 'video-reference':
            record['video_calibration_measurement'] = {
                'completed': summary['performance']['completed'],
                'wall_time_sec': summary['performance']['wall_time_sec'],
            }
        pending.append((source/task_config['platforms']['modal']['baseline'], record))
    return pending


def aggregate_video_baselines(pending):
    """Use every one of three admitted native runs; never select the fastest run."""
    if len(pending) != 3 or len({path for path, _ in pending}) != 1:
        raise ValueError('Video median requires exactly three runs of one task')
    path, first = pending[0]
    records = [record for _, record in pending]
    conditions = first['performance_baseline']['baseline_conditions']
    if 'video_benchmark' not in conditions:
        raise ValueError('Video median requires a video benchmark')
    count = len(conditions['video_benchmark']['case_ids'])
    if count != 4:
        raise ValueError('Video median requires the complete four-case workload')
    seconds = []
    for record in records:
        if record['performance_baseline']['baseline_conditions'] != conditions:
            raise ValueError('Video calibration measurement conditions differ')
        if record['service_image_id'] != first['service_image_id']:
            raise ValueError('Video calibration service images differ')
        measured = record['video_calibration_measurement']
        duration = measured['wall_time_sec']
        if (measured['completed'] != count or not math.isfinite(duration) or duration <= 0
                or not math.isclose(record['performance_baseline']['output_throughput'],
                                    3600 * count / duration)):
            raise ValueError('Invalid full-workset video calibration timing')
        seconds.append(duration)
    result = deepcopy(first)
    result.pop('video_calibration_measurement', None)
    result['performance_baseline'].update(
        output_throughput=3600 * count / statistics.median(seconds),
        source='Median of three independent full native Modal Oracle calibrations',
        aggregation={'method': 'median_full_workset_seconds', 'run_count': 3,
                     'workset_seconds': seconds, 'median_workset_seconds': statistics.median(seconds)},
    )
    result['calibration_runs'] = records
    return [(path, result)]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_dir',type=Path)
    parser.add_argument('--video-median', action='store_true',
                        help='Require three complete native video calibrations and register their median')
    args=parser.parse_args()
    pending=validated_baselines(args.job_dir)
    if args.video_median:
        pending=aggregate_video_baselines(pending)
    elif len({path for path, _ in pending}) != len(pending):
        raise ValueError('Multiple baselines target the same file; use an explicit aggregation policy')
    for path,value in pending:
        if path.exists():
            raise FileExistsError(f'Baseline already exists; do not silently overwrite: {path}')
    for path,value in pending:
        path.write_text(json.dumps(value,indent=2)+'\n')
        print(path)

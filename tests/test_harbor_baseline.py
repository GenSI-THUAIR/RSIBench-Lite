"""正式基线登记的负控：成功smoke、短测伪装标定、step失败都不得登记。"""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.calibration.record_harbor_baseline import validated_baselines, aggregate_video_baselines
from verify.inference.baseline import baseline_conditions

ROOT=Path(__file__).resolve().parents[1]


class BaselineAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.agents/workdir')
        self.addCleanup(self.temp.cleanup)
        self.job=Path(self.temp.name)
        self.trial=self.job/'trial'
        self.verifier=self.trial/'verifier'
        self.verifier.mkdir(parents=True)
        self.result={'finished_at':'2026-10-10T00:00:00Z','exception_info':None,
                     'config':{'environment':{'type':'modal'},'agent':{'name':'oracle'}},'step_results':[]}
        self.write(self.trial/'result.json',self.result)

    def write(self,path,value):
        path.write_text(json.dumps(value))

    def test_successful_smoke_is_not_a_baseline(self):
        self.write(self.verifier/'smoke.json',{'mode':'smoke','smoke_complete':1})
        with self.assertRaisesRegex(ValueError,'full calibration'):
            validated_baselines(self.job)

    def test_optimized_agent_cannot_be_registered_as_oracle_baseline(self):
        for agent in ({'name':'codex'}, {'name':'oracle','import_path':'custom:Agent'}):
            with self.subTest(agent=agent):
                self.result['config']['agent']=agent
                self.write(self.trial/'result.json',self.result)
                with self.assertRaisesRegex(ValueError,'native Oracle'):
                    validated_baselines(self.job)

    def test_short_measurement_cannot_be_renamed_as_calibration(self):
        self.write(self.verifier/'calibrate.json',{'mode':'calibrate','calibrate_complete':1})
        self.write(self.verifier/'benchmark-summary.json',{'gate':{'num_questions':3},'performance':{'wall_time_sec':30}})
        self.write(self.verifier/'runtime-lifecycle.json',{})
        with self.assertRaisesRegex(ValueError,'full workload'):
            validated_baselines(self.job)

    def test_step_failure_is_checked_even_when_trial_exception_is_null(self):
        self.result['step_results']=[{'exception_info':{'exception_type':'TimeoutError'}}]
        self.write(self.trial/'result.json',self.result)
        with self.assertRaisesRegex(ValueError,'step failed'):
            validated_baselines(self.job)

    def test_full_consistent_receipt_binds_image_and_rejects_metric_drift(self):
        repository=self.job/'repo'
        source=repository/'tasks/model'
        (source/'environment').mkdir(parents=True)
        (source/'task.yaml').write_text('verifier:\n  scoring:\n    gate: {center: 0.73, tolerance: 0.05}\nplatforms:\n  modal:\n    baseline: runtimes/modal-baseline.json\n')
        (source/'environment/launch.sh').write_text('# frozen launch\n')
        spec=self.verifier/'runtime.json'
        scoring={
            'gate': {'kind':'longbench-v2-openai-chat', 'data':'/data/gate.jsonl',
                     'center':.73, 'tolerance':.05},
            'bench_serving': {'window_full_duration':True, 'window_duration_sec':3600},
            'performance_baseline': {'unit':'tokens/s'},
        }
        runtime={'backend':'modal','gpu':'H200','gpus':1,'serve_cpus':8,'serve_memory_mb':131072,
                 'benchmark_cpus':4,'benchmark_memory_mb':16384,'image':'frozen-image','inputs':[]}
        conditions=baseline_conditions(runtime,scoring)
        runtime.update(sha256='runtime-id',baseline_conditions=conditions)
        self.write(spec,runtime)
        self.result['task_name']='rsibench-open-lite-modal/model'
        self.write(self.trial/'result.json',self.result)
        receipt={'mode':'calibrate','calibrate_complete':1,'output_throughput':42.0,
                 'gate_accuracy':.73,'runtime_spec_sha256':'runtime-id',
                 'baseline_conditions':conditions,
                 'source_task_sha256':hashlib.sha256((source/'task.yaml').read_bytes()).hexdigest(),
                 'source_launch_sha256':hashlib.sha256((source/'environment/launch.sh').read_bytes()).hexdigest()}
        self.write(self.verifier/'calibrate.json',receipt)
        summary={'gate':{'num_questions':100,'accuracy':.73},
                 'performance':{'wall_time_sec':3600,'output_throughput':42.0,'request_success_rate':1.0}}
        self.write(self.verifier/'benchmark-summary.json',summary)
        self.write(self.verifier/'verifier-config.json',scoring)
        self.write(self.verifier/'inputs.json', {'files':[{'path':'/data/gate.jsonl','records':100}]})
        self.write(self.verifier/'runtime-lifecycle.json',
                   {'measurement_complete':True,'serve_terminal_status':137,'service_image_id':'im-frozen'})
        # 源码与任务说明变化不作废已完成的标定；记录仍保留当时的 runtime ID。
        with (source/'task.yaml').open('a') as stream:
            stream.write('# revised documentation\n')
        (source/'environment/launch.sh').write_text('# revised launch\n')
        with patch('scripts.calibration.record_harbor_baseline.ROOT',repository):
            [(destination,baseline)]=validated_baselines(self.job)
            self.assertEqual(destination,source/'runtimes/modal-baseline.json')
            self.assertEqual(baseline['performance_baseline']['service_image_id'],'im-frozen')
            self.assertEqual(baseline['runtime_spec_sha256'],'runtime-id')
            self.assertEqual(baseline['performance_baseline']['baseline_conditions'],conditions)
            self.assertFalse(destination.exists())
            self.write(spec,{'sha256':'runtime-id','baseline_conditions':{'fixture':'changed workload'}})
            with self.assertRaisesRegex(ValueError,'measurement conditions'):
                validated_baselines(self.job)
            self.write(spec,runtime)
            scoring['bench_serving']['window_duration_sec']=120
            summary['performance']['wall_time_sec']=120
            self.write(self.verifier/'verifier-config.json',scoring)
            self.write(self.verifier/'benchmark-summary.json',summary)
            with self.assertRaisesRegex(ValueError,'measurement conditions'):
                validated_baselines(self.job)
            scoring['bench_serving']['window_duration_sec']=3600
            summary['performance']['wall_time_sec']=3600
            self.write(self.verifier/'verifier-config.json',scoring)
            summary['performance']['output_throughput']=99.0
            self.write(self.verifier/'benchmark-summary.json',summary)
            with self.assertRaisesRegex(ValueError,'disagree'):
                validated_baselines(self.job)

    def test_video_median_uses_all_three_complete_matching_runs(self):
        conditions = {'video_benchmark': {'case_ids': ['a', 'b', 'c', 'd']}}
        records = [(self.job/'baseline.json', {
            'service_image_id': 'same-image', 'evidence_sha256': {'receipt': str(seconds)},
            'performance_baseline': {'baseline_conditions': conditions,
                                     'output_throughput': 14400 / seconds},
            'video_calibration_measurement': {'completed': 4, 'wall_time_sec': seconds},
        }) for seconds in (600, 400, 500)]
        [(path, result)] = aggregate_video_baselines(records)
        self.assertEqual(result['performance_baseline']['output_throughput'], 28.8)
        self.assertEqual(result['performance_baseline']['aggregation']['workset_seconds'], [600, 400, 500])
        self.assertEqual(len(result['calibration_runs']), 3)
        with self.assertRaisesRegex(ValueError, 'exactly three'):
            aggregate_video_baselines(records[:2])
        records[2][1]['video_calibration_measurement']['completed'] = 3
        with self.assertRaisesRegex(ValueError, 'full-workset'):
            aggregate_video_baselines(records)


if __name__=='__main__':
    unittest.main()

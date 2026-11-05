"""CPU-only coordinator state/lifecycle tests; no real subprocesses or GPUs."""
import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from llama_profile import common, runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='llama-runner-test-')
        self.root = Path(self.temporary.name)
        self.output = self.root / 'results'
        self.options = dict(output=str(self.output), data=str(self.root / 'prepared.json'),
            model=str(self.root / 'target'), draft=str(self.root / 'draft'),
            runtime=str(self.root / 'runtime/profile/sglang'), preflight=str(self.root / 'preflight.json'),
            gpus=list(map(str, range(8))), tp=8, batches=[128], repeat=3, threshold=.5,
            memory_fraction=.8, min_steps=10, context=4096, max_new_tokens=1024,
            chunked_prefill=4096, protocol='queued', timeout_seconds=2, poll_seconds=.001,
            shutdown_grace_seconds=.01, preflight_timeout_seconds=20)
        Path(self.options['data']).write_text('{"rows":[]}')
        self.job = dict(name='smoke_b128_chain7', stage='smoke', method='chain7', bs=128, repeat=0, run_id='test-run')
        self.plan = dict(schema_version=1, options=runner._options(self.options), runs=[self.job],
            prepared_sha256=common.sha(self.options['data']),
            preflight_identity={'devices': [dict(uuid=f'GPU-{i}') for i in range(8)]})
        self.child = SimpleNamespace(pid=12345, returncode=0, poll=lambda: 0)
        self.idle = dict(devices=[dict(uuid=f'GPU-{i}') for i in range(8)], processes=[], outsiders=[], time=0)

    def tearDown(self):
        self.temporary.cleanup()

    @contextlib.contextmanager
    def fake_lock(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        yield

    def test_monitor_foreign_process_never_targets_foreign_pid(self):
        foreign = {'pid': 90001, 'uuid': 'GPU-7', 'used_mib': '10'}
        state = dict(self.idle, processes=[foreign], outsiders=[foreign])
        with mock.patch.object(runner.hardware, 'snapshot', return_value=state) as snapshot:
            with self.assertRaisesRegex(RuntimeError, 'Foreign GPU'):
                runner._monitor(self.child, self.plan['options'], self.root / 'monitor.jsonl')
        snapshot.assert_called_once_with(self.options['gpus'], own_group=self.child.pid)

    def test_monitor_timeout(self):
        child = SimpleNamespace(pid=12345, poll=lambda: None)
        with mock.patch.object(runner.hardware, 'snapshot', return_value=self.idle), \
             mock.patch.object(runner.time, 'monotonic', side_effect=[0, 3]):
            with self.assertRaises(TimeoutError):
                runner._monitor(child, self.plan['options'], self.root / 'monitor.jsonl')

    def test_clean_exit_requires_no_surviving_descendants(self):
        with mock.patch.object(runner.hardware, 'snapshot', return_value=self.idle), \
             mock.patch.object(runner, '_group_members', return_value=[12346]), \
             mock.patch.object(runner.time, 'monotonic', side_effect=[0, 0, 1]), \
             mock.patch.object(runner.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'descendants'):
                runner._monitor(self.child, self.plan['options'], self.root / 'monitor.jsonl')
        with mock.patch.object(runner.hardware, 'snapshot', return_value=self.idle), \
             mock.patch.object(runner, '_group_members', return_value=[]):
            runner._monitor(self.child, self.plan['options'], self.root / 'monitor2.jsonl')

    def test_interrupted_worker_stops_only_owned_group_and_preserves_failure(self):
        with mock.patch.object(runner, 'validate_frozen_inputs'), \
             mock.patch.object(runner, '_prerequisites'), \
             mock.patch.object(runner, '_wait_idle', return_value=self.idle), \
             mock.patch.object(runner.subprocess, 'Popen', return_value=self.child) as spawn, \
             mock.patch.object(runner, '_monitor', side_effect=KeyboardInterrupt), \
             mock.patch.object(runner.hardware, 'stop') as stop:
            with self.assertRaises(KeyboardInterrupt):
                runner._execute(self.output, self.job, self.plan, False, False)
        stop.assert_called_once_with(self.child)
        self.assertTrue(spawn.call_args.kwargs['start_new_session'])
        folder = self.output / 'runs' / self.job['name']
        self.assertEqual((folder / 'trace/control.bin').read_bytes(), b'\0' * 8)
        self.assertEqual(common.load(folder / 'failure.json')['error_type'], 'KeyboardInterrupt')
        self.assertTrue((folder / 'worker.log').is_file())

    def test_calibration_environment_has_no_profile_switches(self):
        job = dict(self.job, name='calibration', stage='calibrate', method='ar')
        with mock.patch.object(runner, 'validate_frozen_inputs'), \
             mock.patch.object(runner, '_prerequisites'), \
             mock.patch.object(runner, '_wait_idle', return_value=self.idle), \
             mock.patch.object(runner.subprocess, 'Popen', return_value=self.child) as spawn, \
             mock.patch.object(runner, '_monitor'), \
             mock.patch.object(runner, '_completed', return_value=True), \
             mock.patch.dict(os.environ, {'PROFILE_KERNEL_MODE': '1', 'EAGLE_ACTIVE_LINEAR_CONFIG': '/wrong/8b.json'}):
            runner._execute(self.output, job, self.plan, False, False)
        env = spawn.call_args.kwargs['env']
        self.assertFalse(any(key.startswith('PROFILE_') for key in env))
        self.assertNotIn('EAGLE_ACTIVE_LINEAR_CONFIG', env)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')

    def test_completed_resume_revalidates_sources_and_does_not_spawn(self):
        (self.output / 'runs' / self.job['name']).mkdir(parents=True)
        with mock.patch.object(runner, 'validate_frozen_inputs') as validate, \
             mock.patch.object(runner, '_completed', return_value=True), \
             mock.patch.object(runner.subprocess, 'Popen') as spawn:
            self.assertEqual(runner._execute(self.output, self.job, self.plan, True, False), 'skipped')
        validate.assert_called_once_with(self.plan)
        spawn.assert_not_called()

    def test_failed_directory_requires_explicit_resume_and_is_archived(self):
        folder = self.output / 'runs' / self.job['name']
        folder.mkdir(parents=True)
        (folder / 'important.log').write_text('preserve me')
        with mock.patch.object(runner, 'validate_frozen_inputs'), \
             mock.patch.object(runner, '_completed', return_value=False):
            with self.assertRaises(FileExistsError):
                runner._execute(self.output, self.job, self.plan, False, False)
        with mock.patch.object(runner, 'validate_frozen_inputs'), \
             mock.patch.object(runner, '_completed', side_effect=[False, True]), \
             mock.patch.object(runner, '_prerequisites'), \
             mock.patch.object(runner, '_wait_idle', return_value=self.idle), \
             mock.patch.object(runner.subprocess, 'Popen', return_value=self.child), \
             mock.patch.object(runner, '_monitor'):
            runner._execute(self.output, self.job, self.plan, True, False)
        archives = list((self.output / 'failures').glob(self.job['name'] + '.*'))
        self.assertEqual(len(archives), 1)
        self.assertEqual((archives[0] / 'important.log').read_text(), 'preserve me')
        self.assertFalse((folder / 'important.log').exists())

    def test_busy_gpu_wait_is_explicit(self):
        busy = dict(self.idle, processes=[dict(pid=999)])
        with mock.patch.object(runner.hardware, 'snapshot', return_value=busy):
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                runner._wait_idle(self.plan['options'], False)
        with mock.patch.object(runner.hardware, 'snapshot', side_effect=[busy, self.idle]), \
             mock.patch.object(runner.time, 'sleep') as sleep:
            runner._wait_idle(self.plan['options'], True)
        sleep.assert_called_once()

    def test_kernel_prerequisites_include_all_same_bs_timings(self):
        timing = dict(self.job, stage='timing', name='timing', repeat=1)
        kernel = dict(self.job, stage='kernel', name='kernel')
        plan = dict(self.plan, runs=[self.job, timing, kernel])
        with mock.patch.object(runner, '_completed', side_effect=[True, False]):
            with self.assertRaisesRegex(RuntimeError, 'timing'):
                runner._prerequisites(self.output, kernel, plan)

    def test_input_snapshot_is_immutable_and_required_on_resume(self):
        self.output.mkdir()
        runner._ensure_input_snapshot(self.output, self.plan)
        self.assertEqual((self.output / 'input_snapshot.json').read_bytes(), Path(self.options['data']).read_bytes())
        (self.output / 'input_snapshot.json').write_text('tampered')
        with self.assertRaisesRegex(RuntimeError, 'refusing to replace'):
            runner._ensure_input_snapshot(self.output, self.plan)
        (self.output / 'input_snapshot.json').unlink()
        (self.output / 'plan.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            runner._ensure_input_snapshot(self.output, self.plan)

    def test_fresh_doctor_failure_preserves_first_root_without_plan(self):
        with mock.patch.object(runner.sys, 'platform', 'linux'), \
             mock.patch.object(runner, '_lock', side_effect=self.fake_lock), \
             mock.patch.object(runner.os, 'getuid', return_value=123, create=True), \
             mock.patch.object(runner, 'preflight', side_effect=RuntimeError('doctor mismatch')):
            with self.assertRaisesRegex(RuntimeError, 'doctor mismatch'):
                runner.launch_plan(self.output, {'options': self.options, 'runs': [self.job]})
        self.assertFalse((self.output / 'plan.json').exists())
        self.assertEqual(len(list((self.output / 'failures').glob('coordinator-*.json'))), 1)

    def test_stage_order_and_resume_plan_drift(self):
        jobs = [dict(self.job, stage=stage, name=stage) for stage in ('kernel', 'timing', 'smoke')]
        frozen = dict(self.plan, runs=jobs)
        with mock.patch.object(runner.sys, 'platform', 'linux'), \
             mock.patch.object(runner, '_lock', side_effect=self.fake_lock), \
             mock.patch.object(runner.os, 'getuid', return_value=123, create=True), \
             mock.patch.object(runner, 'preflight', return_value={'audit_pass': True}), \
             mock.patch.object(runner, 'build_plan', return_value=frozen), \
             mock.patch.object(runner, '_execute', return_value='completed') as execute:
            result = runner.launch_plan(self.output, {'options': self.options, 'runs': jobs})
        self.assertEqual(result['completed'], 3)
        self.assertEqual([call.args[1]['stage'] for call in execute.call_args_list], ['smoke', 'timing', 'kernel'])
        changed = dict(frozen, runtime_sha256='changed')
        with mock.patch.object(runner.sys, 'platform', 'linux'), \
             mock.patch.object(runner, '_lock', side_effect=self.fake_lock), \
             mock.patch.object(runner.os, 'getuid', return_value=123, create=True), \
             mock.patch.object(runner, 'preflight', return_value={'audit_pass': True}), \
             mock.patch.object(runner, 'build_plan', return_value=changed):
            with self.assertRaisesRegex(RuntimeError, 'immutable plan'):
                runner.launch_plan(self.output, {'options': self.options, 'runs': jobs}, resume=True)

    def test_prepared_data_and_steady_selection_are_checked(self):
        repo = self.root / 'repo'
        (repo / 'data').mkdir(parents=True)
        raw = repo / 'data/redhat8_4000.jsonl'
        raw.write_text('source')
        target = Path(self.options['model'])
        target.mkdir()
        (target / 'tokenizer_config.json').write_text('{}')
        rows = [dict(id=str(i), prompt_sha256=str(i), input_ids=[1, 2]) for i in range(4000)]
        prepared = dict(rows=rows, count=4000, model_path=str(target.resolve()), context=4096,
                        max_new_tokens=1024, inputs_sha256=common.digest([row['input_ids'] for row in rows]),
                        tokenizer_files={'tokenizer_config.json': common.sha(target / 'tokenizer_config.json')},
                        raw_sha256=common.sha(raw))
        common.save(self.options['data'], prepared)
        calibration = dict(self.job, stage='calibrate')
        with mock.patch.object(common, 'ROOT', repo):
            runner._validate_data(self.plan['options'], [calibration])
            with self.assertRaisesRegex(ValueError, 'steady_selection'):
                runner._validate_data(self.plan['options'], [dict(self.job, stage='timing')])
            (target / 'tokenizer_config.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'Tokenizer'):
                runner._validate_data(self.plan['options'], [calibration])


if __name__ == '__main__':
    unittest.main()

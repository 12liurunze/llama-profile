import copy
import csv
import json
from pathlib import Path
import statistics
import tempfile
import unittest

from llama_profile.common import digest, load, plan_runs, save
from llama_profile.report import IncompleteReportError, report
from llama_profile.runner import _artifact_manifest


def seal_run(root, job):
    folder = root / 'runs' / job['name']
    save(folder / 'completion_audit.json', dict(audit_pass=True, job=job,
         plan_sha256=digest(load(root / 'plan.json')), artifacts=_artifact_manifest(folder), audited_unix=2))


def create_experiment(root, batches=(128, 256, 512)):
    options = dict(batches=list(batches), min_steps=10,
                   data='/original-machine/unavailable/data.json',
                   runtime='/original-machine/unavailable/runtime',
                   model='/original-machine/unavailable/target', output=str(root))
    jobs = plan_runs(options)
    for job in jobs:
        job['run_id'] = job['name'] + '-identity'
    save(root / 'plan.json', dict(options=options, runs=jobs))
    for job in jobs:
        folder = root / 'runs' / job['name']
        save(folder / 'complete.json', dict(job=job, completed_at=1))
        save(folder / 'launch.json', dict(job=job, plan_sha256=digest(load(root / 'plan.json'))))
        save(folder / 'config.json', dict(job=job, source='/original-machine/unavailable/runtime/__init__.py'))
        for rank in range(8):
            trace = folder / 'trace' / f'rank_{rank:03d}'
            rows = [dict(rank=rank, world_size=8, run_id=job['run_id'], step=step) for step in range(10, 20)]
            save(trace / 'rounds.json', rows)
            save(trace / 'metadata.json', dict(schema_version=3, audit_pass=True, rank=rank, world_size=8,
                 run_id=job['run_id'], expected_bs=job['bs'], diagnostic_only=job['stage'] == 'kernel',
                 rows=len(rows), recorded_steps=[row['step'] for row in rows]))
            if job['stage'] == 'kernel':
                save(trace / 'kernel_profile/status.json', dict(status='complete'))
        if job['stage'] != 'timing':
            seal_run(root, job)
            continue
        steps, bs, repeat = 10, job['bs'], job['repeat']
        draft, extend, verify, other, gap = {
            'chain7': (20, 10, 50, 2, 5),
            'adaptive2': (40, 10, 70, 3, 7),
            'eagle728': (50, 10, 60, 4, 6),
        }[job['method']]
        other += repeat - 1
        total = draft + extend + verify + other
        values = dict(draft=draft, verify_total=verify, target_prepare=2, target_forward=verify - 5,
                      accept_and_cache=3, kv_copy=1, draft_extend=extend, total=total)
        stages = {name: dict(total_ms=value * steps, mean_ms_per_step=value) for name, value in values.items()}
        window = (total + gap) * steps
        tokens = bs * steps * (2.5 if job['method'] == 'adaptive2' else 3)
        per_rank = {str(rank): dict(stages=copy.deepcopy(stages), other_worker_ms=other * steps,
                                   inter_worker_gap_ms=gap * steps, window_gpu_elapsed_ms=window) for rank in range(8)}
        primary = dict(audit_pass=True, actual_bs=bs, rank_count=8, steps=steps, request_steps=bs * steps,
                       first_step=10, last_step=19, committed_output_tokens_rank0_only=tokens,
                       accepted_tokens_with_bonus_rank0_only=bs * steps * 3,
                       accepted_draft_tokens_rank0_only=bs * steps * 2,
                       rank0_continuous_gpu_window_ms=window,
                       rank0_continuous_gpu_decode_tokens_per_second=tokens * 1000 / window,
                       per_rank=per_rank,
                       per_stage_rank_maxima={name: dict(mean_rank_max_ms_per_step=value) for name, value in values.items()},
                       context=dict(mean_over_steps=100 + repeat, min=64, max=256))
        analysis = dict(audit_pass=True, primary_window=primary, method=job['method'], run_id=job['run_id'],
                        expected_bs=bs, expected_world_size=8, minimum_consecutive_steps_required=10)
        save(folder / 'analysis.json', analysis)
        seal_run(root, job)
    return jobs


class ReportTests(unittest.TestCase):
    def test_complete_matrix_stats_and_nested_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_experiment(root)
            result = report(root)
            self.assertTrue(result['complete'])
            self.assertTrue(result['full_matrix_complete'])
            self.assertEqual(result['valid_timing_runs'], 27)
            self.assertEqual(len(result['counterfactual_records']), 9)
            group = next(g for g in result['throughput_summary'] if g['bs'] == 128 and g['method'] == 'adaptive2')
            expected = [128 * 2.5 * 1000 / (130 + i) for i in range(3)]
            self.assertAlmostEqual(group['metrics']['throughput_tps']['mean'], statistics.mean(expected))
            self.assertAlmostEqual(group['metrics']['throughput_tps']['sample_sd'], statistics.stdev(expected))
            phase = next(row for row in result['rank0_phase_records'] if row['bs'] == 128 and row['method'] == 'adaptive2' and row['repeat'] == 1)
            self.assertEqual(phase['window_ms_per_step'], 130)
            self.assertEqual(phase['draft_ms_per_step'] + phase['verify_total_ms_per_step'] + phase['draft_extend_ms_per_step'] + phase['other_ms_per_step'] + phase['intergap_ms_per_step'], 130)
            self.assertEqual(phase['kv_copy_ms_per_step'], 1)
            text = (root / 'REPORT.md').read_text(encoding='utf-8')
            self.assertIn('不是客户端端到端吞吐', text)
            self.assertIn('不可相加', text)
            with (root / 'throughput.csv').open(encoding='utf-8-sig', newline='') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 27)

    def test_counterfactual_keeps_adaptive_verify_other_gaps_and_actual_yield(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_experiment(root, (128,))
            result = report(root)
            cf = next(row for row in result['counterfactual_records'] if row['repeat'] == 1)
            self.assertEqual(cf['adaptive_retained_verify_other_intergap_ms'], 800)
            self.assertEqual(cf['counterfactual_window_ms'], 1100)
            self.assertEqual(cf['adaptive_actual_output_tokens'], 3200)
            self.assertAlmostEqual(cf['counterfactual_committed_tps'], 3200 * 1000 / 1100)
            self.assertLess(cf['counterfactual_vs_eagle_percent'], 0)
            self.assertGreater(cf['counterfactual_accept_plus_one_formula_tps'], cf['eagle_actual_tps'])
            self.assertLess(cf['counterfactual_raw_accept_formula_tps'], cf['counterfactual_committed_tps'])
            self.assertTrue(result['complete'])
            self.assertFalse(result['full_matrix_complete'])

    def test_missing_repeat_is_explicit_partial_and_strict_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = create_experiment(root)
            job = next(job for job in jobs if job['stage'] == 'timing' and job['method'] == 'adaptive2' and job['bs'] == 512 and job['repeat'] == 3)
            (root / 'runs' / job['name'] / 'complete.json').unlink()
            with self.assertRaises(IncompleteReportError) as raised:
                report(root)
            self.assertFalse(raised.exception.report['complete'])
            self.assertIn('未完成', (root / 'REPORT.md').read_text(encoding='utf-8'))
            partial = report(root, allow_partial=True)
            self.assertEqual(partial['valid_timing_runs'], 26)
            self.assertEqual(len(partial['counterfactual_records']), 8)
            self.assertEqual(len(partial['counterfactual_skipped']), 1)

    def test_invalid_window_and_unfinished_kernel_cannot_be_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = create_experiment(root, (128,))
            timing = next(job for job in jobs if job['stage'] == 'timing')
            path = root / 'runs' / timing['name'] / 'analysis.json'
            analysis = json.loads(path.read_text(encoding='utf-8'))
            analysis['primary_window']['audit_pass'] = False
            save(path, analysis)
            kernel = next(job for job in jobs if job['stage'] == 'kernel')
            (root / 'runs' / kernel['name'] / 'complete.json').unlink()
            result = report(root, allow_partial=True)
            self.assertFalse(result['plan_complete'])
            self.assertFalse(result['timing_complete'])
            self.assertEqual(result['valid_timing_runs'], 8)
            self.assertEqual(len(result['invalid']), 1)

    def test_plan_cannot_drop_required_repeat_or_duplicate_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_experiment(root, (128,))
            path = root / 'plan.json'
            plan = json.loads(path.read_text(encoding='utf-8'))
            omitted = next(job for job in plan['runs'] if job['stage'] == 'timing')
            plan['runs'].remove(omitted)
            save(path, plan)
            result = report(root, allow_partial=True)
            self.assertFalse(result['complete'])
            self.assertTrue(result['plan_issues'])

    def test_worker_marker_without_seal_or_with_later_failure_is_rejected(self):
        for kind in ('missing_seal', 'later_failure'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                jobs = create_experiment(root, (128,))
                job = next(job for job in jobs if job['stage'] == 'timing')
                folder = root / 'runs' / job['name']
                if kind == 'missing_seal':
                    (folder / 'completion_audit.json').unlink()
                else:
                    save(folder / 'failure.json', dict(error='shutdown failed after worker complete'))
                result = report(root, allow_partial=True)
                self.assertFalse(result['complete'])
                self.assertEqual(result['valid_timing_runs'], 8)

    def test_sealed_analysis_trace_metadata_and_completion_hashes_are_checked(self):
        for relative in ('analysis.json', 'trace/rank_007/metadata.json', 'trace/rank_002/rounds.json', 'complete.json'):
            with self.subTest(artifact=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                jobs = create_experiment(root, (128,))
                job = next(job for job in jobs if job['stage'] == 'timing')
                path = root / 'runs' / job['name'] / relative
                path.write_bytes(path.read_bytes() + b'\n')
                result = report(root, allow_partial=True)
                self.assertFalse(result['complete'])
                self.assertEqual(result['valid_timing_runs'], 8)
                self.assertIn('artifacts', result['invalid'][0]['error'])

    def test_sealed_wrong_trace_identity_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = create_experiment(root, (128,))
            job = next(job for job in jobs if job['stage'] == 'timing')
            path = root / 'runs' / job['name'] / 'trace/rank_003/metadata.json'
            metadata = load(path)
            metadata['run_id'] = 'different-run'
            save(path, metadata)
            seal_run(root, job)
            result = report(root, allow_partial=True)
            self.assertFalse(result['complete'])
            self.assertIn('identity mismatch', result['invalid'][0]['error'])

    def test_moved_results_are_portable_without_original_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'original'
            root.mkdir()
            create_experiment(root, (128,))
            moved = Path(directory) / 'moved'
            root.rename(moved)
            self.assertFalse(root.exists())
            result = report(moved)
            self.assertTrue(result['complete'])
            self.assertEqual(result['valid_timing_runs'], 9)


if __name__ == '__main__':
    unittest.main()

"""CPU-only contract tests: TP accounting, fail-closed windows and hooks."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

from profiler.analyze import analyze_run
from profiler.instrument import instrument
from profiler.kernel_probe import summarize_events
from profiler import kernel_probe


ROOT = Path(__file__).resolve().parents[1]


def fixture_row(rank, step, bs=128):
    draft, verify = (20, 20) if rank % 2 == 0 else (10, 30)
    ms = {'draft': draft, 'verify_total': verify, 'target_prepare': 2,
          'target_forward': verify - 5, 'accept_and_cache': 3,
          'kv_copy': 1, 'draft_extend': 5, 'total': 46}
    start = rank * 1000 + step * (48 + rank)
    return dict(rank=rank, world_size=8, run_id='test-run', step=step,
                phase='puredecode', bs=bs, batch_fingerprint='same-ordered-requests',
                diagnostic_only=False, target_graph=True, draft_graph=True, draft_extend_graph=True,
                draft_forward_calls=6, draft_frontier_width=2, draft_input_rows=2 * bs,
                draft_input_rows_source='prepared_frontier_positions',
                stages={k: {'gpu_ms': value, 'calls': 1} for k, value in ms.items()},
                gpu_start_ms=start, gpu_end_ms=start + 46,
                commit_observed=True, committed_tokens=3 * bs, accepted_tokens=3 * bs,
                accepted_draft_tokens=2 * bs, non_draft_tokens_per_req=1,
                committed_request_count=bs, accept_lens_count=bs, committed_positive_requests=bs,
                committed_negative_deltas=0, finished_requests_before=0,
                context_min=64, context_max=256, context_mean=128,
                start_unix=100 + step, commit_unix=100.5 + step)


def write_fixture(root, bs=128, steps=12, mutate=None, diagnostic=False):
    for rank in range(8):
        rows = [fixture_row(rank, step, bs) for step in range(steps)]
        if mutate:
            rows = mutate(rank, rows)
        directory = root / f'rank_{rank:03d}'
        directory.mkdir(parents=True, exist_ok=True)
        meta = dict(rank=rank, world_size=8, run_id='test-run', expected_bs=bs,
                    rows=len(rows), recorded_steps=[row['step'] for row in rows],
                    audit_pass=True, diagnostic_only=diagnostic, usable_for_formal_speed=not diagnostic)
        (directory / 'rounds.json').write_text(json.dumps(rows), encoding='utf-8')
        (directory / 'metadata.json').write_text(json.dumps(meta), encoding='utf-8')


class MultiRankAnalysisTests(unittest.TestCase):
    def test_all_supported_batches_and_tokens_counted_once(self):
        for bs in (128, 256, 512):
            with self.subTest(bs=bs), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_fixture(root, bs)
                result = analyze_run(root, bs)
                self.assertTrue(result['audit_pass'], result['audit_errors'])
                primary = result['primary_window']
                self.assertEqual(primary['committed_output_tokens_rank0_only'], 12 * bs * 3)
                self.assertEqual(primary['rank0_continuous_gpu_window_ms'], 11 * 48 + 46)
                self.assertAlmostEqual(primary['rank0_continuous_gpu_decode_tokens_per_second'], 12 * bs * 3 * 1000 / 574)
                self.assertEqual(primary['sum_of_slowest_rank_step_durations_ms'], 12 * 46)
                self.assertEqual(primary['max_rank_window_duration_ms'], 11 * 55 + 46)
                self.assertEqual(primary['draft_execution']['entry_input_rows'], {str(2 * bs): 12})
                self.assertEqual(primary['draft_execution']['entry_rows_sources'], {'prepared_frontier_positions': 12})
                maxima = primary['per_stage_rank_maxima']
                self.assertEqual(maxima['draft']['mean_rank_max_ms_per_step'], 20)
                self.assertEqual(maxima['verify_total']['mean_rank_max_ms_per_step'], 30)
                self.assertGreater(20 + 30 + 5, maxima['total']['mean_rank_max_ms_per_step'])

    def test_minimum_is_not_relaxed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root, steps=9)
            result = analyze_run(root, 128)
            self.assertFalse(result['audit_pass'])
            self.assertIsNone(result['primary_window'])
            self.assertEqual(result['minimum_consecutive_steps_required'], 10)
            self.assertTrue(analyze_run(root, 128, min_consecutive_steps=9)['audit_pass'])
            with self.assertRaises(ValueError):
                analyze_run(root, 128, min_consecutive_steps=1)

    def test_prefill_and_nonadvancing_eos_break_windows(self):
        for kind in ('prefill', 'eos', 'request_change'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                def mutate(rank, rows):
                    if kind == 'prefill':
                        rows[6]['phase'] = 'prefill'
                    elif kind == 'eos' and rank == 0:
                        rows[6]['committed_positive_requests'] -= 1
                    elif kind == 'request_change':
                        for row in rows[6:]:
                            row['batch_fingerprint'] = 'different-request-set'
                    return rows
                root = Path(directory)
                write_fixture(root, mutate=mutate)
                result = analyze_run(root, 128)
                self.assertFalse(result['audit_pass'])
                self.assertLess(result['longest_window']['steps'], 10)

    def test_padded_graph_size_does_not_qualify_actual_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            def mutate(rank, rows):
                for row in rows:
                    row['bs'] = 127
                    row['target_graph_bs'] = 128
                return rows
            root = Path(directory)
            write_fixture(root, mutate=mutate)
            result = analyze_run(root, 128)
            self.assertFalse(result['audit_pass'])
            self.assertEqual(result['physical_full_batch_steps'], 0)

    def test_rank_gate_edges_can_match_but_internal_holes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root, mutate=lambda rank, rows: rows[1:] if rank == 7 else rows)
            result = analyze_run(root, 128)
            self.assertTrue(result['audit_pass'], result['audit_errors'])
            self.assertEqual(result['primary_window']['first_step'], 1)
            self.assertTrue(result['warnings'])
            write_fixture(root, mutate=lambda rank, rows: [row for row in rows if row['step'] != 5] if rank == 7 else rows)
            self.assertFalse(analyze_run(root, 128)['audit_pass'])

    def test_missing_rank_or_mismatched_batch_or_graph_fails(self):
        for kind in ('missing', 'mismatch', 'graph', 'commit'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                def mutate(rank, rows):
                    if kind == 'mismatch' and rank == 5:
                        rows[2]['batch_fingerprint'] = 'incorrect'
                    if kind == 'graph' and rank == 5:
                        rows[2]['draft_extend_graph'] = False
                    if kind == 'commit' and rank == 0:
                        rows[2]['commit_observed'] = False
                    return rows
                write_fixture(root, mutate=mutate)
                if kind == 'missing':
                    (root / 'rank_007/metadata.json').unlink()
                self.assertFalse(analyze_run(root, 128)['audit_pass'])

    def test_kernel_diagnostics_never_formal_speed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root, diagnostic=True)
            result = analyze_run(root, 128)
            self.assertFalse(result['audit_pass'])
            self.assertIsNone(result['primary_window'])

    def test_kernel_totals_exclude_launches_and_copies(self):
        events = [dict(ph='X', cat='kernel', name='gemm', dur=1000, args={'graph id': 2}),
                  dict(ph='X', cat='kernel', name='gemm', dur=2000),
                  dict(ph='X', cat='cuda_runtime', name='cudaGraphLaunch', dur=9000),
                  dict(ph='X', cat='gpu_memcpy', name='copy', dur=8000)]
        result = summarize_events(events)
        self.assertEqual(result['kernel_gpu_ms_sum'], 3)
        self.assertEqual(result['kernel_calls'], 2)
        self.assertEqual(result['graph_node_visibility'], 'observed')


class HookTests(unittest.TestCase):
    def load_hooks(self, directory, rank=0):
        state = {'ticks': 0, 'synchronize': 0}
        class Event:
            def __init__(self, **kwargs):
                self.tick = None
            def record(self):
                state['ticks'] += 1
                self.tick = state['ticks']
            def query(self):
                return True
            def synchronize(self):
                state['synchronize'] += 1
            def elapsed_time(self, other):
                return float(other.tick - self.tick)
        fake = types.ModuleType('torch')
        fake.cuda = types.SimpleNamespace(Event=Event)
        fake.distributed = types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True,
                                                 get_rank=lambda: rank, get_world_size=lambda: 8)
        kernel = types.ModuleType('kernel_probe')
        import contextlib
        kernel.ENABLED = False
        kernel.begin = lambda *args: None
        kernel.end = lambda *args: None
        kernel.finish = lambda: None
        kernel.span = lambda name: contextlib.nullcontext()
        package = types.ModuleType('sglang.srt.speculative')
        package.kernel_probe = kernel
        modules = {'torch': fake, 'sglang': types.ModuleType('sglang'),
                   'sglang.srt': types.ModuleType('sglang.srt'), 'sglang.srt.speculative': package}
        control = directory / 'control.bin'
        if not control.exists():
            control.write_bytes(struct.pack('Q', 1))
        env = {'PROFILE_TRACE_ROOT': str(directory / 'trace'), 'PROFILE_CONTROL': str(control),
               'PROFILE_RUN_ID': 'hook-test', 'PROFILE_EXPECTED_WORLD_SIZE': '8', 'PROFILE_EXPECTED_BS': '128'}
        spec = importlib.util.spec_from_file_location('test_stage_hooks', ROOT / 'profiler/stage_hooks.py')
        hooks = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules), patch.dict(os.environ, env):
            spec.loader.exec_module(hooks)
        self.addCleanup(self.close_hooks, hooks)
        return hooks, state

    @staticmethod
    def close_hooks(hooks):
        hooks._flushed = True
        if hooks.CONTROL is not None:
            hooks.CONTROL.release()
            hooks._mapping.close()

    def test_no_new_d2h_and_real_output_count(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(directory)
        hooks, state = self.load_hooks(root)
        class Req:
            def __init__(self, rid):
                self.rid, self.output_ids, self.finished_len = rid, [], None
            def finished(self):
                return self.finished_len is not None
        requests = [Req('one'), Req('two')]
        batch = types.SimpleNamespace(reqs=requests, seq_lens_cpu=None, is_extend_in_batch=False,
                                      forward_mode=types.SimpleNamespace(is_idle=lambda: False, is_extend=lambda: False))
        class Accept:
            device = types.SimpleNamespace(type='cuda')
            def tolist(self):
                assert self.device.type == 'cpu'
                return [3, 2]
        result = types.SimpleNamespace(accept_lens=Accept(), can_run_cuda_graph=True, num_non_draft_tokens_per_req=1)
        worker = types.SimpleNamespace(ps=types.SimpleNamespace(tp_rank=0))
        returned = hooks.record_worker(lambda self, batch: result)(worker, batch)
        self.assertIs(returned, result)
        self.assertEqual(state['synchronize'], 0)
        self.assertEqual(result.accept_lens.device.type, 'cuda')
        def original_processor(self, batch, result):
            result.accept_lens.device = types.SimpleNamespace(type='cpu')
            requests[0].output_ids.extend([1, 2, 3])
            requests[0].finished_len = 2
            requests[1].output_ids.extend([4, 5])
        hooks.record_commit(original_processor)(None, batch, result)
        hooks.flush()
        row = json.loads((root / 'trace/rank_000/rounds.json').read_text())[0]
        self.assertEqual(row['committed_tokens'], 4)
        self.assertEqual(row['accepted_tokens'], 5)
        self.assertEqual(row['accepted_draft_tokens'], 3)
        self.assertEqual(state['synchronize'], 1)

    def test_rank_ownership_rejects_clobber_without_exit_write(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(directory)
        first, _ = self.load_hooks(root)
        first._init_rank(types.SimpleNamespace(tp_rank=0))
        owner = root / 'trace/rank_000/owner.json'
        original = owner.read_bytes()
        second, _ = self.load_hooks(root)
        with self.assertRaises(RuntimeError):
            second._init_rank(types.SimpleNamespace(tp_rank=0))
        second.flush()
        self.assertIsNone(second._identity)
        self.assertEqual(owner.read_bytes(), original)
        self.assertFalse((owner.parent / 'rounds.json').exists())


class InstrumentTests(unittest.TestCase):
    def test_graph_draft_metadata_accepts_absent_input_ids_without_device_read(self):
        # The real prepare_for_draft sets positions before input_ids is created
        # inside draft_forward. Graph dispatch and off-gate warmup may therefore
        # reach the injected note with input_ids=None.
        paths = ('srt/speculative/eagle_worker_v2.py',
                 'srt/speculative/eagle_worker_common.py',
                 'kernels/ops/kvcache/cache_move.py',
                 'srt/managers/scheduler_components/batch_result_processor.py')
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'source', Path(directory) / 'target'
            with tarfile.open(ROOT / 'vendor/original_runtime.tar.gz') as archive:
                for relative in paths:
                    path = source / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.extractfile('sglang/' + relative).read())
            manifest = instrument(source, target)
            self.assertTrue(manifest['ast_equivalent'])
            self.assertEqual(len(manifest['files']), 4)
            tree = ast.parse((target / paths[0]).read_text(encoding='utf-8'))
            notes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == 'note'
                     and any(key.arg == 'draft_input_rows' for key in node.keywords)]
            self.assertEqual(len(notes), 1)
            evaluate_note = compile(ast.Expression(body=notes[0]), '<instrumented draft note>', 'eval')

            class Positions:
                def __init__(self, size):
                    self.size = size
                def numel(self):
                    return self.size  # Tensor metadata; no GPU data access.
                def cpu(self):
                    raise AssertionError('Diagnostics must not add a device-to-host copy')
                def item(self):
                    raise AssertionError('Diagnostics must not synchronize for a scalar')

            for width in (1, 2):
                for bs in (0, 32, 127, 128, 256, 512):
                    with self.subTest(width=width, bs=bs):
                        values = eval(evaluate_note, {
                            '_steady': types.SimpleNamespace(note=lambda **values: values),
                            'can_run_decode_cuda_graph': True,
                            'self': types.SimpleNamespace(topk=width, speculative_num_steps=7),
                            'draft_input': types.SimpleNamespace(positions=Positions(bs * width)),
                            # A padded Graph512 must not supply the observed row count.
                            'forward_batch': types.SimpleNamespace(input_ids=None, batch_size=512),
                        })
                        self.assertEqual(values['draft_input_rows'], bs * width)
                        self.assertEqual(values['draft_input_rows_source'], 'prepared_frontier_positions')
                        self.assertTrue(values['draft_graph'])

    def test_original_source_is_unchanged_and_hooks_self_contained(self):
        worker = '''class Worker:
    def forward_batch_generation(self, batch):
        return batch
    def draft(self, batch: ScheduleBatch):
        with canary_outside_ctx:
            run()
    def _draft_extend_for_decode(self, batch):
        with canary_ctx:
            if can_run_decode_cuda_graph:
                draft_logits_output = execute()
'''
        common = '''def run_eagle_verify(batch):
    # Batch 1: Target verify
    prepare()
    # Run target verify batch in the main compute stream
    forward_batch_output = target_worker.forward_batch_generation(
        batch=None)
    # Generate vocab mask
    grammar = None
    # Sample
    sample()
    next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)
    return next_draft_input
'''
        files = {'srt/speculative/eagle_worker_v2.py': worker,
                 'srt/speculative/eagle_worker_common.py': common,
                 'kernels/ops/kvcache/cache_move.py': 'def copy_all_layer_kv_cache_func():\n    pass\n',
                 'srt/managers/scheduler_components/batch_result_processor.py': 'class Processor:\n    def process_batch_result_decode(self, batch, result):\n        pass\n'}
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'source', Path(directory) / 'target'
            for relative, text in files.items():
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding='utf-8')
            manifest = instrument(source, target)
            self.assertTrue(manifest['ast_equivalent'])
            self.assertEqual(set(manifest['hooks']), {'stage_hooks.py', 'kernel_probe.py'})
            for relative, text in files.items():
                self.assertEqual((source / relative).read_text(encoding='utf-8'), text)
            self.assertTrue((target / 'srt/speculative/kernel_probe.py').is_file())
            with self.assertRaises(FileExistsError):
                instrument(source, target)


class KernelWindowTests(unittest.TestCase):
    def fake_torch(self, events):
        class Profiler:
            def start(self):
                pass
            def stop(self):
                pass
            def export_chrome_trace(self, path):
                Path(path).write_text(json.dumps({'traceEvents': events}), encoding='utf-8')
        fake = types.ModuleType('torch')
        fake.cuda = types.SimpleNamespace(synchronize=lambda: None)
        fake.profiler = types.SimpleNamespace(
            ProfilerActivity=types.SimpleNamespace(CPU='cpu', CUDA='cuda'),
            supported_activities=lambda: {'cpu', 'cuda'},
            profile=lambda **kwargs: Profiler())
        return fake

    @staticmethod
    def row(step):
        return dict(step=step, phase='puredecode', bs=128, batch_fingerprint='same-requests',
                    target_graph=True, draft_graph=True, draft_extend_graph=True)

    def test_only_continuous_full_identical_request_capture_completes(self):
        events = [dict(ph='X', cat='kernel', name='gemm', dur=100, args={'graph id': 1})]
        with tempfile.TemporaryDirectory() as directory, patch.object(kernel_probe, 'EXPECTED_BS', 128), patch.object(kernel_probe, 'MAX_STEPS', 3), patch.dict(sys.modules, {'torch': self.fake_torch(events)}):
            probe = kernel_probe.Probe(directory)
            for step in range(3):
                row = self.row(step)
                probe.begin(row)
                probe.end(row)
            status = json.loads((Path(directory) / 'kernel_profile/status.json').read_text())
            self.assertEqual(status['status'], 'complete')
            self.assertEqual(status['graph_node_visibility'], 'observed')
            self.assertEqual(status['captured_steps'], 3)

    def test_prefill_size_request_step_or_graph_change_invalidates_capture(self):
        events = [dict(ph='X', cat='kernel', name='gemm', dur=100, args={'graph id': 1})]
        for kind in ('prefill', 'size', 'requests', 'step', 'graph'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, patch.object(kernel_probe, 'EXPECTED_BS', 128), patch.object(kernel_probe, 'MAX_STEPS', 3), patch.dict(sys.modules, {'torch': self.fake_torch(events)}):
                probe = kernel_probe.Probe(directory)
                row = self.row(0)
                probe.begin(row)
                probe.end(row)
                changed = self.row(1)
                changed.update({'prefill': {'phase': 'prefill'}, 'size': {'bs': 127},
                                'requests': {'batch_fingerprint': 'new-requests'},
                                'step': {'step': 2}, 'graph': {'draft_graph': False}}[kind])
                probe.begin(changed)
                probe.end(changed)
                status = json.loads((Path(directory) / 'kernel_profile/status.json').read_text())
                self.assertEqual(status['status'], 'invalid_window')
                self.assertTrue(status['invalid_reason'])
                self.assertLess(status['captured_steps'], 3)

    def test_unverified_graph_nodes_cannot_be_complete(self):
        events = [dict(ph='X', cat='kernel', name='gemm', dur=100),
                  dict(ph='X', cat='cuda_runtime', name='cudaGraphLaunch', dur=1)]
        with tempfile.TemporaryDirectory() as directory, patch.object(kernel_probe, 'EXPECTED_BS', 128), patch.object(kernel_probe, 'MAX_STEPS', 3), patch.dict(sys.modules, {'torch': self.fake_torch(events)}):
            probe = kernel_probe.Probe(directory)
            for step in range(3):
                row = self.row(step)
                probe.begin(row)
                probe.end(row)
            status = json.loads((Path(directory) / 'kernel_profile/status.json').read_text())
            self.assertEqual(status['status'], 'graph_nodes_unverified')
            self.assertNotEqual(status['status'], 'complete')

    def test_graph_launch_correlation_is_explicit_visibility_evidence(self):
        result = summarize_events([
            dict(ph='X', cat='cuda_runtime', name='cudaGraphLaunch', dur=1, args={'correlation': 77}),
            dict(ph='X', cat='kernel', name='graph_gemm', dur=100, args={'correlation': 77})])
        self.assertEqual(result['graph_node_visibility'], 'observed')
        self.assertEqual(result['kernels_correlated_to_graph_launch'], 1)


if __name__ == '__main__':
    unittest.main()

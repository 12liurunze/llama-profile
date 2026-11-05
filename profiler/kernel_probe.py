"""Bounded, separate-process Kineto diagnostics; never formal speed evidence.

Enable PROFILE_KERNEL_MODE=1 in a fresh process/trace directory. Each rank
captures the first actual-BS match and PROFILE_KERNEL_STEPS (default 3)
consecutive pure-decode calls with unchanged ordered request IDs. A batch change
stops and invalidates the capture. Graph-node visibility must be evidenced before
status can be complete. Rank-local kernel duration sums are not elapsed time.
"""
from collections import Counter
import contextlib
import gzip
import json
import math
import os
from pathlib import Path
import shutil


ENABLED = os.getenv('PROFILE_KERNEL_MODE', '0') == '1'
EXPECTED_BS = int(os.getenv('PROFILE_EXPECTED_BS', '512'))
MAX_STEPS = int(os.getenv('PROFILE_KERNEL_STEPS', '3'))
_probe = None


def summarize_events(events):
    """Count complete GPU kernel events only, including visible replay nodes."""
    groups = {}
    launches = [event for event in events if event.get('ph') == 'X'
                and event.get('cat') in ('cuda_runtime', 'cuda_driver')
                and 'graphlaunch' in event.get('name', '').lower()]
    def correlation(event):
        arguments = event.get('args', {})
        for key in ('correlation', 'Correlation ID', 'correlation id'):
            if key in arguments:
                return str(arguments[key])
        return None
    launch_correlations = {correlation(event) for event in launches}
    launch_correlations.discard(None)
    graph_ids = graph_correlations = 0
    for event in events:
        if event.get('ph') != 'X' or event.get('cat') != 'kernel':
            continue
        duration = event.get('dur')
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            continue
        name = event.get('name', '<unnamed>')
        item = groups.setdefault(name, {'name': name, 'calls': 0, 'kernel_gpu_ms': 0.0})
        item['calls'] += 1
        item['kernel_gpu_ms'] += duration / 1000
        graph_ids += any('graph' in str(key).lower() and 'id' in str(key).lower()
                         and value not in (None, 0, '0', '')
                         for key, value in event.get('args', {}).items())
        graph_correlations += correlation(event) in launch_correlations
    kernels = sorted(groups.values(), key=lambda item: -item['kernel_gpu_ms'])
    return {
        'kernel_calls': sum(item['calls'] for item in kernels),
        'kernel_gpu_ms_sum': sum(item['kernel_gpu_ms'] for item in kernels),
        'kernels': kernels, 'cuda_graph_launch_calls': len(launches),
        'kernels_with_graph_id': graph_ids,
        'kernels_correlated_to_graph_launch': graph_correlations,
        'graph_node_visibility': 'observed' if graph_ids or graph_correlations else 'unverified',
        'trace_categories': dict(Counter(event.get('cat', '') for event in events)),
        'time_semantics': 'Rank-local kernel busy-time sum; may contain overlaps; not elapsed time, throughput, or multi-GPU critical path.',
    }


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


class Probe:
    def __init__(self, rank_dir):
        if EXPECTED_BS not in (128, 256, 512) or MAX_STEPS < 1:
            raise RuntimeError('Invalid diagnostic batch size or step count')
        self.out = Path(rank_dir) / 'kernel_profile'
        self.profiler = None
        self.torch = None
        self.rows = []
        self.inflight = None
        self.finished = False
        self.invalid_reason = None

    def _window_problem(self, row):
        if row.get('phase') != 'puredecode' or row.get('bs') != EXPECTED_BS:
            return 'A non-decode or non-full-batch call interrupted the diagnostic window'
        step = row.get('step')
        if not isinstance(step, int) or isinstance(step, bool) or step < 0 or not row.get('batch_fingerprint'):
            return 'Missing absolute step or ordered-request fingerprint'
        if self.rows:
            previous = self.rows[-1]
            if step != previous['step'] + 1:
                return 'Diagnostic worker steps are not consecutive'
            if row['batch_fingerprint'] != previous['batch_fingerprint']:
                return 'Ordered request IDs changed inside the diagnostic window'
        return None

    def begin(self, row):
        if self.finished:
            return
        if self.inflight is not None:
            raise RuntimeError('Kernel diagnostic begin called before prior end')
        if self.profiler is None and (row.get('phase') != 'puredecode' or row.get('bs') != EXPECTED_BS):
            return
        problem = self._window_problem(row)
        if problem:
            self.invalid_reason = problem
            self.finish()
            return
        if self.profiler is None:
            import torch
            self.torch = torch
            activity = torch.profiler.ProfilerActivity
            if activity.CUDA not in torch.profiler.supported_activities():
                raise RuntimeError('CUDA Kineto required; no CPU-only diagnostic fallback')
            # Diagnostic-only boundary sync; formal mode does not run this path.
            torch.cuda.synchronize()
            self.profiler = torch.profiler.profile(activities=[activity.CPU, activity.CUDA],
                                                   record_shapes=False, with_stack=False,
                                                   profile_memory=False, with_flops=False)
            self.profiler.start()
        self.inflight = row
        row['kernel_profile_captured'] = True

    def end(self, row):
        if self.finished or self.inflight is None:
            return
        if self.inflight is not row:
            raise RuntimeError('Mismatched kernel diagnostic worker call')
        self.rows.append({key: value for key, value in row.items()
                          if value is None or isinstance(value, (int, float, str, bool))})
        self.inflight = None
        if any(row.get(key) is not True for key in ('target_graph', 'draft_graph', 'draft_extend_graph')):
            self.invalid_reason = 'Captured worker lacks true target/draft/extend graph execution evidence'
            self.finish()
            return
        if len(self.rows) >= MAX_STEPS:
            self.finish()

    def finish(self):
        if self.finished:
            return
        self.finished = True
        status = {'diagnostic_only': True, 'usable_for_formal_speed': False,
                  'captured_steps': len(self.rows), 'requested_steps': MAX_STEPS,
                  'rows': self.rows, 'expected_bs': EXPECTED_BS,
                  'invalid_reason': self.invalid_reason,
                  'selection': 'Consecutive pure decode calls at actual requested BS with identical ordered-request fingerprint; no refill, padding or gap substitution.'}
        if self.profiler is None:
            status['status'] = 'invalid_window' if self.invalid_reason else 'no_actual_batch_match'
            _write(self.out / 'status.json', status)
            return
        try:
            self.torch.cuda.synchronize()
            self.profiler.stop()
            self.out.mkdir(parents=True, exist_ok=True)
            raw = self.out / 'trace.json'
            self.profiler.export_chrome_trace(str(raw))
            with raw.open(encoding='utf-8') as handle:
                totals = summarize_events(json.load(handle)['traceEvents'])
            with raw.open('rb') as source, gzip.open(self.out / 'trace.json.gz', 'wb', compresslevel=1) as target:
                shutil.copyfileobj(source, target)
            raw.unlink()
            totals.update(diagnostic_only=True, usable_for_formal_speed=False, rows=self.rows)
            _write(self.out / 'kernel_totals.json', totals)
            status.update(graph_node_visibility=totals['graph_node_visibility'],
                          kernel_calls=totals['kernel_calls'],
                          kernels_with_graph_id=totals['kernels_with_graph_id'],
                          kernels_correlated_to_graph_launch=totals['kernels_correlated_to_graph_launch'])
            if self.invalid_reason:
                status['status'] = 'invalid_window'
            elif len(self.rows) != MAX_STEPS:
                status['status'] = 'partial_capture'
            elif not totals['kernel_calls']:
                status['status'] = 'missing_cuda_kernels'
            elif totals['graph_node_visibility'] != 'observed':
                status['status'] = 'graph_nodes_unverified'
            else:
                status['status'] = 'complete'
        except Exception as error:
            status.update(status='failed', error=repr(error))
            raise
        finally:
            _write(self.out / 'status.json', status)


def begin(row, rank_dir):
    global _probe
    if ENABLED:
        if _probe is None:
            _probe = Probe(rank_dir)
        _probe.begin(row)


def end(row):
    if ENABLED and _probe is not None:
        _probe.end(row)


def span(name):
    if ENABLED and _probe is not None and _probe.profiler is not None and not _probe.finished:
        return _probe.torch.profiler.record_function('profile.stage.' + name)
    return contextlib.nullcontext()


def finish():
    if ENABLED and _probe is not None:
        _probe.finish()

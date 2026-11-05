"""TP-safe, asynchronous CUDA spans with rank-zero retained-output accounting.

Environment: PROFILE_TRACE_ROOT, PROFILE_CONTROL (shared uint64 gate 0/1/2),
PROFILE_RUN_ID, PROFILE_EXPECTED_WORLD_SIZE (8), PROFILE_EXPECTED_BS (128/256/512).
Set gate=2 and execute an unmeasured probe to flush every rank. There are no
added collectives, per-step CUDA synchronizations, or device-to-host transfers.
"""
import atexit
from collections import deque
import contextlib
import functools
import hashlib
import json
import mmap
import os
from pathlib import Path
import socket
import time

import torch
from sglang.srt.speculative import kernel_probe as _kernel


ROOT = Path(os.environ['PROFILE_TRACE_ROOT']) if os.getenv('PROFILE_TRACE_ROOT') else None
RUN_ID = os.getenv('PROFILE_RUN_ID', '')
EXPECTED_WORLD = int(os.getenv('PROFILE_EXPECTED_WORLD_SIZE', '8'))
EXPECTED_BS = int(os.getenv('PROFILE_EXPECTED_BS', '512'))
CONTROL = None
_mapping = None
if ROOT is not None:
    if not RUN_ID or EXPECTED_WORLD < 1 or EXPECTED_BS not in (128, 256, 512):
        raise RuntimeError('Valid PROFILE_RUN_ID, expected world size and batch size are required')
    with open(os.environ['PROFILE_CONTROL'], 'r+b') as handle:
        _mapping = mmap.mmap(handle.fileno(), 0)
    if len(_mapping) < 8:
        raise RuntimeError('PROFILE_CONTROL needs at least eight bytes')
    CONTROL = memoryview(_mapping).cast('Q')

_identity = None
_out = None
_step = 0
_current = None
_pending = deque()
_rows = []
_events = []
_waiting = {}
_anchor = None
_flushed = False
_failures = []


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def _init_rank(worker):
    global _identity, _out
    if _identity is not None:
        return
    dist = torch.distributed
    if not dist.is_available() or not dist.is_initialized():
        if EXPECTED_WORLD != 1:
            raise RuntimeError('TP profiling requires an initialized default distributed process group')
        rank, world = 0, 1
    else:
        rank, world = dist.get_rank(), dist.get_world_size()
    if world != EXPECTED_WORLD or not 0 <= rank < world:
        raise RuntimeError(f'Expected world {EXPECTED_WORLD}; observed rank/world {rank}/{world}')
    tp_rank = getattr(worker, 'tp_rank', None)
    if tp_rank is None:
        tp_rank = getattr(getattr(worker, 'ps', None), 'tp_rank', None)
    if tp_rank is not None and int(tp_rank) != rank:
        raise RuntimeError(f'Only single-node DP1/PP1 TP runs supported: tp_rank={tp_rank}, global_rank={rank}')
    out = ROOT / f'rank_{rank:03d}'
    out.mkdir(parents=True, exist_ok=True)
    identity = {'rank': rank, 'world_size': world, 'run_id': RUN_ID,
                'pid': os.getpid(), 'host': socket.gethostname(), 'expected_bs': EXPECTED_BS}
    # Never replace another process's output, including a stale prior run.
    try:
        with (out / 'owner.json').open('x', encoding='utf-8') as handle:
            json.dump(identity, handle)
    except FileExistsError:
        raise RuntimeError(f'Rank trace already owned: {out}; use a fresh trace directory') from None
    if any((out / name).exists() for name in ('rounds.json', 'metadata.json')):
        raise RuntimeError(f'Existing trace artifacts in {out}')
    _out, _identity = out, identity


def _event():
    return _events.pop() if _events else torch.cuda.Event(enable_timing=True)


@contextlib.contextmanager
def span(name):
    if _current is None:
        yield
        return
    with _kernel.span(name):
        start, end = _event(), _event()
        start.record()
        try:
            yield
        finally:
            end.record()
            _current['events'].append((name, start, end))


def phase(name):
    def decorate(fn):
        if ROOT is None:
            return fn
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with span(name):
                return fn(*args, **kwargs)
        return wrapped
    return decorate


def note(**values):
    if _current is not None:
        _current['row'].update(values)


def _retained_length(req):
    size = len(req.output_ids)
    finished_len = getattr(req, 'finished_len', None)
    return min(size, int(finished_len)) if finished_len is not None else size


def _digest(values):
    return hashlib.sha256(json.dumps(values, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def record_commit(fn):
    if ROOT is None:
        return fn
    @functools.wraps(fn)
    def wrapped(self, batch, result, *args, **kwargs):
        # Other TP ranks may process CPU results too. Never count them again.
        entry = _waiting.get(id(result)) if _identity is not None and _identity['rank'] == 0 else None
        if entry is None or entry[0] is not result:
            return fn(self, batch, result, *args, **kwargs)
        row = entry[1]
        requests = list(batch.reqs)
        before = [_retained_length(req) for req in requests]
        row['finished_requests_before'] = sum(bool(req.finished()) for req in requests)
        started = time.perf_counter()
        returned = fn(self, batch, result, *args, **kwargs)
        row['scheduler_result_cpu_ms'] = (time.perf_counter() - started) * 1000
        deltas = [_retained_length(req) - old for req, old in zip(requests, before)]
        row.update(commit_observed=True, commit_unix=time.time(), committed_tokens=sum(deltas),
                   committed_request_count=len(requests), committed_positive_requests=sum(n > 0 for n in deltas),
                   committed_negative_deltas=sum(n < 0 for n in deltas),
                   committed_zero_deltas=sum(n == 0 for n in deltas),
                   commit_lengths_sha256=_digest(deltas),
                   finished_requests_after=sum(bool(req.finished()) for req in requests))
        accept = getattr(result, 'accept_lens', None)
        if accept is None or accept.device.type != 'cpu':
            row['accept_audit_error'] = 'Existing CPU result did not supply CPU accept_lens; no additional D2H attempted'
        else:
            values = accept.tolist()
            non_draft = int(getattr(result, 'num_non_draft_tokens_per_req', 1))
            row.update(accepted_tokens=sum(values), accepted_draft_tokens=sum(n - non_draft for n in values),
                       non_draft_tokens_per_req=non_draft, accept_lens_count=len(values),
                       accept_lens_min=min(values) if values else None,
                       accept_lens_max=max(values) if values else None,
                       accept_lens_histogram={str(n): values.count(n) for n in sorted(set(values))})
        del _waiting[id(result)]
        return returned
    return wrapped


def drain(wait=False):
    while _pending:
        frame = _pending[0]
        if wait:
            frame['ready'].synchronize()
        elif not frame['ready'].query():
            break
        _pending.popleft()
        row, stages = frame['row'], {}
        for name, start, end in frame['events']:
            item = stages.setdefault(name, {'gpu_ms': 0.0, 'calls': 0})
            item['gpu_ms'] += start.elapsed_time(end)
            item['calls'] += 1
            if name == 'total':
                row['gpu_start_ms'], row['gpu_end_ms'] = _anchor.elapsed_time(start), _anchor.elapsed_time(end)
            _events.extend((start, end))
        row['stages'] = stages
        _rows.append(row)


def flush():
    global _flushed
    if _identity is None or _flushed:
        return
    drain(wait=True)
    unmatched = [row['step'] for row in _rows if row['phase'] == 'puredecode'
                 and _identity['rank'] == 0 and not row.get('commit_observed')]
    accept_errors = [row['step'] for row in _rows if row.get('accept_audit_error')]
    metadata = dict(_identity, schema_version=3, rows=len(_rows),
                    recorded_steps=[row['step'] for row in _rows],
                    absolute_worker_calls=_step, unmatched_decode_commit_steps=unmatched,
                    acceptance_error_steps=accept_errors, failures=list(_failures),
                    audit_pass=not unmatched and not accept_errors and not _failures,
                    token_counting_rank=0, extra_worker_d2h_copy=False,
                    per_batch_added_cuda_synchronize=False,
                    diagnostic_only=_kernel.ENABLED, usable_for_formal_speed=not _kernel.ENABLED,
                    timing='Rank-local CUDA event spans. Nested stages are inclusive. No global GPU clock alignment.',
                    output_count='Rank 0 CPU-retained output delta after original result processing; excludes EOS/length overshoot.',
                    flush_unix=time.time())
    _write(_out / 'rounds.json', _rows)
    _write(_out / 'metadata.json', metadata)
    _kernel.finish()
    _flushed = True


def record_worker(fn):
    if ROOT is None:
        return fn
    @functools.wraps(fn)
    def wrapped(self, batch, *args, **kwargs):
        global _step, _current, _anchor
        step = _step
        _step += 1  # Count off-gate calls too so rank-edge timing cannot shift IDs.
        gate = int(CONTROL[0])
        if gate in (1, 2):
            _init_rank(self)
        if gate == 2:
            flush()
        if gate != 1:
            return fn(self, batch, *args, **kwargs)
        if _flushed or _current is not None:
            raise RuntimeError('Cannot reactivate a flushed trace or nest recorded worker calls')
        idle = batch.forward_mode.is_idle()
        prefill = batch.forward_mode.is_extend() or batch.is_extend_in_batch
        kind = 'idle' if idle else 'prefill' if prefill else 'puredecode'
        rids = [str(req.rid) for req in batch.reqs]
        row = dict(_identity, step=step, phase=kind, bs=len(rids),
                   batch_fingerprint=_digest([kind, rids]), start_unix=time.time(),
                   diagnostic_only=_kernel.ENABLED, commit_observed=False)
        lengths = getattr(batch, 'seq_lens_cpu', None)
        if lengths is not None and getattr(lengths, 'is_cpu', False):
            lengths = lengths.tolist()
            row.update(context_min=min(lengths) if lengths else None, context_max=max(lengths) if lengths else None,
                       context_mean=sum(lengths) / len(lengths) if lengths else None)
        frame = {'row': row, 'events': []}
        _current = frame
        _kernel.begin(row, _out)
        if _anchor is None:
            _anchor = torch.cuda.Event(enable_timing=True)
            _anchor.record()
        started = time.perf_counter()
        try:
            with span('total'):
                result = fn(self, batch, *args, **kwargs)
            row.update(cpu_submit_ms=(time.perf_counter() - started) * 1000,
                       end_submit_unix=time.time(), target_graph=bool(getattr(result, 'can_run_cuda_graph', False)))
            if kind == 'puredecode' and _identity['rank'] == 0:
                _waiting[id(result)] = (result, row)
            ready = torch.cuda.Event()
            ready.record()
            frame['ready'] = ready
            _pending.append(frame)
        except Exception as error:
            _failures.append({'step': step, 'error': repr(error)})
            raise
        finally:
            _kernel.end(row)
            _current = None
        if len(_pending) >= 8:
            drain()
        return result
    return wrapped


def _at_exit():
    try:
        flush()
    except Exception:
        # Only explicit gate=2 flush plus complete rank audits is authoritative.
        pass


atexit.register(_at_exit)

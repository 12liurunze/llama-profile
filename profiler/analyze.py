"""Match TP ranks and audit actual-BS steady decode without summing GPU times.

The primary candidate rate is rank-zero retained output divided by rank-zero
CUDA window elapsed time, including gaps. It is NOT client end-to-end throughput
or a globally clock-aligned multi-GPU critical path. Default minimum: ten
consecutive steps, never automatically relaxed.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


STAGES = ('draft', 'verify_total', 'target_prepare', 'target_forward',
          'accept_and_cache', 'kv_copy', 'draft_extend', 'total')
REQUIRED = tuple(name for name in STAGES if name != 'kv_copy')
SUPPORTED_BS = (128, 256, 512)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _stage(row, name):
    value = row.get('stages', {}).get(name, {})
    value = value.get('gpu_ms', 0) if isinstance(value, dict) else None
    return value if _number(value) else 0.0


def _rate(tokens, milliseconds):
    return tokens * 1000 / milliseconds if milliseconds > 0 else None


def timing_errors(row, expected_bs, tolerance_ms=0.1):
    errors = []
    if row.get('phase') != 'puredecode' or row.get('bs') != expected_bs:
        errors.append('not pure decode at required actual batch size')
    for name in REQUIRED:
        item = row.get('stages', {}).get(name)
        if not isinstance(item, dict) or not _number(item.get('gpu_ms')) or item['gpu_ms'] < 0:
            errors.append('missing/invalid stage ' + name)
        elif item.get('calls') != 1:
            errors.append('expected exactly one outer call for ' + name)
    for name in ('target_graph', 'draft_graph', 'draft_extend_graph'):
        if row.get(name) is not True:
            errors.append('missing/false graph execution evidence ' + name)
    if row.get('draft_forward_calls') != 6 or row.get('draft_frontier_width') not in (1, 2):
        errors.append('expected seven-step, width-one/two drafting configuration')
    if not _number(row.get('draft_input_rows')) or row['draft_input_rows'] <= 0:
        errors.append('missing draft-entry input row metadata')
    if errors:
        return errors
    total = _stage(row, 'total')
    if total <= 0:
        errors.append('nonpositive worker duration')
    top = sum(_stage(row, name) for name in ('draft', 'verify_total', 'draft_extend'))
    child = sum(_stage(row, name) for name in ('target_prepare', 'target_forward', 'accept_and_cache'))
    if top > total + tolerance_ms:
        errors.append('rank-local top-level spans exceed total')
    if child > _stage(row, 'verify_total') + tolerance_ms:
        errors.append('verify children exceed verify total')
    if _stage(row, 'kv_copy') > _stage(row, 'accept_and_cache') + tolerance_ms:
        errors.append('KV-copy subspan exceeds acceptance parent')
    start, end = row.get('gpu_start_ms'), row.get('gpu_end_ms')
    if not (_number(start) and _number(end) and end >= start):
        errors.append('invalid rank-local GPU endpoints')
    elif abs(end - start - total) > tolerance_ms:
        errors.append('GPU endpoints disagree with worker duration')
    return errors


def token_errors(row):
    errors = []
    bs = row['bs']
    if not row.get('commit_observed'):
        errors.append('rank-zero CPU result object was not matched')
    if row.get('accept_audit_error'):
        errors.append(row['accept_audit_error'])
    for name in ('committed_tokens', 'accepted_tokens', 'accepted_draft_tokens'):
        if not _number(row.get(name)) or row[name] < 0:
            errors.append('missing/invalid ' + name)
    if row.get('committed_request_count') != bs or row.get('accept_lens_count') != bs:
        errors.append('rank-zero CPU result width differs from actual batch')
    if row.get('committed_negative_deltas', 0):
        errors.append('negative retained-output delta')
    if row.get('non_draft_tokens_per_req') != 1:
        errors.append('expected one bonus per request')
    if not errors:
        if row['accepted_tokens'] != row['accepted_draft_tokens'] + bs:
            errors.append('accepted draft plus one bonus disagrees with accepted total')
        if row['committed_tokens'] > row['accepted_tokens']:
            errors.append('retained output exceeds accepted output')
    return errors


def _runs(paired_steps):
    runs, current = [], []
    for paired in paired_steps:
        row = paired[0]
        if current and (row['step'] != current[-1][0]['step'] + 1
                        or row['batch_fingerprint'] != current[-1][0]['batch_fingerprint']):
            runs.append(current)
            current = []
        current.append(paired)
    if current:
        runs.append(current)
    return runs


def _window(paired_steps, expected_bs):
    zero = [paired[0] for paired in paired_steps]
    ranks = sorted(paired_steps[0])
    count = len(zero)
    errors, per_rank = [], {}
    for rank in ranks:
        rows = [paired[rank] for paired in paired_steps]
        for row in rows:
            problems = timing_errors(row, expected_bs)
            if rank == 0:
                problems += token_errors(row)
            if problems:
                errors.append({'rank': rank, 'step': row['step'], 'errors': problems})
        stages = {name: {'total_ms': sum(_stage(row, name) for row in rows),
                         'mean_ms_per_step': sum(_stage(row, name) for row in rows) / count}
                  for name in STAGES}
        endpoints = all(_number(row.get('gpu_start_ms')) and _number(row.get('gpu_end_ms')) for row in rows)
        elapsed = rows[-1]['gpu_end_ms'] - rows[0]['gpu_start_ms'] if endpoints else None
        gap = elapsed - stages['total']['total_ms'] if elapsed is not None else None
        if gap is not None and gap < -0.1:
            errors.append({'rank': rank, 'errors': ['rank-local worker intervals overlap']})
        per_rank[str(rank)] = {
            'stages': stages, 'window_gpu_elapsed_ms': elapsed,
            'inter_worker_gap_ms': gap,
            'other_worker_ms': stages['total']['total_ms'] - sum(stages[n]['total_ms'] for n in ('draft', 'verify_total', 'draft_extend')),
            'verify_unattributed_ms': stages['verify_total']['total_ms'] - sum(stages[n]['total_ms'] for n in ('target_prepare', 'target_forward', 'accept_and_cache')),
            'graph_fractions': {name: sum(row.get(name) is True for row in rows) / count
                                for name in ('target_graph', 'draft_graph', 'draft_extend_graph')},
        }
    has_tokens = all(not token_errors(row) for row in zero)
    committed = sum(row['committed_tokens'] for row in zero) if has_tokens else None
    accepted = sum(row['accepted_tokens'] for row in zero) if has_tokens else None
    draft = sum(row['accepted_draft_tokens'] for row in zero) if has_tokens else None
    zero_elapsed = per_rank['0']['window_gpu_elapsed_ms']
    zero_total = per_rank['0']['stages']['total']['total_ms']
    max_stage = {name: {
        'mean_rank_max_ms_per_step': sum(max(_stage(paired[rank], name) for rank in ranks) for paired in paired_steps) / count,
        'semantics': 'Maximum rank-local duration of this stage at each step; different stages may be slowest on different ranks; do not sum these maxima.',
    } for name in STAGES}
    slowest_step_sum = sum(max(_stage(paired[rank], 'total') for rank in ranks) for paired in paired_steps)
    rank_elapsed = [value['window_gpu_elapsed_ms'] for value in per_rank.values() if value['window_gpu_elapsed_ms'] is not None]
    contexts = [row.get('context_mean') for row in zero if _number(row.get('context_mean'))]
    commit_times = [row['commit_unix'] for row in zero if _number(row.get('commit_unix'))]
    return {
        'steps': count, 'actual_bs': expected_bs, 'request_steps': count * expected_bs,
        'first_step': zero[0]['step'], 'last_step': zero[-1]['step'],
        'batch_fingerprint': zero[0]['batch_fingerprint'], 'rank_count': len(ranks),
        'committed_output_tokens_rank0_only': committed,
        'accepted_tokens_with_bonus_rank0_only': accepted,
        'accepted_draft_tokens_rank0_only': draft,
        'mean_accepted_draft': draft / (count * expected_bs) if draft is not None else None,
        'mean_accepted_plus_one': accepted / (count * expected_bs) if accepted is not None else None,
        'mean_committed_per_request_step': committed / (count * expected_bs) if committed is not None else None,
        'rank0_continuous_gpu_window_ms': zero_elapsed,
        'rank0_continuous_gpu_decode_tokens_per_second': _rate(committed, zero_elapsed) if committed is not None and zero_elapsed is not None else None,
        'rank0_worker_effective_decode_tokens_per_second': _rate(committed, zero_total) if committed is not None else None,
        'raw_accept_formula_rank0_window_tokens_per_second': _rate(draft, zero_elapsed) if draft is not None and zero_elapsed is not None else None,
        'accept_plus_one_formula_rank0_window_tokens_per_second': _rate(accepted, zero_elapsed) if accepted is not None and zero_elapsed is not None else None,
        'per_rank': per_rank, 'per_stage_rank_maxima': max_stage,
        'sum_of_slowest_rank_step_durations_ms': slowest_step_sum,
        'max_rank_window_duration_ms': max(rank_elapsed) if rank_elapsed else None,
        'multi_rank_time_semantics': 'Slowest-rank durations are proxies, not a globally clock-aligned critical path. No card times are added. Stage maxima are non-additive.',
        'primary_rate_scope': 'Rank-zero continuous CUDA-event window including inter-worker gaps; not client end-to-end throughput or synchronized global wall time.',
        'context': {'mean_over_steps': sum(contexts) / len(contexts) if contexts else None,
                    'min': min((row['context_min'] for row in zero if _number(row.get('context_min'))), default=None),
                    'max': max((row['context_max'] for row in zero if _number(row.get('context_max'))), default=None),
                    'steps_recorded': len(contexts)},
        'draft_execution': {
            'frontier_widths': dict(Counter(str(row.get('draft_frontier_width')) for row in zero)),
            'configured_inner_forwards': dict(Counter(str(row.get('draft_forward_calls')) for row in zero)),
            'entry_input_rows': dict(Counter(str(row.get('draft_input_rows')) for row in zero)),
            'entry_rows_sources': dict(Counter(str(row.get('draft_input_rows_source')) for row in zero)),
            'entry_rows_scope': 'Prepared draft frontier positions count (request count times frontier width); not input_ids presence, graph padding, or an independently traced GEMM size.'},
        'cpu_boundaries_reference_only': {
            'first_worker_start_unix': zero[0].get('start_unix'),
            'first_commit_unix': commit_times[0] if commit_times else None,
            'last_commit_unix': commit_times[-1] if commit_times else None,
            'first_to_last_commit_seconds': commit_times[-1] - commit_times[0] if commit_times else None,
            'usable_as_output_rate_denominator': False,
            'reason': 'First output is already produced before first commit; first-to-last commit omits its production interval.'},
        'audit_pass': not errors, 'audit_errors': errors,
    }


def analyze_run(trace_root, expected_bs, min_consecutive_steps=10, expected_world_size=8, method=None):
    """Read rank_NNN/{rounds,metadata}.json and return fail-closed audit/results."""
    if expected_bs not in SUPPORTED_BS:
        raise ValueError('expected_bs must be 128, 256, or 512')
    if min_consecutive_steps < 2 or expected_world_size < 1:
        raise ValueError('At least two consecutive steps and one rank are required')
    root = Path(trace_root)
    maps, metadata, errors, warnings = {}, {}, [], []
    for rank in range(expected_world_size):
        directory = root / f'rank_{rank:03d}'
        try:
            rows = json.loads((directory / 'rounds.json').read_text(encoding='utf-8'))
            meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            errors.append(f'rank {rank}: incomplete/unreadable trace: {error}')
            continue
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows) or not isinstance(meta, dict):
            errors.append(f'rank {rank}: invalid trace document shape')
            continue
        metadata[rank] = meta
        steps = [row.get('step') for row in rows]
        if not all(isinstance(step, int) and not isinstance(step, bool) and step >= 0 for step in steps) or steps != sorted(set(steps)):
            errors.append(f'rank {rank}: duplicate/invalid/out-of-order absolute steps')
            continue
        maps[rank] = {row['step']: row for row in rows}
        if meta.get('rank') != rank or meta.get('world_size') != expected_world_size or meta.get('expected_bs') != expected_bs:
            errors.append(f'rank {rank}: metadata rank/world/batch mismatch')
        if meta.get('rows') != len(rows) or meta.get('recorded_steps') != steps:
            errors.append(f'rank {rank}: metadata row count/steps mismatch')
        if meta.get('audit_pass') is not True:
            errors.append(f'rank {rank}: producer audit failed')
        if meta.get('diagnostic_only') or meta.get('usable_for_formal_speed') is not True:
            errors.append(f'rank {rank}: kernel diagnostic trace cannot be used for formal speed')
        for row in rows:
            if row.get('rank') != rank or row.get('world_size') != expected_world_size or row.get('run_id') != meta.get('run_id'):
                errors.append(f'rank {rank}: row identity mismatch at step {row.get("step")}')
                break
            if row.get('diagnostic_only'):
                errors.append(f'rank {rank}: diagnostic row contamination')
                break
    run_ids = {meta.get('run_id') for meta in metadata.values()}
    if len(run_ids) != 1 or not next(iter(run_ids), None):
        errors.append('all ranks must have one nonempty shared run ID')
    if len(maps) != expected_world_size:
        errors.append('not all expected TP ranks are present')
    common = sorted(set.intersection(*(set(mapping) for mapping in maps.values()))) if len(maps) == expected_world_size else []
    if not common:
        errors.append('no steps shared by all ranks')
    paired = []
    if common:
        union = set.union(*(set(mapping) for mapping in maps.values()))
        internal = [step for step in range(common[0], common[-1] + 1) if step not in common]
        if internal:
            errors.append(f'internal missing rank steps: {internal[:20]}')
        edges = sorted(union - set(common))
        if edges:
            warnings.append({'unmatched_gate_boundary_steps': edges, 'meaning': 'Excluded from all metrics; only trace leading/trailing edges are permitted.'})
        for step in common:
            group = {rank: maps[rank][step] for rank in range(expected_world_size)}
            signatures = {(row.get('phase'), row.get('bs'), row.get('batch_fingerprint')) for row in group.values()}
            if len(signatures) != 1 or not group[0].get('batch_fingerprint'):
                errors.append(f'batch identity/phase/actual-size mismatch across ranks at step {step}')
                continue
            paired.append(group)
    physical = [group for group in paired if group[0]['phase'] == 'puredecode' and group[0]['bs'] == expected_bs]
    strict = [group for group in physical if group[0].get('committed_positive_requests') == expected_bs
              and not group[0].get('finished_requests_before', 0)]
    windows = [_window(run, expected_bs) for run in _runs(strict)]
    for i, window in enumerate(windows):
        window.update(window_index=i, meets_minimum_steps=window['steps'] >= min_consecutive_steps)
    longest = max(windows, key=lambda value: value['steps'], default=None)
    if longest is None or longest['steps'] < min_consecutive_steps:
        errors.append(f'insufficient continuous full-batch evidence: required {min_consecutive_steps} steps; observed {longest["steps"] if longest else 0}')
    elif not longest['audit_pass']:
        errors.append('longest full-batch window failed timing/output audit')
    return {
        'schema_version': 3, 'method': method, 'trace_root': str(root.resolve()),
        'expected_bs': expected_bs, 'expected_world_size': expected_world_size,
        'minimum_consecutive_steps_required': min_consecutive_steps,
        'run_id': next(iter(run_ids), None) if len(run_ids) == 1 else None,
        'rank_row_counts': {str(rank): len(mapping) for rank, mapping in maps.items()},
        'matched_worker_steps': len(paired), 'physical_full_batch_steps': len(physical),
        'live_full_batch_steps': len(strict), 'excluded_nonadvancing_steps': len(physical) - len(strict),
        'windows': windows, 'longest_window': longest,
        'primary_window': longest if not errors else None,
        'audit_pass': not errors, 'audit_errors': errors, 'warnings': warnings,
        'tokens_are_counted_once_from_rank': 0,
        'selection': 'Longest consecutive absolute-step sequence with identical ordered request IDs, pure decode, true requested BS, and every rank-zero request advancing. Prefill/idle/batch changes break a window.',
        'not_end_to_end': True,
    }


analyze_trace = analyze_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace_root', type=Path)
    parser.add_argument('--bs', type=int, choices=SUPPORTED_BS, required=True)
    parser.add_argument('--world-size', type=int, default=8)
    parser.add_argument('--min-steps', type=int, default=10)
    parser.add_argument('--method')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze_run(args.trace_root, args.bs, args.min_steps, args.world_size, args.method)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    else:
        print(rendered)
    raise SystemExit(0 if result['audit_pass'] else 2)


if __name__ == '__main__':
    main()

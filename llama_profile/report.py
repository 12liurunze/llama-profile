"""Validated Chinese reports for the planned three-repeat TP8 experiment."""
import argparse
import csv
import math
from pathlib import Path
import statistics

from .common import BATCHES, METHODS, digest, load, plan_runs, save, sha


STAGES = ('draft', 'verify_total', 'target_prepare', 'target_forward',
          'accept_and_cache', 'kv_copy', 'draft_extend', 'total')
LABELS = {'chain7': 'Chain7', 'adaptive2': 'Adaptive2', 'eagle728': 'EAGLE728'}


class IncompleteReportError(RuntimeError):
    def __init__(self, result):
        self.report = result
        super().__init__('Report is incomplete; see REPORT.md/report.json or explicitly use allow_partial=True')


def _finite(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError('Missing/nonfinite ' + name)
    return value


def _close(actual, expected, name):
    if not math.isclose(_finite(actual, name), expected, rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError('Inconsistent ' + name)


def _artifact_manifest(folder):
    """Same relative artifact rules as runner; no original-machine lookups."""
    ignored = {'completion_audit.json', 'monitor.jsonl', 'worker.log', 'failure.json'}
    result = {}
    if folder.is_symlink():
        raise ValueError('Run directory must not be a symlink')
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            raise ValueError('Run artifact symlinks are not allowed: ' + str(path))
        relative = path.relative_to(folder).as_posix()
        if path.is_file() and relative not in ignored and relative != 'trace/control.bin':
            result[relative] = sha(path)
    return result


def _completion(folder, job, plan):
    if (folder / 'failure.json').exists():
        raise ValueError('Run has failure.json even if worker wrote complete.json')
    value = load(folder / 'complete.json')
    recorded = value.get('job', {})
    if recorded != job:
        raise ValueError('complete.json job identity mismatch')
    completed_at = _finite(value.get('completed_at'), 'worker completion timestamp')
    seal = load(folder / 'completion_audit.json')
    if seal.get('audit_pass') is not True or seal.get('job') != job or seal.get('plan_sha256') != digest(plan):
        raise ValueError('Coordinator completion audit does not match this job/immutable plan')
    if _finite(seal.get('audited_unix'), 'coordinator audit timestamp') < completed_at:
        raise ValueError('Coordinator audit predates the worker completion')
    actual = _artifact_manifest(folder)
    if not isinstance(seal.get('artifacts'), dict) or actual != seal['artifacts']:
        raise ValueError('Sealed run artifacts were added, removed or changed')
    required = {'complete.json', 'launch.json', 'config.json'}
    if job['stage'] != 'calibrate':
        required |= {f'trace/rank_{rank:03d}/{name}' for rank in range(8) for name in ('metadata.json', 'rounds.json')}
    if job['stage'] == 'timing':
        required.add('analysis.json')
    if job['stage'] == 'kernel':
        required |= {f'trace/rank_{rank:03d}/kernel_profile/status.json' for rank in range(8)}
    if not required.issubset(actual):
        raise ValueError('Completion seal omits required evidence: ' + repr(sorted(required - set(actual))))
    launch = load(folder / 'launch.json')
    if launch.get('job') != job or launch.get('plan_sha256') != digest(plan):
        raise ValueError('Launch job or plan identity mismatch')
    if load(folder / 'config.json').get('job') != job:
        raise ValueError('Worker configuration job identity mismatch')
    if job['stage'] != 'calibrate':
        for rank in range(8):
            directory = folder / 'trace' / f'rank_{rank:03d}'
            meta = load(directory / 'metadata.json')
            if (meta.get('schema_version') != 3 or meta.get('audit_pass') is not True
                    or meta.get('rank') != rank or meta.get('world_size') != 8
                    or meta.get('run_id') != job.get('run_id') or meta.get('expected_bs') != job['bs']):
                raise ValueError(f'Rank {rank} metadata audit/job identity mismatch')
            if bool(meta.get('diagnostic_only')) != (job['stage'] == 'kernel'):
                raise ValueError('Kernel and formal trace identities are mixed')
            rows = load(directory / 'rounds.json')
            if not isinstance(rows, list) or meta.get('rows') != len(rows):
                raise ValueError(f'Rank {rank} trace count mismatch')
            for row in rows:
                if (row.get('rank') != rank or row.get('world_size') != 8
                        or row.get('run_id') != job.get('run_id')):
                    raise ValueError(f'Rank {rank} trace row job identity mismatch')
            if meta.get('recorded_steps') != [row.get('step') for row in rows]:
                raise ValueError(f'Rank {rank} trace/metadata steps differ')
            if job['stage'] == 'kernel' and load(directory / 'kernel_profile/status.json').get('status') != 'complete':
                raise ValueError(f'Rank {rank} kernel diagnostic is incomplete')
    return value


def _timing_record(folder, job, minimum):
    analysis = load(folder / 'analysis.json')
    if analysis.get('audit_pass') is not True:
        raise ValueError('analysis audit failed: ' + repr(analysis.get('audit_errors', [])))
    p = analysis.get('primary_window')
    if not isinstance(p, dict) or p.get('audit_pass') is not True:
        raise ValueError('No valid primary window')
    if analysis.get('expected_bs') != job['bs'] or p.get('actual_bs') != job['bs']:
        raise ValueError('Actual/expected batch differs from planned batch')
    if analysis.get('method') != job['method'] or analysis.get('expected_world_size') != 8 or p.get('rank_count') != 8:
        raise ValueError('Method or TP8 identity mismatch')
    if 'run_id' in job and analysis.get('run_id') != job['run_id']:
        raise ValueError('Analysis run ID differs from planned run')
    if analysis.get('minimum_consecutive_steps_required') != minimum:
        raise ValueError('Analysis minimum-step threshold differs from the plan')
    steps = _finite(p.get('steps'), 'steps')
    if int(steps) != steps or steps < minimum:
        raise ValueError('Primary window is shorter than configured minimum')
    steps = int(steps)
    tokens = _finite(p.get('committed_output_tokens_rank0_only'), 'committed output')
    accepted = _finite(p.get('accepted_tokens_with_bonus_rank0_only'), 'accepted output')
    raw = _finite(p.get('accepted_draft_tokens_rank0_only'), 'accepted draft')
    window_ms = _finite(p.get('rank0_continuous_gpu_window_ms'), 'rank0 window')
    if tokens <= 0 or window_ms <= 0 or raw < 0 or tokens > accepted:
        raise ValueError('Invalid output counts or window duration')
    _close(accepted, raw + steps * job['bs'], 'accepted draft plus bonus')
    _close(p.get('request_steps'), steps * job['bs'], 'request steps')
    _close(p.get('rank0_continuous_gpu_decode_tokens_per_second'), tokens * 1000 / window_ms, 'rank0 window TPS')
    ranks = p.get('per_rank', {})
    if set(ranks) != {str(i) for i in range(8)}:
        raise ValueError('Primary window lacks one or more TP ranks')
    rank0 = ranks['0']
    stage_ms = {}
    for stage in STAGES:
        item = rank0.get('stages', {}).get(stage, {})
        stage_ms[stage] = _finite(item.get('total_ms'), stage)
        if stage_ms[stage] < 0:
            raise ValueError('Negative stage duration ' + stage)
        _close(item.get('mean_ms_per_step'), stage_ms[stage] / steps, stage + ' mean')
    other = _finite(rank0.get('other_worker_ms'), 'other worker')
    gap = _finite(rank0.get('inter_worker_gap_ms'), 'inter-worker gap')
    _close(stage_ms['total'], stage_ms['draft'] + stage_ms['verify_total'] + stage_ms['draft_extend'] + other, 'rank0 additive worker accounting')
    _close(window_ms, stage_ms['total'] + gap, 'rank0 window accounting')
    _close(rank0.get('window_gpu_elapsed_ms'), window_ms, 'rank0 elapsed')
    if stage_ms['kv_copy'] > stage_ms['accept_and_cache'] + .1 * steps:
        raise ValueError('KV-copy child exceeds its parent')
    verify_child = sum(stage_ms[n] for n in ('target_prepare', 'target_forward', 'accept_and_cache'))
    if verify_child > stage_ms['verify_total'] + .1 * steps:
        raise ValueError('Verify children exceed their parent')
    maxima = {}
    for stage in STAGES:
        maxima[stage] = _finite(p.get('per_stage_rank_maxima', {}).get(stage, {}).get('mean_rank_max_ms_per_step'), stage + ' rank maximum')
        if maxima[stage] + .1 < stage_ms[stage] / steps:
            raise ValueError('Per-stage rank maximum is below rank0 duration')
    context = p.get('context', {})
    row = {key: job[key] for key in ('name', 'method', 'bs', 'repeat')}
    row.update(steps=steps, output_tokens=tokens, accepted_draft_tokens=raw,
               accepted_tokens_with_bonus=accepted, request_steps=steps * job['bs'],
               throughput_tps=tokens * 1000 / window_ms,
               worker_effective_tps=tokens * 1000 / stage_ms['total'],
               mean_accepted_draft=raw / (steps * job['bs']),
               mean_accepted_plus_one=accepted / (steps * job['bs']),
               mean_committed=tokens / (steps * job['bs']),
               raw_accept_formula_tps=raw * 1000 / window_ms,
               accept_plus_one_formula_tps=accepted * 1000 / window_ms,
               window_ms=window_ms, context_mean=context.get('mean_over_steps'),
               context_min=context.get('min'), context_max=context.get('max'),
               first_step=p.get('first_step'), last_step=p.get('last_step'),
               stages_ms=stage_ms, other_ms=other, intergap_ms=gap,
               rank_maxima_ms_per_step=maxima,
               analysis_path=str(folder / 'analysis.json'))
    return row


def _stats(values):
    values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'sample_sd': statistics.stdev(values) if len(values) > 1 else None}


def _csv(path, rows, fields):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _format(stats, digits=2):
    if not stats or stats['mean'] is None:
        return '—'
    text = f"{stats['mean']:.{digits}f}"
    return text + (f" ± {stats['sample_sd']:.{digits}f}" if stats['sample_sd'] is not None else '（单次）')


def _summaries(rows, batches, fields):
    groups, flat = [], []
    for bs in batches:
        for method in METHODS:
            selected = [row for row in rows if row['bs'] == bs and row['method'] == method]
            summary = dict(bs=bs, method=method, repeats=[row['repeat'] for row in selected], n=len(selected),
                           metrics={field: _stats([row.get(field) for row in selected]) for field in fields})
            groups.append(summary)
            item = dict(bs=bs, method=method, n=len(selected))
            for field, value in summary['metrics'].items():
                item[field + '_mean'] = value['mean']
                item[field + '_sample_sd'] = value['sample_sd']
            flat.append(item)
    return groups, flat


def _counterfactual(rows, batches):
    lookup = {(row['bs'], row['repeat'], row['method']): row for row in rows}
    results, skipped = [], []
    for bs in batches:
        for repeat in (1, 2, 3):
            trio = [lookup.get((bs, repeat, method)) for method in METHODS]
            if not all(trio):
                skipped.append({'bs': bs, 'repeat': repeat, 'missing_methods': [m for m, row in zip(METHODS, trio) if row is None]})
                continue
            chain, adaptive, eagle = trio
            a_de = adaptive['stages_ms']['draft'] + adaptive['stages_ms']['draft_extend']
            chain_de_per_step = (chain['stages_ms']['draft'] + chain['stages_ms']['draft_extend']) / chain['steps']
            residual = adaptive['window_ms'] - a_de
            predicted_ms = residual + adaptive['steps'] * chain_de_per_step
            if predicted_ms <= 0:
                raise ValueError('Nonpositive counterfactual duration')
            predicted_tps = adaptive['output_tokens'] * 1000 / predicted_ms
            results.append(dict(bs=bs, repeat=repeat,
                adaptive_actual_tps=adaptive['throughput_tps'], eagle_actual_tps=eagle['throughput_tps'],
                adaptive_actual_output_tokens=adaptive['output_tokens'], adaptive_actual_steps=adaptive['steps'],
                adaptive_output_tokens_per_step=adaptive['output_tokens'] / adaptive['steps'],
                adaptive_actual_window_ms=adaptive['window_ms'],
                adaptive_actual_draft_extend_ms_per_step=a_de / adaptive['steps'],
                chain_draft_extend_ms_per_step=chain_de_per_step,
                adaptive_retained_verify_other_intergap_ms=residual,
                adaptive_retained_verify_ms=adaptive['stages_ms']['verify_total'],
                adaptive_retained_other_ms=adaptive['other_ms'],
                adaptive_retained_intergap_ms=adaptive['intergap_ms'],
                counterfactual_window_ms=predicted_ms, counterfactual_committed_tps=predicted_tps,
                counterfactual_vs_eagle_percent=(predicted_tps / eagle['throughput_tps'] - 1) * 100,
                counterfactual_raw_accept_formula_tps=adaptive['accepted_draft_tokens'] * 1000 / predicted_ms,
                counterfactual_accept_plus_one_formula_tps=adaptive['accepted_tokens_with_bonus'] * 1000 / predicted_ms))
    return results, skipped


def report(root, allow_partial=False):
    """Write REPORT.md/report.json/CSVs; reject incomplete evidence by default."""
    root = Path(root).resolve()
    plan = load(root / 'plan.json')
    options = plan['options']
    batches = options.get('batches', list(BATCHES))
    if not batches or any(bs not in BATCHES for bs in batches) or len(set(batches)) != len(batches):
        raise ValueError('Invalid planned batch sizes')
    batches = sorted(batches)
    minimum = options.get('min_steps', 10)
    jobs = plan['runs']
    names = [job['name'] for job in jobs]
    if len(names) != len(set(names)) or any(Path(name).name != name or name in ('.', '..') for name in names):
        raise ValueError('Duplicate or unsafe planned run names')
    expected = {(job['bs'], job['method'], job['repeat']) for job in plan_runs({'batches': batches}) if job['stage'] == 'timing'}
    missing, invalid, records, plan_issues = [], [], [], []
    planned_cells = set()
    completed_jobs = 0
    for job in jobs:
        timing = job['stage'] == 'timing'
        if timing:
            cell = (job['bs'], job['method'], job['repeat'])
            if cell not in expected or cell in planned_cells:
                plan_issues.append({'name': job['name'], 'error': 'unexpected or duplicate timing cell'})
                continue
            planned_cells.add(cell)
        folder = root / 'runs' / job['name']
        try:
            _completion(folder, job, plan)
            completed_jobs += 1
            if timing:
                records.append(_timing_record(folder, job, minimum))
        except FileNotFoundError as error:
            missing.append({'name': job['name'], 'stage': job['stage'], 'error': str(error)})
        except (ValueError, TypeError, KeyError, ZeroDivisionError) as error:
            invalid.append({'name': job['name'], 'stage': job['stage'], 'error': str(error)})
    for cell in sorted(expected - planned_cells):
        plan_issues.append({'cell': list(cell), 'error': 'required timing cell omitted from plan'})
    records.sort(key=lambda row: (row['bs'], METHODS.index(row['method']), row['repeat']))
    accepted_cells = {(row['bs'], row['method'], row['repeat']) for row in records}
    timing_complete = accepted_cells == expected and not plan_issues
    plan_complete = completed_jobs == len(jobs) and not missing and not invalid and not plan_issues
    complete = timing_complete and plan_complete
    full_matrix = accepted_cells == {(bs, method, repeat) for bs in BATCHES for method in METHODS for repeat in (1, 2, 3)}

    throughput_fields = ('steps', 'output_tokens', 'throughput_tps', 'worker_effective_tps',
                         'mean_accepted_draft', 'mean_accepted_plus_one', 'mean_committed',
                         'context_mean', 'context_min', 'context_max', 'window_ms')
    throughput_summary, throughput_flat = _summaries(records, batches, throughput_fields)
    phase_rows, max_rows = [], []
    for row in records:
        phase = {key: row[key] for key in ('name', 'method', 'bs', 'repeat', 'steps')}
        phase.update({stage + '_ms_per_step': value / row['steps'] for stage, value in row['stages_ms'].items()})
        phase.update(other_ms_per_step=row['other_ms'] / row['steps'],
                     intergap_ms_per_step=row['intergap_ms'] / row['steps'],
                     window_ms_per_step=row['window_ms'] / row['steps'])
        phase_rows.append(phase)
        maxima = {key: row[key] for key in ('name', 'method', 'bs', 'repeat', 'steps')}
        maxima.update({stage + '_rank_max_ms_per_step': value for stage, value in row['rank_maxima_ms_per_step'].items()})
        max_rows.append(maxima)
    phase_fields = tuple(stage + '_ms_per_step' for stage in STAGES) + ('other_ms_per_step', 'intergap_ms_per_step', 'window_ms_per_step')
    max_fields = tuple(stage + '_rank_max_ms_per_step' for stage in STAGES)
    phase_summary, phase_flat = _summaries(phase_rows, batches, phase_fields)
    maxima_summary, maxima_flat = _summaries(max_rows, batches, max_fields)
    cf_rows, cf_skipped = _counterfactual(records, batches)
    cf_fields = ('counterfactual_committed_tps', 'counterfactual_vs_eagle_percent',
                 'adaptive_actual_tps', 'eagle_actual_tps',
                 'counterfactual_raw_accept_formula_tps', 'counterfactual_accept_plus_one_formula_tps')
    cf_summary = [dict(bs=bs, n=sum(row['bs'] == bs for row in cf_rows),
                       metrics={field: _stats([row[field] for row in cf_rows if row['bs'] == bs]) for field in cf_fields}) for bs in batches]
    result = dict(schema_version=1, complete=complete, timing_complete=timing_complete,
                  plan_complete=plan_complete, full_matrix_complete=full_matrix, allow_partial=allow_partial,
                  status='complete_planned_scope' if complete else 'incomplete',
                  planned_batches=batches, expected_timing_runs=len(expected), valid_timing_runs=len(records),
                  completed_planned_jobs=completed_jobs, planned_jobs=len(jobs), minimum_consecutive_steps=minimum,
                  missing=missing, invalid=invalid, plan_issues=plan_issues,
                  throughput_records=records, throughput_summary=throughput_summary,
                  rank0_phase_records=phase_rows, rank0_phase_summary=phase_summary,
                  stage_rank_max_records=max_rows, stage_rank_max_summary=maxima_summary,
                  counterfactual_records=cf_rows, counterfactual_summary=cf_summary, counterfactual_skipped=cf_skipped,
                  counterfactual_scope='Arithmetic scenario: replace Adaptive2 rank0 draft+extend per-step cost with matched-repeat Chain7; retain Adaptive2 actual output, verify, other work and inter-step gaps. Not a measured speedup.',
                  completion_integrity='Accepted runs require coordinator completion_audit, no failure marker, exact relative artifact SHA manifest, immutable-plan/job identities and eight matching rank traces. No original model/runtime/data paths are read.',
                  statistics='Equal weight per timing repeat; sample SD (n-1). n<2 has no SD. No pooling or silent repeat replacement.',
                  primary_metric='Rank0 retained output / contiguous GPU-event window including gaps; not client end-to-end or a globally synchronized TP8 critical path.')
    save(root / 'report.json', result)
    key_fields = ('name', 'method', 'bs', 'repeat')
    _csv(root / 'throughput.csv', records, key_fields + throughput_fields + ('accepted_draft_tokens', 'accepted_tokens_with_bonus', 'raw_accept_formula_tps', 'accept_plus_one_formula_tps', 'first_step', 'last_step'))
    _csv(root / 'phases_rank0.csv', phase_rows, key_fields + ('steps',) + phase_fields)
    _csv(root / 'stage_rank_maxima.csv', max_rows, key_fields + ('steps',) + max_fields)
    for name, rows in [('throughput_summary.csv', throughput_flat), ('phases_rank0_summary.csv', phase_flat), ('stage_rank_maxima_summary.csv', maxima_flat)]:
        _csv(root / name, rows, list(rows[0]) if rows else ['bs', 'method', 'n'])
    _csv(root / 'counterfactual.csv', cf_rows, list(cf_rows[0]) if cf_rows else ['bs', 'repeat', 'counterfactual_committed_tps'])

    lines = ['# TP8 Llama 70B 稳态 Decode 报告', '',
             '**状态：' + ('计划内完成' if complete else '未完成；以下仅为有效的部分结果') + '。**', '',
             f"有效 timing：{len(records)}/{len(expected)}；计划任务完成标记：{completed_jobs}/{len(jobs)}；完整 27 组矩阵：{'是' if full_matrix else '否'}。",
             f"计划 BS：{', '.join(map(str, batches))}。每方法每 BS 要求 3 次重复，连续真实满批最少 {minimum} 步；未降低门槛。", '',
             '仅纳入协调器完成审计封存、无 failure 标记、全部相对路径产物 SHA 校验及主窗口审计均通过的结果；worker 单独写出的 complete.json 不足以证明成功。均值与样本标准差按重复等权计算；单次不报告标准差。吞吐是 rank0 实际保留输出除以连续 GPU 事件窗口（含步间间隙），不是客户端端到端吞吐，也不是全卡时钟同步的关键路径。', '',
             '## 吞吐、接受与实际输出', '',
             '|BS|方法|有效重复|输出 token/s|接受 draft/请求步|接受 +1/请求步|实际输出/请求步|窗口输出 token|连续步数|平均上下文 token|',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in throughput_summary:
        m = group['metrics']
        lines.append('|'+ '|'.join([str(group['bs']), LABELS[group['method']], f"{group['n']}/3"] + [_format(m[key]) for key in ('throughput_tps', 'mean_accepted_draft', 'mean_accepted_plus_one', 'mean_committed', 'output_tokens', 'steps', 'context_mean')]) + '|')
    lines += ['', '上下文最小值/最大值、每次原始计数和窗口边界见 throughput.csv。接受 draft 不含 bonus；“+1”仅是对照口径，主吞吐始终采用 EOS/长度截断后的实际保留输出。', '',
              '## Rank0 阶段（ms/步）', '',
              '可相加的主项为 Draft + Verify + Extend + Other + 步间间隙 = 连续窗口总耗时。Prepare/Target/Accept 是 Verify 子项；KV 是 Accept 子项，均不另加到总量。', '',
              '|BS|方法|n|Draft|Verify|Prepare 子项|Target 子项|Accept 子项|KV 子项|Extend|Other|间隙|窗口总计|',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    selected_phases = ('draft', 'verify_total', 'target_prepare', 'target_forward', 'accept_and_cache', 'kv_copy', 'draft_extend', 'other', 'intergap', 'window')
    for group in phase_summary:
        lines.append('|'+ '|'.join([str(group['bs']), LABELS[group['method']], str(group['n'])] + [_format(group['metrics'][key + '_ms_per_step']) for key in selected_phases]) + '|')
    lines += ['', '## 每阶段跨 rank 最大值（ms/步，不可相加）', '',
              '每步先分别取该阶段 8 个 rank 中的最大跨度，再在窗口和重复之间求均值。不同阶段最慢 rank 可不同；这些列不能相加，也不能冒充真正的全局关键路径。', '',
              '|BS|方法|n|Draft 最大|Verify 最大|Prepare 最大|Target 最大|Accept 最大|KV 最大|Extend 最大|Worker 总跨度最大|',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in maxima_summary:
        lines.append('|'+ '|'.join([str(group['bs']), LABELS[group['method']], str(group['n'])] + [_format(group['metrics'][key + '_rank_max_ms_per_step']) for key in STAGES]) + '|')
    lines += ['', '## 反事实成本替换', '',
              '按相同 BS、相同 repeat 配对：将 Adaptive2 的每步 Draft+Extend 替换为 Chain7 的实测值，保留 Adaptive2 本次窗口的 Verify、Other、步间间隙和实际输出 token。用预测窗口吞吐与同 repeat EAGLE728 的实测吞吐比较。该算术情景不是实测提速，不假设两个方法 Verify 成本相同。', '',
              '`T_cf = T_A窗口 − T_A(Draft+Extend) + steps_A × mean_C(Draft+Extend)`；`TPS_cf = A实际输出 × 1000 / T_cf`。', '']
    for item in cf_summary:
        m = item['metrics']
        lines.append(f"- BS{item['bs']}：{item['n']}/3 个完整配对；预测实际输出 {_format(m['counterfactual_committed_tps'])} token/s，相对 EAGLE728 {_format(m['counterfactual_vs_eagle_percent'])}%；EAGLE728 实测 {_format(m['eagle_actual_tps'])} token/s。")
    lines += ['', 'raw accepted-draft 与 accepted+1 两种分子的反事实对照保存在 counterfactual.csv；它们不替代实际输出结论。各方法窗口的上下文与请求轨迹可能不同，替换计算不预测运行时相互作用。']
    if missing or invalid or plan_issues:
        lines += ['', '## 未完成或无效项目', '']
        for item in missing + invalid + plan_issues:
            lines.append('- ' + str(item.get('name', item.get('cell', '?'))) + '：' + item['error'].replace('\n', ' '))
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if not complete and not allow_partial:
        raise IncompleteReportError(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    try:
        value = report(args.root, args.allow_partial)
    except IncompleteReportError as error:
        parser.exit(2, str(error) + '\n')
    print('COMPLETE' if value['complete'] else 'INCOMPLETE', args.root / 'REPORT.md')


if __name__ == '__main__':
    main()

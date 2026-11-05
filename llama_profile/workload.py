from collections import defaultdict
from pathlib import Path
from .common import digest, load, save, sha


def build_steady(prepared_path, calibration_path, output, minimum=256):
    """Length-only selection using target AR; never use any tree acceptance label."""
    if Path(output).exists():
        raise FileExistsError(output)
    prepared, calibration = load(prepared_path), load(calibration_path)
    if type(minimum) is not int or not 1 <= minimum <= prepared['max_new_tokens']:
        raise ValueError('Minimum output must be a positive integer within the generation cap')
    if calibration['prepared_sha256'] != sha(prepared_path):
        raise ValueError('Calibration and prepared inputs differ')
    if calibration['selection_signal'] != 'target_ar_natural_output_length':
        raise ValueError('Only target-AR natural output length may select this workload')
    lengths = {r['prompt_sha256']:r['output_tokens'] for r in calibration['rows']}
    groups = defaultdict(list)
    seen = set()
    all_subsets = sorted({r['subset'] for r in prepared['rows']})
    for row in prepared['rows']:
        key = (row['subset'], row['prompt_sha256'])
        if key not in seen and lengths.get(row['prompt_sha256'], -1) >= minimum:
            groups[row['subset']].append(row)
            seen.add(key)
    missing = [s for s in all_subsets if not groups[s]]
    if missing:
        raise ValueError(f'No naturally long prompts at threshold {minimum} in {missing}; '
                         'do not silently drop a subset or reduce the threshold. Inspect calibration.json.')
    rows = []
    for k in range(500):
        for subset in all_subsets:
            source = groups[subset][k % len(groups[subset])]
            rows.append(dict(source, id=f'steady-{subset}-{k:04d}', parent_id=source['id'],
                steady_copy=k//len(groups[subset]),
                calibrated_natural_output_tokens=lengths[source['prompt_sha256']]))
    assert len(rows) == 4000 and len({r['id'] for r in rows}) == 4000
    result = {k:v for k,v in prepared.items() if k != 'rows'}
    result.update(rows=rows, count=4000, inputs_sha256=digest([r['input_ids'] for r in rows]),
        min_input_tokens=min(len(r['input_ids']) for r in rows),
        max_input_tokens=max(len(r['input_ids']) for r in rows),
        mean_input_tokens=sum(len(r['input_ids']) for r in rows)/4000,
        steady_selection=dict(signal='target_ar_natural_output_length', minimum=minimum,
            calibration_sha256=sha(calibration_path), original_prepared_sha256=sha(prepared_path),
            eligible_distinct_prompts_per_subset={s:len(groups[s]) for s in all_subsets},
            distinct_prompt_count=len({r['prompt_sha256'] for r in rows}),
            interpretation='Conditional long-response workload, repeated to4000; not original benchmark distribution; not selected by speculative acceptance or speed'))
    save(output, result)
    return result['steady_selection']

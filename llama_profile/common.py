import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('chain7', 'adaptive2', 'eagle728')
BATCHES = (128, 256, 512)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def runtime_hash(runtime):
    runtime = Path(runtime).resolve()
    receipt = load(runtime.parent.parent/'install_receipt.json')
    actual = {p.relative_to(runtime).as_posix():sha(p) for p in runtime.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')}
    if actual != receipt['profile_manifest']:
        bad = [k for k in set(actual) | set(receipt['profile_manifest'])
               if actual.get(k) != receipt['profile_manifest'].get(k)]
        raise RuntimeError('Instrumented runtime changed: ' + repr(sorted(bad)[:10]))
    return digest(actual)


def validate_options(o):
    if o['tp'] != 8 or len(o['gpus']) != 8 or len(set(o['gpus'])) != 8:
        raise ValueError('This release requires one TP8 / DP1 instance on eight distinct GPUs')
    if any(not str(g).isdigit() for g in o['gpus']):
        raise ValueError('Use physical numeric GPU indices, e.g. 0,1,2,3,4,5,6,7')
    if not o['batches'] or any(b not in BATCHES for b in o['batches']):
        raise ValueError('Supported batch sizes: 128,256,512')
    if len(set(o['batches'])) != len(o['batches']):
        raise ValueError('Duplicate batch sizes')
    if not 0 <= o['threshold'] <= 1 or not .5 <= o['memory_fraction'] <= .9:
        raise ValueError('Invalid threshold or memory fraction')
    if o['repeat'] != 3 or o['min_steps'] < 2:
        raise ValueError('Three repeats and at least two continuous steps are required')
    if o['context'] < 2048 or not 0 < o['max_new_tokens'] < o['context']:
        raise ValueError('Invalid context / output cap')
    if o['chunked_prefill'] <= 0:
        raise ValueError('Use a positive chunked prefill size')
    if o['protocol'] not in ('queued', 'cohort'):
        raise ValueError('Unknown workload protocol')


def engine_config(o, method, bs):
    validate_options(o)
    if method not in METHODS or bs not in BATCHES:
        raise ValueError('Unknown method or batch')
    return dict(model_path=o['model'], speculative_draft_model_path=o['draft'],
        tp_size=8, dp_size=1, pp_size=1, nnodes=1, node_rank=0,
        dtype='bfloat16', trust_remote_code=False, context_length=o['context'],
        max_running_requests=bs, mem_fraction_static=o['memory_fraction'],
        attention_backend='triton', sampling_backend='pytorch',
        disable_radix_cache=False, disable_cuda_graph=False,
        disable_decode_cuda_graph=False, cuda_graph_backend_prefill='breakable',
        cuda_graph_bs_decode=sorted(set([v for v in (1,2,4,8,16,24,32,48,64,96,128,192,256,384,512) if v <= bs])),
        chunked_prefill_size=o['chunked_prefill'], max_prefill_tokens=max(16384, o['chunked_prefill']),
        cuda_graph_max_bs_prefill=4096, enable_mixed_chunk=False,
        prefill_decode_interval=0, enable_prefill_delayer=False,
        min_free_slots_delay=1, schedule_policy='fcfs',
        schedule_conservativeness=.15, random_seed=42,
        speculative_algorithm='EAGLE3', speculative_num_steps=7,
        speculative_num_draft_tokens=8,
        speculative_eagle_topk=1 if method == 'chain7' else 2,
        speculative_adaptive_tree=method == 'adaptive2',
        speculative_adaptive_confidence='top1',
        speculative_adaptive_threshold=o['threshold'] if method == 'adaptive2' else 0.,
        log_level='info')


def plan_runs(o):
    runs = []
    for bs in o['batches']:
        for method in METHODS:
            runs.append(dict(name=f'smoke_b{bs}_{method}', stage='smoke', bs=bs, method=method, repeat=0))
        for r in range(3):
            order = METHODS[r:] + METHODS[:r]
            for method in order:
                runs.append(dict(name=f'timing_b{bs}_r{r+1}_{method}', stage='timing', bs=bs, method=method, repeat=r+1))
        for method in METHODS:
            runs.append(dict(name=f'kernel_b{bs}_{method}', stage='kernel', bs=bs, method=method, repeat=0))
    return runs


def environment(o, trace, run):
    env = dict(os.environ)
    # Remove historical experiment switches; never inherit an 8B-specific kernel plan.
    for key in list(env):
        if key.startswith(('PROFILE_', 'STEADY_', 'DIAG_', 'MATH_', 'EAGLE_ACTIVE_', 'SGLANG_')) or key in (
                'SWEEP_BS_COUNTER', 'ADAPTIVE_PROFILE_OUT', 'FUSION_AUDIT_OUT',
                'CUDA_LAUNCH_BLOCKING', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK',
                'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=','.join(o['gpus']), PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(Path(o['runtime']).parent) + os.pathsep + str(ROOT),
        PROFILE_TRACE_ROOT=str(trace), PROFILE_CONTROL=str(trace/'control.bin'),
        PROFILE_EXPECTED_WORLD_SIZE='8', PROFILE_EXPECTED_BS=str(run['bs']), PROFILE_RUN_ID=run['run_id'],
        SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN='1', SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY='0',
        LLAMA_EXPERIMENT_SKIP_LOAD_MEM_QUERY='1')
    if run['stage'] == 'kernel':
        env['PROFILE_KERNEL_MODE'] = '1'
    return env

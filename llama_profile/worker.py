"""One coordinator calls Engine(tp_size=8); do NOT launch this with torchrun."""
import mmap
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llama_profile.common import digest, engine_config, load, save, sha


def drain(engine):
    for _ in range(100):
        if engine.flush_cache().success:
            return
        time.sleep(.2)
    raise RuntimeError('Scheduler did not fully drain; trace must not be published as complete')


def checked_outputs(outputs, count, cap, speculative=True):
    if len(outputs) != count:
        raise RuntimeError('Output count does not match requests')
    result = []
    for item in outputs:
        ids, meta = item['output_ids'], item['meta_info']
        reason = meta.get('finish_reason')
        if not isinstance(reason, dict) or reason.get('type') not in ('stop','length'):
            raise RuntimeError('Request did not finish normally: ' + repr(reason))
        if len(ids) != meta['completion_tokens'] or not 0 < len(ids) <= cap:
            raise RuntimeError('Output token accounting failed')
        if speculative:
            keys = ('spec_correct_drafts_histogram','spec_verify_ct',
                    'spec_num_correct_drafts','spec_num_proposed_drafts')
            # A request that ends at its prefill-produced first token never enters
            # speculative verification. SGLang intentionally omits these fields.
            if len(ids) == 1 and all(k not in meta for k in keys):
                meta = dict(meta, spec_correct_drafts_histogram=[0]*8,
                    spec_verify_ct=0, spec_num_correct_drafts=0,
                    spec_num_proposed_drafts=0, prefill_only_completion=True)
            else:
                if any(k not in meta for k in keys):
                    raise RuntimeError('Missing speculative counters after decode')
                hist = meta['spec_correct_drafts_histogram']
                numbers = list(hist) + [meta[k] for k in keys[1:]]
                if any(type(x) is not int or x < 0 for x in numbers):
                    raise RuntimeError('Speculative counters must be nonnegative integers')
                if meta['spec_verify_ct'] == 0 and len(ids) > 1:
                    raise RuntimeError('Multi-token speculative output without verification')
                if (sum(hist) != meta['spec_verify_ct'] or sum(i*n for i,n in enumerate(hist)) != meta['spec_num_correct_drafts']
                        or meta['spec_num_proposed_drafts'] != 7*meta['spec_verify_ct'] or any(hist[8:])):
                    raise RuntimeError('Speculative histogram / seven-slot accounting failed')
        result.append(dict(output_tokens=len(ids), output_ids_sha256=digest(ids),
            finish_reason=meta.get('finish_reason'), meta_info=meta))
    return result


def main():
    root, name = Path(sys.argv[1]).resolve(), sys.argv[2]
    plan = load(root/'plan.json')
    job = next(r for r in plan['runs'] if r['name'] == name)
    o = plan['options']
    folder = root/'runs'/name
    data = load(o['data'])
    assert sha(o['data']) == plan['prepared_sha256']
    sys.path.insert(0, str(Path(o['runtime']).parent))
    import sglang as sgl
    assert Path(sgl.__file__).resolve() == Path(o['runtime']).resolve()/'__init__.py'
    calibration = job['stage'] == 'calibrate'
    cfg = engine_config(o, 'chain7' if calibration else job['method'], job['bs'])
    rows = data['rows']
    cap = o['max_new_tokens']
    if calibration:
        cfg = {k:v for k,v in cfg.items() if not k.startswith('speculative_')}
        cfg.update(max_running_requests=64, cuda_graph_bs_decode=[1,2,4,8,16,24,32,48,64])
        unique = {}
        for row in rows:
            unique.setdefault(row['prompt_sha256'], row)
        rows = list(unique.values())
    elif job['stage'] == 'smoke':
        # Exercise long inputs as well as normal ones; graph capture still covers requested BS.
        ordered = sorted(range(len(rows)), key=lambda i:len(rows[i]['input_ids']), reverse=True)[:8]
        ordered += [i for i in range(len(rows)) if i not in ordered][:24]
        rows = [rows[i] for i in ordered]
        cap = min(128, cap)
    inputs = [r['input_ids'] for r in rows]
    save(folder/'config.json', dict(job=job, config=cfg, request_count=len(rows), input_sha256=digest(inputs),
        sampling=dict(temperature=0, max_new_tokens=cap, ignore_eos=False), source=str(sgl.__file__),
        environment={k:v for k,v in os.environ.items() if k.startswith(('PROFILE_','SGLANG_','NCCL_'))
                     or k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','PYTHONPATH','LLAMA_EXPERIMENT_SKIP_LOAD_MEM_QUERY')}))
    engine = sgl.Engine(**cfg)
    try:
        info = engine.get_server_info()
        save(folder/'server_info.json', info)
        effective = info.get('server_args', info)
        for k in ('tp_size','dp_size','pp_size','max_running_requests','context_length',
                  'disable_radix_cache','disable_decode_cuda_graph','cuda_graph_backend_prefill',
                  'speculative_algorithm','speculative_num_steps','speculative_num_draft_tokens',
                  'speculative_eagle_topk','speculative_adaptive_tree','speculative_adaptive_threshold'):
            if k in cfg and effective.get(k) != cfg[k]:
                raise RuntimeError(f'Effective engine parameter differs: {k}: {effective.get(k)} != {cfg[k]}')
        # Warmup is unmeasured. All measured requests retain natural EOS.
        engine.generate(input_ids=inputs[:min(len(inputs), cfg['max_running_requests'])],
            sampling_params=dict(temperature=0, max_new_tokens=24, ignore_eos=True))
        drain(engine)
        if calibration:
            start = time.perf_counter()
            outputs = engine.generate(input_ids=inputs, sampling_params=dict(temperature=0,max_new_tokens=cap,ignore_eos=False))
            checked = checked_outputs(outputs, len(rows), cap, speculative=False)
            save(folder/'calibration.json', dict(selection_signal='target_ar_natural_output_length',
                prepared_sha256=plan['prepared_sha256'], elapsed_seconds=time.perf_counter()-start,
                max_new_tokens=cap, rows=[dict(prompt_sha256=r['prompt_sha256'], **c) for r,c in zip(rows,checked)]))
        else:
            if o['protocol'] == 'cohort' and job['stage'] != 'smoke':
                unique = list(dict.fromkeys(map(tuple, inputs)))
                engine.generate(input_ids=[list(x) for x in unique], sampling_params=dict(temperature=0,max_new_tokens=1))
            with (folder/'trace/control.bin').open('r+b') as handle, mmap.mmap(handle.fileno(),0) as mm:
                gate = memoryview(mm).cast('Q')
                try:
                    gate[0] = 1
                    begin_unix, begin = time.time(), time.perf_counter()
                    chunks = [inputs] if o['protocol']=='queued' or job['stage']=='smoke' else [inputs[i:i+job['bs']] for i in range(0,len(inputs),job['bs'])]
                    outputs, calls = [], []
                    for i, chunk in enumerate(chunks):
                        start = time.time()
                        outputs.extend(engine.generate(input_ids=chunk, sampling_params=dict(temperature=0,max_new_tokens=cap,ignore_eos=False)))
                        calls.append(dict(index=i, start=start, end=time.time(), requests=len(chunk)))
                    elapsed, end_unix = time.perf_counter()-begin, time.time()
                    drain(engine)
                    gate[0] = 2
                    engine.generate(input_ids=[inputs[0]], sampling_params=dict(temperature=0,max_new_tokens=1))
                    gate[0] = 0
                finally:
                    gate.release()
            checked = checked_outputs(outputs, len(rows), cap)
            save(folder/'outputs_compact.json', [dict(id=r['id'], **c) for r,c in zip(rows,checked)])
            save(folder/'whole_workload.json', dict(elapsed_seconds=elapsed, start=begin_unix, end=end_unix,
                calls=calls, count=len(rows), output_tokens=sum(c['output_tokens'] for c in checked),
                includes_prefill_and_tail=True, usable_as_steady_throughput=False))
            metas = [load(folder/'trace'/f'rank_{r:03d}'/'metadata.json') for r in range(8)]
            if not all(m['audit_pass'] for m in metas):
                raise RuntimeError('Rank trace producer audit failed')
            if job['stage'] == 'smoke':
                for rank in range(8):
                    recorded = load(folder/'trace'/f'rank_{rank:03d}'/'rounds.json')
                    decode = [r for r in recorded if r['phase']=='puredecode']
                    if not decode or not all(r.get(k) is True for r in decode for k in ('target_graph','draft_graph','draft_extend_graph')):
                        raise RuntimeError(f'Smoke failed graph evidence on rank {rank}')
            elif job['stage'] == 'timing':
                from profiler.analyze import analyze_run
                analysis = analyze_run(folder/'trace', job['bs'], o['min_steps'],8,job['method'])
                save(folder/'analysis.json', analysis)
                if not analysis['audit_pass']:
                    raise RuntimeError('No qualified steady result; inspect analysis.json. Do not relabel near-full batches.')
            else:
                from profiler.kernel_audit import audit_captures
                save(folder/'kernel_audit.json',audit_captures(folder/'trace',job['bs']))
        save(folder/'complete.json', dict(job=job, completed_at=time.time(), request_count=len(rows)))
        print('RUN_COMPLETE', name, flush=True)
    finally:
        engine.shutdown()


if __name__ == '__main__':
    main()

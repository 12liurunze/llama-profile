"""Linux coordinator: immutable plans, exclusive locks, and owned process groups.

The coordinator never kills a foreign PID. A failed/interfered run is retained
and only moved to a unique failure archive when the caller explicitly resumes.
"""
import contextlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from . import common, hardware
from .common import digest, load, save, sha


def _options(options):
    value = dict(options)
    for key in ('output', 'data', 'model', 'draft', 'runtime', 'preflight'):
        value[key] = str(Path(value[key]).expanduser().resolve())
    value['gpus'] = [str(gpu) for gpu in value['gpus']]
    value.setdefault('timeout_seconds', 7200)
    value.setdefault('poll_seconds', 2)
    value.setdefault('shutdown_grace_seconds', 30)
    value.setdefault('preflight_timeout_seconds', 900)
    common.validate_options(value)
    for key in ('timeout_seconds', 'poll_seconds', 'shutdown_grace_seconds', 'preflight_timeout_seconds'):
        if not isinstance(value[key], (int, float)) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError(key + ' must be a finite positive number')
    report = Path(value['preflight'])
    protected = [Path(value[key]) for key in ('model', 'draft', 'runtime')]
    protected += [common.ROOT / name for name in ('vendor', 'llama_profile', 'profiler', 'tools')]
    if (report.suffix != '.json' or report == Path(value['data'])
            or report in (Path(value['output']) / 'plan.json', Path(value['output']) / 'input_snapshot.json')
            or any(report == path or path in report.parents for path in protected)):
        raise ValueError('Preflight output must not replace inputs, models, source, or the immutable plan')
    return value


def _code_manifest():
    paths = []
    for folder in ('llama_profile', 'profiler', 'tools'):
        paths.extend((common.ROOT / folder).rglob('*.py'))
    paths += [common.ROOT / name for name in ('main.py', 'pyproject.toml', 'requirements.runtime.txt')]
    return {path.relative_to(common.ROOT).as_posix(): sha(path)
            for path in sorted(paths) if path.is_file() and '__pycache__' not in path.parts}


def _model_identity(audit):
    if not audit.get('audit_pass'):
        raise RuntimeError('Model audit failed: ' + repr(audit.get('errors')))
    return {key: audit[key] for key in ('path', 'config_sha256', 'metadata_files', 'weight_files', 'tensor_count')}


def _preflight_identity(report, options):
    if not report.get('audit_pass') or not report.get('ready_for_formal'):
        raise RuntimeError('Doctor did not authorize a run: ' + repr(report.get('errors', [])))
    for key in ('model', 'draft', 'runtime', 'gpus', 'environment'):
        if not report.get(key, {}).get('audit_pass'):
            raise RuntimeError('Incomplete doctor audit: ' + key)
    for key in ('model', 'draft', 'runtime'):
        if Path(report[key]['path']).resolve() != Path(options[key]):
            raise RuntimeError('Doctor ' + key + ' path differs from this plan')
    if list(map(str, report['gpus']['indices'])) != options['gpus']:
        raise RuntimeError('Doctor GPU list differs from this plan')
    if report['gpus'].get('processes'):
        raise RuntimeError('Doctor observed occupied GPUs')
    devices = [{key: item[key] for key in ('index', 'uuid', 'name', 'memory_mib',
                'driver_version', 'compute_capability', 'mig_mode')}
               for item in report['gpus']['devices']]
    return {'model': _model_identity(report['model']), 'draft': _model_identity(report['draft']),
            'runtime': {key: report['runtime'][key] for key in ('path', 'receipt_path', 'receipt_sha256', 'archive_sha256')},
            'devices': devices, 'packages': report['environment']['packages'],
            'torch_build': report['environment']['torch_build'],
            'python': report['environment']['python']}


def _validate_data(options, jobs):
    path = Path(options['data'])
    data = load(path)
    rows = data.get('rows', [])
    if len(rows) != 4000 or data.get('count') != 4000 or len({row['id'] for row in rows}) != 4000:
        raise ValueError('Prepared workload must contain exactly 4000 distinct request IDs')
    if Path(data['model_path']).resolve() != Path(options['model']):
        raise ValueError('Prepared tokenizer model differs from the target model')
    if data['context'] != options['context'] or data['max_new_tokens'] != options['max_new_tokens']:
        raise ValueError('Prepared context/output cap differs from the run')
    if data['inputs_sha256'] != digest([row['input_ids'] for row in rows]):
        raise ValueError('Prepared input token digest mismatch')
    for row in rows:
        ids = row['input_ids']
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError('Prepared input IDs must be nonempty nonnegative integers')
        if len(ids) + options['max_new_tokens'] + 16 > options['context']:
            raise ValueError('Prepared prompt/output cap exceeds context')
    for name, expected in data['tokenizer_files'].items():
        if Path(name).name != name or sha(Path(options['model']) / name) != expected:
            raise ValueError('Tokenizer metadata changed: ' + name)
    if data['raw_sha256'] != sha(common.ROOT / 'data/redhat8_4000.jsonl'):
        raise ValueError('Prepared inputs are not tied to the bundled raw dataset')
    if any(job['stage'] in ('timing', 'kernel') for job in jobs):
        selection = data.get('steady_selection', {})
        if (selection.get('signal') != 'target_ar_natural_output_length'
                or not selection.get('calibration_sha256') or selection.get('minimum', 0) < 1):
            raise ValueError('Formal steady runs require an AR-calibrated steady_selection workload')
    return data


def build_plan(options, runs=None):
    """Build a deterministic plan after a successful doctor report exists."""
    options = _options(options)
    proposed = common.plan_runs(options) if runs is None else runs
    jobs, names = [], set()
    for item in proposed:
        job = {key: value for key, value in item.items() if key != 'run_id'}
        name, stage = job['name'], job['stage']
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', name) or name in names:
            raise ValueError('Duplicate or unsafe run name: ' + str(name))
        if stage not in ('smoke', 'timing', 'kernel', 'calibrate') or job['bs'] not in common.BATCHES:
            raise ValueError('Unsupported run stage or batch size')
        if stage != 'calibrate' and job['method'] not in common.METHODS:
            raise ValueError('Unsupported speculative method')
        names.add(name)
        job['run_id'] = digest({'options': options, 'job': job})[:24]
        jobs.append(job)
    if not jobs or (any(job['stage'] == 'calibrate' for job in jobs) and len(jobs) != 1):
        raise ValueError('Calibration must be a separate one-run plan')
    _validate_data(options, jobs)
    report = load(options['preflight'])
    identity = _preflight_identity(report, options)
    result = {'schema_version': 1, 'options': options, 'runs': jobs,
              'prepared_sha256': sha(options['data']), 'preflight_identity': identity,
              'runtime_sha256': common.runtime_hash(options['runtime']), 'code_manifest': _code_manifest()}
    validate_frozen_inputs(result)
    return result


def validate_frozen_inputs(plan):
    """Revalidate sources and CPU model identities before and after every run."""
    from tools.doctor import audit_model
    options, identity = plan['options'], plan['preflight_identity']
    if sha(options['data']) != plan['prepared_sha256']:
        raise RuntimeError('Prepared workload changed after plan creation')
    if (Path(options['output']) / 'plan.json').exists():
        if sha(Path(options['output']) / 'input_snapshot.json') != plan['prepared_sha256']:
            raise RuntimeError('Immutable result input snapshot changed')
    if common.runtime_hash(options['runtime']) != plan['runtime_sha256']:
        raise RuntimeError('Profile runtime changed after plan creation')
    if sha(identity['runtime']['receipt_path']) != identity['runtime']['receipt_sha256']:
        raise RuntimeError('Runtime installation receipt changed')
    if _code_manifest() != plan['code_manifest']:
        raise RuntimeError('Runner/profiler source changed after plan creation')
    for key, role in (('model', 'target'), ('draft', 'draft')):
        if _model_identity(audit_model(options[key], role)) != identity[key]:
            raise RuntimeError(key + ' metadata, weight headers, size or mtime changed')


@contextlib.contextmanager
def _lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another coordinator owns lock: ' + str(path)) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _wait_idle(options, wait_free):
    announced = False
    while True:
        state = hardware.snapshot(options['gpus'])
        if not state['processes']:
            return state
        if not wait_free:
            raise RuntimeError('Selected GPUs are occupied: ' + repr(state['processes']))
        if not announced:
            print('WAITING_FOR_IDLE_GPUS', ','.join(options['gpus']), flush=True)
            announced = True
        time.sleep(min(options['poll_seconds'], 30))


def _group_members(group):
    """Find surviving non-zombie members even if they have not allocated CUDA."""
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
            if int(fields[2]) == group and fields[0] != 'Z':
                result.append(int(path.name))
        except (OSError, IndexError, ValueError):
            continue
    return result


def _monitor(child, options, monitor_file):
    started, exited_at = time.monotonic(), None
    with Path(monitor_file).open('a', encoding='utf-8') as handle:
        while True:
            state = hardware.snapshot(options['gpus'], own_group=child.pid)
            handle.write(json.dumps(state, separators=(',', ':')) + '\n')
            handle.flush()
            if state['outsiders']:
                raise RuntimeError('Foreign GPU process detected; stopping only this run: ' + repr(state['outsiders']))
            code = child.poll()
            now = time.monotonic()
            if code is not None:
                if code != 0:
                    raise RuntimeError('Worker exited with status ' + str(code))
                members = _group_members(child.pid)
                if not members and not state['processes']:
                    return
                exited_at = now if exited_at is None else exited_at
                if now - exited_at >= options['shutdown_grace_seconds']:
                    raise RuntimeError('Worker exited but owned descendants remain: ' + repr(members))
            elif now - started >= options['timeout_seconds']:
                raise TimeoutError('Worker exceeded timeout_seconds=' + str(options['timeout_seconds']))
            time.sleep(min(options['poll_seconds'], 30))


def preflight(options, wait_free=False, log_path=None):
    """Run a fresh doctor without modifying the machine's package environment."""
    options = _options(options)
    _wait_idle(options, wait_free)
    log_path = Path(log_path or Path(options['output']) / 'preflight.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-B', str(common.ROOT / 'tools/doctor.py'),
               '--model', options['model'], '--draft', options['draft'],
               '--gpus', ','.join(options['gpus']), '--runtime', options['runtime'],
               '--output', options['preflight']]
    child = None
    try:
        with log_path.open('xb') as handle:
            child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                     cwd=common.ROOT, start_new_session=True)
            child.wait(timeout=options['preflight_timeout_seconds'])
        if child.returncode != 0:
            raise RuntimeError('Fresh doctor failed; inspect ' + str(log_path))
        report = load(options['preflight'])
        _preflight_identity(report, options)
        return report
    except BaseException:
        hardware.stop(child)
        raise


def _artifact_manifest(folder):
    ignored = {'completion_audit.json', 'monitor.jsonl', 'worker.log', 'failure.json'}
    result = {}
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():
            raise RuntimeError('Run artifact symlinks are not allowed: ' + str(path))
        relative = path.relative_to(folder).as_posix()
        if path.is_file() and relative not in ignored and relative != 'trace/control.bin':
            result[relative] = sha(path)
    return result


def _semantic_completion(folder, job, plan):
    complete = load(folder / 'complete.json')
    if complete.get('job') != job or not isinstance(complete.get('completed_at'), (int, float)):
        raise RuntimeError('Worker completion record does not match this job')
    if load(folder / 'launch.json').get('plan_sha256') != digest(plan):
        raise RuntimeError('Run launch record does not match the immutable plan')
    config = load(folder / 'config.json')
    if config.get('job') != job:
        raise RuntimeError('Worker configuration job mismatch')
    options = plan['options']
    source_rows = load(options['data'])['rows']
    if job['stage'] == 'calibrate':
        unique = {}
        for row in source_rows:
            unique.setdefault(row['prompt_sha256'], row)
        source_rows = list(unique.values())
    elif job['stage'] == 'smoke':
        selected = sorted(range(len(source_rows)), key=lambda i: len(source_rows[i]['input_ids']), reverse=True)[:8]
        selected += [i for i in range(len(source_rows)) if i not in selected][:24]
        source_rows = [source_rows[i] for i in selected]
    cap = min(128, options['max_new_tokens']) if job['stage'] == 'smoke' else options['max_new_tokens']
    if (config.get('input_sha256') != digest([row['input_ids'] for row in source_rows])
            or config.get('request_count') != len(source_rows)
            or config.get('sampling') != dict(temperature=0, max_new_tokens=cap, ignore_eos=False)
            or Path(config.get('source', '')).resolve() != Path(options['runtime']) / '__init__.py'):
        raise RuntimeError('Worker input, source, request count or sampling identity differs from plan')
    expected = common.engine_config(options, 'chain7' if job['stage'] == 'calibrate' else job['method'], job['bs'])
    if job['stage'] == 'calibrate':
        expected = {key: value for key, value in expected.items() if not key.startswith('speculative_')}
        expected.update(max_running_requests=64, cuda_graph_bs_decode=[1,2,4,8,16,24,32,48,64])
    if config.get('config') != expected:
        raise RuntimeError('Worker engine settings differ from the frozen plan')
    if complete.get('request_count') != config.get('request_count') or not complete.get('request_count'):
        raise RuntimeError('Worker completion request accounting mismatch')
    if job['stage'] == 'calibrate':
        calibration = load(folder / 'calibration.json')
        if (calibration.get('selection_signal') != 'target_ar_natural_output_length'
                or calibration.get('prepared_sha256') != plan['prepared_sha256']
                or len(calibration.get('rows', [])) != complete['request_count']
                or [row['prompt_sha256'] for row in calibration['rows']] != [row['prompt_sha256'] for row in source_rows]):
            raise RuntimeError('Calibration output identity/accounting failed')
        return
    outputs = load(folder / 'outputs_compact.json')
    whole = load(folder / 'whole_workload.json')
    if len(outputs) != complete['request_count'] or whole['count'] != len(outputs):
        raise RuntimeError('Completed output count mismatch')
    if [row['id'] for row in outputs] != [row['id'] for row in source_rows]:
        raise RuntimeError('Completed output request IDs/order mismatch')
    if whole['output_tokens'] != sum(row['output_tokens'] for row in outputs):
        raise RuntimeError('Completed token count mismatch')
    for rank in range(8):
        directory = folder / 'trace' / f'rank_{rank:03d}'
        meta = load(directory / 'metadata.json')
        if (meta.get('schema_version') != 3 or not meta.get('audit_pass') or meta.get('rank') != rank or meta.get('world_size') != 8
                or meta.get('run_id') != job['run_id'] or meta.get('expected_bs') != job['bs']):
            raise RuntimeError('Rank trace audit/identity mismatch on rank ' + str(rank))
        if bool(meta.get('diagnostic_only')) != (job['stage'] == 'kernel'):
            raise RuntimeError('Kernel diagnostics and timing outputs must not be mixed')
        rounds = load(directory / 'rounds.json')
        if len(rounds) != meta['rows']:
            raise RuntimeError('Rank row count differs from metadata')
        if job['stage'] == 'smoke':
            decode = [row for row in rounds if row['phase'] == 'puredecode']
            if not decode or not all(row.get(key) is True for row in decode
                                     for key in ('target_graph', 'draft_graph', 'draft_extend_graph')):
                raise RuntimeError('Smoke graph evidence is incomplete')
        if job['stage'] == 'kernel':
            status = load(directory / 'kernel_profile/status.json')
            if status.get('status') != 'complete':
                raise RuntimeError('Incomplete kernel capture on rank ' + str(rank))
    if job['stage'] == 'timing':
        analysis = load(folder / 'analysis.json')
        if (not analysis.get('audit_pass') or analysis.get('expected_bs') != job['bs']
                or analysis.get('expected_world_size') != 8 or analysis.get('method') != job['method']
                or analysis.get('run_id') != job['run_id']):
            raise RuntimeError('Steady timing analysis did not pass or has the wrong identity')
    if job['stage'] == 'kernel':
        from profiler.kernel_audit import audit_captures
        audit_captures(folder/'trace',job['bs'])


def _completed(root, job, plan, seal=False):
    folder = root / 'runs' / job['name']
    if not (folder / 'complete.json').is_file() or (folder / 'failure.json').exists():
        return False
    _semantic_completion(folder, job, plan)
    manifest = _artifact_manifest(folder)
    audit_path = folder / 'completion_audit.json'
    if audit_path.exists():
        audit = load(audit_path)
        if (not audit.get('audit_pass') or audit.get('plan_sha256') != digest(plan)
                or audit.get('job') != job or audit.get('artifacts') != manifest):
            raise RuntimeError('Completed artifacts changed: ' + job['name'])
    elif seal:
        save(audit_path, dict(audit_pass=True, plan_sha256=digest(plan), job=job,
                              artifacts=manifest, audited_unix=time.time()))
    return True


def _prerequisites(root, job, plan):
    stages = {'timing': ('smoke',), 'kernel': ('smoke', 'timing')}.get(job['stage'], ())
    for stage in stages:
        required = [item for item in plan['runs'] if item['bs'] == job['bs'] and item['stage'] == stage]
        if not required:
            raise RuntimeError(f'Missing {stage} prerequisite jobs for BS{job["bs"]}')
        for earlier in required:
            if not _completed(root, earlier, plan):
                raise RuntimeError('Prerequisite must complete successfully: ' + earlier['name'])


def _archive_failed(root, folder):
    if folder.parent.resolve() != (root / 'runs').resolve() or folder.is_symlink():
        raise RuntimeError('Refusing to archive an unexpected run path')
    destination = root / 'failures' / (folder.name + '.' + str(time.time_ns()) + '.' + uuid.uuid4().hex[:8])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    folder.rename(destination)
    print('ARCHIVED_FAILED_RUN', str(destination), flush=True)
    return destination


def _ensure_input_snapshot(root, plan):
    snapshot = root / 'input_snapshot.json'
    if snapshot.exists():
        if snapshot.is_symlink() or sha(snapshot) != plan['prepared_sha256']:
            raise RuntimeError('Existing input snapshot differs; refusing to replace it')
        return
    if (root / 'plan.json').exists():
        raise RuntimeError('An existing plan is missing its immutable input snapshot')
    with Path(plan['options']['data']).open('rb') as source, snapshot.open('xb') as target:
        shutil.copyfileobj(source, target)
    if sha(snapshot) != plan['prepared_sha256']:
        raise RuntimeError('Prepared input changed while snapshotting; preserve failure and inspect')


def _execute(root, job, plan, resume, wait_free):
    options, folder = plan['options'], root / 'runs' / job['name']
    validate_frozen_inputs(plan)
    if folder.exists():
        if _completed(root, job, plan, seal=True):
            print('SKIP_COMPLETED', job['name'], flush=True)
            return 'skipped'
        if not resume:
            raise FileExistsError('Incomplete run exists; explicit --resume is required: ' + str(folder))
        _archive_failed(root, folder)
    _prerequisites(root, job, plan)
    idle = _wait_idle(options, wait_free)
    expected_uuids = [item['uuid'] for item in plan['preflight_identity']['devices']]
    if [item['uuid'] for item in idle['devices']] != expected_uuids:
        raise RuntimeError('Physical GPUs changed after preflight')
    folder.mkdir(parents=True, exist_ok=False)
    trace = folder / 'trace'
    trace.mkdir()
    with (trace / 'control.bin').open('xb') as handle:
        handle.write(b'\0' * 8)
    env = common.environment(options, trace, job)
    if job['stage'] == 'calibrate':
        env = {key: value for key, value in env.items() if not key.startswith('PROFILE_')}
    command = [sys.executable, '-B', '-m', 'llama_profile.worker', str(root), job['name']]
    save(folder / 'launch.json', dict(plan_sha256=digest(plan), job=job, command=command,
                                     started_unix=time.time(), gpu_indices=options['gpus'],
                                     gpu_uuids=expected_uuids, environment_keys=sorted(key for key in env
                                     if key.startswith(('PROFILE_', 'STEADY_', 'EAGLE_ACTIVE_')))))
    child = None
    try:
        print('RUN_START', job['name'], flush=True)
        with (folder / 'worker.log').open('xb') as handle:
            child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                     cwd=common.ROOT, env=env, start_new_session=True)
            _monitor(child, options, folder / 'monitor.jsonl')
        validate_frozen_inputs(plan)
        if not _completed(root, job, plan, seal=True):
            raise RuntimeError('Worker exited without an audited completion record')
        print('RUN_COMPLETE', job['name'], flush=True)
        return 'completed'
    except BaseException as exc:
        hardware.stop(child)
        save(folder / 'failure.json', dict(status='failed', error_type=type(exc).__name__,
                                          error=str(exc), time=time.time(), job=job,
                                          child_pid=child.pid if child is not None else None,
                                          plan_sha256=digest(plan), preserved=True))
        raise


def launch_plan(root, plan, stage='all', resume=False, wait_free=False):
    """Accept raw {options,runs}; take locks, refresh doctor, freeze, then run."""
    if sys.platform != 'linux':
        raise RuntimeError('GPU experiment coordination requires Linux')
    options = _options(plan['options'])
    root = Path(root).expanduser().resolve()
    if root != Path(options['output']):
        raise ValueError('Plan output and coordinator root differ')
    if stage not in ('all', 'smoke', 'timing', 'kernel', 'calibrate'):
        raise ValueError('Unsupported stage')
    with contextlib.ExitStack() as stack:
        stack.enter_context(_lock(root / '.coordinator.lock'))
        if not resume and any(path.name != '.coordinator.lock' for path in root.iterdir()):
            raise FileExistsError('Output root already contains work; use explicit --resume: ' + str(root))
        lock_root = Path(tempfile.gettempdir()) / ('llama-profile-locks-' + str(os.getuid()))
        for gpu in sorted(options['gpus'], key=int):
            stack.enter_context(_lock(lock_root / ('gpu-' + gpu + '.lock')))
        stamp = str(time.time_ns()) + '-' + uuid.uuid4().hex[:8]
        try:
            report = preflight(options, wait_free, root / 'preflights' / (stamp + '.log'))
            save(root / 'preflights' / (stamp + '.json'), report)
            frozen = build_plan(options, plan['runs'])
            if 'preflight_identity' in plan and plan != frozen:
                raise RuntimeError('Supplied frozen plan differs from current audited inputs')
            _ensure_input_snapshot(root, frozen)
            if (root / 'plan.json').exists():
                if load(root / 'plan.json') != frozen:
                    raise RuntimeError('Resume options, inputs, sources, models, or hardware differ from immutable plan')
            else:
                save(root / 'plan.json', frozen)
            selected = [job for job in frozen['runs'] if stage == 'all' or job['stage'] == stage]
            if not selected:
                raise ValueError('Selected stage has no jobs in this plan')
            ordering = {'calibrate': 0, 'smoke': 1, 'timing': 2, 'kernel': 3}
            selected.sort(key=lambda job: ordering[job['stage']])  # stable: retain method/repeat rotation
            counts = {'completed': 0, 'skipped': 0}
            for job in selected:
                counts[_execute(root, job, frozen, resume, wait_free)] += 1
            result = dict(audit_pass=True, stage=stage, plan_sha256=digest(frozen), **counts)
            save(root / ('stage_' + stage + '_complete.json'), result)
            return result
        except BaseException as exc:
            save(root / 'failures' / ('coordinator-' + stamp + '.json'),
                 dict(status='failed', error_type=type(exc).__name__, error=str(exc),
                      stage=stage, time=time.time(), explicit_resume=bool(resume)))
            raise


def run(options, stage='all', resume=False, wait_free=False):
    return launch_plan(options['output'], {'options': options, 'runs': common.plan_runs(options)},
                       stage=stage, resume=resume, wait_free=wait_free)

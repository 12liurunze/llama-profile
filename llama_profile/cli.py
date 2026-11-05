"""Commands intentionally separate input selection, GPU execution and reporting."""
import argparse
import json
import sys
import tarfile
from pathlib import Path
from .common import ROOT, load, plan_runs, sha, validate_options


def execution_arguments(parser):
    parser.add_argument('--model', type=Path, required=True, help='Existing local target checkpoint directory')
    parser.add_argument('--draft', type=Path, required=True, help='Existing matching EAGLE3 checkpoint directory')
    parser.add_argument('--data', type=Path, required=True, help='Prepared tokenized input JSON')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, default=ROOT/'.runtime/profile/sglang')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--batches', type=int, nargs='+', default=[128,256,512])
    parser.add_argument('--tp', type=int, default=8)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--memory-fraction', type=float, default=.8)
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--chunked-prefill', type=int, default=4096)
    parser.add_argument('--min-steps', type=int, default=10)
    parser.add_argument('--protocol', choices=['queued','cohort'], default='queued')
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    parser.add_argument('--wait-free', action='store_true', help='Wait for all eight selected GPUs to become empty')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print a plan; do not import torch or contact a GPU')


def options(args):
    result = {k:v for k,v in vars(args).items() if k not in
              ('command','stage','wait_free','resume','dry_run','steady_output','minimum_output')}
    for k in ('model','draft','data','output','runtime'):
        result[k] = str(Path(result[k]).expanduser().resolve())
    result['gpus'] = [g.strip() for g in result['gpus'].split(',')]
    result['preflight'] = str(Path(result['output'])/'preflight.json')
    validate_options(result)
    return result


def pack(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if destination.is_relative_to(root):
        raise ValueError('Archive must be outside result directory')
    if not (root/'plan.json').is_file():
        raise ValueError('Not an experiment result directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination,'w:gz') as archive:
        # Include raw traces so the recipient can independently recompute tables.
        for p in sorted(root.rglob('*')):
            if p.is_file() and not p.is_symlink() and p.name not in ('control.bin','.driver.lock'):
                archive.add(p, arcname=str(Path(root.name)/p.relative_to(root)), recursive=False)
    return dict(archive=str(destination), bytes=destination.stat().st_size, sha256=sha(destination))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Llama 70B TP8 steady-state speculative decoding benchmark')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Tokenize bundled 4,000 requests locally; no GPU')
    prepare.add_argument('--model', type=Path, required=True)
    prepare.add_argument('--output', type=Path, default=ROOT/'data/prepared4000.json')
    prepare.add_argument('--context', type=int, default=4096)
    prepare.add_argument('--max-new-tokens', type=int, default=1024)
    calibrate = commands.add_parser('calibrate', help='Target-only natural length calibration, then fix the steady workload')
    execution_arguments(calibrate)
    calibrate.add_argument('--steady-output', type=Path, default=ROOT/'data/steady4000.json')
    calibrate.add_argument('--minimum-output', type=int, default=256)
    select = commands.add_parser('select', help='Rebuild a length-selected workload from an existing AR calibration; no GPU')
    select.add_argument('--prepared',type=Path,required=True)
    select.add_argument('--calibration',type=Path,required=True)
    select.add_argument('--output',type=Path,required=True)
    select.add_argument('--minimum-output',type=int,default=256)
    run = commands.add_parser('run', help='One TP8 coordinator; no torchrun')
    execution_arguments(run)
    run.add_argument('--stage',choices=['all','smoke','timing','kernel'],default='all')
    report = commands.add_parser('report',help='Audit and aggregate completed measurements')
    report.add_argument('--output',type=Path,required=True)
    report.add_argument('--allow-partial',action='store_true')
    package = commands.add_parser('pack',help='Package raw evidence and generated tables')
    package.add_argument('--output',type=Path,required=True)
    package.add_argument('--archive',type=Path,required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            from .data import prepare as prepare_data
            result = prepare_data(args.model,args.output,args.context,args.max_new_tokens)
        elif args.command == 'select':
            from .workload import build_steady
            result = build_steady(args.prepared,args.calibration,args.output,args.minimum_output)
        elif args.command in ('calibrate','run'):
            o = options(args)
            runs = [dict(name='calibration',stage='calibrate',method='ar',bs=128,repeat=0)] if args.command=='calibrate' else plan_runs(o)
            if args.dry_run:
                result = dict(options=o, runs=runs, launches=len(runs), gpu_touched=False)
            else:
                from .runner import launch_plan
                plan = dict(options=o,runs=runs)
                result = launch_plan(Path(o['output']),plan,
                    stage='all' if args.command=='calibrate' else args.stage,
                    resume=args.resume,wait_free=args.wait_free)
                if args.command=='calibrate':
                    from .workload import build_steady
                    if args.steady_output.exists():
                        old = load(args.steady_output)
                        selection = old.get('steady_selection',{})
                        cpath = Path(o['output'])/'runs/calibration/calibration.json'
                        if not args.resume or selection.get('calibration_sha256') != sha(cpath) or selection.get('minimum') != args.minimum_output:
                            raise FileExistsError('Existing steady workload differs; use a new --steady-output')
                        result = selection
                    else:
                        result = build_steady(args.data,Path(o['output'])/'runs/calibration/calibration.json',args.steady_output,args.minimum_output)
        elif args.command == 'report':
            from .report import report as write_report
            result = write_report(args.output,args.allow_partial)
            result = {k:result[k] for k in ('complete','timing_complete','plan_complete','full_matrix_complete') if k in result}
        else:
            result = pack(args.output,args.archive)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        print(type(exc).__name__ + ': ' + str(exc),file=sys.stderr)
        return 2

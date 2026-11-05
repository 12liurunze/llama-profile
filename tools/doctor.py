"""Read-only, fail-closed preflight for local Llama 70B + EAGLE3 on 8 A100 80GB.

Reads safetensors HEADERS, not tensor payloads. Never loads model weights, runs
collectives, installs packages, or allocates CUDA tensors. Busy GPUs fail the
audit; runtime import smoke tests are skipped in that case.
"""
import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT_CONFIG_REFERENCE = ('https://huggingface.co/lmsys/'
                          'SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge/blob/main/config.json')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.install_runtime import file_manifest, sha256, verify_archive


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _header(path):
    """Validate tensor extents and return header identities without payload I/O."""
    stat = path.stat()
    with path.open('rb') as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError('Truncated safetensors length prefix')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= 128 * 1024 * 1024 or 8 + length > stat.st_size:
            raise ValueError('Invalid safetensors header length')
        raw = handle.read(length)
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON header key: ' + key)
            result[key] = value
        return result
    header = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError('Safetensors header is not an object')
    widths = {'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
              'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2, 'I32': 4,
              'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8}
    tensors, intervals = {}, []
    for key, value in header.items():
        if key == '__metadata__':
            continue
        shape, dtype, offsets = value.get('shape'), value.get('dtype'), value.get('data_offsets')
        if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                or dtype not in widths or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets)):
            raise ValueError('Unsupported/invalid tensor header: ' + key)
        begin, end = offsets
        if begin < 0 or end < begin or end > stat.st_size - 8 - length:
            raise ValueError('Tensor extent exceeds file: ' + key)
        if end - begin != math.prod(shape) * widths[dtype]:
            raise ValueError('Tensor extent/shape/dtype mismatch: ' + key)
        if begin != end:
            intervals.append((begin, end))
        tensors[key] = {'shape': shape, 'dtype': dtype}
    cursor = 0
    for begin, end in sorted(intervals):
        if begin != cursor:
            raise ValueError('Overlapping or non-contiguous tensor payload')
        cursor = end
    if cursor != stat.st_size - 8 - length:
        raise ValueError('Tensor extents do not cover file payload')
    return tensors, {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                     'header_sha256': hashlib.sha256(prefix + raw).hexdigest(),
                     'header_bytes': 8 + length, 'tensor_count': len(tensors)}


def audit_model(path, role='target'):
    """Public CPU-only model audit reused by the runner; errors are returned."""
    if role not in ('target', 'draft'):
        raise ValueError('role must be target or draft')
    root = Path(path).expanduser().resolve()
    result = {'audit_pass': False, 'errors': [], 'role': role, 'path': str(root),
              'config': {}, 'config_sha256': None, 'metadata_files': {},
              'weight_files': [], 'tensor_count': 0,
              'weight_identity': 'Header SHA256 plus file size/mtime; tensor payloads not hashed'}
    errors = result['errors']
    try:
        if not root.is_dir():
            raise ValueError('Model must be an existing local checkpoint directory')
        cfg = _load(root / 'config.json')
        result['config'] = cfg
        result['config_sha256'] = sha256(root / 'config.json')
        expected_layers = 80 if role == 'target' else 1
        dimensions = {'hidden_size': 8192, 'num_hidden_layers': expected_layers,
                      'num_attention_heads': 64, 'num_key_value_heads': 8,
                      'intermediate_size': 28672, 'vocab_size': 128256}
        if role == 'draft':
            dimensions['draft_vocab_size'] = 32000
            result['config_reference'] = DRAFT_CONFIG_REFERENCE
        result['expected_dimensions'] = dimensions
        for key, expected in dimensions.items():
            if cfg.get(key) != expected:
                errors.append(f'{role} {key}: expected {expected}, found {cfg.get(key)!r}')
        # Older target configs omit head_dim; Llama derives it as hidden/heads.
        heads, kv_heads = cfg.get('num_attention_heads'), cfg.get('num_key_value_heads')
        derived_head_dim = (cfg.get('hidden_size', 0) // heads
                            if type(heads) is int and heads > 0 else None)
        head_dim = cfg.get('head_dim', derived_head_dim)
        if head_dim != 128 or derived_head_dim != 128:
            errors.append(f'{role} head_dim must be 128 and match hidden_size/num_attention_heads')
        if any(type(value) is not int or value <= 0 or value % 8 for value in (heads, kv_heads)):
            errors.append(f'{role} attention and KV heads must both divide TP8 without replication')
        result['tensor_parallel'] = {'world_size': 8, 'head_dim': head_dim,
                                     'query_heads_per_rank': heads // 8 if type(heads) is int else None,
                                     'kv_heads_per_rank': kv_heads // 8 if type(kv_heads) is int else None}
        if cfg.get('quantization_config'):
            errors.append('Quantized checkpoints are outside this BF16 protocol')
        if cfg.get('target_hidden_size', 8192) != 8192:
            errors.append('Draft target_hidden_size must match target hidden_size 8192')
        architecture = ' '.join(cfg.get('architectures', [])).lower()
        if role == 'draft' and cfg.get('architectures') != ['LlamaForCausalLMEagle3']:
            errors.append('Draft architecture must be LlamaForCausalLMEagle3')
        if role == 'draft' and cfg.get('tie_word_embeddings', False):
            errors.append('EAGLE3 reduced vocabulary requires an independent draft LM head')
        if role == 'target' and (cfg.get('model_type') != 'llama' or 'eagle' in architecture):
            errors.append('Target must be a Llama causal model')
        for path_meta in sorted(root.iterdir()):
            if path_meta.is_file() and path_meta.suffix in ('.json', '.model', '.tiktoken'):
                stat = path_meta.stat()
                result['metadata_files'][path_meta.name] = {
                    'sha256': sha256(path_meta), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if role == 'target' and not any((root / name).is_file() for name in ('tokenizer.json', 'tokenizer.model')):
            errors.append('Local target tokenizer is missing; downloads during timing are forbidden')
        if role == 'target' and not (root / 'tokenizer_config.json').is_file():
            errors.append('Local target tokenizer_config.json is missing')
        indices = sorted(root.glob('*.safetensors.index.json'))
        if len(indices) > 1:
            raise ValueError('Ambiguous multiple safetensors indices')
        weight_map = _load(indices[0]).get('weight_map', {}) if indices else None
        names = sorted(set(weight_map.values())) if weight_map is not None else sorted(p.name for p in root.glob('*.safetensors'))
        if not names:
            raise ValueError('No safetensors weights/index; pickle .bin files are not loaded by preflight')
        tensors, tensor_files = {}, {}
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name:
                raise ValueError('Unsafe weight-index path: ' + name)
            found, metadata = _header(root / relative)
            duplicate = set(tensors) & set(found)
            if duplicate:
                raise ValueError('Duplicate tensor names across shards: ' + repr(sorted(duplicate)[:5]))
            tensors.update(found)
            tensor_files.update({key: name for key in found})
            result['weight_files'].append(dict(path=name, **metadata))
        if weight_map is not None and tensor_files != weight_map:
            errors.append('Safetensors index mapping does not exactly match shard headers')
        result['tensor_count'] = len(tensors)
        def shape(key, expected, dtype=None):
            if key not in tensors:
                errors.append('Missing required tensor: ' + key)
            elif tensors[key]['shape'] != expected:
                errors.append(f'{key}: expected shape {expected}, found {tensors[key]["shape"]}')
            elif dtype is not None and tensors[key]['dtype'] != dtype:
                errors.append(f'{key}: expected dtype {dtype}, found {tensors[key]["dtype"]}')
        if role == 'target' and cfg.get('hidden_size') == 8192:
            hidden, vocab = 8192, cfg.get('vocab_size')
            heads, kv_heads = cfg.get('num_attention_heads'), cfg.get('num_key_value_heads')
            inter = cfg.get('intermediate_size')
            if (heads != 64 or kv_heads != 8 or inter != 28672 or vocab != 128256):
                errors.append('Target is not the expected Llama-3.3-70B dimensions (64/8 heads, intermediate28672, vocab128256)')
            else:
                shape('model.embed_tokens.weight', [vocab, hidden])
                shape('model.norm.weight', [hidden])
                if not cfg.get('tie_word_embeddings', False):
                    shape('lm_head.weight', [vocab, hidden])
                for layer in range(80):
                    prefix = f'model.layers.{layer}.'
                    for part, dimensions in {
                        'self_attn.q_proj.weight': [8192, hidden],
                        'self_attn.k_proj.weight': [1024, hidden],
                        'self_attn.v_proj.weight': [1024, hidden],
                        'self_attn.o_proj.weight': [hidden, 8192],
                        'mlp.gate_proj.weight': [inter, hidden],
                        'mlp.up_proj.weight': [inter, hidden],
                        'mlp.down_proj.weight': [hidden, inter],
                        'input_layernorm.weight': [hidden],
                        'post_attention_layernorm.weight': [hidden],
                    }.items():
                        shape(prefix + part, dimensions)
        if role == 'draft':
            # Match the bundled loader's legacy midlayer -> layers.0 mapping.
            # Arbitrary trainer prefixes cannot satisfy the serving contract:
            # the runtime silently skips unrecognized parameter names.
            def draft_shape(name, expected, dtype=None):
                aliases = {name}
                if name != 'lm_head.weight':
                    aliases.add('model.' + name)
                if name.startswith('layers.0.'):
                    legacy = name.replace('layers.0.', 'midlayer.', 1)
                    aliases.update((legacy, 'model.' + legacy))
                matches = sorted(aliases & tensors.keys())
                if len(matches) != 1:
                    errors.append(f'EAGLE3 requires exactly one serving tensor {name}; found {matches}')
                    return
                shape(matches[0], expected, dtype)

            draft_shape('lm_head.weight', [32000, 8192], 'BF16')
            draft_shape('fc.weight', [8192, 24576], 'BF16')
            draft_shape('norm.weight', [8192])
            for part, expected in {
                'self_attn.q_proj.weight': [8192, 16384],
                'self_attn.k_proj.weight': [1024, 16384],
                'self_attn.v_proj.weight': [1024, 16384],
                'self_attn.o_proj.weight': [8192, 8192],
                'mlp.gate_proj.weight': [28672, 8192],
                'mlp.up_proj.weight': [28672, 8192],
                'mlp.down_proj.weight': [8192, 28672],
                'input_layernorm.weight': [8192],
                'hidden_norm.weight': [8192],
                'post_attention_layernorm.weight': [8192],
            }.items():
                draft_shape('layers.0.' + part, expected,
                            'BF16' if len(expected) == 2 else None)
            draft_shape('d2t', [32000], 'I64')
            draft_shape('t2d', [128256], 'BOOL')
            result['hot_token_mapping'] = {
                'draft_vocab_size': 32000, 'target_vocab_size': 128256,
                'd2t': {'shape': [32000], 'dtype': 'I64',
                        'semantics': 'target_id = draft_id + d2t[draft_id]'},
                't2d': {'shape': [128256], 'dtype': 'BOOL'},
                'values_verified': False,
                'limitation': 'Header-only audit cannot verify mapping range, uniqueness, or inverse consistency',
            }
            if any(int(match.group(1)) != 0 for key in tensors
                   for match in [re.search(r'(?:^|\.)layers\.(\d+)\.', key)] if match):
                errors.append('Draft weight headers contain more than one layer')
    except Exception as exc:
        errors.append(type(exc).__name__ + ': ' + str(exc))
    result['audit_pass'] = not errors
    return result


def audit_runtime(runtime, vendor=ROOT / 'vendor'):
    runtime = Path(runtime).resolve()
    receipt_path = runtime.parent.parent / 'install_receipt.json'
    result = {'audit_pass': False, 'errors': [], 'path': str(runtime),
              'receipt_path': str(receipt_path)}
    try:
        verified = verify_archive(vendor)
        receipt = _load(receipt_path)
        if receipt['archive_sha256'] != verified['archive_sha256']:
            raise ValueError('Installed archive identity differs from vendor')
        if receipt['base_manifest'] != verified['base_manifest']:
            raise ValueError('Receipt baseline differs from vendor manifest')
        if file_manifest(runtime.parent.parent / 'base/sglang') != receipt['base_manifest']:
            raise ValueError('Installed baseline source changed')
        if file_manifest(runtime) != receipt['profile_manifest']:
            raise ValueError('Installed profile source changed')
        if not receipt['instrumentation'].get('ast_equivalent'):
            raise ValueError('Missing instrumentation AST audit')
        result.update(audit_pass=True, receipt_sha256=sha256(receipt_path),
                      archive_sha256=receipt['archive_sha256'],
                      file_count=len(receipt['profile_manifest']))
    except Exception as exc:
        result['errors'].append(type(exc).__name__ + ': ' + str(exc))
    return result


def audit_gpus(gpus, run=subprocess.check_output):
    result = {'audit_pass': False, 'errors': [], 'devices': [], 'processes': [], 'indices': gpus}
    try:
        if len(gpus) != 8 or len(set(gpus)) != 8 or any(not str(g).isdigit() for g in gpus):
            raise ValueError('Exactly eight distinct local GPU indices are required')
        def query(scope, fields):
            text = run(['nvidia-smi', '--query-' + scope + '=' + fields,
                        '--format=csv,noheader,nounits'], text=True, timeout=30)
            return [[value.strip() for value in row] for row in csv.reader(text.splitlines()) if row]
        keys = ['index', 'uuid', 'name', 'memory_mib', 'free_mib', 'driver_version', 'compute_capability', 'mig_mode']
        devices = {row[0]: dict(zip(keys, row)) for row in query(
            'gpu', 'index,uuid,name,memory.total,memory.free,driver_version,compute_cap,mig.mode.current')}
        for gpu in gpus:
            device = devices[gpu]
            result['devices'].append(device)
            if 'A100' not in device['name'] or not 75000 <= float(device['memory_mib']) <= 90000:
                result['errors'].append('GPU ' + gpu + ' must be A100 80GB')
            if device['compute_capability'] != '8.0':
                result['errors'].append('GPU ' + gpu + ' compute capability must be 8.0')
            if device['mig_mode'].lower() != 'disabled':
                result['errors'].append('GPU ' + gpu + ' MIG must be disabled')
            if int(device['driver_version'].split('.')[0]) < 580:
                result['errors'].append('GPU ' + gpu + ' driver is too old for the recorded CUDA13 stack')
        uuids = {device['uuid'] for device in result['devices']}
        for row in query('compute-apps', 'gpu_uuid,pid,used_memory'):
            if row[0] in uuids:
                result['processes'].append({'uuid': row[0], 'pid': int(row[1]), 'memory_mib': row[2]})
        if result['processes']:
            result['errors'].append('Selected GPUs have compute processes; no run is authorized by this preflight')
        result['topology'] = run(['nvidia-smi', 'topo', '-m'], text=True, timeout=30)
        result['audit_pass'] = not result['errors']
    except Exception as exc:
        result['errors'].append(type(exc).__name__ + ': ' + str(exc))
    return result


def audit_environment(runtime, allow_import_smoke, requirements=ROOT / 'requirements.runtime.txt'):
    result = {'audit_pass': False, 'errors': [], 'packages': {}, 'imports': {},
              'python': sys.version, 'platform': platform.platform(), 'gpu_allocations_requested': False}
    try:
        if sys.platform != 'linux' or sys.version_info < (3, 11):
            result['errors'].append('Recorded runtime requires Linux and Python >=3.11')
        for line in Path(requirements).read_text(encoding='utf-8').splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            name, expected = line.strip().split('==')
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual = None
            match = actual is not None and actual.split('+')[0] == expected
            result['packages'][name] = {'expected': expected, 'actual': actual, 'match': match}
            if not match:
                result['errors'].append(f'{name}: expected {expected}, found {actual!r}')
        # CPU-side build queries only. Do not call current_device(), allocate a
        # CUDA tensor, or invoke cudaGetDeviceProperties on an occupied GPU.
        os.environ.setdefault('PYTORCH_NVML_BASED_CUDA_CHECK', '1')
        import torch
        cuda_version = torch.version.cuda
        arch_flags = getattr(torch._C, '_cuda_getArchFlags', lambda: '')() or ''
        nccl = torch.distributed.is_nccl_available()
        result['torch_build'] = {'version': str(torch.__version__), 'path': torch.__file__,
                                 'cuda': cuda_version, 'arch_flags': arch_flags,
                                 'nccl_available': nccl}
        if cuda_version != '13.0':
            result['errors'].append('Expected recorded torch CUDA13.0 build, found ' + str(cuda_version))
        if 'sm_80' not in arch_flags.split():
            result['errors'].append('Torch wheel does not advertise sm_80 support')
        if not nccl:
            result['errors'].append('Torch was built without NCCL support')
        if allow_import_smoke and not result['errors']:
            sys.path.insert(0, str(Path(runtime).resolve().parent))
            for name in ('sglang', 'sglang.srt.entrypoints.engine', 'sglang.srt.models.llama_eagle3',
                         'sglang.kernels.ops.kvcache.cache_move', 'flashinfer', 'sgl_kernel'):
                module = importlib.import_module(name)
                result['imports'][name] = str(getattr(module, '__file__', None))
            if Path(result['imports']['sglang']).resolve() != Path(runtime).resolve() / '__init__.py':
                result['errors'].append('Imported SGLang does not come from the bundled profile runtime')
            result['sglang_reported_version'] = str(sys.modules['sglang'].__version__)
            try:
                result['system_sglang_distribution_version'] = importlib.metadata.version('sglang')
            except importlib.metadata.PackageNotFoundError:
                result['system_sglang_distribution_version'] = None
            result['source_identity_note'] = 'Source identity is path plus manifest, not inherited distribution version'
        else:
            result['import_smoke_skipped'] = 'GPU audit failed/busy or dependency/build mismatch'
            result['errors'].append('Runtime import smoke did not pass; preflight cannot authorize a run')
        result['audit_pass'] = not result['errors']
    except Exception as exc:
        result['errors'].append(type(exc).__name__ + ': ' + str(exc))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--draft', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--runtime', type=Path, default=ROOT / '.runtime/profile/sglang')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/preflight.json')
    args = parser.parse_args(argv)
    gpus = [str(i) for i in range(8)] if args.gpus == '8' else args.gpus.split(',')
    gpus = [gpu.strip() for gpu in gpus]
    report = {'schema_version': 1, 'created_unix': time.time(), 'audit_pass': False,
              'read_only': True, 'model_weights_loaded': False, 'gpu_tensor_allocations_requested': False,
              'model': audit_model(args.model, 'target'), 'draft': audit_model(args.draft, 'draft'),
              'runtime': audit_runtime(args.runtime), 'gpus': audit_gpus(gpus)}
    report['environment'] = audit_environment(args.runtime, report['gpus']['audit_pass'] and report['runtime']['audit_pass'])
    report['audit_pass'] = all(report[key]['audit_pass'] for key in ('model', 'draft', 'runtime', 'gpus', 'environment'))
    report['errors'] = [key + ': ' + error for key in ('model', 'draft', 'runtime', 'gpus', 'environment')
                        for error in report[key]['errors']]
    report['ready_for_formal'] = report['audit_pass']
    report['limitations'] = ['Preflight does not prove BS128/256/512 fits or reaches steady state.',
                             'Model identity uses metadata/header hashes and weight size/mtime, not full payload hashes.',
                             'The runner must recheck source, model identity, and GPU occupancy before each run.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(args.output)
    print(json.dumps({'audit_pass': report['audit_pass'], 'report': str(args.output.resolve()),
                      'errors': report['errors']}, ensure_ascii=False, indent=2))
    return 0 if report['audit_pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

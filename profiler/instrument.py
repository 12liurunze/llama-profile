"""Copy a compatible original SGLang package and inject AST-checked profiling.

Unsupported source layouts fail closed. No original runtime files are edited.
"""
import ast
import hashlib
import shutil
from pathlib import Path


def _inject_import(text):
    tree = ast.parse(text)
    future = [node.end_lineno for node in tree.body
              if isinstance(node, ast.ImportFrom) and node.module == '__future__']
    lines = text.splitlines(keepends=True)
    lines.insert(max(future, default=0), 'from sglang.srt.speculative import stage_hooks as _steady\n')
    return ''.join(lines)


def _decorate(text, signature, decorator):
    assert text.count(signature) == 1, (signature, text.count(signature))
    indent = len(signature) - len(signature.lstrip())
    return text.replace(signature, ' ' * indent + decorator + '\n' + signature, 1)


def _wrap(text, begin, end, name):
    assert text.count(begin) == 1, (name, 'begin', text.count(begin))
    start = text.index(begin)
    stop = text.index(end, start)
    chunk = text[start:stop]
    indent = len(begin) - len(begin.lstrip())
    return (text[:start] + ' ' * indent + f'with _steady.span({name!r}):\n'
            + ''.join('    ' + line if line.strip() else line
                      for line in chunk.splitlines(keepends=True)) + text[stop:])


def _stripped(text):
    class Strip(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == 'sglang.srt.speculative' and any(a.name == 'stage_hooks' for a in node.names):
                return None
            return node

        def visit_FunctionDef(self, node):
            node.decorator_list = [d for d in node.decorator_list if not ast.unparse(d).startswith('_steady.')]
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if ast.unparse(node).startswith('_steady.note('):
                return None
            return self.generic_visit(node)

        def visit_With(self, node):
            self.generic_visit(node)
            if len(node.items) == 1 and ast.unparse(node.items[0].context_expr).startswith('_steady.span('):
                return node.body
            return node

    return ast.dump(Strip().visit(ast.parse(text)), include_attributes=False)


def instrument(src, dst):
    """src/dst are package roots containing srt/ and kernels/; dst must not exist."""
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve() or src.resolve() in dst.resolve().parents:
        raise ValueError('The isolated destination must not be the source or a child of it')
    assert (src / 'srt/speculative/eagle_worker_v2.py').is_file(), src
    worker_source = (src / 'srt/speculative/eagle_worker_v2.py').read_text(encoding='utf-8')
    for prior_hook in ('bs_counter', '_diag.', 'stage_hooks'):
        if prior_hook in worker_source:
            raise ValueError(f'Source already contains profiling hooks ({prior_hook}); use an original runtime')
    if dst.exists():
        raise FileExistsError(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    manifest = {'ast_equivalent': True, 'schema_version': 3, 'files': {}, 'hooks': {}}
    for name in ('stage_hooks.py', 'kernel_probe.py'):
        source_hook = Path(__file__).with_name(name)
        shutil.copy2(source_hook, dst / 'srt/speculative' / name)
        manifest['hooks'][name] = hashlib.sha256(source_hook.read_bytes()).hexdigest()

    def edit(rel, transform):
        path = dst / rel
        original = path.read_text(encoding='utf-8')
        for prior_hook in ('_diag.', 'bs_counter', 'import stage_hooks'):
            if prior_hook in original:
                raise ValueError(f'{rel} already contains profiling hooks ({prior_hook})')
        changed = transform(_inject_import(original))
        assert _stripped(original) == _stripped(changed), ('AST mismatch', rel)
        path.write_text(changed, encoding='utf-8')
        manifest['files'][rel] = {
            'source_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'instrumented_sha256': hashlib.sha256(changed.encode()).hexdigest(),
        }

    def worker(text):
        for signature, decorator in [
            ('    def forward_batch_generation(', '@_steady.record_worker'),
            ('    def draft(self, batch: ScheduleBatch):', "@_steady.phase('draft')"),
            ('    def _draft_extend_for_decode(', "@_steady.phase('draft_extend')"),
        ]:
            text = _decorate(text, signature, decorator)
        needle = '        with canary_outside_ctx:\n'
        assert text.count(needle) == 1, ('draft graph note', text.count(needle))
        text = text.replace(needle,
            '        _steady.note(draft_graph=bool(can_run_decode_cuda_graph),\n'
            '                     draft_frontier_width=self.topk,\n'
            '                     draft_forward_calls=self.speculative_num_steps - 1,\n'
            '                     draft_input_rows=int(draft_input.positions.numel()),\n'
            '                     draft_input_rows_source="prepared_frontier_positions")\n'
            + needle, 1)
        needle = ('            if can_run_decode_cuda_graph:\n'
                  '                draft_logits_output = ')
        assert text.count(needle) == 1, ('extend graph note', text.count(needle))
        text = text.replace(needle,
            '            _steady.note(draft_extend_graph=bool(can_run_decode_cuda_graph))\n'
            + needle, 1)
        return text

    def common(text):
        text = _decorate(text, 'def run_eagle_verify(', "@_steady.phase('verify_total')")
        text = _wrap(text, '    # Batch 1: Target verify\n',
                     '    # Run target verify batch in the main compute stream', 'target_prepare')
        text = _wrap(text, '    forward_batch_output = target_worker.forward_batch_generation(\n',
                     '    # Generate vocab mask', 'target_forward')
        text = _wrap(text, '    # Sample\n',
                     '    next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)', 'accept_and_cache')
        return text

    edit('srt/speculative/eagle_worker_v2.py', worker)
    edit('srt/speculative/eagle_worker_common.py', common)
    edit('kernels/ops/kvcache/cache_move.py', lambda text: _decorate(
        text, 'def copy_all_layer_kv_cache_func(', "@_steady.phase('kv_copy')"))
    edit('srt/managers/scheduler_components/batch_result_processor.py', lambda text: _decorate(
        text, '    def process_batch_result_decode(', '@_steady.record_commit'))
    return manifest

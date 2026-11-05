"""Offline checks against real bundled source; no model/CUDA imports."""
import ast
import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from profiler.instrument import instrument
from tools.install_runtime import verify_archive


def main():
    sources = ('srt/speculative/eagle_worker_v2.py','srt/speculative/eagle_worker_common.py',
               'kernels/ops/kvcache/cache_move.py',
               'srt/managers/scheduler_components/batch_result_processor.py')
    receipt = verify_archive(ROOT/'vendor')
    for parent in ('llama_profile','profiler','tools','tests','scripts'):
        for path in (ROOT/parent).glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    with tempfile.TemporaryDirectory(prefix='llama-release-') as tmp:
        base = Path(tmp)/'base/sglang'
        with tarfile.open(ROOT/'vendor/original_runtime.tar.gz') as tar:
            for relative in sources:
                path = base/relative
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(tar.extractfile('sglang/'+relative).read())
        result = instrument(base,Path(tmp)/'profile/sglang')
        for relative in sources:
            assert hashlib.sha256((base/relative).read_bytes()).hexdigest() == receipt['base_manifest'][relative]
        assert result['ast_equivalent'] and len(result['files']) == 4 and len(result['hooks']) == 2
    print(json.dumps(dict(audit_pass=True, archive_sha256=receipt['archive_sha256'],
        archive_files=receipt['file_count'], instrumented_real_source_files=len(result['files']),
        ast_equivalent=True, gpu_validation=False),indent=2))


if __name__=='__main__':
    main()

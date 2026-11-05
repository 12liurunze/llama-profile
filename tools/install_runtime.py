"""Verify and install the bundled runtime without replacing existing files."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(package):
    result = {}
    for path in sorted(Path(package).rglob('*')):
        if '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo'):
            continue
        if path.is_symlink():
            raise ValueError('Runtime symlink is not allowed: ' + str(path))
        if path.is_file():
            result[path.relative_to(package).as_posix()] = sha256(path)
    return result


def _member_path(member):
    path = PurePosixPath(member.name)
    if (not path.parts or path.parts[0] != 'sglang' or path.is_absolute()
            or '..' in path.parts or '\\' in member.name
            or not (member.isfile() or member.isdir())):
        raise ValueError('Unsafe runtime archive entry: ' + member.name)
    return path


def verify_archive(vendor):
    vendor = Path(vendor)
    archive = vendor / 'original_runtime.tar.gz'
    manifest = json.loads((vendor / 'runtime_manifest.json').read_text(encoding='utf-8'))
    provenance = json.loads((vendor / 'provenance.json').read_text(encoding='utf-8'))
    digest = sha256(archive)
    if digest != provenance['archive_sha256']:
        raise ValueError('Archive SHA256 does not match provenance')
    actual, seen = {}, set()
    with tarfile.open(archive, 'r:gz') as source:
        for member in source.getmembers():
            path = _member_path(member)
            if path.as_posix() in seen:
                raise ValueError('Duplicate runtime archive entry: ' + member.name)
            seen.add(path.as_posix())
            if member.isfile():
                if len(path.parts) < 2:
                    raise ValueError('Package root must be a directory')
                key = PurePosixPath(*path.parts[1:]).as_posix()
                digest_member = hashlib.sha256()
                with source.extractfile(member) as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b''):
                        digest_member.update(block)
                actual[key] = digest_member.hexdigest()
    if actual != manifest:
        changed = sorted(key for key in set(actual) | set(manifest)
                         if actual.get(key) != manifest.get(key))
        raise ValueError('Archive contents differ from manifest: ' + repr(changed[:20]))
    return {'audit_pass': True, 'archive_sha256': digest, 'file_count': len(actual),
            'base_manifest': actual, 'vendor_manifest_sha256': sha256(vendor / 'runtime_manifest.json')}


def install(vendor, output, instrument_fn=None):
    """Install into a NEW root with base/sglang and profile/sglang packages."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing runtime root: ' + str(output))
    receipt = verify_archive(vendor)
    if instrument_fn is None:
        sys.path.insert(0, str(ROOT))
        from profiler.instrument import instrument
        instrument_fn = instrument
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.llama-runtime-', dir=output.parent) as name:
        staging = Path(name)
        base = staging / 'base' / 'sglang'
        base.mkdir(parents=True)
        with tarfile.open(Path(vendor) / 'original_runtime.tar.gz', 'r:gz') as source:
            for member in source.getmembers():
                path = _member_path(member)
                target = staging / 'base' / Path(*path.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
                    # Preserve executable bits, never setuid/setgid/sticky bits.
                    target.chmod(member.mode & 0o777)
        if file_manifest(base) != receipt['base_manifest']:
            raise ValueError('Extracted runtime differs from the verified archive')
        profile = staging / 'profile' / 'sglang'
        instrumentation = instrument_fn(base, profile)
        if not instrumentation.get('ast_equivalent'):
            raise ValueError('Instrumentation did not pass its AST equivalence audit')
        if file_manifest(base) != receipt['base_manifest']:
            raise ValueError('Instrumentation modified the original runtime')
        receipt.update(schema_version=1, base_package='base/sglang', profile_package='profile/sglang',
                       profile_manifest=file_manifest(profile), instrumentation=instrumentation,
                       source_identity='Bundled SHA256 snapshot; not an inferred upstream release')
        if not receipt['profile_manifest']:
            raise ValueError('Instrumentation produced an empty runtime')
        (staging / 'install_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        if output.exists():
            raise FileExistsError('Runtime destination appeared during installation: ' + str(output))
        staging.rename(output)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor', type=Path, default=ROOT / 'vendor')
    parser.add_argument('--output', type=Path, default=ROOT / '.runtime')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    if args.verify_only:
        receipt = verify_archive(args.vendor)
    else:
        if os.name != 'posix':
            parser.error('Install on Linux; Windows supports --verify-only and offline tests')
        receipt = install(args.vendor, args.output)
    print(json.dumps({key: receipt[key] for key in ('audit_pass', 'archive_sha256', 'file_count')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

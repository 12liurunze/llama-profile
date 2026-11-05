"""Offline installation/preflight invariants; no GPU or model payload loads."""
import contextlib
import copy
import hashlib
import io
import json
import shutil
import struct
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.doctor import _header, audit_gpus, audit_model, audit_runtime
from tools.install_runtime import file_manifest, install, main, sha256, verify_archive


class RuntimeInstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='llama-install-test-')
        self.root = Path(self.temp.name)
        self.vendor = self.root / 'vendor'
        self.vendor.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, entries=None):
        entries = entries or [('sglang/__init__.py', b'__version__="test"\n'),
                              ('sglang/srt/example.py', b'VALUE=1\n')]
        archive = self.vendor / 'original_runtime.tar.gz'
        manifest = {}
        with tarfile.open(archive, 'w:gz') as handle:
            for name, data in entries:
                info = tarfile.TarInfo(name)
                if data is None:
                    info.type = tarfile.SYMTYPE
                    info.linkname = '/outside'
                    handle.addfile(info)
                else:
                    info.size = len(data)
                    handle.addfile(info, io.BytesIO(data))
                    manifest[name.removeprefix('sglang/')] = hashlib.sha256(data).hexdigest()
        (self.vendor / 'runtime_manifest.json').write_text(json.dumps(manifest))
        (self.vendor / 'provenance.json').write_text(json.dumps({'archive_sha256': sha256(archive)}))
        return archive

    @staticmethod
    def instrument(source, destination):
        shutil.copytree(source, destination)
        (destination / 'srt/example.py').write_text('VALUE=1\n# instrumented\n')
        return {'ast_equivalent': True}

    def test_verify_only_never_extracts(self):
        self.archive()
        output = self.root / 'runtime'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--vendor', str(self.vendor), '--output', str(output), '--verify-only']), 0)
        self.assertFalse(output.exists())

    def test_complete_install_receipt_and_runtime_audit(self):
        self.archive()
        output = self.root / 'runtime'
        receipt = install(self.vendor, output, self.instrument)
        self.assertEqual(receipt['base_manifest'], file_manifest(output / 'base/sglang'))
        self.assertEqual(receipt['profile_manifest'], file_manifest(output / 'profile/sglang'))
        self.assertTrue(audit_runtime(output / 'profile/sglang', self.vendor)['audit_pass'])
        (output / 'profile/sglang/srt/example.py').write_text('DRIFT=True')
        self.assertFalse(audit_runtime(output / 'profile/sglang', self.vendor)['audit_pass'])

    def test_existing_directory_is_preserved(self):
        self.archive()
        output = self.root / 'runtime'
        output.mkdir()
        (output / 'keep.txt').write_text('do not overwrite')
        with self.assertRaises(FileExistsError):
            install(self.vendor, output, self.instrument)
        self.assertEqual((output / 'keep.txt').read_text(), 'do not overwrite')

    def test_corrupt_archive_and_manifest_rejected(self):
        archive = self.archive()
        with archive.open('ab') as handle:
            handle.write(b'tamper')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            verify_archive(self.vendor)
        self.archive()
        (self.vendor / 'runtime_manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            verify_archive(self.vendor)

    def test_traversal_links_and_duplicates_rejected(self):
        for entries in ([('sglang/../../escape', b'x')], [('sglang/link', None)],
                        [('sglang/a', b'x'), ('sglang/a', b'y')]):
            self.archive(entries)
            with self.assertRaises(ValueError):
                verify_archive(self.vendor)
        self.assertFalse((self.root / 'escape').exists())

    def test_instrumentation_cannot_modify_baseline(self):
        self.archive()
        def invalid(source, destination):
            self.instrument(source, destination)
            (source / '__init__.py').write_text('CHANGED=True')
            return {'ast_equivalent': True}
        output = self.root / 'runtime'
        with self.assertRaisesRegex(ValueError, 'original runtime'):
            install(self.vendor, output, invalid)
        self.assertFalse(output.exists())

    def test_safetensors_header_extents_and_corruption(self):
        path = self.root / 'model.safetensors'
        header = json.dumps({'tensor': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]}}).encode()
        path.write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 4)
        tensors, metadata = _header(path)
        self.assertEqual(tensors['tensor']['shape'], [2])
        self.assertEqual(metadata['tensor_count'], 1)
        path.write_bytes(struct.pack('<Q', len(header)) + header + b'\0')
        with self.assertRaises(ValueError):
            _header(path)

    def test_wrong_model_and_remote_identifier_fail(self):
        folder = self.root / '8b'
        folder.mkdir()
        (folder / 'config.json').write_text(json.dumps({'hidden_size': 4096, 'num_hidden_layers': 32}))
        result = audit_model(folder, 'target')
        self.assertFalse(result['audit_pass'])
        self.assertTrue(any('hidden_size' in error for error in result['errors']))
        self.assertFalse(audit_model(self.root / 'not-downloaded', 'draft')['audit_pass'])

    def draft_fixture(self):
        """CPU header fixture: no 2.81GB checkpoint or model allocation needed."""
        cfg = {'architectures': ['LlamaForCausalLMEagle3'], 'model_type': 'llama',
               'hidden_size': 8192, 'num_hidden_layers': 1, 'head_dim': 128,
               'num_attention_heads': 64, 'num_key_value_heads': 8,
               'intermediate_size': 28672, 'vocab_size': 128256,
               'draft_vocab_size': 32000, 'dtype': 'bfloat16',
               'tie_word_embeddings': False}
        shapes = {'lm_head.weight': [32000, 8192], 'fc.weight': [8192, 24576],
                  'norm.weight': [8192],
                  'midlayer.self_attn.q_proj.weight': [8192, 16384],
                  'midlayer.self_attn.k_proj.weight': [1024, 16384],
                  'midlayer.self_attn.v_proj.weight': [1024, 16384],
                  'midlayer.self_attn.o_proj.weight': [8192, 8192],
                  'midlayer.mlp.gate_proj.weight': [28672, 8192],
                  'midlayer.mlp.up_proj.weight': [28672, 8192],
                  'midlayer.mlp.down_proj.weight': [8192, 28672],
                  'midlayer.input_layernorm.weight': [8192],
                  'midlayer.hidden_norm.weight': [8192],
                  'midlayer.post_attention_layernorm.weight': [8192]}
        tensors = {name: {'shape': dimensions, 'dtype': 'BF16'}
                   for name, dimensions in shapes.items()}
        tensors.update(d2t={'shape': [32000], 'dtype': 'I64'},
                       t2d={'shape': [128256], 'dtype': 'BOOL'})
        return cfg, tensors

    def audit_draft_fixture(self, cfg, tensors):
        folder = self.root / 'draft'
        folder.mkdir(exist_ok=True)
        (folder / 'config.json').write_text(json.dumps(cfg))
        (folder / 'model.safetensors').touch()
        metadata = {'size': 2810000000, 'mtime_ns': 123,
                    'header_sha256': 'fixture', 'header_bytes': 4096,
                    'tensor_count': len(tensors)}
        with mock.patch('tools.doctor._header', return_value=(tensors, metadata)):
            return audit_model(folder, 'draft')

    def test_specforge_70b_draft_shapes_and_tp8_contract(self):
        cfg, tensors = self.draft_fixture()
        result = self.audit_draft_fixture(cfg, tensors)
        self.assertTrue(result['audit_pass'], result['errors'])
        self.assertEqual(result['tensor_parallel'], {'world_size': 8, 'head_dim': 128,
                                                     'query_heads_per_rank': 8, 'kv_heads_per_rank': 1})
        self.assertEqual(result['expected_dimensions']['draft_vocab_size'], 32000)
        self.assertFalse(result['hot_token_mapping']['values_verified'])
        # The bundled loader supports both official legacy and current names.
        renamed = {key.replace('midlayer.', 'model.layers.0.'): value for key, value in tensors.items()}
        result = self.audit_draft_fixture(cfg, renamed)
        self.assertTrue(result['audit_pass'], result['errors'])

    def test_draft_rejects_8b_projection_and_mapping_mismatches(self):
        cfg, tensors = self.draft_fixture()
        changes = [('midlayer.self_attn.q_proj.weight', 'shape', [8192, 8192]),
                   ('midlayer.self_attn.k_proj.weight', 'shape', [8192, 16384]),
                   ('fc.weight', 'shape', [8192, 16384]),
                   ('lm_head.weight', 'shape', [128256, 8192]),
                   ('d2t', 'shape', [128256]), ('d2t', 'dtype', 'F32'),
                   ('t2d', 'shape', [32000]), ('t2d', 'dtype', 'I64')]
        for name, field, bad in changes:
            with self.subTest(name=name, field=field):
                changed = copy.deepcopy(tensors)
                changed[name][field] = bad
                result = self.audit_draft_fixture(cfg, changed)
                self.assertFalse(result['audit_pass'])
                self.assertTrue(any(name in error for error in result['errors']), result['errors'])
        for key, bad in [('head_dim', 64), ('num_key_value_heads', 4),
                         ('num_attention_heads', 32), ('draft_vocab_size', 128256)]:
            with self.subTest(config=key):
                changed = dict(cfg, **{key: bad})
                result = self.audit_draft_fixture(changed, tensors)
                self.assertFalse(result['audit_pass'])

    def test_draft_rejects_silently_skipped_or_duplicate_serving_names(self):
        cfg, tensors = self.draft_fixture()
        changed = dict(tensors)
        changed['trainer.fc.weight'] = changed.pop('fc.weight')
        result = self.audit_draft_fixture(cfg, changed)
        self.assertFalse(result['audit_pass'])
        self.assertTrue(any('fc.weight' in error for error in result['errors']))
        changed = dict(tensors, **{'model.fc.weight': tensors['fc.weight']})
        self.assertFalse(self.audit_draft_fixture(cfg, changed)['audit_pass'])
        changed = dict(tensors)
        del changed['d2t']
        self.assertFalse(self.audit_draft_fixture(cfg, changed)['audit_pass'])

    def test_free_and_busy_eight_gpu_read_only_queries(self):
        calls = []
        def runner(command, **kwargs):
            calls.append(command)
            if '--query-gpu=' in command[1]:
                return '\n'.join(f'{i}, GPU-{i}, NVIDIA A100-SXM4-80GB, 81920, 81000, 580.95.05, 8.0, Disabled' for i in range(8))
            if '--query-compute-apps=' in command[1]:
                return ''
            return 'topology'
        gpus = list(map(str, range(8)))
        self.assertTrue(audit_gpus(gpus, run=runner)['audit_pass'])
        self.assertTrue(all(command[0] == 'nvidia-smi' for command in calls))
        def busy(command, **kwargs):
            if '--query-compute-apps=' in command[1]:
                return 'GPU-3, 123, 6000\n'
            return runner(command, **kwargs)
        result = audit_gpus(gpus, run=busy)
        self.assertFalse(result['audit_pass'])
        self.assertEqual(result['processes'][0]['pid'], 123)


if __name__ == '__main__':
    unittest.main()

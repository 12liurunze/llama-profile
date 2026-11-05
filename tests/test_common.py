"""CPU-only contracts for the TP8 experiment plan and immutable configuration."""
from collections import Counter
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llama_profile import common


def options(**changes):
    result = dict(tp=8, gpus=list(map(str, range(8))), batches=[128, 256, 512],
                  threshold=.8, memory_fraction=.8, repeat=3, min_steps=10,
                  context=4096, max_new_tokens=1024, chunked_prefill=4096,
                  protocol='queued', model='/models/target', draft='/models/draft',
                  runtime='/runtime/profile/sglang')
    result.update(changes)
    return result


class PlanAndConfigurationTests(unittest.TestCase):
    def test_three_repeats_rotate_method_order_and_keep_separate_diagnostics(self):
        runs = common.plan_runs(options())
        self.assertEqual(len(runs), 45)
        self.assertEqual(len({r['name'] for r in runs}), 45)
        for bs in common.BATCHES:
            batch = [r for r in runs if r['bs'] == bs]
            self.assertEqual(Counter(r['stage'] for r in batch),
                             {'smoke': 3, 'timing': 9, 'kernel': 3})
            self.assertEqual([r['stage'] for r in batch[:3]], ['smoke'] * 3)
            self.assertEqual([r['stage'] for r in batch[-3:]], ['kernel'] * 3)
            orders = [[r['method'] for r in batch if r['stage'] == 'timing' and r['repeat'] == repeat]
                      for repeat in (1, 2, 3)]
            self.assertEqual(orders, [['chain7', 'adaptive2', 'eagle728'],
                                      ['adaptive2', 'eagle728', 'chain7'],
                                      ['eagle728', 'chain7', 'adaptive2']])
            for position in range(3):
                self.assertEqual({order[position] for order in orders}, set(common.METHODS))
            self.assertTrue(all(r['repeat'] == 0 for r in batch if r['stage'] != 'timing'))

    def test_requested_batch_order_is_preserved(self):
        runs = common.plan_runs(options(batches=[512, 128]))
        self.assertEqual([r['bs'] for r in runs], [512] * 15 + [128] * 15)

    def test_methods_share_seven_node_budget_and_explicit_prefill_policy(self):
        original = options()
        snapshot = copy.deepcopy(original)
        for bs in common.BATCHES:
            for method in common.METHODS:
                with self.subTest(bs=bs, method=method):
                    cfg = common.engine_config(original, method, bs)
                    self.assertEqual((cfg['tp_size'], cfg['dp_size'], cfg['pp_size']), (8, 1, 1))
                    self.assertEqual(cfg['max_running_requests'], bs)
                    self.assertEqual(cfg['speculative_algorithm'], 'EAGLE3')
                    self.assertEqual(cfg['speculative_num_steps'], 7)
                    self.assertEqual(cfg['speculative_num_draft_tokens'], 8)
                    self.assertEqual(cfg['speculative_eagle_topk'], 1 if method == 'chain7' else 2)
                    self.assertEqual(cfg['speculative_adaptive_tree'], method == 'adaptive2')
                    self.assertEqual(cfg['speculative_adaptive_threshold'], .8 if method == 'adaptive2' else 0.)
                    self.assertEqual(cfg['speculative_adaptive_confidence'], 'top1')
                    self.assertFalse(cfg['disable_cuda_graph'])
                    self.assertFalse(cfg['disable_decode_cuda_graph'])
                    self.assertEqual(cfg['cuda_graph_backend_prefill'], 'breakable')
                    self.assertEqual(cfg['cuda_graph_max_bs_prefill'], 4096)
                    self.assertIn(bs, cfg['cuda_graph_bs_decode'])
                    self.assertTrue(all(0 < n <= bs for n in cfg['cuda_graph_bs_decode']))
                    self.assertEqual(cfg['chunked_prefill_size'], 4096)
                    self.assertEqual(cfg['max_prefill_tokens'], 16384)
                    self.assertFalse(cfg['enable_mixed_chunk'])
                    self.assertEqual(cfg['prefill_decode_interval'], 0)
                    self.assertFalse(cfg['enable_prefill_delayer'])
                    self.assertEqual(cfg['min_free_slots_delay'], 1)
                    self.assertEqual(cfg['schedule_policy'], 'fcfs')
                    self.assertNotIn('ignore_eos', cfg)
                    self.assertNotIn('min_new_tokens', cfg)
        self.assertEqual(original, snapshot)

    def test_invalid_launch_settings_fail_before_loading_an_engine(self):
        cases = [dict(tp=4), dict(gpus=list(map(str, range(7)))),
                 dict(gpus=['0'] * 8), dict(gpus=['GPU-uuid'] + list(map(str, range(1, 8)))),
                 dict(batches=[]), dict(batches=[64]), dict(batches=[128, 128]),
                 dict(threshold=-.01), dict(threshold=1.01), dict(threshold=float('nan')),
                 dict(memory_fraction=.49), dict(memory_fraction=.91),
                 dict(repeat=2), dict(min_steps=1), dict(context=1024),
                 dict(max_new_tokens=0), dict(max_new_tokens=4096),
                 dict(chunked_prefill=-1), dict(protocol='unknown')]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                common.validate_options(options(**changes))
        for method, bs in [('unknown', 128), ('chain7', 127)]:
            with self.subTest(method=method, bs=bs), self.assertRaises(ValueError):
                common.engine_config(options(), method, bs)

    def test_environment_clears_previous_experiment_overrides(self):
        inherited = dict(PATH='normal-path', PROFILE_KERNEL_MODE='1', PROFILE_OLD='stale',
                         STEADY_OLD='stale', DIAG_OLD='stale', MATH_OLD='stale',
                         EAGLE_ACTIVE_OLD='stale', SWEEP_BS_COUNTER='stale',
                         ADAPTIVE_PROFILE_OUT='stale', FUSION_AUDIT_OUT='stale',
                         SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION='128',
                         SGLANG_SIMULATE_ACC_LEN='8', SGLANG_SIMULATE_ACC_TOKEN_MODE='fixed',
                         SGLANG_TRACE_LOGITS_E2E_SYNC='1', CUDA_LAUNCH_BLOCKING='1',
                         RANK='3', LOCAL_RANK='3', WORLD_SIZE='8', LOCAL_WORLD_SIZE='8',
                         MASTER_ADDR='old-launcher', MASTER_PORT='12345')
        with patch.dict(os.environ, inherited, clear=True):
            run = dict(bs=512, stage='timing', run_id='stable-run-id')
            trace = Path('/results/trace')
            env = common.environment(options(), trace, run)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
            self.assertEqual(env['PROFILE_EXPECTED_WORLD_SIZE'], '8')
            self.assertEqual(env['PROFILE_EXPECTED_BS'], '512')
            self.assertEqual(env['PROFILE_RUN_ID'], 'stable-run-id')
            self.assertEqual(env['PROFILE_CONTROL'], str(trace / 'control.bin'))
            self.assertEqual(env['PATH'], 'normal-path')
            self.assertNotIn('PROFILE_KERNEL_MODE', env)
            self.assertTrue(all(key not in env for key in inherited if key != 'PATH'))
            self.assertEqual(common.environment(options(), trace, dict(run, stage='kernel'))['PROFILE_KERNEL_MODE'], '1')
            self.assertEqual(dict(os.environ), inherited)


class RuntimeIdentityTests(unittest.TestCase):
    def test_runtime_manifest_rejects_change_missing_and_extra_files(self):
        with tempfile.TemporaryDirectory(prefix='llama-common-test-') as directory:
            root = Path(directory)
            runtime = root / '.runtime/profile/sglang'
            runtime.mkdir(parents=True)
            source = runtime / '__init__.py'
            source.write_text('ORIGINAL = True\n', encoding='utf-8')
            manifest = {'__init__.py': common.sha(source)}
            common.save(root / '.runtime/install_receipt.json', {'profile_manifest': manifest})
            (runtime / '__pycache__').mkdir()
            (runtime / '__pycache__/ignore.pyc').write_bytes(b'cache')
            # A custom runtime keeps its receipt beside the installation,
            # independently of the checkout containing common.py.
            with patch.object(common, 'ROOT', root / 'unrelated-checkout'):
                self.assertEqual(common.runtime_hash(runtime), common.digest(manifest))
                source.write_text('CHANGED = True\n', encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError, '__init__.py'):
                    common.runtime_hash(runtime)
                source.unlink()
                with self.assertRaisesRegex(RuntimeError, '__init__.py'):
                    common.runtime_hash(runtime)
                source.write_text('ORIGINAL = True\n', encoding='utf-8')
                (runtime / 'extra.py').write_text('EXTRA = True\n', encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError, 'extra.py'):
                    common.runtime_hash(runtime)


if __name__ == '__main__':
    unittest.main()

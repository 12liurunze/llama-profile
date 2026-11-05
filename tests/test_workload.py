"""Offline workload provenance and frozen Engine output-accounting contracts."""
from collections import Counter
import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llama_profile.common import digest, load, save, sha
from llama_profile.workload import build_steady
from llama_profile.worker import checked_outputs, drain

SUBSETS = ('HumanEval', 'math_reasoning', 'qa', 'question', 'rag',
           'summarization', 'translation', 'writing')


class SteadyWorkloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='llama-workload-test-')
        self.root = Path(self.temp.name)
        self.prepared_path = self.root / 'prepared.json'
        self.calibration_path = self.root / 'calibration.json'
        rows, calibrated = [], {}
        for subset_index, subset in enumerate(SUBSETS):
            # The source corpus contains identical prompts in question/writing.
            shared_index = 3 if subset == 'writing' else subset_index
            for i, length in enumerate((255, 256, 300, 400)):
                prompt_hash = digest([shared_index, i])
                row = dict(id=f'{subset}-{i}', subset=subset, source_id=f'source-{subset}-{i}',
                           prompt_sha256=prompt_hash, input_ids=[100 + shared_index] * (i + 2))
                rows.append(row)
                calibrated[prompt_hash] = dict(prompt_sha256=prompt_hash, output_tokens=length,
                                               speculative_acceptance=1000 - length, speedup=-length)
                if i == 1:
                    rows.append(dict(row, id=f'{subset}-{i}-copy'))
        self.prepared = dict(format_version=1, model_path='/models/target', context=4096,
                             max_new_tokens=1024, raw_sha256='source-raw-hash',
                             tokenizer_files={'tokenizer.json': 'tokenizer-hash'},
                             chat_template_sha256='chat-template-hash', count=len(rows), rows=rows)
        save(self.prepared_path, self.prepared)
        self.calibration = dict(selection_signal='target_ar_natural_output_length',
                                prepared_sha256=sha(self.prepared_path), max_new_tokens=1024,
                                rows=list(calibrated.values()))
        save(self.calibration_path, self.calibration)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, filename='steady.json', **kwargs):
        path = self.root / filename
        selection = build_steady(self.prepared_path, self.calibration_path, path, **kwargs)
        return path, load(path), selection

    def test_balanced_deterministic_selection_and_honest_copy_provenance(self):
        path, result, selection = self.build()
        second, repeat, _ = self.build('repeat.json')
        self.assertEqual(path.read_bytes(), second.read_bytes())
        self.assertEqual(result, repeat)
        rows = result['rows']
        self.assertEqual(len(rows), 4000)
        self.assertEqual(len({r['id'] for r in rows}), 4000)
        self.assertEqual(Counter(r['subset'] for r in rows), {s: 500 for s in SUBSETS})
        self.assertEqual([r['subset'] for r in rows[:8]], sorted(SUBSETS))
        self.assertEqual(selection['eligible_distinct_prompts_per_subset'], {s: 3 for s in SUBSETS})
        self.assertEqual(selection['distinct_prompt_count'], 21)
        parents = {r['id']: r for r in self.prepared['rows']}
        for subset in SUBSETS:
            selected = [r for r in rows if r['subset'] == subset]
            self.assertEqual(Counter(r['parent_id'] for r in selected),
                             {f'{subset}-1': 167, f'{subset}-2': 167, f'{subset}-3': 166})
            for k, row in enumerate(selected):
                parent = parents[row['parent_id']]
                self.assertEqual(row['steady_copy'], k // 3)
                self.assertGreaterEqual(row['calibrated_natural_output_tokens'], 256)
                self.assertEqual(row['input_ids'], parent['input_ids'])
                self.assertEqual(row['source_id'], parent['source_id'])
                self.assertEqual(row['prompt_sha256'], parent['prompt_sha256'])

    def test_output_metadata_identifies_conditional_repeated_workload(self):
        _, result, selection = self.build()
        self.assertEqual(result['count'], 4000)
        self.assertEqual(result['inputs_sha256'], digest([r['input_ids'] for r in result['rows']]))
        lengths = [len(r['input_ids']) for r in result['rows']]
        self.assertEqual(result['min_input_tokens'], min(lengths))
        self.assertEqual(result['max_input_tokens'], max(lengths))
        self.assertAlmostEqual(result['mean_input_tokens'], sum(lengths) / 4000)
        self.assertEqual(result['tokenizer_files'], self.prepared['tokenizer_files'])
        self.assertEqual(result['raw_sha256'], self.prepared['raw_sha256'])
        self.assertEqual(result['chat_template_sha256'], self.prepared['chat_template_sha256'])
        self.assertEqual(selection, result['steady_selection'])
        self.assertEqual(selection['minimum'], 256)
        self.assertEqual(selection['signal'], 'target_ar_natural_output_length')
        self.assertEqual(selection['calibration_sha256'], sha(self.calibration_path))
        self.assertEqual(selection['original_prepared_sha256'], sha(self.prepared_path))
        self.assertIn('not original benchmark distribution', selection['interpretation'])
        self.assertIn('not selected by speculative acceptance or speed', selection['interpretation'])

    def test_speculative_gain_labels_cannot_influence_selection(self):
        _, baseline, _ = self.build()
        changed = copy.deepcopy(self.calibration)
        changed['rows'].reverse()
        for i, row in enumerate(changed['rows']):
            row.update(speculative_acceptance=i % 2, speedup=10**9 if i % 2 else -10**9)
        save(self.calibration_path, changed)
        _, result, _ = self.build('different-gains.json')
        self.assertEqual(result['rows'], baseline['rows'])
        self.assertEqual(result['inputs_sha256'], baseline['inputs_sha256'])

    def test_existing_output_is_never_replaced(self):
        output = self.root / 'keep.json'
        output.write_bytes(b'keep this exact payload')
        with self.assertRaises(FileExistsError):
            build_steady(self.root / 'missing.json', self.calibration_path, output)
        self.assertEqual(output.read_bytes(), b'keep this exact payload')

    def test_missing_eligible_subset_fails_without_lowering_threshold(self):
        missing = {r['prompt_sha256'] for r in self.prepared['rows'] if r['subset'] == 'qa'}
        for mode in ('short', 'absent'):
            changed = copy.deepcopy(self.calibration)
            if mode == 'short':
                for row in changed['rows']:
                    if row['prompt_sha256'] in missing:
                        row['output_tokens'] = 255
            else:
                changed['rows'] = [r for r in changed['rows'] if r['prompt_sha256'] not in missing]
            save(self.calibration_path, changed)
            output = self.root / f'missing-{mode}.json'
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'qa'):
                build_steady(self.prepared_path, self.calibration_path, output)
            self.assertFalse(output.exists())

    def test_incompatible_calibration_and_non_ar_signal_are_rejected(self):
        for changes in ({'prepared_sha256': 'wrong'}, {'selection_signal': 'speculative_speedup'}):
            changed = dict(self.calibration, **changes)
            save(self.calibration_path, changed)
            output = self.root / 'invalid.json'
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_steady(self.prepared_path, self.calibration_path, output)
            self.assertFalse(output.exists())


def speculative_output():
    # One prefill token plus three verify rounds, five accepted draft tokens.
    return dict(output_ids=list(range(9)), meta_info=dict(completion_tokens=9,
                finish_reason={'type': 'stop', 'matched': 128009},
                spec_verify_ct=3, spec_num_correct_drafts=5, spec_num_proposed_drafts=21,
                spec_correct_drafts_histogram=[1, 0, 1, 1, 0, 0, 0, 0]))


class WorkerOutputTests(unittest.TestCase):
    def test_output_metadata_and_seven_slot_histogram_are_preserved(self):
        output = speculative_output()
        before = copy.deepcopy(output)
        result = checked_outputs([output], 1, 1024)
        self.assertEqual(output, before)
        self.assertEqual(result[0]['output_tokens'], 9)
        self.assertEqual(result[0]['output_ids_sha256'], digest(output['output_ids']))
        self.assertEqual(result[0]['finish_reason'], output['meta_info']['finish_reason'])
        self.assertEqual(result[0]['meta_info'], output['meta_info'])

    def test_ar_calibration_does_not_require_speculative_metrics(self):
        output = dict(output_ids=[10, 11], meta_info=dict(completion_tokens=2, finish_reason={'type': 'stop'}))
        self.assertEqual(checked_outputs([output], 1, 2, speculative=False)[0]['output_tokens'], 2)

    def test_first_prefill_token_can_naturally_finish_without_verify_metadata(self):
        # Frozen tokenizer_manager emits spec_* metadata only when verify_ct > 0.
        output = dict(output_ids=[128009], meta_info=dict(completion_tokens=1,
                                                       finish_reason={'type': 'stop', 'matched': 128009}))
        checked = checked_outputs([output], 1, 1024)[0]
        self.assertEqual(checked['output_tokens'], 1)
        self.assertTrue(checked['meta_info']['prefill_only_completion'])
        self.assertEqual(checked['meta_info']['spec_verify_ct'], 0)
        self.assertEqual(checked['meta_info']['spec_correct_drafts_histogram'], [0] * 8)
        self.assertNotIn('spec_verify_ct', output['meta_info'])

    def test_missing_verify_metadata_for_multiple_tokens_is_rejected(self):
        output = dict(output_ids=[1, 2], meta_info=dict(completion_tokens=2, finish_reason={'type': 'stop'}))
        with self.assertRaises(RuntimeError):
            checked_outputs([output], 1, 1024)

    def test_output_count_completion_count_and_cap_are_checked(self):
        output = speculative_output()
        with self.assertRaises(RuntimeError):
            checked_outputs([output], 2, 1024)
        with self.assertRaises(RuntimeError):
            checked_outputs([output], 1, 8)
        output['meta_info']['completion_tokens'] = 8
        with self.assertRaises(RuntimeError):
            checked_outputs([output], 1, 1024)

    def test_bad_histogram_counts_and_negative_buckets_are_rejected(self):
        cases = [dict(spec_verify_ct=4), dict(spec_num_correct_drafts=4),
                 dict(spec_num_proposed_drafts=20),
                 dict(spec_correct_drafts_histogram=[0] * 8 + [1], spec_verify_ct=1,
                      spec_num_correct_drafts=8, spec_num_proposed_drafts=7),
                 dict(spec_correct_drafts_histogram=[-1, 2, 0, 0, 0, 0, 0, 0],
                      spec_verify_ct=1, spec_num_correct_drafts=2, spec_num_proposed_drafts=7),
                 dict(spec_correct_drafts_histogram=[0] * 8, spec_verify_ct=0,
                      spec_num_correct_drafts=0, spec_num_proposed_drafts=0),
                 dict(spec_verify_ct=3.0),
                 dict(spec_correct_drafts_histogram=[True, 0, 1, 1, 0, 0, 0, 0])]
        for changes in cases:
            output = speculative_output()
            output['meta_info'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                checked_outputs([output], 1, 1024)

    def test_aborted_outputs_are_not_natural_completed_requests(self):
        output = speculative_output()
        output['meta_info']['finish_reason'] = {'type': 'abort', 'message': 'cancelled'}
        for speculative in (False, True):
            with self.subTest(speculative=speculative), self.assertRaises(RuntimeError):
                checked_outputs([output], 1, 1024, speculative=speculative)

    def test_drain_waits_for_flush_success_and_fails_if_never_idle(self):
        replies = iter([False, False, True])
        calls = []
        def flush():
            calls.append(None)
            return types.SimpleNamespace(success=next(replies))
        with patch('llama_profile.worker.time.sleep') as sleep:
            drain(types.SimpleNamespace(flush_cache=flush))
            self.assertEqual(len(calls), 3)
            self.assertEqual(sleep.call_count, 2)
        with patch('llama_profile.worker.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'did not fully drain'):
                drain(types.SimpleNamespace(flush_cache=lambda: types.SimpleNamespace(success=False)))


if __name__ == '__main__':
    unittest.main()

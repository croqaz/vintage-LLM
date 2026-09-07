"""Unit accounting, missing values, protocol grouping and file workflows."""

import collections
import copy
import io
import json
import math
import re
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from eval import SCHEMA_VERSION
from eval import __main__ as E
from eval import metrics as M
from eval import report as R
from eval.__main__ import collect_results, parse_args, rank_by_chat, rank_by_composite, rank_by_prose, settings_fingerprint
from eval.helpers import ChatItem, TextItem, atomic_json, model_fingerprint, stable_hash, training_curve
from eval.measurements import summarize_result
from eval.metric_guide import describe, undocumented
from eval.tests.test_surface_comparison import model, sample


class ByteTokenizer:
    """One token per UTF-8 character used here; optionally prepend BOS."""

    eos_token_id = None

    def __init__(self, bos=None):
        self.bos_token_id = bos

    def __call__(self, text, add_special_tokens=True):
        ids = [1] * len(text)
        if add_special_tokens and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, **kwargs):
        return ''.join('a' if i == 1 else '<bos>' for i in ids)


class UniformModel:
    device = torch.device('cpu')

    def __init__(self, bos=None):
        self.config = SimpleNamespace(bos_token_id=bos, eos_token_id=None, max_position_embeddings=16)

    def __call__(self, input_ids, **kwargs):
        return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 4)))


class AccountingTests(unittest.TestCase):
    def test_continuation_span_bpb_units(self):
        for bos in (None, 0):
            tokenizer, net = ByteTokenizer(bos), UniformModel(bos)
            self.assertAlmostEqual(M.span_bpb(tokenizer, net, 'a' * 12, 'a' * 8), 2.0, places=5)
        self.assertTrue(math.isnan(M.span_bpb(ByteTokenizer(), UniformModel(), 'a' * 12, '')))

    def test_probe_bytes_follow_predicted_ids_with_and_without_bos(self):
        self.assertEqual(M.scored_span_bytes(ByteTokenizer(), 'abcdefgh'), 7)
        self.assertEqual(M.scored_span_bytes(ByteTokenizer(0), 'abcdefgh'), 8)

    def test_prose_and_judge_same_exact_units_with_and_without_bos(self):
        for bos in (None, 0):
            with self.subTest(bos=bos):
                tokenizer, net = ByteTokenizer(bos), UniformModel(bos)
                records, skipped = M.score_prose_records(tokenizer, net, [TextItem('a', 'a' * 30, 'A')], 32, 0)
                self.assertEqual(skipped, 0)
                row = records[0]
                self.assertTrue(row['truncated'])
                self.assertEqual(row['tokens'], 15)
                self.assertEqual(row['bytes'], 15)
                self.assertAlmostEqual(row['bits'], 30, places=5)
                self.assertAlmostEqual(M.records_bpb(records), 2.0, places=5)
                self.assertAlmostEqual(M.bits_per_byte_of_texts(tokenizer, net, ['a' * 30], 32), 2.0, places=5)
                self.assertEqual(len(row['scored_text_sha256']), 64)
                # A document too short to score is counted, not silently dropped.
                short, skipped = M.score_prose_records(tokenizer, net, [TextItem('s', 'abc', 'A')], 32, 0)
                self.assertEqual((short, skipped), ([], 1))

    def test_aggregate_and_bootstrap_use_identical_valid_pool(self):
        records = model('a', [100, 900])['prose_records']
        records[1]['bytes'] = 300
        bad = [
            {'id': 'bad', 'bits': float('nan'), 'bytes': 100},
            {'id': 'empty', 'bits': 9, 'bytes': 0},
            {'id': 'null', 'bits': None, 'bytes': 100},
        ]
        self.assertEqual(M.records_bpb(records + bad), 2.5)
        self.assertEqual(M.bootstrap_ci(records + bad, 200, 0.9, 2), M.bootstrap_ci(records, 200, 0.9, 2))
        self.assertEqual(M.paired_bootstrap(records + bad, records, 200, 0.9, 2)['delta'], 0.0)
        self.assertEqual(M.count_nonfinite_records(records + bad), 2)
        self.assertTrue(math.isnan(M.bootstrap_ci(records[:1], 200, 0.9, 2)[0]))

    def test_nonfinite_composite_is_partial_and_not_ranked(self):
        r = model('partial', [100, 100])
        r['logic'] = {'accuracy': float('nan')}
        r['chat_records'] = [
            {**row, 'bits': float('inf'), 'target_truncated': False, 'input_text_sha256': row['scored_text_sha256']}
            for row in r['prose_records']
        ]
        summarize_result(r)
        self.assertEqual(r['summary']['bake_status'], 'partial')
        self.assertEqual(r['summary']['bake_weight_covered'], 0.5)
        self.assertEqual(r['bake_score'], M.interp(1.0, M.BPB_LADDER))
        self.assertNotIn('points_logic', r['summary'])
        self.assertEqual(rank_by_composite([r]), [])
        for val in (float('nan'), float('inf'), None):
            self.assertTrue(math.isnan(M.interp(val, M.BPB_LADDER)))
            self.assertTrue(math.isnan(M.logic_points(val)))

    def test_json_has_null_instead_of_nan_inf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.json'
            atomic_json(path, {'a': float('nan'), 'b': [float('inf'), -float('inf')], 'zero': 0})
            self.assertEqual(json.loads(path.read_text()), {'a': None, 'b': [None, None], 'zero': 0})
        self.assertEqual(R._confidence(0.985), '98.5%')
        self.assertEqual(R._confidence(None), 'unknown-confidence')

    def test_training_logs_respect_run_directory_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            atomic_json(parent / 'trainer_state.json', {'log_history': [{'step': 10, 'eval_loss': 2.0}]})
            child = parent / 'unrelated-run'
            child.mkdir()
            self.assertIsNone(training_curve(child)['eval_loss_nats_per_token'])
            self.assertIsNone(training_curve(child)['grad_norm_nonfinite'])
            final = parent / 'final'
            final.mkdir()
            self.assertEqual(training_curve(final)['eval_loss_nats_per_token'], 2.0)

    def test_every_generation_aggregate_is_flat_and_documented(self):
        rows = [sample('A small bird flew over the quiet garden before settling on a branch.'), sample('Yes.')]
        aggregate = M.summarize_generations(rows)
        r = {
            'generation': {
                'greedy_summary': aggregate.copy(),
                'sampled_summary': aggregate.copy(),
                'temperature': 0.8,
                'temperature_sweep': {'1.2': aggregate.copy()},
            }
        }
        summarize_result(r)
        for prefix in ('greedy', 'sampled', 'sweep_t1.2'):
            for key, value in aggregate.items():
                actual = r['summary'][prefix + '_' + key]
                self.assertTrue(actual == value or math.isnan(actual) and math.isnan(value))
        self.assertFalse(undocumented(r['summary']))
        self.assertEqual(describe(['logic_margin_bpb'])['logic_margin_bpb']['unit'], 'bits/byte')

    def test_probe_bpb_and_diagnostics_are_flat(self):
        r = {
            'period_probes': {
                'historical': {'bpb': 1.2, 'ppl': 40, 'sentences_scored': 10},
                'overall': {'bpb': 1.4, 'p10_token_prob': 0.001},
            },
            'logic': {
                'accuracy': 0.6,
                'margin_bpb': 0.01,
                'items_scored': 10,
                'categories': {'causal': {'accuracy': 0.5, 'items_scored': 2}},
            },
            'traps': {'mean_delta_bpb': -0.2, 'min_delta_bpb': -0.5, 'nonpositive_pairs': 2, 'pairs_scored': 6},
            'training_curve': {'eval_loss_nats_per_token': 3.0},
        }
        summarize_result(r)
        s = r['summary']
        self.assertEqual(s['probe_historical_bpb'], 1.2)
        self.assertEqual(s['probe_overall_bpb'], 1.4)
        self.assertEqual(s['trap_nonpositive_pairs'], 2)
        self.assertEqual(s['logic_accuracy'], 0.6)
        self.assertEqual(s['logic_margin_bpb'], 0.01)
        self.assertEqual(s['trainer_eval_loss_nats_per_token'], 3.0)
        self.assertFalse(undocumented(s))
        once = copy.deepcopy(r)
        summarize_result(r)
        self.assertEqual(r, once)

    def test_summary_rebuilds_from_measurements(self):
        r = model('recomputed', [100, 100])
        r['summary']['scratch'] = 123
        r['prose_records'][0]['bits'] = 300
        summarize_result(r)
        self.assertEqual(r['summary']['prose_bpb'], 2.0)
        self.assertNotIn('scratch', r['summary'])
        self.assertEqual(r['points']['bpb'], M.interp(2.0, M.BPB_LADDER))

    def test_uniform_token_baseline_uses_model_vocab_and_finite_scored_pool(self):
        result = model('uniform baseline', [100, 100])
        result['info'] = {'vocab_size_config': 32768}
        result['tokenizer_stats'] = {'vocab_size': 10, 'bytes_per_token': 999}
        result['prose_records'][0].update(tokens=10, bytes=30)
        result['prose_records'][1].update(tokens=20, bytes=70)
        result['prose_records'].append({**result['prose_records'][0], 'bits': None, 'tokens': 1000})
        summarize_result(result)
        self.assertEqual(result['summary']['prose_uniform_token_bpb'], 4.5)
        self.assertEqual(result['summary']['prose_scored_tokens'], 30)
        self.assertEqual(result['summary']['prose_scored_bytes'], 100)
        self.assertFalse(undocumented(result['summary']))
        for vocab in (None, 0, -1, float('nan'), float('inf')):
            result['info']['vocab_size_config'] = vocab
            summarize_result(result)
            self.assertIsNone(result['summary']['prose_uniform_token_bpb'])
        result['info']['vocab_size_config'] = 32768
        result['prose_records'] = []
        summarize_result(result)
        self.assertIsNone(result['summary']['prose_uniform_token_bpb'])

    def test_choice_producers_feed_summary_directly(self):
        items = [
            ('a', 'context', ' good', ' bad'),
            ('a', 'context', ' good', ' bad'),
            ('b', 'context', ' good', ' bad'),
            ('b', 'context', ' good', ' bad'),
        ]
        with patch.object(M, 'span_bpb', side_effect=[1.0, 2.0, 3.0, 2.0, 1.0, 1.0, float('nan'), 1.0]):
            logic = M.run_logic(None, None, items)
        self.assertEqual(logic['accuracy'], 1 / 3)
        self.assertEqual(logic['margin_bpb'], 0.0)
        self.assertEqual(logic['items_scored'], 3)
        self.assertEqual(logic['items_correct'], 1)
        self.assertEqual(logic['items_skipped'], 1)
        self.assertEqual(logic['categories']['a'], {'accuracy': 0.5, 'items_scored': 2})
        self.assertEqual(len(logic['items']), len(items))
        self.assertEqual([row['correct'] for row in logic['items']], [True, False, False, None])
        self.assertEqual([row['margin_bpb'] for row in logic['items']], [1.0, -1.0, 0.0, None])
        self.assertEqual(
            logic['items'][0],
            {
                'category': 'a',
                'context': 'context',
                'good_continuation': ' good',
                'bad_continuation': ' bad',
                'good_bpb': 1.0,
                'bad_bpb': 2.0,
                'margin_bpb': 1.0,
                'correct': True,
            },
        )
        self.assertIsNone(logic['items'][-1]['good_bpb'])
        self.assertEqual(logic['items'][-1]['bad_bpb'], 1.0)
        json.dumps(logic['items'], allow_nan=False)
        pairs = [('modern one', 'one', 'period two', 'two')] * 3
        with patch.object(M, 'span_bpb', side_effect=[3.0, 1.0, 1.0, 2.0, float('inf'), 1.0]):
            traps = M.run_traps(None, None, pairs)
        self.assertEqual(traps['mean_delta_bpb'], 0.5)
        self.assertEqual(traps['min_delta_bpb'], -1.0)
        self.assertEqual(traps['nonpositive_pairs'], 1)
        self.assertEqual(traps['pairs_skipped'], 1)
        result = {'logic': logic, 'traps': traps}
        summarize_result(result)
        self.assertEqual(result['summary']['logic_items_scored'], 3)
        self.assertEqual(result['summary']['trap_pairs_scored'], 2)
        self.assertFalse(undocumented(result['summary']))

    def test_logic_retains_all_skipped_pairs_without_zero_accuracy(self):
        items = [('category', 'context', ' good', ' bad')] * 2
        with patch.object(M, 'span_bpb', side_effect=[float('inf'), 1.0, 2.0, float('nan')]):
            logic = M.run_logic(None, None, items)
        self.assertEqual(logic['items_scored'], 0)
        self.assertEqual(logic['items_skipped'], 2)
        self.assertTrue(math.isnan(logic['accuracy']))
        self.assertEqual(logic['categories'], {})
        self.assertTrue(all(row['correct'] is None for row in logic['items']))
        self.assertEqual(logic['items'][0]['bad_bpb'], 1.0)
        self.assertEqual(logic['items'][1]['good_bpb'], 2.0)
        json.dumps(logic['items'], allow_nan=False)


class ProtocolTests(unittest.TestCase):
    def test_output_and_recovery_file_lifecycle(self):
        for case in ('success', 'slim', 'rerun', 'model_failure', 'report_failure', 'interrupted', 'mixed_failure'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                checkpoints = [root / 'final']
                if case == 'mixed_failure':
                    checkpoints.append(root / 'second')
                for checkpoint in checkpoints:
                    checkpoint.mkdir()
                out_json = root / 'eval-test.json'
                partial_path = root / 'eval-test.partial.json'
                prev_path = root / 'eval-test.prev.json'
                if case not in ('success', 'slim'):
                    atomic_json(out_json, {'schema_version': SCHEMA_VERSION, 'settings': {}, 'results': [model('previous', [200, 200])]})
                    previous_bytes = out_json.read_bytes()
                argv = [
                    str(checkpoints[0]),
                    '--out',
                    str(out_json),
                    '--device',
                    'cpu',
                    '--docs',
                    '2',
                    '--chat-docs',
                    '0',
                    '--gen-mode',
                    'none',
                    '--bootstrap',
                    '100',
                ]
                if case == 'slim':
                    argv.append('--slim')

                def evaluate(checkpoint, *args, snapshot, **kwargs):
                    result = model(checkpoint.name, [100, 100])
                    snapshot('prose', result)
                    self.assertTrue(partial_path.exists())
                    if case == 'interrupted':
                        raise KeyboardInterrupt
                    if case == 'model_failure' or (case == 'mixed_failure' and checkpoint == checkpoints[-1]):
                        raise RuntimeError('evaluation failed')
                    return result

                render = R.render_report

                def render_with_snapshot(payload):
                    self.assertTrue(partial_path.exists(), 'snapshot must survive until the report is saved')
                    if case not in ('success', 'slim'):
                        self.assertEqual(prev_path.read_bytes(), previous_bytes)
                    if case == 'report_failure':
                        raise OSError('report write failed')
                    return render(payload)

                error = {'model_failure': SystemExit, 'report_failure': OSError, 'interrupted': KeyboardInterrupt}.get(case)
                with (
                    patch.object(E, 'resolve_targets', return_value=SimpleNamespace(checkpoints=checkpoints, skipped=[])),
                    patch.object(E, 'resolve_tokenizer', return_value=checkpoints[0]),
                    patch.object(E, 'model_fingerprint', return_value='same-weights'),
                    patch.object(E, 'evaluate_checkpoint', side_effect=evaluate),
                    patch.object(E.R, 'render_report', side_effect=render_with_snapshot),
                    patch.object(E, 'DEFAULT_RESULTS_DIR', root / 'index'),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(error) if error else nullcontext(),
                ):
                    E.main(argv)

                succeeded = case in ('success', 'slim', 'rerun')
                self.assertEqual(partial_path.exists(), not succeeded)
                self.assertEqual(prev_path.exists(), not succeeded)
                if succeeded:
                    self.assertEqual({p.name for p in root.rglob('*') if p.is_file()}, {'eval-test.json', 'eval-test.md'})
                else:
                    self.assertEqual(prev_path.read_bytes(), previous_bytes)
                if succeeded or case == 'mixed_failure':
                    self.assertTrue(out_json.with_suffix('.md').is_file())
                    payload = json.loads(out_json.read_text())
                    result = payload['results'][0]
                    self.assertEqual(result['summary']['prose_bpb'], 1.0)
                    self.assertEqual(set(payload['metric_guide']), set(result['summary']))
                    if case != 'slim':
                        self.assertEqual(result['prose_records'], model('fixture', [100, 100])['prose_records'])
                if case == 'model_failure':
                    self.assertEqual(out_json.read_bytes(), previous_bytes)
                if case == 'slim':
                    self.assertTrue(payload['slim'])
                    self.assertNotIn('prose_records', result)

    def test_collection_writes_only_json_and_markdown(self):
        for explicit_output in (False, True):
            with self.subTest(explicit_output=explicit_output), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / 'input' / 'eval-models.json'
                records = [model('a', [100, 100]), model('b', [200, 200])]
                atomic_json(source, {'schema_version': SCHEMA_VERSION, 'results': records})
                source_bytes = source.read_bytes()
                output_dir = root / 'output'
                out_json = output_dir / ('comparison.json' if explicit_output else 'eval-collected.json')
                argv = ['--collect', str(source.parent), '--bootstrap', '100']
                argv += ['--out', str(out_json)] if explicit_output else ['--results-dir', str(output_dir)]
                with redirect_stdout(io.StringIO()), patch.object(E, 'load_model_and_tokenizer') as load_model:
                    E.main(argv)
                load_model.assert_not_called()
                self.assertEqual(
                    {p.relative_to(root) for p in root.rglob('*') if p.is_file()},
                    {source.relative_to(root), out_json.relative_to(root), out_json.with_suffix('.md').relative_to(root)},
                )
                self.assertEqual(source.read_bytes(), source_bytes)
                payload = json.loads(out_json.read_text())
                self.assertEqual(len(payload['rankings']['prose']), 2)
                for result, expected in zip(payload['results'], records):
                    self.assertEqual(result['prose_records'], expected['prose_records'])
                    self.assertEqual(result['summary']['prose_bpb'], expected['summary']['prose_bpb'])
                    self.assertEqual(set(payload['metric_guide']), set(result['summary']))
                self.assertEqual(out_json.with_suffix('.md').read_text(), R.render_report(payload))

    def test_matching_cache_does_not_load_model_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'final'
            checkpoint.mkdir()
            fixture = model('fixture', [100, 100])
            summarize_result(fixture)
            argv = [
                str(checkpoint),
                '--out',
                str(root / 'eval-test.json'),
                '--device',
                'cpu',
                '--docs',
                '2',
                '--chat-docs',
                '0',
                '--gen-mode',
                'none',
                '--bootstrap',
                '100',
            ]
            output = io.StringIO()
            with (
                patch.object(E, 'resolve_targets', return_value=SimpleNamespace(checkpoints=[checkpoint], skipped=[])),
                patch.object(E, 'resolve_tokenizer', return_value=checkpoint),
                patch.object(E, 'model_fingerprint', return_value='same-weights'),
                patch.object(E, 'evaluate_checkpoint', return_value=fixture) as evaluate,
                patch.object(E, 'DEFAULT_RESULTS_DIR', root / 'index'),
                redirect_stdout(output),
            ):
                E.main(argv)
                partial_path = root / 'eval-test.partial.json'
                atomic_json(partial_path, {'checkpoint': str(checkpoint), 'result': fixture})
                E.main(argv)
            evaluate.assert_called_once()
            self.assertIn('[cached]', output.getvalue())
            self.assertFalse(partial_path.exists())
            self.assertEqual({p.name for p in root.rglob('*') if p.is_file()}, {'eval-test.json', 'eval-test.md'})

    def test_composite_groups_require_matching_chat_context_and_template(self):
        a = model('a', [100, 100])
        aggregate = M.summarize_generations(
            [sample('Yes.'), sample('A small bird flew over the quiet garden before settling on a branch.')]
        )
        a.update(
            chat_records=[{**row, 'target_truncated': False, 'input_text_sha256': row['scored_text_sha256']} for row in a['prose_records']],
            logic={'accuracy': 0.6, 'items_scored': 40},
            generation={'temperature': 0.8, 'greedy_summary': aggregate, 'sampled_summary': aggregate},
        )
        summarize_result(a)
        for change in ('context', 'template', 'incomplete'):
            b = copy.deepcopy(a)
            b['label'] = 'b'
            if change == 'context':
                b['chat_records'][0]['input_text_sha256'] = 'different context'
            if change == 'template':
                b['generation']['chat_template_sha256'] = 'different template'
            if change == 'incomplete':
                b['logic']['accuracy'] = None
            summarize_result(b)
            rows = rank_by_composite([a, b])
            self.assertEqual(len(rows), 1 if change == 'incomplete' else 2)
            self.assertTrue(all(r['rank'] == 1 for r in rows))

    def test_scored_spans_windows_and_scorers_split_groups(self):
        a = model('a', [100, 100])
        a['evaluation_settings']['scoring_code_sha256'] = 'x'
        for change in ('bytes', 'hash', 'window', 'code', 'coverage'):
            with self.subTest(change=change):
                b = copy.deepcopy(a)
                b['label'] = 'b'
                if change == 'bytes':
                    b['prose_records'][0]['bytes'] += 1
                if change == 'hash':
                    b['prose_records'][0]['scored_text_sha256'] = 'different'
                if change == 'window':
                    b['evaluation_settings']['max_tokens'] = 256
                if change == 'code':
                    b['evaluation_settings']['scoring_code_sha256'] = 'y'
                if change == 'coverage':
                    b['prose_records'][0]['bits'] = None
                rows = rank_by_prose([a, b], parse_args(['--bootstrap', '100']))
                self.assertEqual(len({r['comparison_group'] for r in rows}), 2)

    def test_duplicate_document_ids_do_not_silently_collapse(self):
        a = model('a', [100, 100], ['same', 'same'])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(rank_by_prose([a], parse_args([])), [])
        self.assertTrue(math.isnan(M.paired_bootstrap(a['prose_records'], a['prose_records'], 100, 0.9, 1)['delta']))

    def test_collection_retains_protocols_and_distinguishes_same_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r = model('same-model', [100, 100])
            for i in (128, 256):
                r['evaluation_settings']['max_tokens'] = i
                atomic_json(root / f'eval-{i}.json', {'schema_version': SCHEMA_VERSION, 'results': [r]})
            r['evaluation_settings']['max_tokens'] = 128
            atomic_json(root / 'eval-duplicate.json', {'schema_version': SCHEMA_VERSION, 'results': [r]})
            results = collect_results([root])
            self.assertEqual(len(results), 2)
            self.assertEqual(len({r['label'] for r in results}), 2)
            self.assertEqual({r['evaluation_settings']['max_tokens'] for r in results}, {128, 256})
            self.assertTrue(all(r['evaluation_environment']['dtype'] == 'float32' for r in results))

    def test_cache_tracks_skips_template_and_prompt_contents(self):
        args = parse_args(['--device', 'cpu'])
        args.generation_modes, args.temp_sweep = ['greedy'], []
        base = settings_fingerprint(args, 'prose', 'chat')
        for flag in ('chat', 'skip_probes', 'skip_embeddings'):
            other = copy.deepcopy(args)
            setattr(other, flag, True)
            self.assertNotEqual(base, settings_fingerprint(other, 'prose', 'chat'))
        with patch('eval.__main__.PROBE_SENTENCES', ['changed text, same code']):
            self.assertNotEqual(base, settings_fingerprint(args, 'prose', 'chat'))

    def test_tokenizer_settings_invalidate_cache_but_eval_output_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / 'config.json', {'model_type': 'llama'})
            atomic_json(root / 'tokenizer_config.json', {'bos_token': 'a'})
            before = model_fingerprint(root, root)
            atomic_json(root / 'eval-test.json', {'unrelated': 'result'})
            self.assertEqual(before, model_fingerprint(root, root))
            atomic_json(root / 'tokenizer_config.json', {'bos_token': 'b'})
            self.assertNotEqual(before, model_fingerprint(root, root))

    def test_invalid_bootstrap_options_are_rejected(self):
        for options in (['--bootstrap', '0'], ['--confidence', '1'], ['--equivalence', '-1']):
            args = parse_args([])
            key = options[0].removeprefix('--')
            setattr(args, key, int(options[1]) if key == 'bootstrap' else float(options[1]))
            with self.assertRaises(ValueError):
                rank_by_prose([], args)


if __name__ == '__main__':
    unittest.main()


class OffsetByteTokenizer(ByteTokenizer):
    """Byte-per-token tokenizer that also reports character offsets, for chat scoring."""

    is_fast = True

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        ids = [1] * len(text)
        if add_special_tokens and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        encoded = SimpleNamespace(input_ids=ids)
        if return_offsets_mapping:
            offset = 1 if (add_special_tokens and self.bos_token_id is not None) else 0
            encoded.offset_mapping = [(0, 0)] * offset + [(i, i + 1) for i in range(len(text))]
        return encoded


class ChatScoringTests(unittest.TestCase):
    def _chat(self, bos, context='ctx ', target='target text'):
        tokenizer, net = OffsetByteTokenizer(bos), UniformModel(bos)
        return M.score_chat_records(tokenizer, net, [ChatItem('c', context, target, 'A')], 16, 0)

    def test_a_model_without_bos_or_eos_still_gets_chat_records(self):
        """The prose scorer falls back to no prefix; chat used to return nothing at all."""
        with_bos, skipped_bos = self._chat(0)
        without, skipped_none = self._chat(None)
        self.assertEqual((skipped_bos, skipped_none), (0, 0))
        self.assertEqual(len(with_bos), 1)
        self.assertEqual(len(without), 1, 'chat suite silently vanished for a model with no BOS/EOS')
        # Same target span, same accounting; only the prepended prefix differs.
        self.assertEqual(without[0]['tokens'], with_bos[0]['tokens'])
        self.assertEqual(without[0]['bytes'], with_bos[0]['bytes'])
        self.assertEqual(without[0]['scored_text_sha256'], with_bos[0]['scored_text_sha256'])
        self.assertAlmostEqual(without[0]['bits'], with_bos[0]['bits'], places=5)

    def test_no_prefix_and_no_context_drops_only_the_unpredictable_first_token(self):
        records, skipped = self._chat(None, context='', target='target text')
        self.assertEqual(skipped, 0)
        # Without a prefix nothing predicts the very first token, so it is not scored.
        self.assertEqual(records[0]['tokens'], len('target text') - 1)
        self.assertTrue(records[0]['target_truncated'])

    def test_items_with_no_scoreable_target_are_counted_not_dropped(self):
        tokenizer, net = OffsetByteTokenizer(0), UniformModel(0)
        records, skipped = M.score_chat_records(tokenizer, net, [ChatItem('c', 'ctx', '', 'A')], 16, 0)
        self.assertEqual((records, skipped), ([], 1))

    def test_skipped_counts_reach_the_flat_summary(self):
        result = {'prose_docs_skipped': 3, 'chat_target_docs_skipped': 2, 'logic': {}}
        summarize_result(result)
        self.assertEqual(result['summary']['prose_docs_skipped'], 3)
        self.assertEqual(result['summary']['chat_target_docs_skipped'], 2)
        self.assertEqual(undocumented(result['summary'].keys()), [])


class SmallSetUncertaintyTests(unittest.TestCase):
    def test_wilson_interval_brackets_the_estimate_and_clamps_at_the_ends(self):
        low, high = M.wilson_interval(24, 40)
        self.assertAlmostEqual(low, 0.44596, places=4)
        self.assertAlmostEqual(high, 0.73652, places=4)
        self.assertLess(low, 0.6)
        self.assertGreater(high, 0.6)
        self.assertEqual(M.wilson_interval(0, 10)[0], 0.0)
        self.assertAlmostEqual(M.wilson_interval(10, 10)[1], 1.0)
        self.assertTrue(all(math.isnan(v) for v in M.wilson_interval(0, 0)))
        # A smaller set must produce a wider interval.
        narrow, wide = M.wilson_interval(60, 100), M.wilson_interval(6, 10)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_logic_reports_an_accuracy_interval_in_the_summary(self):
        items = [('cat', 'The ship ', 'sailed.', 'sailed.')] * 4
        with patch.object(M, 'span_bpb', side_effect=[1.0, 2.0] * 4):
            result = {'logic': M.run_logic(None, None, items)}
        summarize_result(result)
        s = result['summary']
        self.assertEqual(s['logic_accuracy'], 1.0)
        self.assertEqual(s['logic_accuracy_ci_confidence'], 0.95)
        self.assertLess(s['logic_accuracy_ci_low'], 1.0)
        self.assertEqual(s['logic_accuracy_ci_high'], 1.0)
        self.assertEqual(undocumented(s.keys()), [])

    def test_traps_skip_malformed_pairs_instead_of_aborting_the_checkpoint(self):
        good = ('a period phrase here', 'period phrase', 'a modern phrase here', 'modern phrase')
        broken = ('a period phrase here', 'absent phrase', 'a modern phrase here', 'modern phrase')
        with patch.object(M, 'span_bpb', side_effect=[1.5, 1.0, 2.5, 1.0]):
            traps = M.run_traps(None, None, [good, broken, good])
        self.assertEqual(traps['pairs_scored'], 2)
        self.assertEqual(traps['pairs_skipped'], 1)
        self.assertAlmostEqual(traps['mean_delta_bpb'], 1.0)
        self.assertAlmostEqual(traps['mean_delta_stderr_bpb'], 0.5)
        result = {'traps': traps}
        summarize_result(result)
        self.assertAlmostEqual(result['summary']['trap_mean_delta_stderr_bpb'], 0.5)
        self.assertEqual(undocumented(result['summary'].keys()), [])

    def test_single_scored_pair_has_no_standard_error(self):
        pair = ('a period phrase here', 'period phrase', 'a modern phrase here', 'modern phrase')
        with patch.object(M, 'span_bpb', side_effect=[1.5, 1.0]):
            traps = M.run_traps(None, None, [pair])
        self.assertTrue(math.isnan(traps['mean_delta_stderr_bpb']))


class PositionalPoolTests(unittest.TestCase):
    def test_early_and_late_bpb_use_the_same_documents(self):
        long_doc, short_doc = model('x', [100])['prose_records'][0], model('y', [40])['prose_records'][0]
        # A document with no second half must be in NEITHER position, not just `late`.
        short_doc.update(id='short', late_bits=0.0, late_bytes=0, early_bits=40.0, early_bytes=50)
        result = {'prose_records': [long_doc, short_doc]}
        summarize_result(result)
        s = result['summary']
        self.assertEqual(s['prose_bpb_position_docs'], 1)
        self.assertAlmostEqual(s['prose_bpb_early'], 1.0)
        self.assertAlmostEqual(s['prose_bpb_late'], 1.0)
        self.assertEqual(undocumented(s.keys()), [])


class ScoringIdentityScopeTests(unittest.TestCase):
    def test_only_modules_that_change_a_number_are_in_the_scoring_identity(self):
        self.assertEqual(set(E.SCORING_MODULES), {'helpers.py', 'measurements.py', 'metrics.py', 'prompts.py'})
        package = Path(E.__file__).parent
        for name in E.SCORING_MODULES:
            self.assertTrue((package / name).is_file(), name)
        # Rendering, documentation and the standalone side scripts must NOT be here:
        # this digest is part of prose_comparison_key, so including them would split
        # comparison groups on a cosmetic edit.
        for name in ('report.py', 'metric_guide.py', 'memorize.py', 'probe_memorization.py', '__main__.py'):
            self.assertNotIn(name, E.SCORING_MODULES)

    def test_a_cosmetic_edit_keeps_results_in_the_same_comparison_group(self):
        args = parse_args(['--device', 'cpu'])
        args.generation_modes, args.temp_sweep = ['greedy'], []
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / 'eval'
            package.mkdir()
            for name in (*E.SCORING_MODULES, 'report.py', 'memorize.py'):
                (package / name).write_text(f'# {name}\n')
            with patch.object(E, '__file__', str(package / '__main__.py')):
                base = settings_fingerprint(args, 'prose', 'chat')
                (package / 'report.py').write_text('# report.py\n# cosmetic\n')
                (package / 'memorize.py').write_text('# memorize.py\n# unrelated script\n')
                self.assertEqual(settings_fingerprint(args, 'prose', 'chat')['scoring_code_sha256'], base['scoring_code_sha256'])
                (package / 'metrics.py').write_text('# metrics.py\n# a real scoring change\n')
                self.assertNotEqual(settings_fingerprint(args, 'prose', 'chat')['scoring_code_sha256'], base['scoring_code_sha256'])


class ChatRankingTests(unittest.TestCase):
    @staticmethod
    def _with_chat(label, bits):
        result = model(label, [100] * len(bits))
        result['chat_records'] = [
            {
                'id': str(i),
                'split': 'A',
                'bits': value,
                'bytes': 100,
                'tokens': 20,
                'truncated': False,
                'target_truncated': False,
                'scored_text_sha256': stable_hash(f'target {i}'.encode()),
                'input_text_sha256': stable_hash(f'context {i}'.encode()),
            }
            for i, value in enumerate(bits)
        ]
        result['evaluation_settings']['chat_max_tokens'] = 128
        summarize_result(result)
        return result

    def test_chat_targets_get_the_same_paired_treatment_as_prose(self):
        args = parse_args(['--bootstrap', '200'])
        results = [self._with_chat('leader', [80, 90, 95, 95]), self._with_chat('worse', [130, 140, 145, 145])]
        rows = rank_by_chat(results, args)
        self.assertEqual([row['metric'] for row in rows], ['chat_target_bpb'] * 2)
        self.assertEqual([row['comparison_to_leader'] for row in rows], ['leader', 'worse'])
        self.assertEqual(rows[0]['label'], 'leader')
        summaries = {r['label']: r['summary'] for r in results}
        self.assertEqual(summaries['worse']['chat_target_comparison_to_leader'], 'worse')
        self.assertLess(summaries['leader']['chat_target_bpb_ci_low'], summaries['leader']['chat_target_bpb'])
        self.assertGreater(summaries['leader']['chat_target_bpb_ci_high'], summaries['leader']['chat_target_bpb'])
        self.assertAlmostEqual(summaries['worse']['chat_target_delta_vs_leader_bpb'], 0.5)
        # Prose and chat verdicts live side by side without overwriting each other.
        # Prose bits are identical here, so the two models tie on prose and differ on chat.
        rank_by_prose(results, args)
        worse = {r['label']: r['summary'] for r in results}['worse']
        self.assertEqual(worse['chat_target_comparison_to_leader'], 'worse')
        self.assertEqual(worse['prose_comparison_to_leader'], 'equivalent')
        self.assertEqual(undocumented(worse.keys()), [])

    def test_the_leader_reports_no_fraction_against_itself(self):
        rows = rank_by_prose([model('a', [100] * 4), model('b', [110] * 4)], parse_args(['--bootstrap', '200']))
        self.assertIsNone(rows[0]['bootstrap_fraction_lower_than_leader'])
        self.assertIsInstance(rows[1]['bootstrap_fraction_lower_than_leader'], float)

    def test_duplicate_ids_say_why_they_are_unranked(self):
        duplicate = model('dup', [100, 100], ['same', 'same'])
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(rank_by_prose([duplicate], parse_args(['--bootstrap', '100'])), [])
        self.assertIn('duplicate document IDs', out.getvalue())
        s = duplicate['summary']
        self.assertEqual(s['prose_comparison_to_leader'], 'unavailable')
        self.assertIn('duplicate document IDs', s['prose_comparison_unavailable_reason'])
        self.assertEqual(undocumented(s.keys()), [])

    def test_every_candidate_is_compared_on_the_same_resampled_documents(self):
        args = parse_args(['--bootstrap', '200'])
        results = [model('a', [100] * 6), model('b', [110] * 6), model('c', [120] * 6)]
        rows = {row['label']: row for row in rank_by_prose(results, args)}
        # Identical per-document ratios, so a shared resample must give identical
        # bounds; a per-rank seed would make these differ.
        self.assertAlmostEqual(rows['b']['delta_ci_low'], rows['b']['delta_ci_high'], places=9)
        self.assertAlmostEqual(rows['c']['delta_ci_low'], rows['c']['delta_ci_high'], places=9)
        self.assertAlmostEqual(rows['c']['delta_vs_leader'], 2 * rows['b']['delta_vs_leader'], places=9)


class CheckpointOrderTests(unittest.TestCase):
    def test_checkpoints_are_ordered_by_step_not_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('checkpoint-2', 'checkpoint-9', 'checkpoint-10', 'checkpoint-100', 'checkpoint-49501'):
                (root / name).mkdir()
                atomic_json(root / name / 'config.json', {'model_type': 'llama'})
                (root / name / 'model.safetensors').write_bytes(b'')
            found = E.resolve_targets([root], include_checkpoints=True).checkpoints
            self.assertEqual(
                [p.name for p in found],
                ['checkpoint-2', 'checkpoint-9', 'checkpoint-10', 'checkpoint-100', 'checkpoint-49501'],
            )


class LogicItemFairnessTests(unittest.TestCase):
    """Enforce the rules in the LOGIC_ITEMS header, so additions cannot regress them.

    BPB is per BYTE. A wrong option that is longer than the right one spreads its
    error over more text and can win on length alone, which is exactly what these
    checks exist to prevent.
    """

    @classmethod
    def setUpClass(cls):
        from eval.prompts import LOGIC_ITEMS

        cls.items = LOGIC_ITEMS

    def test_options_are_length_matched(self):
        for category, context, good, bad in self.items:
            with self.subTest(context=context):
                delta = len(bad.encode()) - len(good.encode())
                self.assertLessEqual(abs(delta), 2, f'{category}: options differ by {delta} bytes')

    def test_length_bias_does_not_favour_either_option_overall(self):
        deltas = [len(bad.encode()) - len(good.encode()) for _, _, good, bad in self.items]
        longer = sum(d > 0 for d in deltas)
        shorter = sum(d < 0 for d in deltas)
        self.assertAlmostEqual(sum(deltas) / len(deltas), 0.0, delta=0.5)
        # Neither direction may dominate; a lopsided set biases accuracy either way.
        self.assertLessEqual(abs(longer - shorter), max(3, len(self.items) // 10))

    @staticmethod
    def _words(text):
        return [re.sub(r"[^a-z']", '', word) for word in text.lower().split()]

    def test_options_are_a_minimal_edit_or_a_pure_permutation(self):
        """Options either swap a few words in place, or reorder the very same words.

        Both forms hold vocabulary and length fixed, so the BPB gap comes from the
        swap. Options differing throughout would measure which reads more fluently.
        """
        for category, context, good, bad in self.items:
            with self.subTest(context=context):
                good_words, bad_words = self._words(good), self._words(bad)
                if collections.Counter(good_words) == collections.Counter(bad_words):
                    continue  # a pure reordering, e.g. the two names in a coref item
                self.assertEqual(len(good_words), len(bad_words), f'{category}: options have different word counts')
                swapped = sum(a != b for a, b in zip(good_words, bad_words))
                self.assertLessEqual(swapped, 3, f'{category}: {swapped} words differ, so the item measures fluency too')

    def test_most_items_turn_on_a_single_word(self):
        """The cleanest item form; keep it dominant so the metric stays interpretable."""
        single = 0
        for _, _, good, bad in self.items:
            good_words, bad_words = self._words(good), self._words(bad)
            if len(good_words) == len(bad_words) and sum(a != b for a, b in zip(good_words, bad_words)) == 1:
                single += 1
        self.assertGreater(single / len(self.items), 0.60)

    def test_items_are_well_formed_and_distinct(self):
        contexts = [context for _, context, _, _ in self.items]
        self.assertEqual(len(set(contexts)), len(contexts), 'duplicate context')
        for category, context, good, bad in self.items:
            with self.subTest(context=context):
                self.assertNotEqual(good, bad)
                # span_bpb cuts the context at its last non-space character, so each
                # option must own its own leading space.
                self.assertTrue(good.startswith(' ') and bad.startswith(' '))
                self.assertFalse(context.endswith(' '))
                self.assertNotIn('’', context + good + bad, 'use a straight apostrophe: the corpus and tokenizer prefer it')

    def test_every_category_has_enough_items_to_report(self):
        counts = collections.Counter(category for category, _, _, _ in self.items)
        self.assertGreaterEqual(len(self.items), 40)
        for category, count in counts.items():
            self.assertGreaterEqual(count, 6, f'{category} is too small to report an accuracy for')

    def test_the_composite_ceiling_matches_the_test_ceiling(self):
        self.assertEqual(M.logic_points(0.5), 0.0)
        self.assertEqual(M.logic_points(1.0), 100.0)
        self.assertEqual(M.logic_points(0.0), 0.0)
        # A model near the current state of the art here must still have headroom.
        self.assertLess(M.logic_points(0.89), 90.0)

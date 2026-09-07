"""Regression tests for surface flags and practical BPB equivalence."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from eval import SCHEMA_VERSION
from eval import metrics as M
from eval import report as R
from eval.__main__ import main, parse_args, rank_by_prose
from eval.helpers import atomic_json, stable_hash
from eval.measurements import summarize_result
from eval.metric_guide import describe, undocumented


def sample(text):
    return {'prompt': 'Continue:', 'continuation': text, **M.text_stats('Continue:', text)}


def model(label, bits, ids=None):
    records = []
    for i, value in enumerate(bits):
        item_id = str(i) if ids is None else ids[i]
        text_hash = stable_hash(('document ' + item_id).ljust(100).encode())
        records.append(
            {
                'id': item_id,
                'split': 'A',
                'bits': value,
                'bytes': 100,
                'tokens': 20,
                'early_bits': value / 2,
                'early_bytes': 50,
                'late_bits': value / 2,
                'late_bytes': 50,
                'truncated': False,
                'prefix_token_used': True,
                'scored_text_sha256': text_hash,
            }
        )
    result = {
        'label': label,
        'checkpoint': label,
        'fingerprint': 'weights-' + label,
        'prose_records': records,
        'evaluation_settings': {'scoring_code_sha256': 'test-scorer', 'max_tokens': 128, 'load_8bit': False},
        'evaluation_environment': {'dtype': 'float32'},
    }
    summarize_result(result)
    return result


class SurfaceTests(unittest.TestCase):
    def test_valid_short_text_explains_flags(self):
        row = sample('may be considered with advantage.')
        self.assertTrue(row['surface_failure'])
        self.assertEqual(row['surface_failure_reasons'], ['short_text', 'back_matter_like'])

    def test_nonsense_can_pass(self):
        row = sample('amber birch cobalt dapple elm fern granite hazel iris juniper kestrel larch.')
        self.assertFalse(row['surface_failure'])
        self.assertEqual(row['surface_failure_reasons'], [])

    def test_union_counts_once_and_report_includes_formatting_only(self):
        formatting = sample('First we went to the station.\nThen we caught the train.\nSoon we arrived in London.')
        short = sample('may be considered with advantage.')
        clean = sample('A small bird flew over the quiet garden before settling on a branch.')
        self.assertFalse(formatting['degenerate'])
        self.assertEqual(formatting['surface_failure_reasons'], ['back_matter_like'])
        rows = [formatting, short, clean]
        summary = M.summarize_generations(rows)
        self.assertEqual(summary['surface_failure_rate'], 2 / 3)
        self.assertEqual(summary['degenerate_rate'], 1 / 3)
        result = {'generation': {'sampled_samples': rows, 'sampled_summary': summary}}
        markdown = '\n'.join(R.render_generation_section(result) + R.render_checkpoint_examples(result))
        self.assertIn('2 of 3', markdown)
        self.assertIn('back_matter_like', markdown)
        self.assertIn('short_text', markdown)

    def test_unknown_rate_is_not_zero(self):
        result = {'generation': {'sampled_summary': {'mean_words': 10}}}
        markdown = '\n'.join(R.render_generation_section(result))
        self.assertIn('| Flagged completions (`surface_failure_rate`) | — |', markdown)
        self.assertEqual(M.summarize_generations([]), {})

    def test_flat_summary_and_guide_for_each_mode_and_sweep(self):
        for mode in ('greedy', 'sampled'):
            result = {
                'generation': {
                    'temperature': 0.8,
                    f'{mode}_summary': {'surface_failure_rate': 0.5},
                    'temperature_sweep': {'1.2': {'surface_failure_rate': 0.25}},
                }
            }
            with redirect_stdout(io.StringIO()):
                summarize_result(result)
            self.assertEqual(result['summary'][f'{mode}_surface_failure_rate'], 0.5)
            self.assertEqual(result['summary']['sweep_t1.2_surface_failure_rate'], 0.25)
            self.assertFalse(undocumented(result['summary']))
            self.assertEqual(describe([f'{mode}_surface_failure_rate'])[f'{mode}_surface_failure_rate']['unit'], 'fraction')


class ReportTitleTests(unittest.TestCase):
    def test_title_uses_architecture_metadata_not_checkpoint_path(self):
        result = model('experiment/final', [100, 100])
        result.update(
            params_millions=74.72,
            info={
                'model_type': 'llama',
                'num_layers': 8,
                'hidden_size': 768,
                'num_attention_heads': 8,
                'num_key_value_heads': 2,
                'max_position_embeddings': 1024,
                'vocab_size_config': 32768,
                'tie_word_embeddings': True,
            },
        )
        payload = {'target_label': '/full/path/to/experiment/final', 'results': [result]}
        before = copy.deepcopy(payload)
        markdown = R.render_report(payload)
        self.assertEqual(
            markdown.splitlines()[0],
            '# llama · 74.72M params · depth 8 · width 768 · heads 8 · KV 2 · ctx 1024 · vocab 32768 · tied embeddings',
        )
        self.assertIn('- Checkpoint: `experiment/final`', markdown)
        self.assertNotIn(payload['target_label'], markdown)
        self.assertEqual(payload, before)

    def test_missing_metadata_is_not_invented(self):
        for tied, suffix in ((True, ' · tied embeddings'), (False, ' · untied embeddings'), (None, '')):
            with self.subTest(tied=tied):
                result = {'info': {'model_type': 'llama', 'tie_word_embeddings': tied}}
                self.assertEqual(R.model_title(result), 'llama' + suffix)
        self.assertEqual(R.model_title({}), 'Unknown model')
        self.assertEqual(R.model_title({'params_millions': float('nan')}), 'Unknown model')

    def test_comparison_title_and_individual_model_titles(self):
        results = [model('experiment-a/final', [100, 100]), model('experiment-b/final', [200, 200])]
        for i, result in enumerate(results, 1):
            result['info'] = {'model_type': 'llama', 'num_layers': i * 4}
        markdown = R.render_report({'target_label': 'collected: /full/path', 'results': results})
        self.assertEqual(markdown.splitlines()[0], '# Evaluation comparison: 2 checkpoints')
        self.assertIn('# llama · depth 4', markdown)
        self.assertIn('# llama · depth 8', markdown)
        for result in results:
            self.assertIn(f'- Checkpoint: `{result["checkpoint"]}`', markdown)


class ComparisonTests(unittest.TestCase):
    def test_interval_classifications(self):
        cases = [
            (-0.01, 0.01, 'equivalent'),
            (0, 0, 'equivalent'),
            (0.002, 0.009, 'equivalent'),
            (0.011, 0.03, 'worse'),
            (-0.03, -0.011, 'better'),
            (-0.5, 1.0, 'inconclusive'),
            (0.005, 0.02, 'inconclusive'),
            (-0.02, 0.005, 'inconclusive'),
            (0.01, 0.02, 'inconclusive'),
            (float('nan'), 0.01, 'unavailable'),
            (-0.01, float('inf'), 'unavailable'),
            (None, None, 'unavailable'),
            (0.02, 0.01, 'unavailable'),
        ]
        for low, high, expected in cases:
            with self.subTest(low=low, high=high):
                self.assertEqual(M.classify_bpb_interval(low, high, 0.01), expected)
        self.assertEqual(M.classify_bpb_interval(0, 0, None), 'unavailable')
        self.assertEqual(M.classify_bpb_interval(0, 0, -0.01), 'unavailable')

    def test_twice_the_loss_is_not_equivalent_when_uncertain(self):
        args = parse_args([])
        rows = rank_by_prose([model('leader', [100] * 4), model('candidate', [10, 10, 10, 770])], args)
        candidate = rows[1]
        self.assertEqual(candidate['bpb'], 2.0)
        self.assertEqual(candidate['comparison_to_leader'], 'inconclusive')
        self.assertFalse(candidate['equivalent_to_leader'])
        self.assertLess(candidate['delta_ci_low'], 0)
        self.assertGreater(candidate['delta_ci_high'], candidate['equivalence_margin_bpb'])

    def test_equivalent_worse_and_confidence(self):
        args = parse_args(['--confidence', '.9', '--bootstrap', '200'])
        results = [model('leader', [100] * 4), model('close', [100.05] * 4), model('worse', [110] * 4)]
        rows = rank_by_prose(results, args)
        self.assertEqual([row['comparison_to_leader'] for row in rows], ['leader', 'equivalent', 'worse'])
        self.assertTrue(rows[1]['equivalent_to_leader'])
        self.assertFalse(rows[2]['equivalent_to_leader'])
        self.assertEqual(rows[1]['equivalence_margin_bpb'], 0.001)
        markdown = R.render_report({'results': results, 'rankings': {'prose': rows}, 'bootstrap_confidence': 0.9})
        self.assertIn('90% CI', markdown)
        self.assertIn('90% paired confidence intervals', markdown)
        self.assertIn('| equivalent |', markdown)
        self.assertIn('| worse |', markdown)

    def test_unpaired_documents_are_not_ranked_against_each_other(self):
        results = [model('leader', [100, 100]), model('other', [110, 110], ['x', 'y'])]
        rows = rank_by_prose(results, parse_args(['--bootstrap', '200']))
        self.assertEqual(len({r['comparison_group'] for r in rows}), 2)
        self.assertTrue(all(r['comparison_to_leader'] == 'leader' for r in rows))
        self.assertTrue(all(r['leader_label'] == r['label'] for r in rows))

    def test_report_reads_results_without_mutating_them(self):
        results = [model('leader', [100] * 4), model('candidate', [10, 10, 10, 770])]
        rows = rank_by_prose(results, parse_args(['--bootstrap', '200']))
        payload = {'results': results, 'rankings': {'prose': rows}, 'bootstrap_confidence': 0.95}
        snapshot = copy.deepcopy(payload)
        markdown = R.render_report(payload)
        self.assertEqual(payload, snapshot)
        self.assertIn('| inconclusive |', markdown)

    def test_report_cli_writes_markdown_without_modifying_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'eval-test.json'
            results = [model('leader', [100] * 4), model('candidate', [10, 10, 10, 770])]
            rows = rank_by_prose(results, parse_args(['--bootstrap', '200']))
            atomic_json(
                path, {'schema_version': SCHEMA_VERSION, 'results': results, 'rankings': {'prose': rows}, 'bootstrap_confidence': 0.95}
            )
            snapshot = path.read_bytes()
            with redirect_stdout(io.StringIO()):
                main(['--render-report', str(path)])
            self.assertEqual(path.read_bytes(), snapshot)
            self.assertIn('| inconclusive |', path.with_suffix('.md').read_text())
            self.assertEqual(json.loads(snapshot)['results'][1]['summary']['prose_comparison_to_leader'], 'inconclusive')


if __name__ == '__main__':
    unittest.main()

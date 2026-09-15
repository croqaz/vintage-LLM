"""Human-readable report selection, measured values, escaping and missing data."""

import copy
import unittest
from unittest.mock import patch

from eval import metrics as M
from eval import report as R
from eval.measurements import summarize_result


def completion(prompt, text=None, flagged=False):
    return {
        'prompt': prompt,
        'continuation': text if text is not None else 'A measured continuation.',
        'surface_failure': flagged,
        'surface_failure_reasons': None if flagged is None else ['repeat_loop'] if flagged else [],
        'distinct_2': 0.9,
        'echo_rate': 0.05,
        'longest_loop_words': 12 if flagged else 0,
    }


class GenerationReportTests(unittest.TestCase):
    def test_gallery_contains_unflagged_and_flagged_samples_without_repeating_prompts(self):
        rows = [completion(f'prompt{i}', flagged=i >= 8) for i in range(11)]
        gen = {'sampled_samples': rows}
        before = copy.deepcopy(gen)
        markdown = '\n'.join(R._generation_examples(gen))
        self.assertIn('3 of 11 saved continuations flagged; 8 unflagged', markdown)
        self.assertIn('Unflagged sampled continuations (4 examples)', markdown)
        self.assertIn('Flagged sampled continuations (2 examples)', markdown)
        self.assertEqual(markdown.count('**Sampled continuation'), 6)
        for i in (0, 2, 4, 7, 8, 10):
            self.assertEqual(markdown.count(f'> prompt{i}\n'), 1)
        self.assertNotIn('> prompt1\n', markdown)
        self.assertIn('not an estimate of typical quality', markdown)
        self.assertEqual(gen, before)

    def test_shared_prompts_match_by_text_not_sample_position_or_outcome(self):
        greedy = [completion(f'prompt{i}', f'greedy output {i}', flagged=i % 2 == 0) for i in range(4)]
        sampled = [completion(f'prompt{i}', f'sampled output {i}', flagged=i % 2 != 0) for i in reversed(range(4))]
        gen = {'greedy_samples': greedy, 'sampled_samples': sampled}
        markdown = '\n'.join(R._generation_examples(gen))
        pairs = markdown.split('\n#### ')[1]
        self.assertIn('Same prompts: greedy vs sampled', pairs)
        for i in (0, 1):
            self.assertIn(f'> greedy output {i}', pairs)
            self.assertIn(f'> sampled output {i}', pairs)
            self.assertEqual(markdown.count(f'> prompt{i}\n'), 1)
        self.assertNotIn('> prompt2', pairs)
        for row in greedy + sampled:
            row['surface_failure'] = not row['surface_failure']
        changed = '\n'.join(R._generation_examples(gen)).split('\n#### ')[1]
        self.assertEqual(
            [line for line in pairs.splitlines() if line.startswith('> prompt')],
            [line for line in changed.splitlines() if line.startswith('> prompt')],
        )

    def test_greedy_only_and_uniform_flag_groups(self):
        for flagged, label in ((False, 'Unflagged'), (True, 'Flagged'), (None, 'Not assessed')):
            with self.subTest(flagged=flagged):
                gen = {'greedy_samples': [completion('prompt', 'retained words', flagged)]}
                markdown = '\n'.join(R.render_checkpoint_examples({'generation': gen}))
                self.assertIn(f'{label} greedy continuations', markdown)
                self.assertIn('> retained words', markdown)
                self.assertNotIn('Same prompts:', markdown)
                self.assertNotIn('Sampling: temperature', markdown)
                if flagged is not False:
                    self.assertNotIn('#### Unflagged', markdown)
                if flagged is None:
                    self.assertIn('0 unflagged; 1 not assessed', markdown)
                    self.assertIn('Flags: not recorded', markdown)

    def test_paragraphs_literal_text_and_explicit_excerpt_truncation(self):
        text = 'First paragraph.\n\nSecond | paragraph <details> [link](https://example.invalid)\n' + 'x' * 1300
        row = completion('PROMPT-ONLY', text)
        markdown = '\n'.join(R._generation_examples({'sampled_samples': [row]}))
        self.assertIn('> First paragraph\\.\n> \n> Second \\| paragraph &lt;details&gt;', markdown)
        self.assertIn('\\[link\\]', markdown)
        self.assertIn('Excerpt truncated at 1200 characters', markdown)
        self.assertNotIn('x' * 1200, markdown)
        self.assertEqual(markdown.count('PROMPT\\-ONLY'), 1)
        self.assertNotRegex(markdown, r'</?(?:details|summary)\b')
        empty = '\n'.join(R._completion(completion('prompt', ''), 'Greedy'))
        self.assertIn('(empty continuation)', empty)
        self.assertNotIn('truncated', empty)
        exact = '\n'.join(R._completion(completion('prompt', 'x' * 1200), 'Greedy'))
        self.assertNotIn('truncated', exact)

    def test_comparison_table_uses_saved_values_and_exposes_each_reason(self):
        gen = {
            'temperature': 0.8,
            'top_p': 0.95,
            'top_k': 40,
            'greedy_summary': {'n_prompts': 4, 'surface_failure_rate': 0.75, 'mean_loop_words': 12.5},
            'sampled_summary': {'n_prompts': 4, 'surface_failure_rate': 0.25, 'mean_loop_words': 2.5},
            'temperature_sweep': {'1.2': {'surface_failure_rate': 0.5}, '0.8': {'surface_failure_rate': 0.25}},
        }
        r = {'generation': gen, 'evaluation_settings': {'seed': 42, 'generation_tokens': 256}}
        markdown = '\n'.join(R.render_generation_section(r))
        self.assertIn('| Flagged completions (`surface_failure_rate`) | 75.0% | 25.0% |', markdown)
        self.assertIn('| Mean longest loop (words) (`mean_loop_words`) | 12.5 | 2.5 |', markdown)
        self.assertIn('Generation budget: 256 new tokens; seed: 42', markdown)
        self.assertIn('temperature 0.80, top-p 0.95, top-k 40', markdown)
        self.assertIn('| Short-text or repetition flags (`degenerate_rate`) | — | — |', markdown)
        for reason in M.SURFACE_REASONS:
            self.assertIn(f'| `{reason}_rate` | — | — |', markdown)
        self.assertLess(markdown.index('| t=0.8'), markdown.index('| t=1.2'))
        self.assertEqual(R.render_checkpoint_examples(r), [])


class MeasurementReportTests(unittest.TestCase):
    def test_recorded_training_time_uses_runtime_and_keeps_layout(self):
        lengths = set()
        for minutes, expected in (
            (None, '—'),
            (float('nan'), '—'),
            (float('inf'), '—'),
            (-1, '—'),
            (0, '0.0 min'),
            (57.98, '58.0 min'),
            (60, '1 h 0.0 min'),
            (125.5, '2 h 5.5 min'),
            (119.99, '2 h 0.0 min'),
        ):
            with self.subTest(minutes=minutes):
                result = {
                    'checkpoint': 'fixture',
                    'info': {'architecture': 'LlamaForCausalLM'},
                    'lineage': {'runtime_minutes': minutes, 'max_train_minutes': 999, 'line': 'fixture lineage'},
                }
                lines = R.render_info_section(result)
                lengths.add(len(lines))
                self.assertEqual(lines[3], f'- Recorded training time: {expected}')
                self.assertEqual(lines[4], '- Recorded lineage: fixture lineage')
                self.assertNotIn('Model class', '\n'.join(lines))
                self.assertIn('configured time limit: 16 h 39.0 min per process', '\n'.join(lines))
        self.assertEqual(lengths, {11})

    def test_model_metadata_separates_available_template_from_applied_template(self):
        result = {
            'checkpoint': 'fixture',
            'info': {'disk_size_mb': 308.3, 'has_chat_template': True},
            'lineage': {
                'lr_scheduler': 'cosine',
                'warmup_steps': 200,
                'stable_steps': None,
                'decay_steps': 100,
                'max_steps': 510,
                'tokens_per_step': 262144,
                'train_seq_length': 1024,
                'config_file': '/run/recipe.toml',
            },
            'evaluation_settings': {'chat_template_enabled': False},
        }
        markdown = R.render_report({'results': [result]})
        self.assertIn('Model storage: 308.3 MB; chat template available: yes', markdown)
        self.assertIn('warmup / stable / decay: 200 / — / 100 steps; max optimizer steps: 510', markdown)
        self.assertIn('262144 tokens/optimizer step; sequence: 1024 tokens', markdown)
        self.assertIn('/run/recipe\\.toml', markdown)
        self.assertIn('Generation chat template applied: no', markdown)

    def _reference_tables(self, markdown):
        """Rows of each rendered table, in order, without the header rows."""
        tables, rows = [], None
        for line in markdown.splitlines():
            if line.startswith('| model'):
                rows = []
                tables.append(rows)
            elif line.startswith('|---'):
                continue
            elif line.startswith('| ') and rows is not None:
                rows.append(line)
            elif not line.strip():
                rows = None
        return tables

    def _leaderboard(self, result, peers=(), hidden=0):
        """Render the section against a fixed leaderboard instead of the real file."""
        original = R.entries_for
        R.entries_for = lambda h: (list(peers), hidden)
        try:
            return '\n'.join(R.render_reference_section(result))
        finally:
            R.entries_for = original

    def test_the_leaderboard_is_descriptive_and_does_not_affect_scores(self):
        result = {
            'params_millions': 77.08,
            'summary': {'prose_bpb': 1.51552, 'prose_uniform_token_bpb': 3.51201, 'bake_score': 25.4},
            'evaluation_settings': {'heldout_sha256': 'abc'},
        }
        before = copy.deepcopy(result)
        peers = [{'name': 'peer', 'display': 'A Peer', 'params_millions': 340, 'prose_bpb': 1.11899, 'note': 'BF16'}]
        markdown = self._leaderboard(result, peers)
        self.assertIn('| **This checkpoint** | 77.08 | **1.51552** |', markdown)
        self.assertIn('Uniform-token baseline | — | 3.51201', markdown)
        self.assertIn('| A Peer | 340 | 1.11899 |', markdown)
        self.assertIn('not an evaluated untrained network', markdown)
        self.assertIn('do not affect the composite', markdown)
        self.assertEqual(result, before)

    def test_an_empty_result_still_renders_one_table(self):
        missing = self._leaderboard({})
        self.assertIn('| **This checkpoint** | — | **—** |', missing)
        self.assertEqual(len(self._reference_tables(missing)), 1)
        self.assertNotIn('NOT comparable', missing)

    def test_the_leaderboard_sorts_the_checkpoint_and_baseline_among_the_peers(self):
        peers = [
            {'name': 'high', 'display': 'High', 'params_millions': 7240.0, 'prose_bpb': 1.44380},
            {'name': 'low', 'display': 'Low', 'params_millions': 152.0, 'prose_bpb': 1.36439},
        ]
        for bpb in (4.0, 1.4, 0.8, None, float('nan'), float('inf')):
            with self.subTest(bpb=bpb):
                result = {
                    'summary': {'prose_bpb': bpb, 'prose_uniform_token_bpb': 3.5},
                    'evaluation_settings': {'heldout_sha256': 'abc'},
                }
                markdown = self._leaderboard(result, peers)
                (table,) = self._reference_tables(markdown)
                values = [line.split('|')[3].strip().strip('*') for line in table]
                finite = [float(value) for value in values if value != '—']
                self.assertEqual(finite, sorted(finite, reverse=True))
                self.assertEqual(len(table), 4)
                if not M.is_finite(bpb):
                    self.assertIn('**This checkpoint**', table[-1])
                elif bpb == 1.4:
                    self.assertLess(markdown.index('| High |'), markdown.index('| **This checkpoint** |'))
                    self.assertLess(markdown.index('| **This checkpoint** |'), markdown.index('| Low |'))

    def test_rendering_the_leaderboard_twice_gives_the_same_text(self):
        result = {
            'summary': {'prose_bpb': 1.36439, 'prose_uniform_token_bpb': None},
            'evaluation_settings': {'heldout_sha256': 'abc'},
        }
        peers = [{'name': 'low', 'display': 'Low', 'params_millions': 152, 'prose_bpb': 1.11899}]
        self.assertEqual(self._leaderboard(result, peers), self._leaderboard(result, peers))

    def test_rows_from_another_corpus_are_counted_not_listed(self):
        # The regression this section was rewritten for: a row scored elsewhere
        # must never appear beside one that can be ranked, whatever its number.
        result = {'summary': {'prose_bpb': 1.36439, 'prose_uniform_token_bpb': 3.5}}
        markdown = self._leaderboard(result, peers=(), hidden=7)
        (table,) = self._reference_tables(markdown)
        self.assertEqual(len(table), 2)
        self.assertIn('Uniform-token baseline', table[0])
        self.assertIn('**This checkpoint**', table[1])
        self.assertIn('7 further entries', markdown)
        self.assertNotIn('NOT comparable', markdown)

    def test_composite_shows_the_actual_inputs_with_units(self):
        result = {
            'summary': {
                'prose_bpb': 1.5,
                'logic_accuracy': 0.475,
                'chat_target_bpb': 1.4,
                'greedy_mean_loop_words': 101.7,
                'sampled_mean_punct_issues_p100': 0.08,
            }
        }
        markdown = '\n'.join(R.render_bake_section(result))
        self.assertIn('Prose loss (`points_bpb`) | `prose_bpb`: 1.5000 bits/byte', markdown)
        self.assertIn('Fixed-choice logic (`points_logic`) | `logic_accuracy`: 47.5%', markdown)
        self.assertIn('`chat_target_bpb`: 1.4000 bits/byte', markdown)
        self.assertIn('`greedy_mean_loop_words`: 101.7 words', markdown)
        self.assertIn('`sampled_mean_punct_issues_p100`: 0.08/100 words', markdown)

    def test_probe_token_diagnostics_and_weakest_phrase_keep_units_and_counts(self):
        result = {
            'period_probes': {'overall': {}},
            'traps': {'min_delta_phrase': 'aeroplane'},
            'summary': {
                'trap_mean_delta_bpb': 1.56,
                'probe_historical_sentences_scored': 10,
                'probe_modern_sentences_scored': 10,
                'probe_overall_tokens_scored': 270,
                'probe_overall_mean_token_prob': 0.0447,
                'probe_overall_token_prob_below_0_01_rate': 0.648,
                'probe_overall_mean_entropy_nats': 5.439,
            },
        }
        markdown = '\n'.join(R.render_probe_section(result))
        self.assertIn('Smallest delta belongs to aeroplane (`traps.min_delta_phrase`)', markdown)
        for key, value in (
            ('probe_historical_sentences_scored', '10'),
            ('probe_modern_sentences_scored', '10'),
            ('probe_overall_tokens_scored', '270'),
            ('probe_overall_mean_token_prob', '0.0447'),
            ('probe_overall_token_prob_below_0_01_rate', '64.8%'),
            ('probe_overall_mean_entropy_nats', '5.4390'),
        ):
            self.assertIn(f'(`{key}`) | {value} |', markdown)

    def test_word_cosines_have_fixed_alphabetical_rows_without_json_column(self):
        values = {f'sense_shift_{word}_cosine': (index + 1) / 10 for index, word in enumerate(R.HISTORICAL_WORDS)}
        markdown = '\n'.join(R.render_diagnostics_section({'summary': values}))
        section = markdown.split('### Paired-context word similarities\n', 1)[1]
        for word in R.HISTORICAL_WORDS:
            key = f'sense_shift_{word}_cosine'
            self.assertIn(f'| {word} | {values[key]:.4f} | — | — |', section)
        words = [line.split('|')[1].strip() for line in section.splitlines() if line.startswith('| ')][1:]
        self.assertEqual(words, sorted(R.HISTORICAL_WORDS))
        self.assertNotIn('JSON key', section)
        self.assertNotIn('sense_shift_', section)
        self.assertIn('not established better sense understanding', markdown)
        self.assertIn('among these ten probes only', section)
        missing = '\n'.join(R.render_diagnostics_section({}))
        self.assertEqual(len(markdown.splitlines()), len(missing.splitlines()))

    def test_word_neighbors_use_both_pair_orientations_and_rank_finite_nonself_matches(self):
        result = {
            'summary': {'sense_shift_gay_cosine': 0.4567},
            'embeddings': {
                'pairwise_historical': {
                    'awful|gay': 0.9,
                    'gay|nice': 0.8,
                    'gay|meat': 0.8,
                    'gay|python': 0.0,
                    'gay|want': -0.4,
                    'parliament|gay': -0.9,
                    'gay|science': None,
                    'gay|commerce': float('nan'),
                    'manufacture|gay': float('inf'),
                    'gay|gay': 1.0,
                    'gay|outsider': 0.99,
                }
            },
        }
        before = copy.deepcopy(result)
        markdown = '\n'.join(R.render_diagnostics_section(result))
        self.assertIn(
            '| gay | 0.4567 | awful (0.900), meat (0.800), nice (0.800) | parliament (-0.900), want (-0.400), python (0.000) |',
            markdown,
        )
        self.assertIn('| awful | — | gay (0.900) | gay (0.900) |', markdown)
        self.assertIn('| science | — | — | — |', markdown)
        self.assertNotIn('outsider', markdown)
        self.assertEqual(result, before)
        result['embeddings']['pairwise_historical'] = dict(reversed(list(result['embeddings']['pairwise_historical'].items())))
        self.assertEqual(markdown, '\n'.join(R.render_diagnostics_section(result)))

    def test_perplexity_ratio_explains_the_direction_in_prose(self):
        lengths = set()
        for ratio in (7.7, 1.0, 0.5, None, float('nan'), float('inf')):
            with self.subTest(ratio=ratio):
                result = {'period_probes': {'overall': {}}, 'summary': {'probe_modern_historical_ppl_ratio': ratio}}
                lines = R.render_probe_section(result)
                lengths.add(len(lines))
                markdown = '\n'.join(lines)
                self.assertIn('modern perplexity divided by historical perplexity', markdown)
                self.assertIn('`probe_modern_historical_ppl_ratio`', markdown)
                if M.is_finite(ratio):
                    self.assertIn(f'{ratio:.2f} times the token perplexity of the historical set', markdown)
                else:
                    self.assertIn('ratio is unavailable', markdown)
                    self.assertNotIn('times the token perplexity', markdown)
        self.assertEqual(len(lengths), 1)

    def test_logic_categories_and_correct_incorrect_tied_skipped_choices(self):
        items = [('causal', f'context{i}', f' good{i}', f' bad{i}') for i in range(4)]
        with patch.object(M, 'span_bpb', side_effect=[1.0, 2.0, 2.0, 1.0, 1.0, 1.0, float('nan'), 1.0]):
            result = {'logic': M.run_logic(None, None, items)}
        summarize_result(result)
        measurements = '\n'.join(R.render_logic_section(result))
        self.assertNotIn('context0', measurements)
        markdown = measurements + '\n' + '\n'.join(R.render_logic_examples(result))
        self.assertIn('| causal | 33.3% | 3 | `logic_category_causal_accuracy` |', markdown)
        self.assertIn('**Correct · causal**', markdown)
        self.assertIn('**Incorrect · causal**', markdown)
        self.assertIn('**Incorrect (tie) · causal**', markdown)
        self.assertIn('**Skipped · causal**', markdown)
        self.assertIn('| good | good0 | 1.0000 |', markdown)
        self.assertIn('| bad | bad0 | 2.0000 |', markdown)
        self.assertIn('| good | good3 | — |', markdown)
        self.assertIn('not generated answers', markdown)

    def test_all_sentence_probes_render_with_bytes_and_literal_text(self):
        rows = [
            {
                'sentence': f'sentence{i} | <tag>',
                'label': 'historical' if i < 10 else 'modern',
                'ppl': 125.0,
                'bpb': 1.0,
                'scored_bytes': 20,
            }
            for i in range(20)
        ]
        rows[-1]['bpb'] = None
        result = {'period_probes': {'per_sentence': rows}}
        markdown = '\n'.join(R.render_probe_examples(result))
        self.assertIn('All 20 recorded probe sentences', markdown)
        for i in range(20):
            self.assertIn(f'sentence{i} \\| &lt;tag&gt;', markdown)
        self.assertIn('| 125.00 | 1.0000 | 20 |', markdown)
        self.assertIn('| — | 20 |', markdown)
        self.assertNotRegex(markdown, r'</?(?:details|summary)\b')

    def test_probe_sentences_sort_by_raw_perplexity_descending_with_stable_ties(self):
        rows = [
            {'sentence': name, 'ppl': ppl, 'bpb': 1000.0 - index, 'label': 'historical' if index % 2 else 'modern'}
            for index, (name, ppl) in enumerate(
                [
                    ('low', 10.0),
                    ('missing', None),
                    ('tiefirst', 300.0),
                    ('highest', 900.0),
                    ('nan', float('nan')),
                    ('tiesecond', 300.0),
                    ('infinity', float('inf')),
                    ('roundedlower', 20.001),
                    ('roundedhigher', 20.002),
                    ('zero', 0.0),
                ]
            )
        ]
        result = {'period_probes': {'per_sentence': rows}}
        before = copy.deepcopy(result)
        markdown = '\n'.join(R.render_probe_examples(result))
        order = [line.split('|')[2].strip() for line in markdown.splitlines() if line.startswith('| ')][1:]
        self.assertEqual(
            order,
            [
                'highest',
                'tiefirst',
                'tiesecond',
                'roundedhigher',
                'roundedlower',
                'low',
                'zero',
                'missing',
                'nan',
                'infinity',
            ],
        )
        self.assertIn('Highest token perplexity first', markdown)
        self.assertEqual(result, before)

    def test_logic_category_rows_are_alphabetical_not_dictionary_order(self):
        result = {
            'logic': {
                'categories': {
                    'social': {'accuracy': 0.9, 'items_scored': 10},
                    'causal': {'accuracy': 0.8, 'items_scored': 10},
                    'physical': {'accuracy': 0.7, 'items_scored': 10},
                }
            }
        }
        markdown = '\n'.join(R.render_logic_section(result))
        self.assertLess(markdown.index('| causal |'), markdown.index('| physical |'))
        self.assertLess(markdown.index('| physical |'), markdown.index('| social |'))

    def test_training_progress_keeps_first_latest_and_actual_steps(self):
        rows = [[i * 10, 6.0 - i / 10] for i in range(11)]
        result = {'training_curve': {'eval_curve': rows, 'train_curve': [[1, 7.0], [109, None]]}}
        before = copy.deepcopy(result)
        markdown = '\n'.join(R.render_training_section(result))
        for index in (0, 2, 5, 7, 10):
            self.assertIn(f'| `eval_curve` | {index * 10} | {6.0 - index / 10:.4f} |', markdown)
        self.assertEqual(markdown.count('| `eval_curve`'), 5)
        self.assertIn('| `train_curve` | 109 | — |', markdown)
        self.assertIn('nats/token', markdown)
        self.assertIn('without interpolation', markdown)
        self.assertIn('Retained entries: 11 validation, 2 training', markdown)
        self.assertEqual(result, before)

    def test_selection_limits_and_missing_suites(self):
        for n in (0, 1, 2, 5, 100):
            for limit in (0, 1, 2, 5):
                rows = list(range(n))
                selected = R._spread(rows, limit)
                self.assertEqual(len(selected), min(n, limit))
                self.assertEqual(len(set(selected)), len(selected))
                if selected:
                    self.assertEqual(selected[0], 0)
                if len(selected) > 1:
                    self.assertEqual(selected[-1], n - 1)
        training = '\n'.join(R.render_training_section({}))
        self.assertEqual(training.count('| `eval_curve` | — | — |'), 5)
        self.assertEqual(training.count('| `train_curve` | — | — |'), 5)
        self.assertEqual(R.render_logic_section({}), [])
        self.assertEqual(R.render_probe_section({}), [])
        self.assertEqual(R.render_generation_section({}), [])

    def test_full_report_is_deterministic_and_does_not_rescore_or_mutate(self):
        result = {
            'checkpoint': 'fixture',
            'label': 'fixture',
            'logic': {'accuracy': 0.5, 'items_scored': 2, 'categories': {'physical': {'accuracy': 0.5, 'items_scored': 2}}},
            'generation': {
                'temperature': 0.8,
                'sampled_samples': [completion('one'), completion('two', flagged=True)],
                'sampled_summary': {'n_prompts': 2, 'surface_failure_rate': 0.5},
            },
            'period_probes': {'per_sentence': [{'sentence': 'A fixed sentence.', 'bpb': 1.0, 'scored_bytes': 17}]},
            'training_curve': {'eval_curve': [[1, 2.0], [2, 1.5]]},
        }
        summarize_result(result)
        payload = {'results': [result]}
        before = copy.deepcopy(payload)
        with (
            patch.object(M, 'text_stats', side_effect=AssertionError('must not rescore')),
            patch.object(M, 'span_bpb', side_effect=AssertionError('must not rescore')),
        ):
            markdown = R.render_report(payload)
            self.assertEqual(markdown, R.render_report(payload))
        self.assertEqual(payload, before)
        self.assertNotRegex(markdown, r'</?(?:details|summary)\b')
        for heading in (
            'Fixed-choice logic',
            'Sentence and phrase likelihood',
            'Autocomplete and generation',
            'Recorded training progress',
        ):
            self.assertIn(f'## {heading}', markdown)


class ReportLayoutTests(unittest.TestCase):
    @staticmethod
    def fixture():
        result = {
            'checkpoint': 'fixture',
            'label': 'fixture',
            'logic': {
                'accuracy': 1.0,
                'categories': {'physical': {'accuracy': 1.0, 'items_scored': 1}},
                'items': [
                    {
                        'category': 'physical',
                        'context': 'logic prompt',
                        'good_continuation': ' good',
                        'bad_continuation': ' bad',
                        'good_bpb': 1.0,
                        'bad_bpb': 2.0,
                        'margin_bpb': 1.0,
                        'correct': True,
                    }
                ],
            },
            'generation': {
                'temperature': 0.8,
                'sampled_samples': [completion('generation prompt')],
                'sampled_summary': {'n_prompts': 1, 'surface_failure_rate': 0.0},
            },
            'period_probes': {'per_sentence': [{'sentence': 'probe text', 'bpb': 1.0, 'scored_bytes': 10}]},
            'training_curve': {'eval_curve': [[1, 2.0], [2, 1.5]]},
        }
        summarize_result(result)
        return result

    def test_section_order_and_examples_are_followed_only_by_reproducibility(self):
        markdown = R.render_report({'results': [self.fixture()]})
        headings = [line for line in markdown.splitlines() if line.startswith('## ')]
        self.assertEqual(
            headings,
            [
                '## Model and training records',
                '## Recorded training progress',
                '## Experimental composite',
                '## Other recorded diagnostics',
                '## Fixed-choice logic',
                '## Sentence and phrase likelihood',
                '## Autocomplete and generation',
                '## Text examples',
                '## Reproducibility',
            ],
        )
        measurements, examples = markdown.split('## Text examples\n', 1)
        for text in ('logic prompt', 'probe text', 'generation prompt'):
            self.assertNotIn(text, measurements)
            self.assertIn(text, examples)
        self.assertNotRegex(markdown, r'</?(?:details|summary)\b')

    def test_example_length_cannot_change_the_measurement_prefix(self):
        result = self.fixture()
        before = R.render_report({'results': [result]})
        result['logic']['items'][0]['context'] = 'Long context\n\n' * 100
        result['period_probes']['per_sentence'][0]['sentence'] = 'Long sentence ' * 100
        result['generation']['sampled_samples'][0]['continuation'] = 'Long continuation\n\n' * 100
        after = R.render_report({'results': [result]})
        self.assertEqual(before.split('## Text examples\n')[0], after.split('## Text examples\n')[0])
        self.assertGreater(len(after), len(before))

    def test_priority_sections_have_fixed_rows_when_measurements_are_missing(self):
        positions = []
        for count in (0, 1, 100):
            result = self.fixture()
            result['training_curve'] = {'eval_curve': [[i, 2.0] for i in range(count)]}
            result['summary'] = {'embedding_mean_norm': 6.0} if count else {}
            markdown = '\n'.join(R.render_checkpoint_measurements(result))
            positions.append(
                [(index, line) for index, line in enumerate(markdown.splitlines()) if line.startswith('## ') or line.startswith('| `')]
            )

        # Values may differ, but the headers and table keys occupy identical lines.
        def layout(rows):
            return [(index, line.split('|', 2)[1] if line.startswith('| `') else line) for index, line in rows]

        self.assertEqual(layout(positions[0]), layout(positions[1]))
        self.assertEqual(layout(positions[0]), layout(positions[2]))
        diagnostics = '\n'.join(R.render_diagnostics_section({}))
        self.assertIn('| `embedding_mean_norm` | — |', diagnostics)

    def test_collections_put_all_measurements_and_failures_before_any_examples(self):
        results = [self.fixture(), self.fixture()]
        results[1].update(label='second', checkpoint='second')
        markdown = R.render_report(
            {
                'results': results,
                'failures': [{'path': 'failed', 'reason': 'test failure'}],
            }
        )
        first_example = markdown.index('## Text examples:')
        self.assertLess(markdown.rindex('## Autocomplete and generation'), first_example)
        self.assertLess(markdown.index('## Failures'), first_example)
        self.assertIn('## Text examples: fixture', markdown)
        self.assertIn('## Text examples: second', markdown)
        self.assertEqual([line for line in markdown.splitlines() if line.startswith('## ')][-1], '## Reproducibility')

    def test_reproducibility_uses_each_evaluation_not_the_collection_environment(self):
        results = [self.fixture(), self.fixture()]
        for index, result in enumerate(results):
            result.update(label=f'run{index}', checkpoint=f'run{index}')
            result['evaluation_environment'] = {
                'device': f'device{index}',
                'dtype': f'dtype{index}',
                'python': f'python{index}',
                'torch': f'torch{index}',
                'transformers': f'transformers{index}',
            }
            result['evaluation_settings'] = {
                'seed': index,
                'heldout_sha256': 'a' * 64,
                'chat_sha256': 'b' * 64,
                'docs': 200,
                'chat_docs': 100,
                'max_tokens': 1024,
                'chat_max_tokens': 768,
                'chat_template_enabled': bool(index),
            }
        payload = {'results': results, 'environment': {'device': 'collection-device'}}
        section = R.render_report(payload).split('## Reproducibility\n', 1)[1]
        self.assertNotIn('collection-device', section)
        for index in range(2):
            self.assertIn(f'Seed: {index}; device/dtype: device{index} / dtype{index}', section)
            self.assertIn(f'Python python{index}, PyTorch torch{index}, Transformers transformers{index}', section)
        self.assertEqual(section.count('`' + 'a' * 64 + '`; requested documents: 200; token limit: 1024'), 2)
        self.assertEqual(section.count('`' + 'b' * 64 + '`; requested documents: 100; token limit: 768'), 2)
        self.assertIn('Generation chat template applied: yes', section)
        self.assertIn('Generation chat template applied: no', section)


if __name__ == '__main__':
    unittest.main()

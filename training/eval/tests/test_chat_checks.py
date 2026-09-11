"""Constraint checks, repetition, reliability bands and the capability composite.

These exist because of a bug that scored "a most romantic question" as a
correct answer to "what is the capital of Italy?": `contains_any` is substring
matching, and "romantic" contains "roma". A check that silently passes is
worse than no check, so every trap that bit us has a test.
"""

import unittest

from eval.chat_eval import CHECKS, chat_capability, reliability, repetition_rate, run_checks, score


def passes(kind, text, value):
    return CHECKS[kind](text, value)


class SubstringTraps(unittest.TestCase):
    def test_substring_matching_is_still_substring_matching(self):
        self.assertTrue(passes('contains_any', 'a most romantic question', ['roma']))

    def test_word_matching_rejects_the_word_inside_a_word(self):
        self.assertFalse(passes('contains_word_any', 'a most romantic question', ['rome', 'roma']))
        self.assertTrue(passes('contains_word_any', 'The capital is Rome.', ['rome', 'roma']))

    def test_the_traps_that_bit_us(self):
        # Each pair is (text that must NOT match, text that must).
        for value, wrong, right in [
            (['king'], 'I was thinking of asking', 'the King reigns'),
            (['four'], 'fourteen shillings', 'four legs'),
            (['seven'], 'seventeen days', 'seven days'),
            (['red'], 'two hundred men', 'red and blue'),
            (['tea'], 'a steam engine instead', 'tea and cake'),
            (['ice'], 'a fair price for service', 'it turns to ice'),
            (['east'], 'not in the least', 'in the east'),
            (['day'], 'today and always', 'by day'),
            (['3'], 'in 1830', '3 words'),
        ]:
            self.assertFalse(passes('contains_word_any', wrong, value), f'{value} matched {wrong!r}')
            self.assertTrue(passes('contains_word_any', right, value), f'{value} missed {right!r}')

    def test_case_sensitive_check_is_actually_case_sensitive(self):
        self.assertTrue(passes('contains_cased_any', 'THUNDER', ['THUNDER']))
        self.assertFalse(passes('contains_cased_any', 'thunder', ['THUNDER']))
        # The lowercasing check would have made this impossible to fail.
        self.assertTrue(passes('contains_any', 'thunder', ['thunder']))

    def test_gold_symbol_is_not_found_in_because(self):
        self.assertFalse(passes('contains_cased_any', 'because autumn', ['Au']))
        self.assertTrue(passes('contains_cased_any', 'The symbol is Au.', ['Au']))

    def test_negative_word_constraint_ignores_the_word_inside_a_word(self):
        self.assertTrue(passes('contains_word_none', 'a seahorse is not one', ['horse']))
        self.assertFalse(passes('contains_word_none', 'a fine horse', ['horse']))


class Shape(unittest.TestCase):
    def test_exact_answer_ignores_punctuation_and_case(self):
        self.assertTrue(passes('equals_any', '  London. ', ['london']))
        self.assertFalse(passes('equals_any', 'London is the capital', ['london']))

    def test_line_counts_ignore_blank_lines(self):
        self.assertTrue(passes('max_lines', 'one\n\n', 1))
        self.assertTrue(passes('min_lines', '1. a\n2. b\n\n3. c', 3))

    def test_numbered_list_requires_every_line_numbered(self):
        self.assertTrue(passes('every_line_starts_with_digit', '1. oak\n2. elm', 1))
        self.assertFalse(passes('every_line_starts_with_digit', 'Here you are:\n1. oak', 1))

    def test_is_upper_needs_letters(self):
        self.assertTrue(passes('is_upper', 'THUNDER', 1))
        self.assertFalse(passes('is_upper', 'Thunder', 1))
        self.assertFalse(passes('is_upper', '1234', 1))


class Repetition(unittest.TestCase):
    def test_clean_prose_scores_zero(self):
        text = 'A locomotive is a steam engine designed to draw carriages along iron rails between distant towns.'
        self.assertEqual(repetition_rate(text), 0.0)

    def test_a_loop_scores_high(self):
        self.assertGreater(repetition_rate('the cat sat on the mat ' * 6), 0.5)

    def test_short_replies_are_not_penalised(self):
        self.assertEqual(repetition_rate('Paris.'), 0.0)


class ReliabilityBands(unittest.TestCase):
    def make(self, passed_counts, attempts=10):
        return [
            {'id': f'p{i}', 'n_passed': k, 'n_attempts': attempts, 'stop_count': attempts, 'checks': [{'passed': True}]}
            for i, k in enumerate(passed_counts)
        ]

    def test_a_greedy_run_gets_no_reliability_score(self):
        self.assertIsNone(reliability([{'id': 'p', 'checks': []}]))

    def test_the_three_bands_partition_the_probes(self):
        r = reliability(self.make([10, 9, 0]))
        self.assertAlmostEqual(r['always_right'], 1 / 3)
        self.assertAlmostEqual(r['sometimes_right'], 1 / 3)
        self.assertAlmostEqual(r['never_right'], 1 / 3)
        self.assertAlmostEqual(r['always_right'] + r['sometimes_right'] + r['never_right'], 1.0)

    def test_nine_out_of_ten_is_not_reported_as_a_failure(self):
        r = reliability(self.make([9, 9, 9]))
        self.assertAlmostEqual(r['expected_pass_rate'], 0.9)
        self.assertEqual(r['never_right'], 0.0)
        self.assertEqual(r['always_right'], 0.0)
        self.assertEqual(r['sometimes_right'], 1.0)

    def test_retry_premium_is_zero_when_the_model_is_never_lucky(self):
        self.assertAlmostEqual(reliability(self.make([10, 10, 0]))['retry_premium'], 0.0)

    def test_retry_premium_grows_with_flakiness(self):
        steady = reliability(self.make([10, 10]))['retry_premium']
        flaky = reliability(self.make([5, 5]))['retry_premium']
        self.assertGreater(flaky, steady)

    def test_worst_probe_is_named(self):
        samples = self.make([10, 2, 8])
        self.assertEqual(reliability(samples)['worst_probe'], 'p1')


class Capability(unittest.TestCase):
    def sample(self, reply, checks, stopped=True, turns=1):
        return {
            'id': 'x',
            'tag': 'length',
            'reply': reply,
            'stopped': stopped,
            'truncated': not stopped,
            'new_tokens': len(reply.split()),
            'turns': turns,
            'repetition': repetition_rate(reply),
            'checks': run_checks(reply, checks),
        }

    def test_a_model_that_never_stops_cannot_score_well(self):
        never = [self.sample('a reply that goes on.', [{'type': 'max_words', 'value': 20}], stopped=False)] * 5
        always = [self.sample('a reply that goes on.', [{'type': 'max_words', 'value': 20}], stopped=True)] * 5
        low = chat_capability(score(never), never)['chat_capability']
        high = chat_capability(score(always), always)['chat_capability']
        self.assertLess(low, high)

    def test_self_dialogue_is_a_multiplier_not_an_average(self):
        clean = [self.sample('Paris.', [{'type': 'max_words', 'value': 3}])] * 4
        leaky = list(clean)
        leaky[0] = self.sample('Paris. <|user|> and you?', [{'type': 'max_words', 'value': 3}])
        penalised = chat_capability(score(leaky), leaky)
        self.assertLess(penalised['chat_capability_turn_safety'], 1.0)
        self.assertLess(penalised['chat_capability'], chat_capability(score(clean), clean)['chat_capability'])

    def test_missing_axes_renormalise_rather_than_score_zero(self):
        no_multiturn = [self.sample('Paris.', [{'type': 'max_words', 'value': 3}])] * 3
        out = chat_capability(score(no_multiturn), no_multiturn)
        self.assertIsNone(out['chat_capability_parts']['multiturn'])
        self.assertLess(out['chat_capability_coverage'], 1.0)
        self.assertGreater(out['chat_capability'], 90)

    def test_looping_lowers_the_score(self):
        clean = [self.sample('A market is a place of trade.', [{'type': 'max_words', 'value': 40}])] * 4
        looping = [self.sample('the cat sat on the mat ' * 6, [{'type': 'max_words', 'value': 40}])] * 4
        self.assertLess(
            chat_capability(score(looping), looping)['chat_capability'],
            chat_capability(score(clean), clean)['chat_capability'],
        )


if __name__ == '__main__':
    unittest.main()

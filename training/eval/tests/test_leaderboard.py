"""The standing leaderboard: what gets in, what stays out, what survives a refresh.

This section is the first thing anyone reads in a report, and it broke once by
splitting into a comparable half and an incomparable half when the default
held-out file changed. The rule these tests pin down is narrow: a row may sit
in the ranked column only if it was scored on the same held-out file.
"""

import json
import tempfile
import unittest
from pathlib import Path

from eval import leaderboard as L
from eval import report as R

PISTON = '559ac03c26747498b57cc6afd8ee3b01f87c430ca6e723fadf8d099f56c965f6'
SPROCKET = '51b30a75d40a40d3499b3372ab08b87752b756dc6b4ebccad51f507c89765f00'


def result(name, bpb, heldout=PISTON, checkpoint=None, params=100.0):
    return {
        'checkpoint': checkpoint or f'/models/{name}',
        'name': name,
        'params_millions': params,
        'summary': {'prose_bpb': bpb, 'probe_historical_bpb': 1.0, 'probe_modern_bpb': 1.7},
        'evaluation_settings': {'heldout_sha256': heldout, 'schema': 2},
    }


class Registering(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'leaderboard.json'
        self.addCleanup(self.tmp.cleanup)

    def test_a_result_without_a_prose_score_is_not_recorded(self):
        broken = result('x', None)
        L.register([broken], self.path)
        self.assertEqual(L.load(self.path)['entries'], [])

    def test_rerunning_a_model_updates_its_row_rather_than_adding_one(self):
        L.register([result('m', 1.2)], self.path)
        added, updated = L.register([result('m', 1.1)], self.path)
        entries = L.load(self.path)['entries']
        self.assertEqual((added, updated), (0, 1))
        self.assertEqual(len(entries), 1)
        self.assertAlmostEqual(entries[0]['prose_bpb'], 1.1)

    def test_a_moved_checkpoint_is_still_the_same_row(self):
        L.register([result('m', 1.2, checkpoint='/old/place/m')], self.path)
        L.register([result('m', 1.2, checkpoint='/new/place/m')], self.path)
        self.assertEqual(len(L.load(self.path)['entries']), 1)

    def test_curated_fields_survive_a_refresh(self):
        L.register([result('m', 1.2)], self.path)
        data = L.load(self.path)
        data['entries'][0].update(display='Pretty Name', kind='base', origin='from scratch', note='hand written')
        self.path.write_text(json.dumps(data))

        L.register([result('m', 0.9)], self.path)
        entry = L.load(self.path)['entries'][0]
        self.assertEqual(entry['display'], 'Pretty Name')
        self.assertEqual(entry['note'], 'hand written')
        self.assertAlmostEqual(entry['prose_bpb'], 0.9, msg='the measurement must still be refreshed')

    def test_entries_are_stored_best_first(self):
        L.register([result('worse', 1.4), result('better', 0.8)], self.path)
        names = [e['name'] for e in L.load(self.path)['entries']]
        self.assertEqual(names, ['better', 'worse'])

    def test_plumbing_directories_are_not_used_as_names(self):
        # Publishers name the weights directory whatever they like. A row
        # called "checkpoints" tells a reader nothing.
        for name, path in (
            ('final', '/x/ft08/attnonly-r16/sft_checkpoints/final'),
            ('final_merged', '/x/ft08/full-epoch/sft_checkpoints/final_merged'),
            ('checkpoints', '/x/MODELS/Bartholomew-sft/checkpoints'),
        ):
            L.register([result(name, 1.0, checkpoint=path)], self.path)
        names = {e['name'] for e in L.load(self.path)['entries']}
        self.assertEqual(names, {'attnonly-r16', 'full-epoch', 'Bartholomew-sft'})


class Comparability(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'leaderboard.json'
        self.addCleanup(self.tmp.cleanup)
        L.register([result('piston_a', 0.9), result('piston_b', 1.1), result('sprocket', 0.5, heldout=SPROCKET)], self.path)

    def test_only_rows_scored_on_the_same_heldout_are_returned(self):
        keep, hidden = L.entries_for(PISTON, self.path)
        self.assertEqual({e['name'] for e in keep}, {'piston_a', 'piston_b'})
        self.assertEqual(hidden, 1)

    def test_a_run_with_no_recorded_heldout_gets_no_peers_at_all(self):
        keep, hidden = L.entries_for(None, self.path)
        self.assertEqual(keep, [])
        self.assertEqual(hidden, 3)

    def test_the_lower_scoring_incomparable_row_cannot_win_the_ranking(self):
        # sprocket has the best number and must never appear, or it would top
        # a table it was not measured for.
        keep, _ = L.entries_for(PISTON, self.path)
        self.assertNotIn('sprocket', {e['name'] for e in keep})


class Rendering(unittest.TestCase):
    def render(self, summary, heldout=PISTON, peers=(), hidden=0):
        original = R.entries_for
        R.entries_for = lambda h: (list(peers), hidden)
        try:
            return '\n'.join(
                R.render_reference_section(
                    {
                        'summary': summary,
                        'params_millions': 141.0,
                        'name': 'me',
                        'evaluation_settings': {'heldout_sha256': heldout},
                    }
                )
            )
        finally:
            R.entries_for = original

    def test_there_is_exactly_one_table(self):
        out = self.render({'prose_bpb': 0.9, 'prose_uniform_token_bpb': 3.1}, peers=[{'name': 'peer', 'prose_bpb': 1.0}])
        self.assertEqual(out.count('| model |'), 1)
        self.assertNotIn('NOT comparable', out)

    def test_hidden_rows_are_counted_in_words_not_listed(self):
        out = self.render({'prose_bpb': 0.9}, hidden=3)
        self.assertIn('3 further entries', out)
        out_one = self.render({'prose_bpb': 0.9}, hidden=1)
        self.assertIn('1 further entry is', out_one)

    def test_the_checkpoint_appears_once_even_if_it_is_also_on_file(self):
        out = self.render({'prose_bpb': 0.9}, peers=[{'name': 'me', 'prose_bpb': 0.9, 'display': 'Me'}])
        self.assertEqual(out.count('This checkpoint'), 1)
        self.assertNotIn('| Me |', out)

    def test_the_register_ratio_is_shown_and_flags_a_modern_base(self):
        out = self.render(
            {'prose_bpb': 0.83, 'probe_historical_bpb': 1.4028, 'probe_modern_bpb': 1.1506},
        )
        self.assertIn('0.82', out)

    def test_a_missing_ratio_does_not_print_a_number(self):
        out = self.render({'prose_bpb': 0.9, 'probe_historical_bpb': None, 'probe_modern_bpb': None})
        self.assertNotIn('nan', out.lower())


if __name__ == '__main__':
    unittest.main()

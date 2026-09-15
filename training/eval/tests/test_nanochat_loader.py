"""Discovery, offset mapping and chat rendering for nanochat checkpoints.

No weights are loaded here. These cover the parts that can go wrong silently:
a checkpoint that is not recognised, a character offset that is off by a byte,
and a chat rendering that does not match what the model was trained on.
"""

import pickle
import tempfile
import unittest
from pathlib import Path

import torch
from eval import nanochat_models as NC


class FakeEncoding:
    """Enough of a tiktoken Encoding to exercise the tokenizer wrapper.

    One token per UTF-8 byte for ordinary text, which is the worst case for
    offset mapping: every multi-byte character is split across tokens.
    """

    n_vocab = 300
    _SPECIALS = {'<|bos|>': 256, '<|user_start|>': 257, '<|user_end|>': 258, '<|assistant_start|>': 259, '<|assistant_end|>': 260}

    special_tokens_set = frozenset(_SPECIALS)

    def encode(self, text, allowed_special=None):
        ids, rest = [], text
        while rest:
            for name, token in self._SPECIALS.items():
                if rest.startswith(name):
                    ids.append(token)
                    rest = rest[len(name) :]
                    break
            else:
                ids.extend(rest[0].encode('utf-8'))
                rest = rest[1:]
        return ids

    def encode_single_token(self, text):
        return self._SPECIALS[text]

    def decode_single_token_bytes(self, token):
        for name, value in self._SPECIALS.items():
            if value == token:
                return name.encode()
        return bytes([token])

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            out += self.decode_single_token_bytes(i)
        return out.decode('utf-8', 'replace')


class Discovery(unittest.TestCase):
    def test_pairs_weights_with_meta_and_prefers_the_last_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for step in ('000100', '009600'):
                (d / f'model_{step}.pt').write_bytes(b'')
                (d / f'meta_{step}.json').write_text('{}')
            self.assertTrue(NC.is_nanochat_checkpoint(d))
            weights, meta = NC.nanochat_parts(d)
            self.assertEqual(weights.name, 'model_009600.pt')
            self.assertEqual(meta.name, 'meta_009600.json')

    def test_a_lone_weight_file_is_not_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'model_000100.pt').write_bytes(b'')
            self.assertFalse(NC.is_nanochat_checkpoint(d))
            self.assertIsNone(NC.nanochat_parts(d))

    def test_an_hf_checkpoint_is_not_mistaken_for_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'config.json').write_text('{}')
            (d / 'model.safetensors').write_bytes(b'')
            self.assertFalse(NC.is_nanochat_checkpoint(d))

    def test_a_missing_tokenizer_says_what_to_do(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'model_09.pt').write_bytes(b'')
            (d / 'meta_09.json').write_text('{}')
            with self.assertRaises(FileNotFoundError) as caught:
                NC.resolve_nanochat_tokenizer(d)
            self.assertIn('--tokenizer', str(caught.exception))


class Stage(unittest.TestCase):
    def test_sft_is_recognised_from_the_chat_only_knobs(self):
        self.assertEqual(NC.nanochat_stage({'user_config': {'mmlu_epochs': 0, 'gsm8k_epochs': 0}}), 'sft')

    def test_base_is_read_straight_from_the_config_when_present(self):
        self.assertEqual(NC.nanochat_stage({'user_config': {'stage': 'base'}}), 'base')

    def test_an_unlabelled_run_is_not_guessed(self):
        self.assertEqual(NC.nanochat_stage({'user_config': {'depth': 20}}), 'unknown')


class Offsets(unittest.TestCase):
    def setUp(self):
        self.tok = NC.NanochatTokenizer(FakeEncoding(), Path('.'))
        self.split = self.tok

    def test_ascii_offsets_reconstruct_the_string_exactly(self):
        text = 'The Duke said so, and left.'
        offsets = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)['offset_mapping']
        self.assertEqual(''.join(text[a:b] for a, b in offsets), text)

    def test_a_character_split_across_tokens_still_maps_inside_the_string(self):
        text = 'café «x» 日本'
        offsets = self.split(text, return_offsets_mapping=True, add_special_tokens=False)['offset_mapping']
        covered = {i for a, b in offsets for i in range(a, b)}
        self.assertEqual(covered, set(range(len(text))))
        self.assertTrue(all(0 <= a <= b <= len(text) for a, b in offsets))

    def test_offsets_are_non_decreasing(self):
        text = 'Naïve café — twice.'
        offsets = self.split(text, return_offsets_mapping=True, add_special_tokens=False)['offset_mapping']
        self.assertEqual(offsets, sorted(offsets))

    def test_a_special_token_claims_no_source_text(self):
        offsets = self.tok('abc', return_offsets_mapping=True, add_special_tokens=True)['offset_mapping']
        self.assertEqual(offsets[0], (0, 0))

    def test_padding_and_offsets_together_are_refused(self):
        with self.assertRaises(ValueError):
            self.tok(['a', 'bcd'], return_offsets_mapping=True, padding=True)


class Encoding(unittest.TestCase):
    def setUp(self):
        self.tok = NC.NanochatTokenizer(FakeEncoding(), Path('.'))

    def test_bos_is_added_once_and_only_once(self):
        first = self.tok.encode('abc', add_special_tokens=True)
        self.assertEqual(first[0], self.tok.bos_token_id)
        self.assertEqual(self.tok.encode(self.tok.decode(first), add_special_tokens=True), first)

    def test_left_padding_puts_the_zeros_on_the_left(self):
        out = self.tok(['a', 'abcd'], padding=True, add_special_tokens=False)
        self.assertEqual(out['attention_mask'][0][:3], [0, 0, 0])
        self.assertEqual(out['attention_mask'][1], [1, 1, 1, 1])

    def test_pt_tensors_come_back_batched(self):
        out = self.tok('abc', return_tensors='pt')
        self.assertEqual(out['input_ids'].shape[0], 1)
        self.assertIs(out['input_ids'].dtype, torch.long)

    def test_skip_special_tokens_drops_the_markers(self):
        ids = self.tok.encode('<|user_start|>hi<|user_end|>', add_special_tokens=True)
        self.assertEqual(self.tok.decode(ids, skip_special_tokens=True), 'hi')


class ChatRendering(unittest.TestCase):
    def setUp(self):
        self.tok = NC.NanochatTokenizer(FakeEncoding(), Path('.'))

    def test_matches_nanochat_render_for_completion(self):
        out = self.tok.apply_chat_template([{'role': 'user', 'content': 'Hi'}], add_generation_prompt=True)
        self.assertEqual(out, '<|bos|><|user_start|>Hi<|user_end|><|assistant_start|>')

    def test_a_system_message_is_folded_into_the_user_turn(self):
        out = self.tok.apply_chat_template(
            [{'role': 'system', 'content': 'Be brief'}, {'role': 'user', 'content': 'Hi'}], add_generation_prompt=True
        )
        self.assertEqual(out, '<|bos|><|user_start|>Be brief\n\nHi<|user_end|><|assistant_start|>')

    def test_a_dangling_system_message_is_an_error_not_a_guess(self):
        with self.assertRaises(ValueError):
            self.tok.apply_chat_template([{'role': 'system', 'content': 'Be brief'}], add_generation_prompt=True)

    def test_an_unknown_role_is_refused(self):
        with self.assertRaises(ValueError):
            self.tok.apply_chat_template([{'role': 'tool', 'content': 'x'}])

    def test_the_rendered_prompt_re_encodes_to_its_own_special_ids(self):
        text = self.tok.apply_chat_template([{'role': 'user', 'content': 'Hi'}], add_generation_prompt=True)
        ids = self.tok.encode(text, add_special_tokens=False)
        self.assertEqual(ids[0], self.tok.bos_token_id)
        self.assertEqual(ids[-1], self.tok.enc.encode_single_token('<|assistant_start|>'))


class VictorianDialect(unittest.TestCase):
    """Mr. Chatterbox renamed every role token and dropped the user-end marker.

    Its own tokenizer_wrapper.py is the reference: a turn is
    `<|endoftext|><human>...<victorian>...<|endoftext|>`, and <|endoftext|>
    does triple duty as BOS, user-turn terminator and assistant stop.
    """

    class VictorianEncoding(FakeEncoding):
        _SPECIALS = {'<|endoftext|>': 256, '<|pad|>': 257, '<human>': 258, '<victorian>': 259}
        special_tokens_set = frozenset(_SPECIALS)

    def setUp(self):
        self.tok = NC.NanochatTokenizer(self.VictorianEncoding(), Path('.'))

    def test_the_dialect_is_detected_from_the_vocabulary(self):
        self.assertIs(self.tok.markers, NC.VICTORIAN_MARKERS)
        self.assertEqual(self.tok.bos_token, '<|endoftext|>')
        self.assertEqual(self.tok.eos_token, '<|endoftext|>')

    def test_pad_is_its_own_token_not_the_stop_token(self):
        self.assertEqual(self.tok.pad_token, '<|pad|>')
        self.assertNotEqual(self.tok.pad_token_id, self.tok.eos_token_id)

    def test_rendering_matches_the_authors_wrapper(self):
        out = self.tok.apply_chat_template([{'role': 'user', 'content': 'Good day.'}], add_generation_prompt=True)
        self.assertEqual(out, '<|endoftext|><human>Good day.<victorian>')

    def test_a_user_turn_has_no_closing_marker(self):
        out = self.tok.apply_chat_template([{'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Ho'}])
        self.assertEqual(out, '<|endoftext|><human>Hi<victorian>Ho<|endoftext|>')

    def test_the_upstream_dialect_is_not_applied_here(self):
        self.assertNotIn('<|user_start|>', self.tok.apply_chat_template([{'role': 'user', 'content': 'Hi'}]))


class Pairing(unittest.TestCase):
    def test_a_vocabulary_of_the_wrong_size_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'tokenizer.pkl').write_bytes(pickle.dumps(FakeEncoding()))
            loaded = NC.load_nanochat_tokenizer(d)
            self.assertEqual(loaded.vocab_size, 300)


if __name__ == '__main__':
    unittest.main()

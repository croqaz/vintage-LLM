"""The t-v5 chat template must mask user turns without changing the wire format.

Two things can break independently and both are silent:

1. The rendered string changes, so a fine-tune no longer matches what the base
   model and the eval harness expect.
2. The {% generation %} block drifts, so assistant_only_loss either masks
   nothing (back to training on "hey dude") or masks the assistant EOS (which
   removes the stop token, the main thing SFT buys).
"""

import unittest
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / 'tokenizers/t-v5/chat_template.jinja'
TOKENIZER = ROOT / 'tokenizers/t-v5'

# The format every model in this project was trained and evaluated against.
EXPECTED = '<|bos|><|user|>\nHey dude, what about steam?<|eos|>\n<|bos|><|assistant|>\nA most agreeable subject.<|eos|>\n'

MESSAGES = [
    {'role': 'user', 'content': 'Hey dude, what about steam?'},
    {'role': 'assistant', 'content': 'A most agreeable subject.'},
]


def load():
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    tok.chat_template = TEMPLATE.read_text(encoding='utf-8')
    return tok


class ChatTemplate(unittest.TestCase):
    def setUp(self):
        if not TEMPLATE.is_file():
            self.skipTest(f'no template at {TEMPLATE}')
        self.tok = load()

    def test_render_is_unchanged(self):
        self.assertEqual(self.tok.apply_chat_template(MESSAGES, tokenize=False), EXPECTED)

    def test_generation_prompt_ends_open(self):
        out = self.tok.apply_chat_template(MESSAGES[:1], tokenize=False, add_generation_prompt=True)
        self.assertTrue(out.endswith('<|bos|><|assistant|>\n'), out)

    def _masked(self):
        out = self.tok.apply_chat_template(MESSAGES, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
        ids, mask = out['input_ids'], out['assistant_masks']
        keep = self.tok.decode([i for i, m in zip(ids, mask) if m])
        drop = self.tok.decode([i for i, m in zip(ids, mask) if not m])
        return keep, drop

    def test_assistant_content_is_scored(self):
        keep, _ = self._masked()
        self.assertIn('A most agreeable subject.', keep)

    def test_user_turn_is_not_scored(self):
        keep, drop = self._masked()
        self.assertNotIn('Hey dude', keep)
        self.assertIn('Hey dude', drop)

    def test_assistant_eos_is_scored(self):
        """SFT mostly buys a stop token. Masking EOS out would remove it."""
        keep, _ = self._masked()
        self.assertTrue(keep.rstrip().endswith('<|eos|>'), f'assistant EOS not scored: {keep!r}')

    def test_something_is_masked(self):
        """Guards against a template that marks everything as assistant."""
        out = self.tok.apply_chat_template(MESSAGES, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
        mask = out['assistant_masks']
        self.assertTrue(0 < sum(mask) < len(mask), f'{sum(mask)} of {len(mask)} scored')


if __name__ == '__main__':
    unittest.main()

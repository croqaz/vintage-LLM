"""Generation-batch persistence without loading checkpoint weights."""

import copy
import io
import json
import re
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from eval import __main__ as E
from eval import metrics as M
from eval.helpers import atomic_json, json_safe


class Batch(dict):
    @property
    def input_ids(self):
        return self['input_ids']

    def to(self, device):
        return self


class Tokenizer:
    padding_side = 'right'
    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def __len__(self):
        return 16

    def __call__(self, prompts, **kwargs):
        return Batch(input_ids=torch.full((len(prompts), 2), 2, dtype=torch.long))

    def decode(self, ids, **kwargs):
        return ' '.join('word' + chr(ord('a') + int(i)) for i in ids)


class Model(torch.nn.Module):
    device = torch.device('cpu')

    def __init__(self, interrupt_at=None):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.config = SimpleNamespace(
            model_type='llama',
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            vocab_size=16,
            max_position_embeddings=16,
            tie_word_embeddings=True,
        )
        self.interrupt_at = interrupt_at
        self.calls = 0

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, input_ids, max_new_tokens, do_sample, **kwargs):
        self.calls += 1
        if self.calls == self.interrupt_at:
            raise KeyboardInterrupt
        shape = (len(input_ids), max_new_tokens)
        new = torch.randint(2, 16, shape) if do_sample else torch.full(shape, 3, dtype=torch.long)
        return torch.cat([input_ids, new], dim=1)


@contextmanager
def checkpoint_fixture(interrupt_at=None):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model, tokenizer = Model(interrupt_at), Tokenizer()
        args = E.parse_args(['--device', 'cpu', '--skip-probes', '--skip-embeddings', '--gen-tokens', '3'])
        args.generation_modes, args.generation_batch_size = ['greedy', 'sample'], 2
        args.temp_sweep = [0.8, 1.2]
        prompts = {'generation': [f'prompt {i}' for i in range(5)], 'logic_items': [], 'trap_pairs': []}
        output = io.StringIO()
        with (
            patch.object(E, 'load_model_and_tokenizer', return_value=(model, tokenizer)),
            patch.object(E, 'checkpoint_lineage', return_value={}),
            patch.object(E, 'provenance_line', return_value='fixture'),
            patch.object(E, 'training_curve', return_value={}),
            patch.object(E, 'free_model') as free,
            redirect_stdout(output),
        ):
            yield root, model, tokenizer, args, prompts, output
        free.assert_called_once_with(model, model.device)
        assert tokenizer.padding_side == 'right'


class GenerationSnapshotTests(unittest.TestCase):
    def test_every_progress_update_has_a_batch_callback_without_changing_generation(self):
        kwargs = dict(
            prompts=[f'prompt {i}' for i in range(5)],
            modes=['greedy', 'sample'],
            max_new_tokens=3,
            temperature=0.8,
            top_p=0.9,
            top_k=5,
            seed=42,
            batch_size=2,
        )
        with redirect_stdout(io.StringIO()):
            baseline = M.generate_continuations(Tokenizer(), Model(), **kwargs)
        events, output = [], io.StringIO()

        def on_batch(mode, rows):
            self.assertEqual(len(re.findall(r'g\d+/5', output.getvalue())), len(events))
            events.append((mode, copy.deepcopy(rows)))

        tokenizer, model = Tokenizer(), Model()
        with redirect_stdout(output):
            result = M.generate_continuations(tokenizer, model, **kwargs, on_batch=on_batch)
        self.assertEqual(result, baseline)
        self.assertEqual(
            [(mode, len(rows)) for mode, rows in events],
            [
                ('greedy', 2),
                ('greedy', 4),
                ('greedy', 5),
                ('sample', 2),
                ('sample', 4),
                ('sample', 5),
            ],
        )
        self.assertEqual(re.findall(r'g\d+/5', output.getvalue()), ['g2/5', 'g4/5', 'g5/5'] * 2)
        self.assertEqual(model.calls, 6)
        self.assertEqual(tokenizer.padding_side, 'right')

    def test_checkpoint_writes_cumulative_batches_for_both_modes_and_sweeps(self):
        with checkpoint_fixture() as (root, model, tokenizer, args, prompts, output):
            partial = root / 'eval-test.partial.json'
            snapshots = []

            def snapshot(stage, result):
                atomic_json(partial, {'stage_completed': stage, 'result': result})
                snapshots.append(json.loads(partial.read_text()))

            result = E.evaluate_checkpoint(root, root, model.device, torch.float32, args, prompts, {'heldout': [], 'chat': []}, snapshot)
            batches = [p for p in snapshots if re.search(r'_g\d+/5$', p['stage_completed'])]
            self.assertEqual(len(batches), 9)
            self.assertEqual(len(re.findall(r' g\d+/5', output.getvalue())), len(batches))
            for index, payload in enumerate(batches):
                gen = payload['result']['generation']
                count = [2, 4, 5][index % 3]
                self.assertIn('logic', payload['result'])
                if index < 6:
                    prefix = 'greedy' if index < 3 else 'sampled'
                    self.assertEqual(len(gen[prefix + '_samples']), count)
                    self.assertEqual(gen[prefix + '_summary']['n_prompts'], count)
                if index >= 3:
                    self.assertEqual(len(gen['greedy_samples']), 5)
                if index >= 6:
                    self.assertEqual(len(gen['sampled_samples']), 5)
                    self.assertEqual(gen['temperature_sweep']['0.8']['n_prompts'], 5)
                    self.assertEqual(gen['temperature_sweep']['1.2']['n_prompts'], count)
                    active = gen['temperature_sweep_in_progress']
                    self.assertEqual(active['temperature'], 1.2)
                    self.assertEqual(len(active['samples']), count)
            self.assertEqual(model.calls, 9)  # The base temperature is reused, not generated twice.
            self.assertNotIn('temperature_sweep_in_progress', result['generation'])
            self.assertNotIn('temperature_sweep_in_progress', json.loads(partial.read_text())['result']['generation'])
            baseline = M.generate_continuations(
                Tokenizer(),
                Model(),
                prompts['generation'],
                args.generation_modes,
                args.gen_tokens,
                args.temperature,
                args.top_p,
                args.top_k,
                args.seed,
                batch_size=args.generation_batch_size,
            )
            for mode, prefix in (('greedy', 'greedy'), ('sample', 'sampled')):
                self.assertEqual(result['generation'][prefix + '_samples'], baseline[mode])
                self.assertEqual(json_safe(result['generation'][prefix + '_summary']), json_safe(M.summarize_generations(baseline[mode])))

    def test_interruption_retains_the_last_completed_batch_and_prior_modes(self):
        for interrupt_at, stage in (
            (2, 'generation_greedy_g2/5'),
            (5, 'generation_sampled_g2/5'),
            (8, 'sweep_t1.2_g2/5'),
        ):
            with self.subTest(stage=stage), checkpoint_fixture(interrupt_at) as (root, model, tokenizer, args, prompts, output):
                partial = root / 'eval-test.partial.json'

                def snapshot(stage, result):
                    atomic_json(partial, {'stage_completed': stage, 'result': result})

                with self.assertRaises(KeyboardInterrupt):
                    E.evaluate_checkpoint(root, root, model.device, torch.float32, args, prompts, {'heldout': [], 'chat': []}, snapshot)
                saved = json.loads(partial.read_text())
                self.assertEqual(saved['stage_completed'], stage)
                gen = saved['result']['generation']
                self.assertIn('logic', saved['result'])
                self.assertEqual(len(gen['greedy_samples']), 2 if interrupt_at == 2 else 5)
                if interrupt_at >= 5:
                    self.assertEqual(len(gen['sampled_samples']), 2 if interrupt_at == 5 else 5)
                if interrupt_at == 8:
                    self.assertEqual(len(gen['temperature_sweep_in_progress']['samples']), 2)
                    self.assertEqual(gen['temperature_sweep']['1.2']['n_prompts'], 2)

    def test_snapshot_write_error_does_not_stop_generation_or_later_snapshots(self):
        with checkpoint_fixture() as (root, model, tokenizer, args, prompts, output):
            stages = []

            def snapshot(stage, result):
                stages.append(stage)
                if stage == 'generation_greedy_g2/5':
                    raise OSError('disk unavailable')

            result = E.evaluate_checkpoint(root, root, model.device, torch.float32, args, prompts, {'heldout': [], 'chat': []}, snapshot)
            self.assertIn('snapshot after generation_greedy_g2/5 failed: disk unavailable', output.getvalue())
            self.assertIn('generation_greedy_g4/5', stages)
            self.assertEqual(result['generation']['sampled_summary']['n_prompts'], 5)
            self.assertEqual(model.calls, 9)


if __name__ == '__main__':
    unittest.main()


class ChatTemplateBosTests(unittest.TestCase):
    """A rendered chat template already carries BOS; the tokenizer must not add a second."""

    class TemplateTokenizer(Tokenizer):
        bos_token = '<s>'
        chat_template = '{{ messages }}'

        def __init__(self, template_emits_bos=True):
            self.template_emits_bos = template_emits_bos
            self.add_special_tokens_seen = []

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            body = messages[0]['content'] + ' [assistant]'
            return ('<s>' + body) if self.template_emits_bos else body

        def __call__(self, prompts, **kwargs):
            self.add_special_tokens_seen.append(kwargs.get('add_special_tokens'))
            return Batch(input_ids=torch.full((len(prompts), 2), 2, dtype=torch.long))

    def _run(self, tokenizer, chat=True):
        with redirect_stdout(io.StringIO()):
            self._generate(tokenizer, chat)
        return tokenizer.add_special_tokens_seen

    def _generate(self, tokenizer, chat):
        M.generate_continuations(
            tokenizer,
            Model(),
            ['hello', 'there'],
            modes=['greedy'],
            max_new_tokens=2,
            temperature=0.8,
            top_p=0.9,
            top_k=10,
            seed=1,
            batch_size=2,
            chat=chat,
        )

    def test_template_bos_is_not_duplicated(self):
        self.assertEqual(self._run(self.TemplateTokenizer(template_emits_bos=True)), [False])

    def test_template_without_bos_still_gets_one(self):
        self.assertEqual(self._run(self.TemplateTokenizer(template_emits_bos=False)), [True])

    def test_plain_prompts_are_unaffected(self):
        tokenizer = self.TemplateTokenizer(template_emits_bos=True)
        self.assertEqual(self._run(tokenizer, chat=False), [True])

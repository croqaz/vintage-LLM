#!/usr/bin/env python3
"""
Generate creative text completions from seed prompts.

Two backends are supported (one is required):
  --model-folder PATH     Load a local HuggingFace checkpoint (transformers).
  --api-url URL           Call an OpenAI-compatible server (llama.cpp / vLLM / etc.).

Output formats:
  text  (default) — samples separated by "\n\n-----\n\n"
  jsonl             — one JSON object per line: {"text": "...", "seed": "..."}
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import requests

# torch and transformers are imported lazily inside LocalModelBackend so the
# API backend (and the Docker image built around it) doesn't need the heavy
# torch stack installed.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device(name: str) -> 'torch.device':  # type: ignore[valid-type]
    import torch

    if name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(name)


def _dtype(name: str, dev: 'torch.device') -> 'torch.dtype':  # type: ignore[valid-type]
    import torch

    mapping = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    if name in mapping:
        return mapping[name]
    return torch.float16 if dev.type == 'cuda' else torch.float32


def _parse_word_range(spec: str) -> 'tuple[int, int]':
    """Parse a "--seed-words" spec into an inclusive (lo, hi) word range.

    Accepts "MIN-MAX" (e.g. "2-4"), a single count (e.g. "4" -> (4, 4)), or
    "0" / "0-0" to disable truncation entirely.
    """
    parts = spec.split('-')
    try:
        if len(parts) == 1:
            lo = hi = int(parts[0])
        elif len(parts) == 2:
            lo, hi = int(parts[0]), int(parts[1])
        else:
            raise ValueError
    except ValueError:
        raise ValueError(f'--seed-words must be "MIN-MAX" or a single integer, got {spec!r}')
    if lo < 0 or hi < 0:
        raise ValueError('--seed-words values must be >= 0')
    if lo > hi:
        raise ValueError(f'--seed-words min ({lo}) must be <= max ({hi})')
    return lo, hi


# Spaces are stripped *before* these (closing punctuation/brackets/quotes)…
_SPACE_BEFORE_PUNCT = re.compile(r' +([,.!?;:%)\]}»”’])')
# …and *after* these (opening brackets/quotes)…
_SPACE_AFTER_OPEN = re.compile(r'([(\[{«“]) +')
# …and before a contraction/possessive suffix ("don 't" -> "don't", "it 's" -> "it's").
_SPACE_BEFORE_CONTRACTION = re.compile(r" +'(s|t|re|ve|ll|d|m|clock)\b")
# …and before the split "n't" negation ("is n't" -> "isn't", "could n't" -> "couldn't").
_SPACE_BEFORE_NT = re.compile(r" +n't\b")


def _tidy_spacing(text: str) -> str:
    """Remove the spurious whitespace tokenizers leave around punctuation."""
    text = _SPACE_BEFORE_PUNCT.sub(r'\1', text)
    text = _SPACE_AFTER_OPEN.sub(r'\1', text)
    text = _SPACE_BEFORE_CONTRACTION.sub(r"'\1", text)
    text = _SPACE_BEFORE_NT.sub("n't", text)
    text = re.sub(r' {2,}', ' ', text)  # collapse runs of spaces
    return text


def _smart_join(prompt: str, completion: str) -> str:
    """Join a prompt and its completion with natural spacing.

    A single space separates the two, then punctuation glue-ups are repaired so
    that e.g. "About two in the morning" + ", the guards…" renders as
    "About two in the morning, the guards…" rather than "morning , the guards".
    """
    left, right = prompt.rstrip(), completion.lstrip()
    if not right:
        return left
    if not left:
        return _tidy_spacing(right)
    return _tidy_spacing(f'{left} {right}')


def _truncate_words(text: str, lo: int, hi: int, rng: 'random.Random') -> str:
    """Keep the first N whitespace-delimited words of *text*.

    N is drawn uniformly from [lo, hi] per call so seeds get varied lengths.
    A range of (0, 0) leaves the text unchanged.
    """
    if hi == 0:
        return text
    words = text.split()
    n = rng.randint(lo, hi)
    return ' '.join(words[:n])


# ---------------------------------------------------------------------------
# Local model backend
# ---------------------------------------------------------------------------


class LocalModelBackend:
    """Generate completions with a local HuggingFace checkpoint in eager mode."""

    def __init__(
        self,
        model_folder: Path,
        device_str: str = 'auto',
        dtype_str: str = 'auto',
    ) -> None:
        import torch  # noqa: F401  (used via _device/_dtype and torch.no_grad below)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = _device(device_str)
        self.dtype = _dtype(dtype_str, self.device)

        print(f'[sample] Loading model from {model_folder} ...')
        self.model = AutoModelForCausalLM.from_pretrained(model_folder, dtype=self.dtype)
        self.model.config.use_cache = True
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_folder, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f'[sample] Loaded {self.model.config.model_type} · {n_params:,} params · {self.device}')

    def generate_iter(
        self,
        prompts: Sequence[str],
        *,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        chat: bool = False,
    ) -> 'Iterator[tuple[int, str]]':
        """Yield (index, completion) pairs as each batch finishes.

        Generation runs in batches of ``batch_size`` rather than one giant
        forward pass over every prompt. This bounds GPU memory *and* — crucially —
        lets the caller persist each batch's results to disk as soon as they are
        ready, so a crash midway through a long run doesn't lose everything.
        """
        import torch

        do_sample = temperature > 0.0
        # Use the model's configured stop tokens (generation_config may list
        # several, e.g. [4, 2]). The tokenizer only knows about a single
        # eos_token_id, so relying on it here would miss the others and let
        # generation run all the way to max_new_tokens.
        eos_token_id = self.model.generation_config.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            min_p=min_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs['temperature'] = temperature

        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            formatted: List[str] = []
            for p in batch:
                if chat:
                    msgs = [{'role': 'user', 'content': p}]
                    formatted.append(self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                else:
                    formatted.append(p)

            inputs = self.tokenizer(formatted, return_tensors='pt', padding=True).to(self.device)
            # With left padding every row shares the same prompt width, so the
            # newly generated tokens start at this offset for all sequences.
            input_len = inputs['input_ids'].shape[1]

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)

            for j, out_ids in enumerate(output_ids):
                # Decode only the continuation — the prompt is added back by the
                # caller, so returning it here would duplicate it.
                gen_ids = out_ids[input_len:]
                text = self.tokenizer.decode(
                    gen_ids,
                    clean_up_tokenization_spaces=False,
                )
                yield start + j, text.strip()


# ---------------------------------------------------------------------------
# API backend  (OpenAI-compatible: llama.cpp server, vLLM, etc.)
# ---------------------------------------------------------------------------


class ApiBackend:
    """Generate completions via an OpenAI-compatible /v1/completions endpoint."""

    def __init__(self, api_url: str, model_name: Optional[str] = None, seed: int = 42) -> None:
        self.api_url = api_url.rstrip('/')
        self.model_name = model_name
        self.seed = seed
        # Quick health-check — non-fatal if it fails, just warn.
        try:
            resp = requests.get(f'{self.api_url}/health', timeout=5)
            if resp.status_code == 200:
                print(f'[sample] API reachable at {self.api_url}')
            else:
                print(f'[sample] ⚠ API health-check returned {resp.status_code}; will try anyway')
        except Exception:
            print(f'[sample] ⚠ Could not reach {self.api_url}/health — will try completions endpoint directly')

    def generate_iter(
        self,
        prompts: Sequence[str],
        *,
        batch_size: int,  # unused — the API is queried one prompt at a time
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        chat: bool = False,
    ) -> 'Iterator[tuple[int, str]]':
        timeout = 120

        for i, prompt in enumerate(prompts):
            payload: dict = {
                'prompt': prompt,
                'max_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'min_p': min_p,
                'repeat_penalty': repetition_penalty,
                'seed': self.seed + i,
                'stream': False,
            }
            if self.model_name:
                payload['model'] = self.model_name

            resp = requests.post(f'{self.api_url}/v1/completions', json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            choice = data['choices'][0]
            text = choice.get('text', '')
            yield i, text.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='sample.py — creative text generation from seed prompts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Local checkpoint (raw completions)
  python sample.py sample_seeds.txt --model-folder checkpoints/checkpoint-10155

  # Local checkpoint (chat-template mode)
  python sample.py sample_seeds.txt --model-folder checkpoints/checkpoint-10155 --chat

  # Remote llama.cpp server
  python sample.py sample_seeds.txt --api-url http://localhost:8080

  # JSON-lines output, 5 samples per seed
  python sample.py sample_seeds.txt --model-folder checkpoints/checkpoint-10155 --format jsonl --samples-per-seed 5

  # Ultra-creative sweep
  python sample.py sample_seeds.txt --model-folder checkpoints/checkpoint-10155 --temperature 1.4 --top-p 0.98 --top-k 80
""",
    )

    # ── Required (mutually exclusive group with simple validation) ──────────
    backend = p.add_argument_group('backend (one required)')
    backend.add_argument(
        '--model-folder',
        type=Path,
        default=None,
        help='Path to a local HuggingFace checkpoint directory.',
    )
    backend.add_argument(
        '--api-url',
        type=str,
        default=None,
        help='URL of an OpenAI-compatible server (e.g. http://localhost:8080).',
    )
    backend.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name to pass to the API server (optional; some servers require it).',
    )

    # ── Input ───────────────────────────────────────────────────────────────
    p.add_argument(
        'seeds_file',
        type=Path,
        help='Text file with one seed prompt per line.',
    )

    # ── Output ──────────────────────────────────────────────────────────────
    p.add_argument(
        '-o',
        '--output',
        type=Path,
        default=None,
        help='Write output to this file instead of stdout.',
    )
    p.add_argument(
        '--format',
        choices=('text', 'jsonl'),
        default='text',
        help="Output format: 'text' (default) or 'jsonl' (one JSON object per line).",
    )
    p.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit the number of seeds to process (for testing).',
    )
    p.add_argument(
        '--shuffle',
        action='store_true',
        default=True,
        help='Shuffle the seed prompts before generating (default: preserve order).',
    )
    p.add_argument(
        '--seed-words',
        type=str,
        default='2-4',
        help='Keep only the first N words of each seed before generating, where '
        'N is chosen at random per seed within this range (default: "2-4"). '
        'Pass a single number (e.g. "4") for a fixed count, or "0" to disable.',
    )

    # ── Generation parameters ───────────────────────────────────────────────
    gen = p.add_argument_group('generation')
    gen.add_argument(
        '--samples-per-seed',
        type=int,
        default=1,
        help='How many completions to generate for each seed prompt (default: 1).',
    )
    gen.add_argument('--max-tokens', type=int, default=1024, help='Max new tokens per sample (default: 1024).')
    gen.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Prompts generated per forward pass for the local backend; results are flushed to disk after each batch (default: 8).',
    )
    gen.add_argument(
        '--temperature',
        type=float,
        default=1.25,
        help='Sampling temperature — higher = more creative (default: 1.2).',
    )
    gen.add_argument('--top-p', type=float, default=0.95, help='Nucleus sampling threshold (default: 0.95).')
    gen.add_argument(
        '--top-k',
        type=int,
        default=60,
        help='Top-K sampling (default: 60).',
    )
    gen.add_argument(
        '--min-p',
        type=float,
        default=0.033,
        help='Minimum probability for token consideration (default: 0.033).',
    )
    gen.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.075,
        help='Repetition penalty > 1 discourages repeats (default: 1.075).',
    )
    gen.add_argument('--chat', action='store_true', help='Apply chat template to seeds before generation.')
    gen.add_argument(
        '--show-special-tokens',
        action='store_true',
        help='Keep special tokens in the decoded output (debug).',
    )

    # ── Local-model specific ────────────────────────────────────────────────
    local = p.add_argument_group('local model options')
    local.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda', 'mps'))
    local.add_argument('--dtype', default='auto', choices=('auto', 'float32', 'float16', 'bfloat16'))

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    # ── Validate backend selection ──────────────────────────────────────────
    if args.model_folder is None and args.api_url is None:
        print('ERROR: you must provide either --model-folder or --api-url.', file=sys.stderr)
        sys.exit(1)
    if args.model_folder is not None and args.api_url is not None:
        print('ERROR: provide only one of --model-folder or --api-url, not both.', file=sys.stderr)
        sys.exit(1)

    # ── Validate parameters ─────────────────────────────────────────────────
    if args.samples_per_seed < 1:
        raise ValueError('--samples-per-seed must be >= 1')
    if args.max_tokens < 5:
        raise ValueError('--max-tokens must be >= 5')
    if args.temperature < 0:
        raise ValueError('--temperature must be >= 0')
    if args.repetition_penalty <= 0:
        raise ValueError('--repetition-penalty must be > 0')
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError('--top-p must be in (0, 1]')
    if args.top_k < 1:
        raise ValueError('--top-k must be >= 1')
    if not 0.0 <= args.min_p <= 1.0:
        raise ValueError('--min-p must be in [0, 1]')

    # ── Read seeds ───────────────────────────────────────────────────────────
    seeds_path: Path = args.seeds_file
    if not seeds_path.exists():
        print(f'ERROR: seeds file not found: {seeds_path}', file=sys.stderr)
        sys.exit(1)

    with open(seeds_path, encoding='utf-8') as fh:
        raw_seeds = [line.rstrip('\n') for line in fh if line.strip()]
    if not raw_seeds:
        print('ERROR: seeds file is empty.', file=sys.stderr)
        sys.exit(1)

    if args.limit is not None:
        raw_seeds = raw_seeds[: args.limit]

    # ── Truncate each seed to a short, randomly-sized prefix ──────────────────
    seed_lo, seed_hi = _parse_word_range(args.seed_words)
    if seed_hi > 0:
        rng = random.Random()
        raw_seeds = [_truncate_words(s, seed_lo, seed_hi, rng) for s in raw_seeds]
        raw_seeds = [s for s in raw_seeds if s]  # drop any that became empty
        if not raw_seeds:
            print('ERROR: all seeds became empty after --seed-words truncation.', file=sys.stderr)
            sys.exit(1)

    print(f'[sample] Loaded {len(raw_seeds)} seed(s) from {seeds_path}')
    print(
        f'[sample] Params: temp={args.temperature}  top-p={args.top_p}  '
        f'top-k={args.top_k}  min-p={args.min_p}  rep-pen={args.repetition_penalty}  '
        f'max-tokens={args.max_tokens}  samples/seed={args.samples_per_seed}'
    )

    # ── Build prompt list (repeat each seed N times) ────────────────────────
    prompts: List[str] = []
    seed_labels: List[str] = []  # track which original seed each prompt came from
    if args.shuffle:
        random.shuffle(raw_seeds)

    for seed_text in raw_seeds:
        for _ in range(args.samples_per_seed):
            prompts.append(seed_text)
            seed_labels.append(seed_text)

    # ── Select backend ──────────────────────────────────────────────────────
    if args.model_folder is not None:
        if not args.model_folder.is_dir():
            print(f'ERROR: model folder not found: {args.model_folder}', file=sys.stderr)
            sys.exit(1)
        backend = LocalModelBackend(
            model_folder=args.model_folder,
            device_str=args.device,
            dtype_str=args.dtype,
        )
    else:
        assert args.api_url is not None
        backend = ApiBackend(
            api_url=args.api_url,
            model_name=args.model,
        )

    # ── Open output sink ──────────────────────────────────────────────────────
    # The file is written incrementally and flushed after every sample, so if
    # the process dies partway through a long run, everything generated up to
    # that point is already safely on disk.
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_fh = open(args.output, 'w', encoding='utf-8')
        sink_desc = str(args.output)
    else:
        out_fh = sys.stdout
        sink_desc = '<stdout>'

    delimiter = '\n\n-----\n\n'

    # ── Generate + stream to disk ─────────────────────────────────────────────
    t0 = time.perf_counter()
    count = 0
    try:
        for idx, text in backend.generate_iter(
            prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            chat=args.chat,
        ):
            full = _smart_join(seed_labels[idx], text)
            if args.format == 'jsonl':
                obj = {'seed': seed_labels[idx], 'text': full}
                out_fh.write(json.dumps(obj, ensure_ascii=False) + '\n')
            else:
                if count > 0:
                    out_fh.write(delimiter)
                out_fh.write(full)
            out_fh.flush()  # push to the OS so a crash can't lose this sample
            count += 1
            print(f'[sample] {count}/{len(prompts)} samples written', end='\r', file=sys.stderr)
    except KeyboardInterrupt:
        print(f'\n[sample] Interrupted — {count} sample(s) saved to {sink_desc}', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'\nERROR: generation failed after {count} sample(s) — {exc}', file=sys.stderr)
        print(f'[sample] {count} sample(s) already saved to {sink_desc}', file=sys.stderr)
        sys.exit(1)
    finally:
        if args.format == 'text' and count > 0:
            out_fh.write('\n')
            out_fh.flush()
        if out_fh is not sys.stdout:
            out_fh.close()

    elapsed = time.perf_counter() - t0
    print(f'\n[sample] Generated {count} sample(s) in {elapsed:.1f}s → {sink_desc}', file=sys.stderr)


if __name__ == '__main__':
    main()

import argparse
import operator
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

PROJECT_DIR = Path(__file__).resolve().parent

# Operands tested: every pair in [0, 12] for each operation, mirroring the dataset.
OPERANDS = list(range(0, 13))

# Operation symbol → (callable, human name). The prompt is always
# "Calculate {x} {symbol} {y}." to match the dataset's "Calculate N OP N." phrasing.
OPS = {
    '+': (operator.add, 'addition'),
    '-': (operator.sub, 'subtraction'),
    '*': (operator.mul, 'multiplication'),
    '/': (operator.truediv, 'division'),
}

# The dataset answers in two phrasings — "X OP Y = Z." and "X OP Y is Z." — plus
# word forms ("The sum of X and Y is Z."). In every case the *result* is the number
# that follows "=" or the word "is", so a single regex grabs it for all templates.
RESULT_RE = re.compile(r'(?:=|\bis)\s*(-?\d+(?:\.\d+)?)')

# Numerical tolerance when comparing the model's number to the true value.
# Division answers are rounded to 3 decimals in the dataset, so a small slack
# absorbs "0.909" vs the exact 0.9090909…; integer ops match well within it.
TOLERANCE = 0.01


def select_device(name: str) -> torch.device:
    if name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(name)


def select_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == 'float32':
        return torch.float32
    if name == 'float16':
        return torch.float16
    if name == 'bfloat16':
        return torch.bfloat16
    if device.type == 'cuda':
        return torch.float16
    return torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Memorization test: every +/-/*// over operands 0..12.')
    parser.add_argument('--checkpoint', type=Path, help='Specific checkpoint directory to load.')
    parser.add_argument(
        '--checkpoints-dir',
        type=Path,
        default=PROJECT_DIR / 'checkpoints',
        help='Directory containing checkpoints (used when --checkpoint is not given).',
    )
    parser.add_argument('--tokens', type=int, default=32, help='Number of new tokens to generate per prompt.')
    # Decoding defaults to greedy (temperature 0) so the score is reproducible.
    # The sampling flags mirror vibe_check.py for when you want to probe variance.
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--top-k', type=int, default=25)
    parser.add_argument('--repetition-penalty', type=float, default=1.1)
    parser.add_argument('--batch-size', type=int, default=64, help='Prompts per generation batch.')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda', 'mps'))
    parser.add_argument('--dtype', default='auto', choices=('auto', 'float32', 'float16', 'bfloat16'))
    parser.add_argument('--show-failures', type=int, default=50, help='How many failing cases to print (0 = none).')
    return parser.parse_args()


def build_cases() -> list[dict]:
    """Every (x, op, y) over OPERANDS, skipping division by zero."""
    cases = []
    for symbol, (fn, _name) in OPS.items():
        for x in OPERANDS:
            for y in OPERANDS:
                if symbol == '/' and y == 0:
                    continue  # division by zero — not in the dataset, skip
                cases.append(
                    {
                        'symbol': symbol,
                        'x': x,
                        'y': y,
                        'prompt': f'Calculate {x} {symbol} {y}.',
                        'expected': fn(x, y),
                    }
                )
    return cases


def grade(generated: str, expected: float) -> tuple[bool, float | None]:
    """Extract the result number from the model's answer and compare numerically."""
    m = RESULT_RE.search(generated)
    if m is None:
        return False, None
    got = float(m.group(1))
    return abs(got - expected) <= TOLERANCE, got


def generate_batch(model, tokenizer, prompts: list[str], args, device) -> list[str]:
    """Render prompts through the chat template, generate, return ONLY the new text."""
    formatted = [
        tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in prompts
    ]
    inputs = tokenizer(formatted, return_tensors='pt', padding=True).to(device)
    input_len = inputs['input_ids'].shape[1]
    do_sample = args.temperature > 0

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            top_k=args.top_k if do_sample else None,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt so grading only ever sees the generated answer.
    new_ids = output_ids[:, input_len:]
    return tokenizer.batch_decode(new_ids, skip_special_tokens=True)


def fmt_expected(symbol: str, expected: float) -> str:
    """Render the expected value the way the dataset does (ints plain, division as float)."""
    if symbol == '/':
        return str(round(expected, 3))
    return str(int(expected))


def print_grid(symbol: str, results: list[dict]) -> None:
    """13x13 pass/fail grid for one operation. '·' marks skipped (division by zero)."""
    by_xy = {(r['x'], r['y']): r['correct'] for r in results}
    header = '   y│' + ''.join(f'{y:>4}' for y in OPERANDS)
    print(header)
    print('  ──┼' + '─' * (4 * len(OPERANDS)))
    for x in OPERANDS:
        cells = []
        for y in OPERANDS:
            if (x, y) not in by_xy:
                cells.append('   ·')
            else:
                cells.append('   ✓' if by_xy[(x, y)] else '   ✗')
        print(f'{x:>3} │' + ''.join(cells))


def main() -> None:
    args = parse_args()
    if args.tokens < 5:
        raise ValueError('--tokens must be at least 5')
    if args.temperature < 0:
        raise ValueError('--temperature must be >= 0')
    if args.repetition_penalty <= 0:
        raise ValueError('--repetition-penalty must be > 0')
    if args.batch_size < 1:
        raise ValueError('--batch-size must be >= 1')

    torch.manual_seed(args.seed)

    if args.checkpoint is not None:
        checkpoint_dir = args.checkpoint
    else:
        checkpoints_dir = str(args.checkpoints_dir)
        checkpoint_dir = get_last_checkpoint(checkpoints_dir) if os.path.isdir(checkpoints_dir) else None
        if checkpoint_dir is None:
            raise FileNotFoundError(f'No checkpoint found in {args.checkpoints_dir}')
        checkpoint_dir = Path(checkpoint_dir)

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)

    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, dtype=dtype)
    model.config.use_cache = True
    model.to(device)
    model.eval()

    cfg = model.config
    num_params = sum(p.numel() for p in model.parameters())
    print(f'model type   : {cfg.model_type}')
    print(f'architecture : {type(model).__name__}')
    print(f'parameters   : {num_params:,}')
    print(f'checkpoint   : {checkpoint_dir}')
    decode = 'greedy' if args.temperature == 0 else (f'sample(t={args.temperature}, top_p={args.top_p}, top_k={args.top_k})')
    print(f'decoding     : {decode}, rep_penalty={args.repetition_penalty}')

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # required for correct batch generation

    cases = build_cases()
    print(f'\nRunning {len(cases):,} prompts ("Calculate x OP y.") ...\n')

    for start in range(0, len(cases), args.batch_size):
        batch = cases[start : start + args.batch_size]
        outputs = generate_batch(model, tokenizer, [c['prompt'] for c in batch], args, device)
        for case, gen in zip(batch, outputs):
            correct, got = grade(gen, case['expected'])
            case['correct'] = correct
            case['got'] = got
            case['generated'] = gen.strip()

    # ── Per-operation grids + tallies ────────────────────────────────────────
    print('=' * 60)
    print('PER-OPERATION RESULTS')
    print('=' * 60)
    op_scores: dict[str, tuple[int, int]] = {}
    for symbol, (_fn, name) in OPS.items():
        results = [c for c in cases if c['symbol'] == symbol]
        correct = sum(c['correct'] for c in results)
        total = len(results)
        op_scores[symbol] = (correct, total)
        pct = 100 * correct / total if total else 0.0
        print(f'\n[ {symbol} ]  {name}: {correct}/{total} ({pct:.1f}%)')
        print_grid(symbol, results)

    # ── Summary table ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    total_correct = total_all = 0
    for symbol, (_fn, name) in OPS.items():
        c, t = op_scores[symbol]
        total_correct += c
        total_all += t
        bar = '█' * int(20 * c / t) if t else ''
        print(f'  {symbol}  {name:<16} {c:>4}/{t:<4} ({100 * c / t:5.1f}%)  {bar}')
    overall = 100 * total_correct / total_all if total_all else 0.0
    print('  ' + '-' * 50)
    print(f'     {"TOTAL":<16} {total_correct:>4}/{total_all:<4} ({overall:5.1f}%)')

    # ── Sample failures ──────────────────────────────────────────────────────
    if args.show_failures:
        failures = [c for c in cases if not c['correct']]
        if failures:
            print('\n' + '=' * 60)
            print(f'SAMPLE FAILURES (showing {min(args.show_failures, len(failures))} of {len(failures)})')
            print('=' * 60)
            for c in failures[: args.show_failures]:
                exp = fmt_expected(c['symbol'], c['expected'])
                got = '<no number>' if c['got'] is None else c['got']
                print(f'  {c["prompt"]:<22} expected {exp:<8} got {got!s:<10} | {c["generated"]!r}')


if __name__ == '__main__':
    main()

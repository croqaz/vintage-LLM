from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent


def checkpoint_step(path: Path) -> int:
    state_path = path / 'trainer_state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        return int(state.get('step', 0) or 0)

    match = re.fullmatch(r'checkpoint-(\d+)', path.name)
    if match is not None:
        return int(match.group(1))

    if path.name == 'final':
        return 10**18

    return -1


def find_latest_checkpoint(checkpoints_dir: Path) -> Path:
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f'checkpoint directory does not exist: {checkpoints_dir}')

    candidates = [
        path
        for path in checkpoints_dir.iterdir()
        if path.is_dir() and (path / 'config.json').exists() and (path.name == 'final' or path.name.startswith('checkpoint-'))
    ]
    if not candidates:
        raise FileNotFoundError(f'no model checkpoints found in {checkpoints_dir}')

    return max(candidates, key=checkpoint_step)


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
    parser = argparse.ArgumentParser(description='Generate text from the latest tiny GPT-NeoX checkpoint.')
    parser.add_argument('prompt', nargs='?', help='Text prompt to continue.')
    parser.add_argument('--checkpoint', type=Path, help='Specific checkpoint directory to load.')
    parser.add_argument('--checkpoints-dir', type=Path, default=PROJECT_DIR / 'checkpoints')
    parser.add_argument('--tokenizer', type=Path, default=PROJECT_DIR / 'pythia-14m')
    parser.add_argument('--tokens', type=int, default=100, help='Number of new tokens to generate.')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--repetition-penalty', type=float, default=1.1)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda', 'mps'))
    parser.add_argument('--dtype', default='auto', choices=('auto', 'float32', 'float16', 'bfloat16'))
    parser.add_argument('--show-special-tokens', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prompt is None:
        raise ValueError('provide a prompt as a positional argument')
    if args.tokens < 1:
        raise ValueError('--tokens must be at least 1')
    if args.temperature < 0:
        raise ValueError('--temperature must be >= 0')
    if args.repetition_penalty <= 0:
        raise ValueError('--repetition-penalty must be > 0')

    torch.manual_seed(args.seed)

    checkpoint_dir = args.checkpoint if args.checkpoint is not None else find_latest_checkpoint(args.checkpoints_dir)
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)

    tokenizer_source = checkpoint_dir if (checkpoint_dir / 'tokenizer.json').exists() else args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, dtype=dtype)
    model.config.use_cache = True
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(args.prompt, return_tensors='pt').to(device)
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

    text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=not args.show_special_tokens,
        clean_up_tokenization_spaces=False,
    )
    print(f'checkpoint: {checkpoint_dir}')
    print(text)


if __name__ == '__main__':
    main()

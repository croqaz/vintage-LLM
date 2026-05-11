from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

PROJECT_DIR = Path(__file__).resolve().parent


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
    parser = argparse.ArgumentParser(description='Generate text from the latest checkpoint.')
    parser.add_argument('prompt', nargs='?', help='Text prompt to continue.')
    parser.add_argument('--checkpoint', type=Path, help='Specific checkpoint directory to load.')
    parser.add_argument(
        '--checkpoints-dir',
        type=Path,
        default=PROJECT_DIR / 'checkpoints',
        help='Directory containing checkpoints (used when --checkpoint is not given).',
    )
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
    print(f'model type      : {cfg.model_type}')
    print(f'architecture    : {type(model).__name__}')
    print(f'hidden layers   : {getattr(cfg, "num_hidden_layers", "n/a")}')
    print(f'attention heads : {getattr(cfg, "num_attention_heads", "n/a")}')
    print(f'parameters      : {num_params:,}')

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
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
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=not args.show_special_tokens,
        clean_up_tokenization_spaces=False,
    )
    print(f'checkpoint: {checkpoint_dir}')
    print('-' * 80)
    print(text)


if __name__ == '__main__':
    main()

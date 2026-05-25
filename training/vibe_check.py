import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_PROMPTS = [
    'Hello!',
    'Who are you?',
    'Greetings, my friend',
    "What's this place?",
    'What is love? Love is',
    'What is God? God is',
    'Are you a human?',
    'I am and idiot and',
    'White bread.',
]


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
    parser = argparse.ArgumentParser(description='Vibe check a model checkpoint with pre-defined prompts.')
    parser.add_argument('--chat', action='store_true', help='Use chat template for the prompts.')
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
    if args.tokens < 5:
        raise ValueError('--tokens must be at least 5')
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
    print(f'model type   : {cfg.model_type}')
    print(f'architecture : {type(model).__name__}')
    print(f'parameters   : {num_params:,}')
    print(f'checkpoint   : {checkpoint_dir}')

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Crucial for batch generation
    tokenizer.padding_side = 'left'

    prompts = DEFAULT_PROMPTS
    formatted_prompts = []

    for p in prompts:
        if args.chat:
            messages = [{'role': 'user', 'content': p}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted)
        else:
            formatted_prompts.append(p)

    inputs = tokenizer(formatted_prompts, return_tensors='pt', padding=True).to(device)
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

    print(f'\n=== VIBE CHECK MODEL ({len(prompts)} prompts) ===\n')
    for i, (original_prompt, out_ids) in enumerate(zip(prompts, output_ids)):
        text = tokenizer.decode(
            out_ids,
            skip_special_tokens=not args.show_special_tokens,
            clean_up_tokenization_spaces=False,
        )
        print(f'--- Prompt {i + 1} ---')
        print(f'User: {original_prompt}')
        print('\nModel:')
        # .lstrip() is used mostly for left pad tokens if include_special_tokens is true
        print(text.lstrip() if args.show_special_tokens else text.strip())
        print('\n' + '=' * 80 + '\n')


if __name__ == '__main__':
    main()

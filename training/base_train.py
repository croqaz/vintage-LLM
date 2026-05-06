import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_DIR = Path(__file__).resolve().parent


class TokenBlockDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, bin_path: Path, block_size: int, dtype: np.dtype) -> None:
        self.bin_path = bin_path
        self.block_size = block_size
        self.tokens = np.memmap(bin_path, dtype=dtype, mode='r')
        self.length = max(0, (len(self.tokens) - 1) // block_size)
        if self.length <= 1:
            raise ValueError(f'{bin_path} does not contain enough tokens for block size {block_size}')

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.block_size
        stop = start + self.block_size
        input_ids = torch.from_numpy(np.asarray(self.tokens[start:stop], dtype=np.int64))
        labels = torch.from_numpy(np.asarray(self.tokens[start + 1 : stop + 1], dtype=np.int64))
        return {'input_ids': input_ids, 'labels': labels}


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: object | None,
    checkpoint_dir: Path,
    step: int,
    epoch: int,
    train_loss: float | None,
    val_loss: float | None,
) -> None:
    accelerator.wait_for_everyone()
    accelerator.save_state(checkpoint_dir)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(checkpoint_dir, save_function=accelerator.save)
        if tokenizer is not None:
            tokenizer.save_pretrained(checkpoint_dir)

        state = {
            'step': step,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        (checkpoint_dir / 'trainer_state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        print(f'Saved checkpoint: {checkpoint_dir}')


def checkpoint_step(path: Path) -> int:
    if path.name == 'final':
        state_path = path / 'trainer_state.json'
        if state_path.exists():
            return int(json.loads(state_path.read_text(encoding='utf-8')).get('step', 0))
        return -1

    match = re.fullmatch(r'checkpoint-(\d+)', path.name)
    if match is None:
        return -1
    return int(match.group(1))


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None

    candidates = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and (path.name == 'final' or path.name.startswith('checkpoint-')) and (path / 'trainer_state.json').exists()
    ]
    if not candidates:
        return None

    return max(candidates, key=checkpoint_step)


def load_trainer_state(checkpoint_dir: Path) -> dict[str, object]:
    state_path = checkpoint_dir / 'trainer_state.json'
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding='utf-8'))


@torch.no_grad()
def evaluate(model: torch.nn.Module, valid_loader: DataLoader, accelerator: Accelerator, max_batches: int) -> float:
    model.eval()
    losses: list[float] = []
    for batch_index, batch in enumerate(valid_loader):
        if batch_index >= max_batches:
            break
        outputs = model(**batch)
        loss = accelerator.gather_for_metrics(outputs.loss.detach()).mean()
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a tiny GPT-NeoX model on tokenized .bin files.')
    parser.add_argument('--train-bin', type=Path, default=PROJECT_DIR / 'train.bin')
    parser.add_argument('--valid-bin', type=Path, default=PROJECT_DIR / 'valid.bin')
    parser.add_argument('--model-config', type=Path, default=PROJECT_DIR / 'pythia-14m')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_DIR / 'checkpoints')
    parser.add_argument('--dtype', choices=('uint16', 'uint32'), default='uint16')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--save-every-iterations', type=int, default=500)
    parser.add_argument('--eval-every-iterations', type=int, default=100)
    parser.add_argument('--eval-batches', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--mixed-precision', choices=('no', 'fp16', 'bf16'), default='no')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError('--epochs must be at least 1')
    if args.block_size < 1:
        raise ValueError('--block-size must be at least 1')

    accelerator = Accelerator(mixed_precision=None if args.mixed_precision == 'no' else args.mixed_precision)
    torch.manual_seed(args.seed)

    np_dtype = np.dtype(args.dtype)
    train_dataset = TokenBlockDataset(args.train_bin, args.block_size, np_dtype)
    valid_dataset = TokenBlockDataset(args.valid_bin, args.block_size, np_dtype)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    config = AutoConfig.from_pretrained(args.model_config)
    config.max_position_embeddings = max(config.max_position_embeddings, args.block_size)
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(config).float()
    model.gradient_checkpointing_enable()

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_config, use_fast=True)
    except OSError:
        tokenizer = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(args.warmup_steps, max(0, total_steps - 1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, valid_loader, scheduler)

    resume_checkpoint = find_latest_checkpoint(args.output_dir)
    global_step = 0
    last_train_loss: float | None = None
    last_val_loss: float | None = None

    if resume_checkpoint is not None:
        accelerator.load_state(resume_checkpoint)
        state = load_trainer_state(resume_checkpoint)
        global_step = int(state.get('step', 0) or 0)
        last_train_loss = state.get('train_loss')  # type: ignore[assignment]
        last_val_loss = state.get('val_loss')  # type: ignore[assignment]
        if accelerator.is_main_process:
            print(f'Resumed from checkpoint: {resume_checkpoint} at step {global_step}')

    if accelerator.is_main_process:
        print(f'device: {accelerator.device}')
        print(f'train tokens: {len(train_dataset.tokens):,}')
        print(f'valid tokens: {len(valid_dataset.tokens):,}')
        print(f'train batches per epoch: {len(train_loader):,}')
        print(f'total optimizer steps: {total_steps:,}')
        print('-' * 60)

    model.train()

    for epoch in range(1, args.epochs + 1):
        for batch_index, batch in enumerate(train_loader):
            target_step = (epoch - 1) * len(train_loader) + batch_index + 1
            if target_step <= global_step:
                continue

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            if args.gradient_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.gradient_clip)

            optimizer.step()
            scheduler.step()

            global_step = target_step
            last_train_loss = float(accelerator.gather_for_metrics(loss.detach()).mean().item())

            if accelerator.is_main_process and global_step % 10 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f'Epoch {epoch}/{args.epochs} | Step {global_step}/{total_steps} | Loss {last_train_loss:.4f} | LR {lr:.2e}')

            if args.eval_every_iterations > 0 and global_step % args.eval_every_iterations == 0:
                last_val_loss = evaluate(model, valid_loader, accelerator, args.eval_batches)
                if accelerator.is_main_process:
                    ppl = math.exp(last_val_loss) if last_val_loss < 20 else float('inf')
                    print(f'Validation step {global_step}: Loss {last_val_loss:.4f} | PPL {ppl:.2f}')
                    print('-' * 60)

            if args.save_every_iterations > 0 and global_step % args.save_every_iterations == 0:
                checkpoint_dir = args.output_dir / f'checkpoint-{global_step:06d}'
                save_checkpoint(accelerator, model, tokenizer, checkpoint_dir, global_step, epoch, last_train_loss, last_val_loss)

    save_checkpoint(accelerator, model, tokenizer, args.output_dir / 'final', global_step, args.epochs, last_train_loss, last_val_loss)


if __name__ == '__main__':
    main()

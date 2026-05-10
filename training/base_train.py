#!/usr/bin/env python3
"""
LLM Pre-training Script.
Trains a causal language model from scratch using HuggingFace Transformers Trainer.
"""

import glob
import math
import os
import shutil
import sys
import time
import tomllib
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    ProgressCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

BF16_SUPPORTED = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# ============================================================================
# Binary Dataset Loader
# ============================================================================


class ConcatMemmap:
    """Virtual concatenation of multiple memmaps without copying into RAM."""

    def __init__(self, arrs):
        self.arrs = arrs
        self.lens = np.array([len(a) for a in arrs])
        self.offsets = np.concatenate([[0], np.cumsum(self.lens)])
        self._total = int(self.offsets[-1])

    def __len__(self):
        return self._total

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._total)
            if step != 1:
                return np.concatenate([self.arrs[0]])[key]  # fallback
            parts = []
            while start < stop:
                i = int(np.searchsorted(self.offsets, start, side='right')) - 1
                local_start = start - int(self.offsets[i])
                local_stop = min(int(self.lens[i]), stop - int(self.offsets[i]))
                parts.append(self.arrs[i][local_start:local_stop])
                start = int(self.offsets[i]) + local_stop
            return np.concatenate(parts) if parts else np.array([], dtype=self.arrs[0].dtype)
        else:
            if key < 0:
                key += self._total
            i = int(np.searchsorted(self.offsets, key, side='right')) - 1
            return self.arrs[i][key - int(self.offsets[i])]


class BinaryTokenDataset(Dataset):
    """
    Document-aware dataset for pre-tokenized binary data.

    Instead of slicing the token stream into fixed-length chunks (which would
    let the model attend across unrelated document boundaries), this dataset
    respects document boundaries marked by EOS tokens in the binary data.

    Each sequence is guaranteed to come from a single document.  Sequences at
    document boundaries are shorter than seq_length and get padded: input_ids
    is padded with pad_token_id and labels is padded with -1 (ignored by
    cross-entropy loss).

    Per-epoch, chunk offsets within each document are randomised so the model
    sees different windows across epochs.
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_length: int,
        eos_token_id: int,
        pad_token_id: int,
        base_seed: int = 42,
    ):
        self.data = data
        self.seq_length = seq_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.base_seed = base_seed

        # Build document boundary index by finding all EOS positions.
        # Each document spans [doc_start, doc_end) where doc_end is one past EOS.
        print(f'  → Scanning for document boundaries (EOS id={eos_token_id})...')
        eos_positions = (
            np.where(np.frombuffer(data, dtype=np.uint16) == eos_token_id)[0]
            if isinstance(data, np.memmap)
            else np.where(np.array(data[:], dtype=np.uint16) == eos_token_id)[0]
        )

        if len(eos_positions) == 0:
            # Fallback: treat entire data as one document
            print('  → WARNING: No EOS tokens found, treating all data as one document')
            self.doc_starts = np.array([0], dtype=np.int64)
            self.doc_ends = np.array([len(data)], dtype=np.int64)
        else:
            starts = np.concatenate([[0], eos_positions[:-1] + 1]).astype(np.int64)
            ends = (eos_positions + 1).astype(np.int64)
            # Filter out empty documents
            mask = ends > starts
            self.doc_starts = starts[mask]
            self.doc_ends = ends[mask]

        self.num_docs = len(self.doc_starts)
        self.doc_lengths = self.doc_ends - self.doc_starts
        total_doc_tokens = int(self.doc_lengths.sum())

        print(f'  → Found {self.num_docs:,} documents ({total_doc_tokens:,} tokens)')
        print(
            f'  → Doc lengths: min={int(self.doc_lengths.min()):,}, '
            f'max={int(self.doc_lengths.max()):,}, '
            f'median={int(np.median(self.doc_lengths)):,}'
        )

        # Build the initial chunk index (epoch 0)
        self._chunks: list[tuple[int, int]] = []  # (start, length) pairs
        self._build_chunks(epoch=0)

    def _build_chunks(self, epoch: int):
        """Build list of (start, length) chunks respecting document boundaries.

        Each document is divided into seq_length-sized chunks.  A per-epoch
        random offset shifts the chunk grid within each document so the model
        sees different windows across epochs.  The final chunk of each document
        may be shorter than seq_length (will be padded in __getitem__).
        """
        rng = np.random.RandomState(seed=hash((self.base_seed, epoch)) & 0xFFFFFFFF)
        chunks: list[tuple[int, int]] = []

        for doc_idx in range(self.num_docs):
            doc_start = int(self.doc_starts[doc_idx])
            doc_len = int(self.doc_lengths[doc_idx])

            # Random offset within [0, min(seq_length, doc_len) - 1]
            max_offset = min(self.seq_length, doc_len) - 1
            offset = rng.randint(0, max_offset + 1) if max_offset > 0 else 0

            pos = offset
            while pos < doc_len:
                chunk_len = min(self.seq_length, doc_len - pos)
                chunks.append((doc_start + pos, chunk_len))
                pos += self.seq_length

            # Also add the skipped prefix as a chunk if offset > 0 and large enough
            if offset > 0:
                chunks.append((doc_start, min(offset, self.seq_length)))

        # Shuffle the chunk order
        rng.shuffle(chunks)
        self._chunks = chunks
        self._num_sequences = len(chunks)

    def set_epoch(self, epoch: int):
        """Rebuild chunks with a new random offset and shuffle order."""
        self._build_chunks(epoch)

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start, length = self._chunks[idx]

        # Read tokens from memmap
        raw = self.data[start : start + length]
        tokens = np.array(raw, dtype=np.int64)

        if length >= self.seq_length:
            # Full-length chunk, no padding needed
            input_ids = torch.from_numpy(tokens)
            labels = torch.from_numpy(tokens.copy())
        else:
            # Partial chunk at document boundary — pad the remainder
            input_ids = torch.full((self.seq_length,), self.pad_token_id, dtype=torch.long)
            labels = torch.full((self.seq_length,), -1, dtype=torch.long)
            input_ids[:length] = torch.from_numpy(tokens)
            labels[:length] = torch.from_numpy(tokens.copy())

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


def load_binary_files(
    file_pattern: Union[str, List[str]],
    seq_length: int,
    eos_token_id: int,
    pad_token_id: int,
    base_seed: int = 42,
    _preloaded_data=None,
) -> BinaryTokenDataset:
    """
    Load binary files from a list of paths or glob pattern.

    Args:
        file_pattern: str (glob pattern) or list of str (file paths)
        seq_length: sequence length for each sample
        eos_token_id: EOS token ID for document boundary detection
        pad_token_id: pad token ID for padding short sequences
        base_seed: base seed for random offset per epoch
        _preloaded_data: if provided, skip file loading and use this data directly

    Returns:
        BinaryTokenDataset instance.
    """
    if _preloaded_data is not None:
        return BinaryTokenDataset(_preloaded_data, seq_length, eos_token_id=eos_token_id, pad_token_id=pad_token_id, base_seed=base_seed)

    # Resolve file patterns to actual file paths
    if isinstance(file_pattern, str):
        # Single file or glob pattern
        if '*' in file_pattern or '?' in file_pattern:
            # Glob pattern
            files = sorted(glob.glob(file_pattern))
        else:
            # Single file
            files = [file_pattern]
    elif isinstance(file_pattern, list):
        # List of files or patterns
        all_files = []
        for pattern in file_pattern:
            if '*' in pattern or '?' in pattern:
                all_files.extend(sorted(glob.glob(pattern)))
            else:
                all_files.append(pattern)
        files = all_files
    else:
        raise ValueError(f'Invalid file_pattern type: {type(file_pattern)}')

    if not files:
        raise ValueError(f'No files found matching pattern: {file_pattern}')

    print(f'Loading {len(files)} file(s):')
    for f in files:
        print(f'  → {f}')

    # Memory-map all files
    memmaps: List[np.ndarray] = []
    total_tokens = 0

    for f in files:
        if not Path(f).exists():
            raise FileNotFoundError(f'File not found: {f}')

        arr = np.memmap(f, dtype=np.uint16, mode='r')
        file_tokens = len(arr)
        total_tokens += file_tokens
        memmaps.append(arr)
        print(f'  → Mapped {file_tokens:,} tokens from {Path(f).name}')

    # For a single file, use the memmap directly (zero-copy)
    # For multiple files, we must concatenate (copies into RAM)
    if len(memmaps) == 1:
        data = memmaps[0]
    else:
        data = ConcatMemmap(memmaps)

    print(f'Total tokens: {total_tokens:,}')
    return BinaryTokenDataset(data, seq_length, eos_token_id=eos_token_id, pad_token_id=pad_token_id, base_seed=base_seed)


# ============================================================================
# Custom Callbacks
# ============================================================================


class DetailedLoggingCallback(TrainerCallback):
    """Log detailed training metrics per batch."""

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None
        self.current_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(total=state.max_steps, dynamic_ncols=True)
            self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.close()
            self.training_bar = None

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and len(eval_dataloader):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(total=len(eval_dataloader), leave=self.training_bar is None, dynamic_ncols=True)
            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_predict(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float],
        **kwargs,
    ):
        """Called when logging occurs."""
        if state.is_world_process_zero and logs:
            # Logs contains:
            # - loss
            # - grad_norm
            # - learning_rate
            # - epoch
            epoch = state.epoch if state.epoch is not None else 0
            step = state.global_step
            total_steps = state.max_steps
            loss = logs.get('loss')
            lr = logs.get('learning_rate')

            # Only log if we have the essential metrics
            if loss is not None and lr is not None:
                # Calculate progress percentage
                progress = (step / total_steps * 100) if total_steps > 0 else 0
                grad_norm = logs.get('grad_norm')
                grad_str = f'grad_norm={grad_norm:.4f}' if grad_norm is not None else ''

                # GPU memory
                gpu_mem_str = ''
                if torch.cuda.is_available():
                    gpu_gb = torch.cuda.max_memory_allocated() / 1e9
                    gpu_mem_str = f'GPU={gpu_gb:.1f}GB'
                    torch.cuda.reset_peak_memory_stats()

                print(
                    f'[TRAIN] Epoch {epoch:.2f}/{args.num_train_epochs} | '
                    f'Step {step}/{total_steps} ({progress:.1f}%) | '
                    f'Loss {loss:.4f} | '
                    f'LR {lr:.2e} | '
                    f'{grad_str} | '
                    f'{gpu_mem_str}'
                )


class DetailedEvaluationCallback(TrainerCallback):
    """Log detailed validation metrics after each evaluation."""

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float],
        **kwargs,
    ):
        """Called after evaluation is complete."""
        if state.is_world_process_zero and metrics:
            # Metrics contains:
            # - eval_loss
            # - eval_runtime
            # - eval_samples_per_second
            # - eval_steps_per_second
            # - epoch
            # Extract metrics
            eval_loss = metrics.get('eval_loss', 0)
            eval_runtime = metrics.get('eval_runtime', 0)
            eval_samples_per_sec = metrics.get('eval_samples_per_second', 0)
            eval_steps_per_sec = metrics.get('eval_steps_per_second', 0)

            # Calculate perplexity
            try:
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float('inf')

            # Log detailed validation info
            print('\n' + '=' * 80)
            print(f'[VALIDATION] Step {state.global_step}')
            print('=' * 80)
            print(f'  Epoch:          {metrics.get("epoch", 0):.2f}')
            print(f'  Loss:           {eval_loss:.4f}')
            print(f'  Perplexity:     {perplexity:.2f}')
            print(f'  Runtime:        {eval_runtime:.2f}s')
            print(f'  Samples/sec:    {eval_samples_per_sec:.2f}')
            print(f'  Steps/sec:      {eval_steps_per_sec:.2f}')

            # Show additional metrics if available
            for key, value in metrics.items():
                if key not in [
                    'eval_loss',
                    'eval_runtime',
                    'eval_samples_per_second',
                    'eval_steps_per_second',
                    'epoch',
                ]:
                    print(f'  {key}: {value}')

            print('=' * 80 + '\n')


class EpochOffsetCallback(TrainerCallback):
    """Apply a random chunk offset at the start of each epoch so boundaries vary."""

    def __init__(self, dataset: 'BinaryTokenDataset'):
        self.dataset = dataset

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        epoch = int(state.epoch) if state.epoch is not None else 0
        self.dataset.set_epoch(epoch)
        if state.is_world_process_zero:
            print(f'[EPOCH] Epoch {epoch}: rebuilt {len(self.dataset):,} chunks with new random offsets')


# ============================================================================
# Configuration Loading
# ============================================================================


def load_config(config_path: str) -> Dict:
    """Load configuration from TOML file."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    return config


def validate_config(cfg: Dict, accelerator: Accelerator):
    """Validate configuration and check for common issues."""
    accelerator.print('\n' + '=' * 80)
    accelerator.print('CONFIGURATION VALIDATION')
    accelerator.print('=' * 80)

    # Check data files exist
    train_files = cfg['data']['train_files']
    valid_files = cfg['data']['valid_files']

    # Check if any files exist
    if isinstance(train_files, str) and not glob.glob(train_files):
        raise FileNotFoundError(f'No training files found: {train_files}')
    if isinstance(valid_files, str) and not glob.glob(valid_files):
        raise FileNotFoundError(f'No validation files found: {valid_files}')

    # Check precision settings
    if cfg['training']['bf16'] and not torch.cuda.is_bf16_supported():
        accelerator.print('⚠️ WARNING: bf16 requested but not supported on this GPU')
        accelerator.print('   Consider setting bf16=false and fp16=true in config.toml')

    # Check if CUDA is available for GPU training
    if not torch.cuda.is_available():
        accelerator.print('⚠️ WARNING: CUDA not available, training on CPU will be slow')

    accelerator.print('✓ Configuration validated')
    accelerator.print('=' * 80 + '\n')


# ============================================================================
# Main Training Function
# ============================================================================


def main():
    """Main training function."""

    # ========================================================================
    # Parse config and initialize Accelerator
    # ========================================================================

    config_path = 'training/config.toml'
    if not Path(config_path).exists():
        print(f'ERROR: Config file not found at {config_path}')
        print('Please create training/config.toml with your training configuration.')
        sys.exit(1)

    print(f'Loading configuration from {config_path}...')
    cfg = load_config(config_path)

    # Initialize Accelerator for distributed detection and printing
    # Note: Trainer will create its own internal Accelerator
    accelerator = Accelerator()
    # Capture before TrainingArguments resets AcceleratorState
    num_processes = accelerator.num_processes

    accelerator.print('\n' + '=' * 80)
    accelerator.print('PRE-TRAINING')
    accelerator.print('=' * 80)
    accelerator.print(f'Distributed setup: {num_processes} process(es)')
    accelerator.print(f'Device: {accelerator.device}')
    accelerator.print(f'Mixed precision: {accelerator.mixed_precision}')
    accelerator.print('=' * 80 + '\n')

    # ========================================================================
    # Set random seeds for reproducibility
    # ========================================================================

    seed = cfg['training'].get('seed', 42)
    set_seed(seed)
    accelerator.print(f'Random seed set to: {seed}')

    # ========================================================================
    # Enable TF32 matmul if supported (Ampere+ GPUs)
    # ========================================================================

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        # TF32 requires compute capability >= 8.0 (Ampere)
        if props.major >= 8:
            torch.set_float32_matmul_precision('high')
            accelerator.print(f'TF32 matmul enabled ({props.name}, compute {props.major}.{props.minor})')
        else:
            accelerator.print(f'TF32 matmul not supported ({props.name}, compute {props.major}.{props.minor}), skipping')

    # ========================================================================
    # Validate configuration
    # ========================================================================

    validate_config(cfg, accelerator)

    # ========================================================================
    # Load tokenizer
    # ========================================================================

    accelerator.print('Loading tokenizer...')
    tokenizer_name = cfg['data']['tokenizer']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Set pad token if not set (required for DataCollator)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    accelerator.print(f'✓ Loaded tokenizer: {tokenizer_name}')
    accelerator.print(f'  Vocab size: {len(tokenizer)}')
    accelerator.print(f'  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})')
    accelerator.print(f'  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})')
    accelerator.print()

    # ========================================================================
    # Create model from config
    # ========================================================================

    accelerator.print('Initializing model...')

    # Create model configuration
    model_kwargs = dict(cfg['model'])
    model_type = model_kwargs.pop('model_type')
    attn_implementation = model_kwargs.pop('attn_implementation', None)
    model_kwargs['use_cache'] = False
    model_config = AutoConfig.for_model(model_type=model_type, **model_kwargs)

    # Verify vocab size matches tokenizer
    if model_config.vocab_size != len(tokenizer):
        accelerator.print(f'⚠️ WARNING: Model vocab_size ({model_config.vocab_size}) != tokenizer vocab_size ({len(tokenizer)})')

    # Use attn_implementation if specified (e.g. "flash_attention_2", "sdpa")
    model_init_kwargs = {}
    if attn_implementation:
        model_init_kwargs['attn_implementation'] = attn_implementation
        accelerator.print(f'  Attention implementation: {attn_implementation}')
    # Initialize model with AutoModelForCausalLM
    model = AutoModelForCausalLM.from_config(model_config)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    accelerator.print(f'✓ Model initialized: {model.__class__.__name__}')
    accelerator.print(f'  Architecture: {model_config.model_type}')
    accelerator.print(f'  Total parameters: {num_params:,} ({num_params / 1e6:.2f}M)')
    accelerator.print(f'  Trainable parameters: {num_trainable_params:,} ({num_trainable_params / 1e6:.2f}M)')
    accelerator.print(f'  Hidden size: {model_config.hidden_size}')
    accelerator.print(f'  Layers: {model_config.num_hidden_layers}')
    accelerator.print(f'  Attention heads: {model_config.num_attention_heads}')
    accelerator.print(f'  FFN size: {model_config.intermediate_size}')
    accelerator.print(f'  Max position embeddings: {model_config.max_position_embeddings}')
    accelerator.print()

    # ========================================================================
    # Load datasets
    # ========================================================================

    accelerator.print('Loading datasets...')
    seq_length = cfg['data']['max_seq_length']

    accelerator.print('\n[TRAINING DATA]')
    train_dataset = load_binary_files(
        cfg['data']['train_files'], seq_length, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, base_seed=seed
    )

    accelerator.print('\n[VALIDATION DATA]')
    eval_dataset = load_binary_files(
        cfg['data']['valid_files'], seq_length, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, base_seed=seed
    )
    accelerator.print()

    # ========================================================================
    # Setup TrainingArguments
    # ========================================================================

    accelerator.print('Setting up training arguments...')

    output_dir = cfg['training']['output_dir']

    training_args = TrainingArguments(
        # Output
        output_dir=output_dir,
        # Training duration
        num_train_epochs=cfg['training']['num_train_epochs'],
        max_steps=-1,  # Train for full epochs
        # Batch sizes
        per_device_train_batch_size=cfg['training'].get('per_device_train_batch_size', 8),
        per_device_eval_batch_size=cfg['training'].get('per_device_eval_batch_size', 8),
        gradient_accumulation_steps=cfg['training'].get('gradient_accumulation_steps', 1),
        # Optimizer with tuned defaults
        optim=cfg['training'].get('optim', 'adamw_torch_fused'),
        learning_rate=cfg['training'].get('learning_rate', 5e-5),
        weight_decay=cfg['training'].get('weight_decay', 0.1),
        adam_beta1=cfg['training'].get('adam_beta1', 0.9),
        adam_beta2=cfg['training'].get('adam_beta2', 0.95),
        adam_epsilon=cfg['training'].get('adam_epsilon', 1e-8),
        max_grad_norm=cfg['training'].get('max_grad_norm', 1.0),
        # Learning rate scheduler
        lr_scheduler_type=cfg['training'].get('lr_scheduler_type', 'cosine_with_min_lr'),
        lr_scheduler_kwargs=cfg['training'].get('lr_scheduler_kwargs', {'min_lr_rate': 0.05}),
        warmup_steps=cfg['training'].get('warmup_steps', 100),
        # Precision
        bf16=cfg['training'].get('bf16', BF16_SUPPORTED),
        fp16=cfg['training'].get('fp16', False),
        # Compilation
        torch_compile=cfg['training'].get('torch_compile', True),
        # Gradient checkpointing
        gradient_checkpointing=cfg['training'].get('gradient_checkpointing', False),
        gradient_checkpointing_kwargs={'use_reentrant': False},
        # Checkpointing
        save_strategy=cfg['training'].get('save_strategy', 'steps'),
        save_steps=cfg['training'].get('save_steps', 500),
        save_total_limit=cfg['training'].get('save_total_limit', 3),
        # Evaluation
        eval_strategy=cfg['training'].get('eval_strategy', 'steps'),
        eval_steps=cfg['training'].get('eval_steps', 500),
        # Logging
        logging_strategy=cfg['training'].get('logging_strategy', 'steps'),
        logging_steps=cfg['training'].get('logging_steps', 10),
        logging_first_step=cfg['training'].get('logging_first_step', True),
        report_to=cfg['training'].get('report_to', ['trackio', 'tensorboard']),
        # Performance
        dataloader_num_workers=cfg['training'].get('dataloader_num_workers', 4),
        dataloader_pin_memory=cfg['training'].get('dataloader_pin_memory', True),
        dataloader_prefetch_factor=cfg['training'].get('dataloader_prefetch_factor', 2),
        # dataloader_persistent_workers intentionally disabled by default
        # Reproducibility
        seed=seed,
        data_seed=seed,
    )

    accelerator.print('✓ Training arguments configured')
    effective_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    accelerator.print(f'  Effective batch size: {effective_batch}')
    accelerator.print(f'  Epochs: {training_args.num_train_epochs}')
    accelerator.print(f'  Learning rate: {training_args.learning_rate}')
    accelerator.print(f'  Warmup steps: {training_args.warmup_steps}')
    accelerator.print(f'  Save steps: {training_args.save_steps}')
    accelerator.print(f'  Eval steps: {training_args.eval_steps}')
    accelerator.print()

    # ========================================================================
    # Create Trainer
    # ========================================================================

    accelerator.print('Creating Trainer...')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=[
            DetailedLoggingCallback(),
            DetailedEvaluationCallback(),
            EpochOffsetCallback(train_dataset),
        ],
    )
    # Remove default print for clean logging
    trainer.remove_callback(ProgressCallback)

    accelerator.print('✓ Trainer created with custom callbacks\n')

    # ========================================================================
    # Train! 🚂
    # ========================================================================

    accelerator.print('=' * 80)
    accelerator.print('STARTING TRAINING')
    accelerator.print('=' * 80)
    accelerator.print(f'Training samples: {len(train_dataset):,}')
    accelerator.print(f'Validation samples: {len(eval_dataset):,}')
    accelerator.print(f'Sequence length: {seq_length}')
    accelerator.print('=' * 80 + '\n')

    start_time = time.time()

    # Try to resume from last checkpoint if it exists, otherwise start fresh
    last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    if last_checkpoint:
        accelerator.print(f'Resuming from checkpoint: {last_checkpoint}')
    else:
        accelerator.print('No checkpoint found, training from scratch.')

    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)

    except KeyboardInterrupt:
        accelerator.print('\n' + '=' * 80)
        accelerator.print('TRAINING INTERRUPTED BY USER')
        accelerator.print('=' * 80)
        accelerator.print('Checkpoint saved. Resume training by running the script again.')
        accelerator.print('=' * 80 + '\n')
        sys.exit(0)

    except Exception as e:
        accelerator.print('\n' + '=' * 80)
        accelerator.print('TRAINING ERROR')
        accelerator.print('=' * 80)
        accelerator.print(f'Error: {e}')
        accelerator.print('=' * 80 + '\n')
        raise

    training_time = time.time() - start_time

    # ========================================================================
    # Save final model
    # ========================================================================

    accelerator.print('\n' + '=' * 80)
    accelerator.print('TRAINING COMPLETE')
    accelerator.print('=' * 80)
    accelerator.print(f'Training time: {training_time / 60:.2f} minutes')
    accelerator.print('=' * 80 + '\n')

    final_model_dir = cfg['training']['final_model_dir']
    accelerator.print(f'Saving final model to {final_model_dir}...')

    if accelerator.is_main_process:
        os.makedirs(final_model_dir, exist_ok=True)

        # Save model and tokenizer
        trainer.save_model(final_model_dir)
        trainer.save_state()
        tokenizer.save_pretrained(final_model_dir)
        # Remove training_args.bin which is not needed and can cause confusion
        os.remove(os.path.join(final_model_dir, 'training_args.bin'))

        # Also save the training config for reference
        shutil.copy(config_path, os.path.join(final_model_dir, 'training_config.toml'))

        accelerator.print(f'✓ Final model saved to {final_model_dir}')
        accelerator.print('  - Model weights: model.safetensors')
        accelerator.print('  - Model config: config.json')

    # Wait for all processes to finish
    accelerator.wait_for_everyone()

    accelerator.print('\n' + '=' * 80)
    accelerator.print('ALL DONE!')
    accelerator.print('=' * 80)
    accelerator.print(f'Checkpoints: {output_dir}')
    accelerator.print(f'Final model: {final_model_dir}')
    accelerator.print('=' * 80 + '\n')


if __name__ == '__main__':
    main()

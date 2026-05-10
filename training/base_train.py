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

# ============================================================================
# Binary Dataset Loader
# ============================================================================


class BinaryTokenDataset(Dataset):
    """
    Dataset for pre-tokenized binary data.
    Each sample is a fixed-length sequence of token IDs.
    Uses memmap directly to avoid loading entire files into RAM.
    Supports a per-epoch random offset so chunk boundaries vary across epochs.
    """

    def __init__(self, data: np.ndarray, seq_length: int):
        """
        Args:
            data: numpy memmap (or array) of tokens (uint16)
            seq_length: sequence length for each sample
        """
        self.data = data
        self.seq_length = seq_length
        self.offset = 0  # random offset applied per epoch

        # Calculate number of complete sequences (with no offset)
        self.num_sequences = len(self.data) // seq_length

        if self.num_sequences == 0:
            raise ValueError(f'Data too short: {len(self.data)} tokens < {seq_length} seq_length')

        # Identity mapping by default (no shuffle until set_epoch is called)
        self._index_map = np.arange(self.num_sequences)

        print(f'  → Created dataset: {self.num_sequences:,} sequences of length {seq_length}')

    def set_epoch(self, epoch: int):
        """Set a per-epoch random offset and shuffle index order."""
        rng = np.random.RandomState(seed=epoch)
        self.offset = rng.randint(0, self.seq_length)
        self.num_sequences = (len(self.data) - self.offset) // self.seq_length
        # Shuffle the order sequences are read so the model sees different
        # mini-batch compositions each epoch (the Trainer sampler already
        # shuffles, but this also shuffles the logical-to-physical mapping).
        self._index_map = rng.permutation(self.num_sequences)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        physical_idx = self._index_map[idx]
        start = self.offset + physical_idx * self.seq_length
        end = start + self.seq_length

        # Read from memmap and convert to int64 for PyTorch
        tokens = torch.from_numpy(self.data[start:end].astype(np.int64))

        return {
            'input_ids': tokens,
            'labels': tokens,
        }


def load_binary_files(file_pattern: Union[str, List[str]], seq_length: int) -> BinaryTokenDataset:
    """
    Load binary files from a list of paths or glob pattern.

    Args:
        file_pattern: str (glob pattern) or list of str (file paths)
        seq_length: sequence length for each sample

    Returns:
        BinaryTokenDataset instance
    """
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
    memmaps = []
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
        # If we switch to multiple shards, replace np.concatenate
        # with a virtual concatenation that reads from the correct memmap
        data = np.concatenate(memmaps)

    print(f'Total tokens: {total_tokens:,}')

    return BinaryTokenDataset(data, seq_length)


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

                print(
                    f'[TRAIN] Epoch {epoch:.2f}/{args.num_train_epochs} | '
                    f'Step {step}/{total_steps} ({progress:.1f}%) | '
                    f'Loss {loss:.4f} | '
                    f'LR {lr:.2e}'
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
            print(f'[OFFSET] Epoch {epoch}: chunk offset = {self.dataset.offset}')


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
    model_config = AutoConfig.for_model(
        model_type=cfg['model']['model_type'],
        vocab_size=cfg['model']['vocab_size'],
        hidden_size=cfg['model']['hidden_size'],
        num_hidden_layers=cfg['model']['num_hidden_layers'],
        num_attention_heads=cfg['model']['num_attention_heads'],
        intermediate_size=cfg['model']['intermediate_size'],
        max_position_embeddings=cfg['model']['max_position_embeddings'],
        rotary_pct=cfg['model']['rotary_pct'],
        rotary_emb_base=cfg['model']['rotary_emb_base'],
        use_cache=False,
    )

    # Verify vocab size matches tokenizer
    if model_config.vocab_size != len(tokenizer):
        accelerator.print(f'⚠️ WARNING: Model vocab_size ({model_config.vocab_size}) != tokenizer vocab_size ({len(tokenizer)})')

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
    train_dataset = load_binary_files(cfg['data']['train_files'], seq_length)

    accelerator.print('\n[VALIDATION DATA]')
    eval_dataset = load_binary_files(cfg['data']['valid_files'], seq_length)
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
        # Optimizer
        optim=cfg['training'].get('optim', 'adamw_torch_fused'),
        learning_rate=cfg['training'].get('learning_rate', 5e-5),
        weight_decay=cfg['training'].get('weight_decay', 0.0),
        adam_beta1=cfg['training'].get('adam_beta1', 0.9),
        adam_beta2=cfg['training'].get('adam_beta2', 0.999),
        adam_epsilon=cfg['training'].get('adam_epsilon', 1e-8),
        max_grad_norm=cfg['training'].get('max_grad_norm', 1.0),
        # Learning rate scheduler
        lr_scheduler_type=cfg['training'].get('lr_scheduler_type', 'linear'),
        warmup_steps=cfg['training'].get('warmup_steps', 100),
        # Precision
        bf16=cfg['training'].get('bf16', False),
        fp16=cfg['training'].get('fp16', False),
        # Compilation
        torch_compile=cfg['training'].get('torch_compile', False),
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
        report_to=cfg['training'].get('report_to', 'trackio'),
        # Performance
        dataloader_num_workers=cfg['training'].get('dataloader_num_workers', 2),
        dataloader_pin_memory=cfg['training'].get('dataloader_pin_memory', True),
        dataloader_prefetch_factor=cfg['training'].get('dataloader_prefetch_factor', 2),
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

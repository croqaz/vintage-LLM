#!/usr/bin/env python3
"""
LLM Supervised Fine-Tuning Script.

Fine-tunes a causal language model produced by base_train.py using TRL's SFTTrainer.

Supported training methods (set via [training] method in fine_tune_config.toml):
  - "full" : All parameters trained — best quality, recommended for small models.
  - "lora" : Only LoRA adapter parameters trained — fast iteration, ~1-5% of params.

Supported JSONL dataset formats (auto-detected from column names):
  - Conversational : {"messages": [{"role": "user", "content": "..."}, ...]}
  - Pre-formatted  : {"text": "...already templated text..."}

Usage:
    # Single GPU
    python fine_tune.py

    # Multi-GPU (DDP)
    accelerate launch fine_tune.py
"""

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import (
    AutoConfig as HFAutoConfig,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    ProgressCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

# ── Shared utilities from base_train ─────────────────────────────────────────
# Both scripts live in the same directory; import directly to avoid duplication.
sys.path.insert(0, str(Path(__file__).parent))
from base_train import (
    BF16_SUPPORTED,
    DetailedEvaluationCallback,
    DetailedLoggingCallback,
    load_config,
)

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'


# =============================================================================
# Dataset
# =============================================================================


def detect_format(dataset: Dataset) -> tuple[str, Optional[str]]:
    """
    Infer whether the dataset is conversational ("messages" column) or
    pre-formatted text ("text" column).

    Returns:
        (format_name, dataset_text_field)
        - ("conversational", None)   → SFTTrainer applies the chat template to
                                       the "messages" column automatically.
        - ("standard",      "text")  → SFTTrainer uses the "text" column as-is.
    """
    cols = dataset.column_names
    if 'messages' in cols:
        return 'conversational', None
    if 'text' in cols:
        return 'standard', 'text'
    raise ValueError(f'Dataset must have a "messages" column (conversational) or a "text" column (pre-formatted). Found columns: {cols}')


def load_sft_dataset(cfg: Dict, seed: int) -> tuple[Dataset, Dataset]:
    """
    Load the SFT dataset from a local JSONL file or a HuggingFace Hub dataset
    and produce a train / validation split.

    Config keys used (under [data]):
        dataset_path   : path to local .jsonl file (or list of paths)
        dataset_name   : HuggingFace Hub dataset name (alternative to dataset_path)
        dataset_split  : Hub split to load, default "train"
        val_fraction   : fraction held out for validation, default 0.05
    """
    data_cfg = cfg['data']

    if dataset_path := data_cfg.get('dataset_path'):
        paths = [dataset_path] if isinstance(dataset_path, str) else list(dataset_path)
        print(f'Loading local JSONL: {paths}')
        raw = load_dataset('json', data_files={'train': paths}, split='train')
    elif dataset_name := data_cfg.get('dataset_name'):
        split = data_cfg.get('dataset_split', 'train')
        print(f'Loading HF Hub dataset: {dataset_name} ({split})')
        raw = load_dataset(dataset_name, split=split)
    else:
        raise ValueError('Config [data] must provide either dataset_path (local JSONL) or dataset_name (HF Hub).')

    print(f'  → {len(raw):,} total examples')

    val_fraction = data_cfg.get('val_fraction', 0.05)
    splits = raw.train_test_split(test_size=val_fraction, seed=seed)
    train_ds = splits['train']
    eval_ds = splits['test']
    print(f'  → Train: {len(train_ds):,} | Val: {len(eval_ds):,}')
    return train_ds, eval_ds


# =============================================================================
# Model + PEFT
# =============================================================================


def load_base_model(cfg: Dict, accelerator: Accelerator) -> tuple:
    """
    Load the pre-trained base model and tokenizer from a local HF checkpoint
    (the output of base_train.py).

    If the tokenizer has no chat_template and the dataset is conversational,
    a minimal Alpaca-style fallback template is applied so that SFTTrainer
    can render the "messages" column into a string.
    """
    model_cfg = cfg['model']
    base_model = model_cfg['base_model']

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        raise ValueError('Tokenizer is missing chat_template!')

    load_kwargs: Dict = {
        'dtype': torch.bfloat16 if BF16_SUPPORTED else torch.float16,
    }
    if attn_impl := model_cfg.get('attn_implementation'):
        load_kwargs['attn_implementation'] = attn_impl
        accelerator.print(f'  Attention implementation: {attn_impl}')

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    model.config.use_cache = False  # Must be off during training

    num_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f'✓ Loaded: {model.__class__.__name__}')
    accelerator.print(f'  Parameters:     {num_params:,} ({num_params / 1e6:.2f}M)')
    accelerator.print(f'  Vocab size:     {len(tokenizer):,}')
    accelerator.print(f'  Dtype:          {load_kwargs["dtype"]}')
    return model, tokenizer


def build_peft_config(cfg: Dict, accelerator: Accelerator):
    """
    Return a LoraConfig when method="lora", or None for full fine-tuning.

    LoRA guidance (from TRL docs):
      - Use ~10× the full-FT learning rate (e.g. 2e-4 instead of 2e-5).
      - target_modules should cover all linear projections in attention + FFN.
      - r=32, lora_alpha=64 (alpha = 2×r) is a solid starting point.
      - merge_after_training=true saves a fused model for easy deployment.
    """
    method = cfg['training'].get('method', 'full').lower()
    if method != 'lora':
        accelerator.print('Method: full fine-tuning (all parameters trained)')
        return None

    lora_cfg = cfg.get('lora', {})
    peft_config = LoraConfig(
        r=lora_cfg.get('r', 32),
        lora_alpha=lora_cfg.get('lora_alpha', 64),
        lora_dropout=lora_cfg.get('lora_dropout', 0.05),
        target_modules=lora_cfg.get(
            'target_modules',
            ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        ),
        bias=lora_cfg.get('bias', 'none'),
        task_type='CAUSAL_LM',
    )
    accelerator.print('Method: LoRA fine-tuning')
    accelerator.print(f'  r={peft_config.r}, lora_alpha={peft_config.lora_alpha}')
    accelerator.print(f'  target_modules={peft_config.target_modules}')
    return peft_config


# =============================================================================
# Validation
# =============================================================================


def validate_sft_config(cfg: Dict, accelerator: Accelerator):
    """Pre-flight checks before training starts."""
    accelerator.print('\n' + '=' * 80)
    accelerator.print('CONFIGURATION VALIDATION')
    accelerator.print('=' * 80)

    # ── Model ────────────────────────────────────────────────────────────────
    base_model = cfg['model'].get('base_model')
    if not base_model:
        raise ValueError('Missing [model] base_model in fine_tune_config.toml')
    if not Path(base_model).exists():
        raise FileNotFoundError(f'Base model path does not exist: {base_model}')
    if not (Path(base_model) / 'config.json').exists():
        raise FileNotFoundError(f'No config.json found in {base_model} — make sure this is a valid HuggingFace checkpoint directory.')

    # ── Data ─────────────────────────────────────────────────────────────────
    data_cfg = cfg['data']
    if not data_cfg.get('dataset_path') and not data_cfg.get('dataset_name'):
        raise ValueError('[data] must set either dataset_path (local JSONL) or dataset_name (HF Hub).')
    if path := data_cfg.get('dataset_path'):
        for p in [path] if isinstance(path, str) else path:
            if not Path(p).exists():
                raise FileNotFoundError(f'Dataset file not found: {p}')

    # ── Sequence length vs model max ─────────────────────────────────────────
    seq_len = data_cfg.get('max_seq_length', 2048)
    try:
        hf_cfg = HFAutoConfig.from_pretrained(base_model)
        model_max = getattr(hf_cfg, 'max_position_embeddings', None)
        if model_max and seq_len > model_max:
            raise ValueError(f'max_seq_length ({seq_len}) > model max_position_embeddings ({model_max}). Lower max_seq_length in [data].')
        if model_max:
            accelerator.print(f'  seq_length {seq_len} ≤ model max {model_max} ✓')
    except Exception as e:
        if 'max_seq_length' in str(e):
            raise
        accelerator.print(f'  ⚠️  Could not verify max_seq_length against model: {e}')

    # ── Packing needs Flash Attention ────────────────────────────────────────
    train_cfg = cfg['training']
    if train_cfg.get('packing', False):
        attn_impl = cfg['model'].get('attn_implementation', '')
        if 'flash' not in attn_impl.lower():
            accelerator.print(
                '  ⚠️  packing=true works best with Flash Attention. Set attn_implementation = "flash_attention_2" in [model].'
            )

    # ── Precision ────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        accelerator.print('  ⚠️  CUDA not available — training on CPU will be slow')
    if train_cfg.get('bf16', False) and not BF16_SUPPORTED:
        accelerator.print('  ⚠️  bf16 requested but not supported on this GPU!')

    # ── load_best_model_at_end consistency ───────────────────────────────────
    if train_cfg.get('load_best_model_at_end', True):
        save_steps = train_cfg.get('save_steps', 100)
        eval_steps = train_cfg.get('eval_steps', 100)
        if train_cfg.get('save_strategy', 'steps') == 'steps' and save_steps != eval_steps:
            accelerator.print(
                f'  ⚠️  load_best_model_at_end=true requires save_steps == eval_steps. '
                f'Got save_steps={save_steps}, eval_steps={eval_steps}. '
                'Trainer will raise an error — fix the config.'
            )

    accelerator.print('✓ Configuration validated')
    accelerator.print('=' * 80 + '\n')


# =============================================================================
# Main
# =============================================================================


def main():
    """Main fine-tuning entry point."""

    config_path = 'training/fine_tune_config.toml'
    if not Path(config_path).exists():
        print(f'ERROR: Config file not found at {config_path}')
        print('Please create training/fine_tune_config.toml')
        sys.exit(1)

    print(f'Loading configuration from {config_path}...')
    cfg = load_config(config_path)

    # Resolve paths relative to the config directory
    config_dir = Path(config_path).parent

    if 'base_model' in cfg.get('model', {}):
        cfg['model']['base_model'] = str(config_dir / cfg['model']['base_model'])

    if 'dataset_path' in cfg.get('data', {}):
        dp = cfg['data']['dataset_path']
        if isinstance(dp, str):
            cfg['data']['dataset_path'] = str(config_dir / dp)
        else:
            cfg['data']['dataset_path'] = [str(config_dir / p) for p in dp]

    if 'output_dir' in cfg.get('training', {}):
        cfg['training']['output_dir'] = str(config_dir / cfg['training']['output_dir'])

    if 'final_model_dir' in cfg.get('training', {}):
        cfg['training']['final_model_dir'] = str(config_dir / cfg['training']['final_model_dir'])

    accelerator = Accelerator()
    num_processes = accelerator.num_processes

    accelerator.print('\n' + '=' * 80)
    accelerator.print('SUPERVISED FINE-TUNING')
    accelerator.print('=' * 80)
    accelerator.print(f'Method:          {cfg["training"].get("method", "full").upper()}')
    accelerator.print(f'Base model:      {cfg["model"]["base_model"]}')
    accelerator.print(f'Distributed:     {num_processes} process(es)')
    accelerator.print(f'Device:          {accelerator.device}')
    accelerator.print(f'Mixed precision: {accelerator.mixed_precision}')
    accelerator.print('=' * 80)

    validate_sft_config(cfg, accelerator)

    seed = cfg['training'].get('seed', 42)
    set_seed(seed)
    accelerator.print(f'Random seed: {seed}')

    # ── TF32 (Ampere+) ───────────────────────────────────────────────────────
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        if props.major >= 8:
            torch.set_float32_matmul_precision('high')
            accelerator.print(f'TF32 matmul enabled ({props.name})')
        else:
            accelerator.print(f'TF32 not supported ({props.name}, compute {props.major}.{props.minor})')

    # ── Dataset ──────────────────────────────────────────────────────────────
    accelerator.print('\nLoading dataset...')
    train_dataset, eval_dataset = load_sft_dataset(cfg, seed)

    fmt, dataset_text_field = detect_format(train_dataset)
    accelerator.print(f'  Format: {fmt}')
    if fmt == 'conversational':
        accelerator.print('  → SFTTrainer will apply the chat template to the "messages" column')
    else:
        accelerator.print(f'  → SFTTrainer will use the "{dataset_text_field}" column as-is')

    # ── Model + tokenizer ────────────────────────────────────────────────────
    accelerator.print('\nLoading base model...')
    model, tokenizer = load_base_model(cfg, accelerator)

    # ── PEFT config ──────────────────────────────────────────────────────────
    peft_config = build_peft_config(cfg, accelerator)

    # ── Training arguments (SFTConfig extends TrainingArguments) ─────────────
    accelerator.print('\nSetting up training arguments...')
    train_cfg = cfg['training']
    output_dir = train_cfg['output_dir']

    # neftune_noise_alpha: 0.0 → None (disabled); any other float → enabled.
    neftune = train_cfg.get('neftune_noise_alpha', 5.0)
    neftune = float(neftune) if neftune else None

    sft_args = SFTConfig(
        # ── Output ───────────────────────────────────────────────────────────
        output_dir=output_dir,
        # ── SFT-specific ─────────────────────────────────────────────────────
        max_length=cfg['data'].get('max_seq_length', 2048),
        dataset_text_field=dataset_text_field,  # None → auto-detect "messages"
        packing=train_cfg.get('packing', False),
        packing_strategy=train_cfg.get('packing_strategy', 'bfd_split'),
        neftune_noise_alpha=neftune,
        # ── Training duration ────────────────────────────────────────────────
        num_train_epochs=train_cfg['num_train_epochs'],
        max_steps=-1,
        # ── Batch sizes ──────────────────────────────────────────────────────
        per_device_train_batch_size=train_cfg.get('per_device_train_batch_size', 4),
        per_device_eval_batch_size=train_cfg.get('per_device_eval_batch_size', 8),
        gradient_accumulation_steps=train_cfg.get('gradient_accumulation_steps', 2),
        # ── Optimizer ────────────────────────────────────────────────────────
        optim=train_cfg.get('optim', 'adamw_torch_fused'),
        learning_rate=train_cfg.get('learning_rate', 2e-5),
        weight_decay=train_cfg.get('weight_decay', 0.01),
        adam_beta1=train_cfg.get('adam_beta1', 0.9),
        adam_beta2=train_cfg.get('adam_beta2', 0.999),
        adam_epsilon=train_cfg.get('adam_epsilon', 1e-8),
        max_grad_norm=train_cfg.get('max_grad_norm', 1.0),
        # ── LR schedule ──────────────────────────────────────────────────────
        lr_scheduler_type=train_cfg.get('lr_scheduler_type', 'cosine_with_min_lr'),
        lr_scheduler_kwargs=train_cfg.get('lr_scheduler_kwargs', {'min_lr_rate': 0.1}),
        warmup_steps=cfg['training'].get('warmup_steps', 100),
        # ── Precision ────────────────────────────────────────────────────────
        bf16=train_cfg.get('bf16', BF16_SUPPORTED),
        fp16=train_cfg.get('fp16', False),
        # ── Performance ──────────────────────────────────────────────────────
        torch_compile=train_cfg.get('torch_compile', False),
        gradient_checkpointing=train_cfg.get('gradient_checkpointing', False),
        gradient_checkpointing_kwargs={'use_reentrant': False},
        # ── Checkpointing ────────────────────────────────────────────────────
        save_strategy=train_cfg.get('save_strategy', 'steps'),
        save_steps=train_cfg.get('save_steps', 100),
        save_total_limit=train_cfg.get('save_total_limit', 3),
        load_best_model_at_end=train_cfg.get('load_best_model_at_end', True),
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        # ── Evaluation ───────────────────────────────────────────────────────
        eval_strategy=train_cfg.get('eval_strategy', 'steps'),
        eval_steps=train_cfg.get('eval_steps', 100),
        # ── Logging ──────────────────────────────────────────────────────────
        logging_strategy=train_cfg.get('logging_strategy', 'steps'),
        logging_steps=train_cfg.get('logging_steps', 10),
        logging_first_step=train_cfg.get('logging_first_step', True),
        report_to=train_cfg.get('report_to', ['tensorboard']),
        # ── Data loading ─────────────────────────────────────────────────────
        dataloader_num_workers=train_cfg.get('dataloader_num_workers', 4),
        dataloader_prefetch_factor=train_cfg.get('dataloader_prefetch_factor', 2),
        dataloader_pin_memory=train_cfg.get('dataloader_pin_memory', True),
        # ── Reproducibility ───────────────────────────────────────────────────
        seed=seed,
        data_seed=seed,
    )

    effective_batch = sft_args.per_device_train_batch_size * sft_args.gradient_accumulation_steps * num_processes
    accelerator.print('✓ Training arguments configured')
    accelerator.print(f'  Effective batch size: {effective_batch}')
    accelerator.print(f'  Learning rate:        {sft_args.learning_rate}')
    accelerator.print(f'  Epochs:               {sft_args.num_train_epochs}')
    accelerator.print(f'  Warmup ratio:         {sft_args.warmup_ratio}')
    accelerator.print(f'  Max sequence length:  {sft_args.max_length}')
    accelerator.print(f'  Packing:              {sft_args.packing}')
    accelerator.print(f'  NEFTune alpha:        {sft_args.neftune_noise_alpha}')
    accelerator.print(f'  Load best at end:     {sft_args.load_best_model_at_end}')
    accelerator.print()

    # ── Create trainer ───────────────────────────────────────────────────────
    accelerator.print('Creating SFTTrainer...')
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[
            DetailedLoggingCallback(),
            DetailedEvaluationCallback(),
        ],
    )
    trainer.remove_callback(ProgressCallback)

    # Print LoRA trainable-param ratio after SFTTrainer applies the adapter.
    if peft_config is not None:
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in trainer.model.parameters())
        accelerator.print(f'  LoRA trainable: {trainable:,} / {total:,} params ({100 * trainable / total:.2f}%)')

    accelerator.print('✓ SFTTrainer created\n')

    # ── Train ────────────────────────────────────────────────────────────────
    accelerator.print('=' * 80)
    accelerator.print('STARTING FINE-TUNING')
    accelerator.print('=' * 80)
    accelerator.print(f'Train examples:  {len(train_dataset):,}')
    accelerator.print(f'Val examples:    {len(eval_dataset):,}')
    accelerator.print('=' * 80 + '\n')

    last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
    if last_checkpoint:
        accelerator.print(f'Resuming from checkpoint: {last_checkpoint}')
    else:
        accelerator.print('No checkpoint found — starting from the base model.')

    start_time = time.time()

    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)

    except KeyboardInterrupt:
        accelerator.print('\n' + '=' * 80)
        accelerator.print('FINE-TUNING INTERRUPTED BY USER')
        accelerator.print('=' * 80)
        accelerator.print('Checkpoint saved. Resume by running the script again.')
        accelerator.print('=' * 80 + '\n')
        sys.exit(0)

    except Exception as e:
        accelerator.print('\n' + '=' * 80)
        accelerator.print('FINE-TUNING ERROR')
        accelerator.print('=' * 80)
        accelerator.print(f'Error: {e}')
        accelerator.print('=' * 80 + '\n')
        raise

    elapsed = time.time() - start_time

    # ── Save final model ─────────────────────────────────────────────────────
    accelerator.print('\n' + '=' * 80)
    accelerator.print('FINE-TUNING COMPLETE')
    accelerator.print('=' * 80)
    accelerator.print(f'Training time: {elapsed / 60:.2f} minutes')
    accelerator.print('=' * 80 + '\n')

    final_model_dir = train_cfg['final_model_dir']
    accelerator.print(f'Saving final model to {final_model_dir}...')

    # save_model must be called by ALL ranks so FSDP can gather weight shards.
    # The Trainer handles rank-0-only file writes internally.
    trainer.save_model(final_model_dir)

    if accelerator.is_main_process:
        os.makedirs(final_model_dir, exist_ok=True)
        tokenizer.save_pretrained(final_model_dir)
        Path(final_model_dir, 'training_args.bin').unlink(missing_ok=True)
        shutil.copy(config_path, os.path.join(final_model_dir, 'fine_tune_config.toml'))
        accelerator.print(f'✓ Model saved to {final_model_dir}')

    # ── LoRA merge ───────────────────────────────────────────────────────────
    if peft_config is not None:
        if cfg.get('lora', {}).get('merge_after_training', True):
            accelerator.print('\nMerging LoRA adapter into base weights...')
            raw_model = trainer.model
            if hasattr(raw_model, '_orig_mod'):
                raw_model = raw_model._orig_mod
            merged_model = raw_model.merge_and_unload()
            merged_dir = final_model_dir.rstrip('/') + '_merged'
            if accelerator.is_main_process:
                os.makedirs(merged_dir, exist_ok=True)
                merged_model.save_pretrained(merged_dir)
                tokenizer.save_pretrained(merged_dir)
                accelerator.print(f'✓ Merged model saved to {merged_dir}')
        else:
            accelerator.print(
                '\nNote: LoRA adapter saved separately (merge_after_training=false).\n'
                'To merge manually:\n'
                '    from peft import PeftModel\n'
                f'    model = PeftModel.from_pretrained(base_model, "{final_model_dir}")\n'
                '    model = model.merge_and_unload()\n'
                f'    model.save_pretrained("{final_model_dir}_merged")'
            )

    accelerator.wait_for_everyone()

    accelerator.print('\n' + '=' * 80)
    accelerator.print('ALL DONE!')
    accelerator.print('=' * 80)
    accelerator.print(f'Checkpoints:   {output_dir}')
    accelerator.print(f'Final model:   {final_model_dir}')
    if peft_config is not None and cfg.get('lora', {}).get('merge_after_training', True):
        accelerator.print(f'Merged model:  {final_model_dir}_merged')
    accelerator.print('=' * 80 + '\n')


if __name__ == '__main__':
    main()

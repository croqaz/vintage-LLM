#!/usr/bin/env python3
"""
Pre-flight doctor for base_train.py and fine_tune.py.

Validates everything about the CURRENT config + model + dataset + GPU
combination that can be checked without running the real training, including
things that a green "config loads fine" cannot catch:

  - Will the configured per_device_train_batch_size OOM? (measured, not guessed)
  - Is torch_compile numerically stable for THIS model on THIS stack?
    (eager vs compiled trained side-by-side from identical weights)
  - How long will the run actually take?
  - Is the dataset healthy? (token ranges, EOS structure, vocab coverage, ...)
  - Will the finished model actually decode? (KV cache, stop tokens, tok/s)

Checks are grouped into five categories: config, model, training, dataset,
decode. Every category runs by default; the probes that cost more than a few
seconds sit behind --deep.

Usage:
    python doctor.py                          # base training (config.toml)
    python doctor.py fine_tune_config.toml    # fine-tuning (auto-detected)
    python doctor.py --skip dataset           # skip the slow dataset scans
    python doctor.py --only config,model      # run only some categories
    python doctor.py --deep                   # full dataset scan + slow decode probes
    python doctor.py --check-compile          # force the compile probe even if torch_compile=false
    python doctor.py --only decode            # inference-side checks alone (~10s)
"""

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import statistics
import sys
import time
import traceback
from glob import glob
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, default_data_collator
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, str(Path(__file__).parent))
from base_train import BF16_SUPPORTED, load_binary_files, load_config, make_model_id

GB = 1024**3
MB = 1024**2
IS_ROCM = torch.version.hip is not None
CATEGORIES = ('config', 'model', 'training', 'dataset', 'decode')

# Dataset scan budgets (tokens) when not running with --deep
TRAIN_SCAN_BUDGET = 400_000_000
VALID_SCAN_BUDGET = 150_000_000
SCAN_CHUNK = 8_000_000


# ============================================================================
# Reporting
# ============================================================================


class Reporter:
    def __init__(self):
        self.records = []  # (level, category, message)
        self.cat = '?'
        self._cat_t0 = None

    def section(self, title):
        self._close_section()
        self.cat = title
        self._cat_t0 = time.time()
        print(f'\n{"═" * 72}')
        print(f'  {title}')
        print('═' * 72)

    def _close_section(self):
        if self._cat_t0 is not None:
            print(f'  ({self.cat.lower()} took {time.time() - self._cat_t0:.1f}s)')
            self._cat_t0 = None

    def _rec(self, level, msg):
        self.records.append((level, self.cat, msg))

    def ok(self, msg):
        print(f'  ✓ {msg}')
        self._rec('ok', msg)

    def warn(self, msg):
        print(f'  ⚠ {msg}')
        self._rec('warn', msg)

    def fail(self, msg):
        print(f'  ✗ {msg}')
        self._rec('fail', msg)

    def info(self, msg):
        print(f'      {msg}')

    def summary(self):
        self._close_section()
        oks = [r for r in self.records if r[0] == 'ok']
        warns = [r for r in self.records if r[0] == 'warn']
        fails = [r for r in self.records if r[0] == 'fail']
        print(f'\n{"═" * 72}')
        print('  SUMMARY')
        print('═' * 72)
        print(f'  ✓ {len(oks)} passed   ⚠ {len(warns)} warnings   ✗ {len(fails)} failures')
        if warns:
            print('\n  Warnings:')
            for _, cat, msg in warns:
                print(f'    ⚠ [{cat.split()[0].lower()}] {msg}')
        if fails:
            print('\n  Failures:')
            for _, cat, msg in fails:
                print(f'    ✗ [{cat.split()[0].lower()}] {msg}')
        print()
        if fails:
            print('  ❌ NOT READY TO TRAIN — fix the failures above first.')
        elif warns:
            print('  ⚠️  Ready to train, but read the warnings above.')
        else:
            print('  ✅ All clear. Go train.')
        print('═' * 72)
        return 1 if fails else 0


# ============================================================================
# Small helpers
# ============================================================================


def resolve_files(pattern):
    """Expand a str-or-list of paths/globs into a sorted file list."""
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    files = []
    for p in patterns:
        if '*' in p or '?' in p:
            files.extend(sorted(glob(p)))
        else:
            files.append(p)
    return [f for f in files if Path(f).exists()]


def gb(nbytes):
    return f'{nbytes / GB:.2f} GB'


def build_model_config(cfg):
    model_kwargs = dict(cfg['model'])
    model_type = model_kwargs.pop('model_type')
    attn_implementation = model_kwargs.pop('attn_implementation', None)
    model_kwargs['use_cache'] = False
    return AutoConfig.for_model(model_type=model_type, **model_kwargs), attn_implementation


def count_params_from_config(model_config):
    """Cheap analytic parameter estimate (used when the model section was skipped)."""
    h = model_config.hidden_size
    v = model_config.vocab_size
    layers = model_config.num_hidden_layers
    inter = model_config.intermediate_size
    kv = getattr(model_config, 'num_key_value_heads', model_config.num_attention_heads)
    heads = model_config.num_attention_heads
    attn = h * h * 2 + 2 * h * h * kv / heads  # q,o + k,v (GQA-scaled)
    mlp = 3 * h * inter
    embed = v * h * (1 if getattr(model_config, 'tie_word_embeddings', False) else 2)
    return int(embed + layers * (attn + mlp))


def longest_run(arr):
    """Length and value of the longest run of identical tokens in `arr`."""
    if len(arr) < 2:
        return len(arr), int(arr[0]) if len(arr) else None
    change = np.flatnonzero(arr[1:] != arr[:-1])
    starts = np.r_[0, change + 1]
    ends = np.r_[change, len(arr) - 1]
    lens = ends - starts + 1
    i = int(np.argmax(lens))
    return int(lens[i]), int(arr[starts[i]])


# ============================================================================
# CONFIG checks
# ============================================================================


def config_checks(cfg, mode, R, ctx, args):
    R.section('1. CONFIG CHECKS')
    tcfg = cfg.get('training', {})

    # ── Precision flags (shared) ─────────────────────────────────────────────
    bf16 = tcfg.get('bf16', BF16_SUPPORTED)
    fp16 = tcfg.get('fp16', False)
    if bf16 and fp16:
        R.fail('bf16 = true AND fp16 = true — pick one (Trainer will error)')
    if tcfg.get('bf16', False) and not BF16_SUPPORTED:
        R.fail('bf16 = true but this GPU does not support bf16 — use fp16 or fp32')
    elif bf16:
        R.ok('precision: bf16 (supported by this GPU)')
    elif fp16:
        lvl = R.warn if mode == 'base' else R.ok
        lvl('precision: fp16' + (' — fp16 pre-training is unstable; prefer bf16 if the GPU supports it' if mode == 'base' else ''))
    elif mode == 'sft' and BF16_SUPPORTED:
        R.warn(
            'bf16 = false, but fine_tune.py loads the model weights in bf16 regardless — compute runs in bf16 '
            'either way, just without Trainer-managed autocast. Set bf16 = true to make this explicit.'
        )
    else:
        R.warn('precision: fp32 — training will be several times slower than bf16')
    ctx['amp_dtype'] = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optim = tcfg.get('optim', 'adamw_torch_fused' if mode == 'sft' else 'adamw_torch_fused')
    lr = tcfg.get('learning_rate', 5e-5)
    beta2 = tcfg.get('adam_beta2', 0.95 if mode == 'base' else 0.999)
    R.info(
        f'optimizer: {optim}, lr={lr}, betas=({tcfg.get("adam_beta1", 0.9)}, {beta2}), '
        f'weight_decay={tcfg.get("weight_decay", 0.1)}, max_grad_norm={tcfg.get("max_grad_norm", 1.0)}'
    )
    if 'adamw' not in optim:
        R.warn(f'optim = "{optim}" — doctor probes assume an AdamW variant')
    # Muon-family optimizers (muon, muonq, normuon) orthogonalize the update, so
    # their step size is NOT on AdamW's scale and these AdamW bands do not apply.
    # Our own 5-arm sweep (autoresearch2/optimizer/REPORT.md) measured the optimum
    # at 1.6e-2 for both MuonQ and NorMuon, which the AdamW band would have called
    # "guaranteed divergence". Band the two families separately rather than fail a
    # value we have trained successfully many times.
    _is_muon = any(k in optim for k in ('muon', 'normuon'))
    _hi_fail, _hi_warn, _typical = (1e-1, 4e-2, '4e-3 … 3.2e-2') if _is_muon else (1e-2, 2e-3, '3e-4 … 1e-3')
    if mode == 'base':
        if lr > _hi_fail:
            R.fail(f'learning_rate {lr} is extremely high — guaranteed divergence territory')
        elif lr > _hi_warn:
            R.warn(f'learning_rate {lr} is high for pre-training (typical for a ~50M model: {_typical})')
        elif lr < 1e-5:
            R.warn(f'learning_rate {lr} is very low for pre-training from scratch')
        else:
            R.ok(f'learning_rate {lr} is in a sane pre-training range' + (f' for a Muon-family optimizer ({optim})' if _is_muon else ''))
        if tcfg.get('adam_beta2', 0.95) > 0.98:
            R.warn(f'adam_beta2 = {tcfg["adam_beta2"]} — 0.95 is the usual choice for pre-training stability')
    else:
        method = tcfg.get('method', 'full').lower()
        if method == 'full' and lr > 1e-4:
            R.warn(f'learning_rate {lr} is high for FULL fine-tuning (typical: 1e-5 … 5e-5); this can wipe the base model')
        elif method == 'lora' and lr < 5e-5:
            R.warn(f'learning_rate {lr} is low for LoRA (typical: 1e-4 … 2e-4)')
        else:
            R.ok(f'learning_rate {lr} is reasonable for method = "{method}"')

    # ── LR scheduler kwargs vs type ──────────────────────────────────────────
    sched_type = tcfg.get('lr_scheduler_type', 'cosine_with_min_lr' if mode == 'base' else 'linear')
    sched_kwargs = tcfg.get('lr_scheduler_kwargs', {}) or {}
    if 'min_lr_rate' in sched_kwargs and sched_type != 'cosine_with_min_lr':
        if mode == 'base':
            R.fail(
                f'lr_scheduler_kwargs.min_lr_rate is only valid with "cosine_with_min_lr", got "{sched_type}" '
                '— base_train.py will crash creating the scheduler'
            )
        else:
            R.warn(f'lr_scheduler_kwargs.min_lr_rate with "{sched_type}" — fine_tune.py silently DROPS it (no LR floor)')
    else:
        R.ok(f'lr schedule: {sched_type} {sched_kwargs or ""}')

    # ── torch_compile flag ───────────────────────────────────────────────────
    default_compile = mode == 'base'  # base_train defaults torch_compile=True, fine_tune False
    if tcfg.get('torch_compile', default_compile):
        R.info('torch_compile = true → its numerical stability will be probed in TRAINING CHECKS')
        ctx['compile_requested'] = True
    else:
        ctx['compile_requested'] = False

    if mode == 'base':
        _config_checks_base(cfg, R, ctx)
    else:
        _config_checks_sft(cfg, R, ctx)

    # ── Output dir: disk space + resume behaviour (shared) ───────────────────
    output_dir = tcfg.get('output_dir')
    if output_dir:
        probe_dir = Path(output_dir)
        while not probe_dir.exists() and probe_dir != probe_dir.parent:
            probe_dir = probe_dir.parent
        free_disk = shutil.disk_usage(probe_dir).free
        n_params = ctx.get('num_params') or count_params_from_config(ctx['model_config']) if ctx.get('model_config') else 60e6
        ckpt_bytes = n_params * 16  # fp32 weights + AdamW states + margin
        need = ckpt_bytes * (tcfg.get('save_total_limit', 3) + 2)
        if free_disk < need:
            R.fail(
                f'only {gb(free_disk)} free on disk for checkpoints; ~{gb(need)} needed '
                f'({tcfg.get("save_total_limit", 3)}+ checkpoints of ~{gb(ckpt_bytes)})'
            )
        else:
            R.ok(f'disk space: {gb(free_disk)} free, ~{gb(need)} needed for checkpoints')

        last_ckpt = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
        if last_ckpt:
            R.warn(f'output_dir contains {Path(last_ckpt).name} — training will RESUME from it, not start fresh')
            if mode == 'base' and ctx.get('model_config') is not None:
                try:
                    ck = AutoConfig.from_pretrained(last_ckpt)
                    keys = ('model_type', 'vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size')
                    bad = [k for k in keys if getattr(ck, k, None) != getattr(ctx['model_config'], k, None)]
                    if bad:
                        R.fail(
                            f'checkpoint in output_dir is a DIFFERENT architecture (mismatch: {", ".join(bad)}) '
                            '— base_train.py will refuse to start'
                        )
                except Exception as e:
                    R.warn(f'could not read checkpoint config in output_dir: {e}')

    # ── Dataloader workers (shared) ──────────────────────────────────────────
    workers = tcfg.get('dataloader_num_workers', 4)
    ncpu = os.cpu_count() or 1
    if workers > ncpu:
        R.warn(f'dataloader_num_workers = {workers} > {ncpu} CPU cores')
    else:
        R.ok(f'dataloader_num_workers = {workers} ({ncpu} CPU cores available)')

    if not torch.cuda.is_available():
        R.fail('CUDA is not available — training on CPU is not realistic')


def _config_checks_base(cfg, R, ctx):
    mcfg = cfg.get('model', {})
    dcfg = cfg.get('data', {})
    tcfg = cfg.get('training', {})

    # ── Model architecture invariants ────────────────────────────────────────
    required = ['model_type', 'vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size']
    missing = [k for k in required if k not in mcfg]
    if missing:
        R.fail(f'missing required [model] keys: {missing}')
        return
    R.ok(
        f'model config: {mcfg["model_type"]}, {mcfg["num_hidden_layers"]}L × {mcfg["hidden_size"]}h, '
        f'{mcfg["num_attention_heads"]} heads, vocab {mcfg["vocab_size"]}'
    )

    if mcfg['hidden_size'] % mcfg['num_attention_heads'] != 0:
        R.fail(f'hidden_size {mcfg["hidden_size"]} not divisible by num_attention_heads {mcfg["num_attention_heads"]}')
    kv = mcfg.get('num_key_value_heads', mcfg['num_attention_heads'])
    if mcfg['num_attention_heads'] % kv != 0:
        R.fail(f'num_attention_heads {mcfg["num_attention_heads"]} not divisible by num_key_value_heads {kv}')

    seq = dcfg.get('max_seq_length')
    max_pos = mcfg.get('max_position_embeddings', seq)
    if seq is None:
        R.fail('missing [data] max_seq_length')
        return
    if seq > max_pos:
        R.fail(f'max_seq_length {seq} > model max_position_embeddings {max_pos}')
    else:
        R.ok(f'max_seq_length {seq} ≤ max_position_embeddings {max_pos}')

    try:
        ctx['model_config'], ctx['attn_implementation'] = build_model_config(cfg)
    except Exception as e:
        R.fail(f'model config does not instantiate: {e}')

    attn = ctx.get('attn_implementation')
    if attn:
        if attn not in ('sdpa', 'eager', 'flash_attention_2', 'flex_attention'):
            R.warn(f'unusual attn_implementation "{attn}"')
        if attn == 'flash_attention_2':
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                R.fail('attn_implementation = "flash_attention_2" but the flash-attn package is not installed')

    # ── Tokenizer vs model vocab ─────────────────────────────────────────────
    tok_name = dcfg.get('tokenizer')
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        ctx['tokenizer'] = tokenizer
        R.ok(f'tokenizer loads: {tok_name} (vocab {len(tokenizer)}, eos id {tokenizer.eos_token_id})')
    except Exception as e:
        R.fail(f'tokenizer failed to load from {tok_name}: {e}')
        tokenizer = None

    if tokenizer is not None:
        if len(tokenizer) > mcfg['vocab_size']:
            R.fail(
                f'tokenizer vocab ({len(tokenizer)}) > model vocab_size ({mcfg["vocab_size"]}) — out-of-range embedding lookups will crash'
            )
        elif mcfg['vocab_size'] > len(tokenizer):
            R.warn(
                f'model vocab_size ({mcfg["vocab_size"]}) > tokenizer vocab ({len(tokenizer)}) '
                f'— {mcfg["vocab_size"] - len(tokenizer)} embedding rows are dead weight'
            )
        else:
            R.ok('model vocab_size == tokenizer vocab')
        if mcfg['vocab_size'] > 65536:
            R.fail(f'vocab_size {mcfg["vocab_size"]} > 65536 does not fit the uint16 .bin token format')
        if tokenizer.eos_token_id is None:
            R.fail('tokenizer has no EOS token — document boundaries cannot exist in the data')

    # ── Data files + schedule math (from file sizes only — instant) ─────────
    train_files = resolve_files(dcfg.get('train_files', []))
    valid_files = resolve_files(dcfg.get('valid_files', []))
    if not train_files:
        R.fail(f'no training files found: {dcfg.get("train_files")}')
        return
    if not valid_files:
        R.warn(f'no validation files found: {dcfg.get("valid_files")}')
    ctx['train_files'], ctx['valid_files'] = train_files, valid_files

    total_tokens = sum(os.path.getsize(f) // 2 for f in train_files)
    valid_tokens = sum(os.path.getsize(f) // 2 for f in valid_files)
    ctx['total_train_tokens'] = total_tokens
    B = tcfg.get('per_device_train_batch_size', 8)
    gas = tcfg.get('gradient_accumulation_steps', 1)
    epochs = tcfg.get('num_train_epochs', 1)
    seqs = max(0, (total_tokens - (seq - 1)) // seq)
    micro_per_epoch = math.ceil(seqs / B)
    opt_steps_per_epoch = math.ceil(micro_per_epoch / gas)
    total_steps = opt_steps_per_epoch * epochs
    eff_batch_tokens = B * gas * seq
    ctx['sched'] = {
        'seqs': seqs,
        'micro_per_epoch': micro_per_epoch,
        'opt_steps_per_epoch': opt_steps_per_epoch,
        'total_steps': total_steps,
        'B': B,
        'gas': gas,
        'epochs': epochs,
        'seq': seq,
    }
    R.ok(
        f'data: {len(train_files)} train file(s) = {total_tokens / 1e9:.2f}B tokens, '
        f'{len(valid_files)} valid file(s) = {valid_tokens / 1e6:.1f}M tokens'
    )
    R.info(f'{seqs:,} sequences of {seq} → {opt_steps_per_epoch:,} optimizer steps/epoch × {epochs} epoch(s) = {total_steps:,} total steps')
    R.info(f'effective batch: {B} × {gas} accum = {B * gas} sequences = {eff_batch_tokens:,} tokens/optimizer-step')

    # ── Warmup vs schedule ───────────────────────────────────────────────────
    warmup = tcfg.get('warmup_steps', 200)
    if total_steps and warmup >= total_steps:
        R.fail(f'warmup_steps ({warmup}) ≥ total optimizer steps ({total_steps:,}) — LR never reaches peak, cosine decay never happens')
    elif total_steps and warmup > 0.2 * total_steps:
        R.warn(f'warmup_steps ({warmup}) is {100 * warmup / total_steps:.0f}% of the schedule ({total_steps:,} steps)')
    else:
        R.ok(f'warmup_steps {warmup} = {100 * warmup / max(total_steps, 1):.1f}% of {total_steps:,} total steps')
    max_min = tcfg.get('max_train_minutes')
    if max_min:
        R.info(f'max_train_minutes = {max_min} — actual coverage of the schedule is estimated in TRAINING CHECKS')

    # ── Save / eval cadence ──────────────────────────────────────────────────
    save_strategy = tcfg.get('save_strategy', 'steps')
    save_steps = tcfg.get('save_steps', 500)
    eval_strategy = tcfg.get('eval_strategy', 'steps')
    if save_strategy == 'minutes':
        if max_min and save_steps > max_min:
            R.warn(f'save every {save_steps} min but max_train_minutes = {max_min} — only the final forced checkpoint will be written')
        else:
            R.ok(f'save strategy: every {save_steps} wall-clock minutes')
    elif save_strategy == 'steps' and total_steps and save_steps > total_steps:
        R.warn(f'save_steps ({save_steps}) > total steps ({total_steps:,}) — no periodic checkpoint will ever be saved')
    if eval_strategy == 'minutes':
        eval_used = min(tcfg.get('max_eval_samples', 0) or 10**12, max(0, (valid_tokens - (seq - 1)) // seq))
        ctx['eval_sequences'] = eval_used
        R.info(
            f'eval strategy: every {tcfg.get("eval_steps", 500)} min on {eval_used:,} sequences (eval duration checked in TRAINING CHECKS)'
        )
    else:
        ctx['eval_sequences'] = min(tcfg.get('max_eval_samples', 0) or 10**12, max(0, (valid_tokens - (seq - 1)) // seq))

    # ── S3 ───────────────────────────────────────────────────────────────────
    s3 = cfg.get('s3', {})
    if s3.get('bucket'):
        if not os.environ.get('HF_TOKEN'):
            R.warn('s3.bucket configured but HF_TOKEN is not set in the environment — uploads will fail')
        if s3.get('upload_steps') and save_strategy == 'steps' and s3['upload_steps'] % save_steps != 0:
            R.warn(
                f's3.upload_steps ({s3["upload_steps"]}) is not a multiple of save_steps ({save_steps}) — uploads only happen on save steps'
            )


def _config_checks_sft(cfg, R, ctx):
    mcfg = cfg.get('model', {})
    dcfg = cfg.get('data', {})
    tcfg = cfg.get('training', {})

    base_model = mcfg.get('base_model')
    if not base_model:
        R.fail('missing [model] base_model')
        return
    if not Path(base_model).exists() or not (Path(base_model) / 'config.json').exists():
        R.fail(f'base_model is not a valid HF checkpoint directory: {base_model}')
        return
    R.ok(f'base_model checkpoint exists: {base_model}')
    ctx['base_model'] = base_model

    try:
        ctx['model_config'] = AutoConfig.from_pretrained(base_model)
    except Exception as e:
        R.fail(f'base model config.json unreadable: {e}')
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        ctx['tokenizer'] = tokenizer
        R.ok(f'tokenizer loads from checkpoint (vocab {len(tokenizer)})')
    except Exception as e:
        R.fail(f'tokenizer failed to load from {base_model}: {e}')
        tokenizer = None
    if tokenizer is not None and tokenizer.chat_template is None:
        R.fail('tokenizer has NO chat_template — fine_tune.py raises immediately for conversational data')

    # ── Data source ──────────────────────────────────────────────────────────
    seq = dcfg.get('max_seq_length', 2048)
    ctx['seq'] = seq
    max_pos = getattr(ctx['model_config'], 'max_position_embeddings', None)
    if max_pos and seq > max_pos:
        R.fail(f'max_seq_length ({seq}) > model max_position_embeddings ({max_pos})')
    else:
        R.ok(f'max_seq_length {seq} ≤ model max_position_embeddings {max_pos}')

    ds_paths = dcfg.get('dataset_path')
    if not ds_paths and not dcfg.get('dataset_name'):
        R.fail('[data] needs dataset_path (local JSONL) or dataset_name (HF Hub)')
        return
    if ds_paths:
        paths = [ds_paths] if isinstance(ds_paths, str) else list(ds_paths)
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            R.fail(f'dataset files not found: {missing}')
            return
        n_rows = 0
        for p in paths:
            with open(p, 'rb') as f:
                n_rows += sum(1 for _ in f)
        ctx['sft_paths'] = paths
        ctx['sft_rows'] = n_rows
        R.ok(f'dataset: {len(paths)} JSONL file(s), {n_rows:,} examples')
    else:
        R.info(f'dataset from HF Hub: {dcfg["dataset_name"]} (schema checked at load time only)')

    vf = dcfg.get('val_fraction', 0.05)
    if not 0 < vf < 0.5:
        R.warn(f'val_fraction = {vf} is outside the sane (0, 0.5) range')
    n_train = int(ctx.get('sft_rows', 0) * (1 - vf))
    if n_train:
        B = tcfg.get('per_device_train_batch_size', 4)
        gas = tcfg.get('gradient_accumulation_steps', 2)
        epochs = tcfg.get('num_train_epochs', 1)
        total_steps = math.ceil(n_train / (B * gas)) * epochs
        max_steps = int(tcfg.get('max_steps', -1))
        if max_steps > 0:
            R.info(f'max_steps = {max_steps} overrides epochs (would otherwise be {total_steps:,} steps)')
            total_steps = max_steps
        ctx['sched'] = {'total_steps': total_steps, 'B': B, 'gas': gas, 'epochs': epochs, 'seq': seq, 'n_train': n_train}
        R.info(f'{n_train:,} train examples → ~{total_steps:,} optimizer steps (effective batch {B * gas})')
        warmup = tcfg.get('warmup_steps', 100)
        if warmup >= total_steps:
            R.fail(f'warmup_steps ({warmup}) ≥ total steps ({total_steps:,})')

    # ── Method / LoRA ────────────────────────────────────────────────────────
    method = tcfg.get('method', 'full').lower()
    if method not in ('full', 'lora'):
        R.fail(f'unknown training method "{method}" (use "full" or "lora")')
    ctx['method'] = method
    if method == 'lora':
        lcfg = cfg.get('lora', {})
        r = lcfg.get('r', 32)
        alpha = lcfg.get('lora_alpha', 64)
        R.ok(f'LoRA: r={r}, alpha={alpha} (α/r = {alpha / r:.1f}), dropout={lcfg.get("lora_dropout", 0.05)}')
        if alpha / r > 4 or alpha / r < 0.5:
            R.warn(f'lora_alpha/r ratio {alpha / r:.1f} is unusual (rule of thumb: alpha = 2×r)')

    # ── The pure-bf16 full-FT trap ───────────────────────────────────────────
    if method == 'full' and BF16_SUPPORTED:
        R.warn(
            'fine_tune.py loads weights in PURE bf16 for full fine-tuning — AdamW updates smaller than '
            'bf16 resolution (~1e-2 relative) are lost and can cause stagnation or loss spikes. '
            'If the run misbehaves, this (not the data) is a prime suspect.'
        )

    # ── Packing / flash ──────────────────────────────────────────────────────
    if tcfg.get('packing', False) and 'flash' not in mcfg.get('attn_implementation', '').lower():
        R.warn('packing = true without attn_implementation = "flash_attention_2" — packed samples may cross-attend')

    # ── load_best_model_at_end ───────────────────────────────────────────────
    if tcfg.get('load_best_model_at_end', True):
        ss, es = tcfg.get('save_steps', 100), tcfg.get('eval_steps', 100)
        sstrat, estrat = tcfg.get('save_strategy', 'steps'), tcfg.get('eval_strategy', 'steps')
        if sstrat != estrat or (sstrat == 'steps' and ss != es):
            R.fail(
                f'load_best_model_at_end = true requires save and eval to align '
                f'(save: {sstrat}/{ss}, eval: {estrat}/{es}) — Trainer raises at startup'
            )
        else:
            R.ok(f'load_best_model_at_end: save/eval aligned every {ss} steps')

    # ── NEFTune ──────────────────────────────────────────────────────────────
    neftune = tcfg.get('neftune_noise_alpha', 5.0)
    if neftune and neftune < 1.0:
        R.warn(f'neftune_noise_alpha = {neftune} — effective range is ~5-15; below 1 it is nearly a no-op')
    elif neftune and neftune > 30:
        R.warn(f'neftune_noise_alpha = {neftune} is very high (typical 5-15)')

    if tcfg.get('report_to') and 'tensorboard' in (tcfg['report_to'] if isinstance(tcfg['report_to'], list) else [tcfg['report_to']]):
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            R.warn('report_to includes "tensorboard" but the package is not installed — Trainer will error')


# ============================================================================
# MODEL checks
# ============================================================================


def model_checks(cfg, mode, R, ctx, args):
    R.section('2. MODEL CHECKS')
    if mode == 'base':
        _model_checks_base(cfg, R, ctx)
    else:
        _model_checks_sft(cfg, R, ctx)


def _model_checks_base(cfg, R, ctx):
    if ctx.get('model_config') is None:
        try:
            ctx['model_config'], ctx['attn_implementation'] = build_model_config(cfg)
        except Exception as e:
            R.fail(f'cannot build model config: {e}')
            return
    model_config = ctx['model_config']

    from_pretrained = cfg['training'].get('from_pretrained')
    t0 = time.time()
    try:
        if from_pretrained:
            model = AutoModelForCausalLM.from_pretrained(from_pretrained, config=model_config, dtype=torch.float32)
            R.ok(f'model loads from from_pretrained = {from_pretrained} ({time.time() - t0:.1f}s)')
        else:
            model = AutoModelForCausalLM.from_config(model_config, dtype=torch.float32)
            R.ok(f'model instantiates from scratch ({time.time() - t0:.1f}s)')
    except Exception as e:
        R.fail(f'model failed to instantiate: {e}')
        return

    num_params = sum(p.numel() for p in model.parameters())
    ctx['num_params'] = num_params
    tied = getattr(model_config, 'tie_word_embeddings', False)
    embed_params = model_config.vocab_size * model_config.hidden_size * (1 if tied else 2)
    R.ok(f'{num_params / 1e6:.1f}M parameters ({make_model_id(model_config.model_type, num_params)})')
    R.info(
        f'embeddings: {embed_params / 1e6:.1f}M params = {100 * embed_params / num_params:.0f}% of the model '
        f'({"tied" if tied else "UNTIED input/output — set tie_word_embeddings=true to halve this"})'
    )
    R.info(f'static training footprint (fp32 weights+grads+AdamW): ~{gb(num_params * 16)} + activations')

    # ── Weight sanity ────────────────────────────────────────────────────────
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        R.fail(f'{len(bad)} parameter tensors contain NaN/Inf (e.g. {bad[:3]}) — corrupted init/checkpoint')
    else:
        R.ok('all weights finite')

    # ── Forward pass + init loss ─────────────────────────────────────────────
    seq = min(cfg['data'].get('max_seq_length', 512), 256)
    x = torch.randint(0, model_config.vocab_size, (2, seq))
    try:
        with torch.no_grad():
            out = model(input_ids=x, labels=x)
        loss = out.loss.item()
        expected = math.log(model_config.vocab_size)
        if not math.isfinite(loss):
            R.fail(f'forward pass produced non-finite loss: {loss}')
        elif from_pretrained:
            R.ok(f'forward pass OK, loss {loss:.2f} on random tokens (pretrained weights)')
        elif abs(loss - expected) > 1.5:
            R.warn(f'initial loss {loss:.2f} far from ln(vocab)={expected:.2f} — suspicious weight init')
        else:
            R.ok(f'forward pass OK, initial loss {loss:.2f} ≈ ln(vocab_size) = {expected:.2f} (healthy random init)')
    except Exception as e:
        R.fail(f'forward pass crashed: {e}')
        return

    # Keep the exact init so TRAINING checks (eager vs compiled) start identical.
    ctx['init_state'] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    del model


def _model_checks_sft(cfg, R, ctx):
    base_model = ctx.get('base_model')
    tokenizer = ctx.get('tokenizer')
    if not base_model:
        R.fail('no valid base_model — skipping model checks')
        return

    t0 = time.time()
    try:
        dtype = torch.bfloat16 if BF16_SUPPORTED else torch.float16
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
        R.ok(f'base model loads in {dtype} ({time.time() - t0:.1f}s)')
    except Exception as e:
        R.fail(f'base model failed to load: {e}')
        return

    num_params = sum(p.numel() for p in model.parameters())
    ctx['num_params'] = num_params
    R.ok(f'{num_params / 1e6:.1f}M parameters')

    # ── THE critical check: is the base checkpoint healthy? ──────────────────
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        R.fail(
            f'base checkpoint contains NaN/Inf in {len(bad)} tensors (e.g. {bad[:3]}) — '
            'this checkpoint is DIVERGED; fine-tuning it is pointless'
        )
        return
    R.ok('all base weights finite (checkpoint is not diverged)')

    if tokenizer is not None:
        embed_rows = model.get_input_embeddings().weight.shape[0]
        if embed_rows < len(tokenizer):
            R.fail(f'model embedding rows ({embed_rows}) < tokenizer vocab ({len(tokenizer)}) — token IDs will index out of range')
        else:
            R.ok(f'embedding rows ({embed_rows}) cover tokenizer vocab ({len(tokenizer)})')

    # ── Chat template render + forward ───────────────────────────────────────
    if tokenizer is not None and tokenizer.chat_template:
        msgs = [{'role': 'user', 'content': 'Who are you?'}, {'role': 'assistant', 'content': 'I am an assistant.'}]
        try:
            train_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            infer_text = tokenizer.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
            R.ok('chat template renders')
            R.info(f'TRAINING format : {train_text!r}')
            R.info(f'INFERENCE format: {infer_text!r}')
            if tokenizer.eos_token and not train_text.rstrip().endswith(tokenizer.eos_token.rstrip()):
                R.warn(f'training render does not end with EOS ({tokenizer.eos_token!r}) — generation may never stop after SFT')
            ids = tokenizer(train_text, return_tensors='pt').input_ids
            with torch.no_grad():
                loss = model.float()(input_ids=ids, labels=ids).loss.item()
            if not math.isfinite(loss):
                R.fail(f'base model forward on a chat sample gives non-finite loss: {loss}')
            else:
                R.ok(
                    f'base model loss on a chat-formatted sample: {loss:.2f} '
                    f'(ppl {math.exp(min(loss, 20)):.0f}) — high before SFT is normal'
                )
        except Exception as e:
            R.fail(f'chat template render/forward failed: {e}')

    # ── LoRA target modules actually exist ───────────────────────────────────
    if ctx.get('method') == 'lora':
        targets = cfg.get('lora', {}).get('target_modules', ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
        module_names = {n.rsplit('.', 1)[-1] for n, _ in model.named_modules()}
        missing = [t for t in targets if t not in module_names]
        if missing == list(targets):
            R.fail(f'NONE of the LoRA target_modules exist in this model: {targets}')
        elif missing:
            R.warn(f'LoRA target_modules not found in model: {missing}')
        else:
            R.ok(f'all {len(targets)} LoRA target_modules exist in the model')

    del model


# ============================================================================
# TRAINING checks (GPU dry-run)
# ============================================================================


def _train_step_fn(model, opt, amp_dtype, scaler, max_grad_norm):
    def step(batch):
        with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            loss = model(**batch).loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        return loss.item(), float(gn)

    return step


def _build_optimizer(tcfg, model, mode):
    name = tcfg.get('optim', 'adamw_torch_fused')
    kwargs = {
        'lr': tcfg.get('learning_rate', 5e-5),
        'betas': (tcfg.get('adam_beta1', 0.9), tcfg.get('adam_beta2', 0.95 if mode == 'base' else 0.999)),
        'eps': tcfg.get('adam_epsilon', 1e-8),
        'weight_decay': tcfg.get('weight_decay', 0.1 if mode == 'base' else 0.01),
    }
    if 'fused' in name:
        kwargs['fused'] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _populate_ctx_quietly(cfg, mode, ctx):
    """Fill ctx fields normally produced by the config category (for --only training)."""
    tcfg = cfg.get('training', {})
    if mode == 'base':
        if 'tokenizer' not in ctx and cfg.get('data', {}).get('tokenizer'):
            try:
                tok = AutoTokenizer.from_pretrained(cfg['data']['tokenizer'], use_fast=True)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                ctx['tokenizer'] = tok
            except Exception:
                ctx['tokenizer'] = None
        if 'train_files' not in ctx:
            ctx['train_files'] = resolve_files(cfg['data'].get('train_files', []))
            ctx['valid_files'] = resolve_files(cfg['data'].get('valid_files', []))
        if 'sched' not in ctx and ctx['train_files']:
            seq = cfg['data']['max_seq_length']
            total_tokens = sum(os.path.getsize(f) // 2 for f in ctx['train_files'])
            valid_tokens = sum(os.path.getsize(f) // 2 for f in ctx.get('valid_files', []))
            B = tcfg.get('per_device_train_batch_size', 8)
            gas = tcfg.get('gradient_accumulation_steps', 1)
            epochs = tcfg.get('num_train_epochs', 1)
            seqs = max(0, (total_tokens - (seq - 1)) // seq)
            micro = math.ceil(seqs / B)
            ctx['sched'] = {
                'seqs': seqs,
                'micro_per_epoch': micro,
                'opt_steps_per_epoch': math.ceil(micro / gas),
                'total_steps': math.ceil(micro / gas) * epochs,
                'B': B,
                'gas': gas,
                'epochs': epochs,
                'seq': seq,
            }
            ctx['eval_sequences'] = min(tcfg.get('max_eval_samples', 0) or 10**12, max(0, (valid_tokens - (seq - 1)) // seq))
    else:
        if 'base_model' not in ctx and cfg.get('model', {}).get('base_model'):
            bm = cfg['model']['base_model']
            if Path(bm, 'config.json').exists():
                ctx['base_model'] = bm
                try:
                    tok = AutoTokenizer.from_pretrained(bm, use_fast=True)
                    if tok.pad_token is None:
                        tok.pad_token = tok.eos_token
                    ctx['tokenizer'] = tok
                except Exception:
                    pass
        if 'sft_paths' not in ctx and (dp := cfg.get('data', {}).get('dataset_path')):
            paths = [dp] if isinstance(dp, str) else list(dp)
            ctx['sft_paths'] = [p for p in paths if Path(p).exists()]
        ctx.setdefault('method', tcfg.get('method', 'full').lower())
    if 'compile_requested' not in ctx:
        ctx['compile_requested'] = tcfg.get('torch_compile', mode == 'base')
    if 'amp_dtype' not in ctx:
        bf16 = tcfg.get('bf16', BF16_SUPPORTED)
        fp16 = tcfg.get('fp16', False)
        ctx['amp_dtype'] = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)


def training_checks(cfg, mode, R, ctx, args):
    R.section('3. TRAINING CHECKS (live GPU dry-run)')
    if not torch.cuda.is_available():
        R.fail('CUDA not available — cannot dry-run training')
        return
    _populate_ctx_quietly(cfg, mode, ctx)
    tcfg = cfg['training']
    amp_dtype = ctx.get('amp_dtype', torch.bfloat16 if BF16_SUPPORTED else torch.float32)
    max_grad_norm = tcfg.get('max_grad_norm', 1.0)
    seq = ctx.get('sched', {}).get('seq') or cfg['data'].get('max_seq_length', 1024)
    B = tcfg.get('per_device_train_batch_size', 8 if mode == 'base' else 4)
    B_eval = tcfg.get('per_device_eval_batch_size', B)

    # ── Batch + model factories per mode ─────────────────────────────────────
    if mode == 'base':
        make_batch = _base_batch_maker(ctx, cfg, seq)
        build_model = _base_gpu_model_builder(cfg, ctx)
    else:
        make_batch = _sft_batch_maker(ctx, cfg, seq)
        build_model = _sft_gpu_model_builder(cfg, ctx)
        if make_batch is None or build_model is None:
            R.fail('cannot build probe batches/model (see MODEL/CONFIG failures)')
            return

    free0, total = torch.cuda.mem_get_info()
    R.info(f'GPU free VRAM before probe: {gb(free0)} of {gb(total)}')

    # ── Build model + optimizer on GPU ───────────────────────────────────────
    t0 = time.time()
    try:
        model = build_model()
    except Exception as e:
        R.fail(f'model failed to move to GPU: {e}')
        return
    R.ok(
        f'model on GPU with training settings (dtype {next(model.parameters()).dtype}, '
        f'grad_ckpt={"on" if tcfg.get("gradient_checkpointing", False) else "off"}, {time.time() - t0:.1f}s)'
    )

    try:
        opt = _build_optimizer(tcfg, model, mode)
    except Exception as e:
        R.fail(f'optimizer "{tcfg.get("optim")}" failed to build: {e} — try optim = "adamw_torch"')
        return
    scaler = torch.amp.GradScaler('cuda') if amp_dtype == torch.float16 else None
    step = _train_step_fn(model, opt, amp_dtype, scaler, max_grad_norm)

    # ── First step + per-sample memory measurement ───────────────────────────
    try:
        loss, gn = step(make_batch(1, 0))
    except Exception as e:
        R.fail(f'first training step crashed: {type(e).__name__}: {e}')
        return
    if not math.isfinite(loss) or not math.isfinite(gn):
        R.fail(f'first training step: loss={loss}, grad_norm={gn} — non-finite from step one')
        return
    R.ok(f'optimizer step works: loss {loss:.3f}, grad_norm {gn:.2f} (optim = {tcfg.get("optim", "adamw")})')
    if gn > 100:
        R.warn(f'grad_norm {gn:.0f} at init is very high — expect heavy clipping early on')

    torch.cuda.reset_peak_memory_stats()
    step(make_batch(1, 1))
    peak1 = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step(make_batch(2, 1))
    peak2 = torch.cuda.max_memory_allocated()
    per_sample = max(peak2 - peak1, 1)
    R.info(f'memory: {gb(peak1)} fixed (weights+grads+optimizer+1 sample) + {per_sample / MB:.0f} MB per extra sample of seq {seq}')

    budget = free0 * 0.95
    predicted = peak1 + per_sample * (B - 1) * 1.05
    max_safe = max(1, int((budget - peak1) // per_sample) + 1)
    micro_s = None
    if predicted > budget:
        R.fail(
            f'per_device_train_batch_size = {B} needs ~{gb(predicted)} but only {gb(free0)} VRAM is free → guaranteed OOM. '
            f'Max safe batch ≈ {max_safe} (or use gradient_accumulation_steps to keep the effective batch)'
        )
    else:
        R.ok(f'predicted peak at batch={B}: {gb(predicted)} — fits (max safe batch ≈ {max_safe})')
        # ── Live steps at the real batch size ────────────────────────────────
        try:
            step(make_batch(B, 2))  # warm-up at full size
            torch.cuda.reset_peak_memory_stats()
            times, losses = [], []
            n_timed = max(3, args.steps)
            for i in range(n_timed):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                loss_i, gn_i = step(make_batch(B, 3 + i))
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                losses.append(loss_i)
                if not math.isfinite(loss_i) or not math.isfinite(gn_i):
                    R.fail(f'non-finite loss/grad at live step {i}: loss={loss_i} grad_norm={gn_i}')
                    return
            peak_live = torch.cuda.max_memory_allocated()
            headroom = free0 - peak_live
            micro_s = statistics.median(times)
            R.ok(
                f'{n_timed} live steps at batch={B}: peak {gb(peak_live)}, headroom {gb(headroom)} ({100 * headroom / free0:.0f}% of free)'
            )
            if headroom < 0.08 * free0:
                R.warn('less than 8% VRAM headroom — fragmentation or an eval pass can still OOM mid-run')
            tokens_s = B * seq / micro_s
            R.ok(f'throughput: {tokens_s:,.0f} tokens/s ({micro_s * 1000:.0f} ms per micro-batch of {B}×{seq})')
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            R.fail(f'OOM during live probe at batch={B} despite the prediction — max safe batch ≈ {max_safe}')

    # ── Time budget estimate ─────────────────────────────────────────────────
    sched = ctx.get('sched')
    if micro_s and sched:
        gas = sched.get('gas', 1)
        if mode == 'base':
            epoch_s = sched['micro_per_epoch'] * micro_s
            total_s = epoch_s * sched['epochs']
            R.info(f'estimated: {micro_s * gas:.1f} s/optimizer-step → {epoch_s / 3600:.1f} h/epoch, {total_s / 3600:.1f} h total')
            max_min = tcfg.get('max_train_minutes')
            if max_min:
                cover = 100 * (max_min * 60) / total_s
                steps_done = int(max_min * 60 / (micro_s * gas))
                R.info(f'max_train_minutes = {max_min} → ~{steps_done:,} optimizer steps = {min(cover, 100):.1f}% of the full schedule')
                if cover < 100 and tcfg.get('warmup_steps', 200) > steps_done:
                    R.warn(
                        f'the time limit stops training INSIDE warm-up ({steps_done:,} steps < '
                        f'warmup_steps {tcfg.get("warmup_steps")}) — LR never reaches peak'
                    )
        else:
            total_s = sched['total_steps'] * gas * micro_s
            R.info(
                f'estimated total fine-tune time: {total_s / 60:.0f} min for {sched["total_steps"]:,} optimizer steps '
                '(batches padded to max length — real runs are usually faster)'
            )

    # ── Eval pass memory + duration ──────────────────────────────────────────
    if micro_s:
        try:
            with torch.no_grad():
                torch.cuda.reset_peak_memory_stats()
                with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                    b = make_batch(min(B_eval, max_safe), 20)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model(**b)
                    torch.cuda.synchronize()
                    fwd_s = time.perf_counter() - t0
            peak_eval = torch.cuda.max_memory_allocated()
            if B_eval > max_safe:
                R.warn(f'per_device_eval_batch_size = {B_eval} likely exceeds VRAM (max safe ≈ {max_safe}); probed at {max_safe}')
            else:
                R.ok(f'eval forward at batch={B_eval}: peak {gb(peak_eval)}, {fwd_s * 1000:.0f} ms/batch')
            eval_seqs = ctx.get('eval_sequences')
            if eval_seqs:
                eval_min = math.ceil(eval_seqs / B_eval) * fwd_s / 60
                R.info(f'full eval pass ({eval_seqs:,} sequences) ≈ {eval_min:.1f} min')
                if tcfg.get('eval_strategy') == 'minutes' and eval_min > 0.5 * tcfg.get('eval_steps', 5):
                    R.warn(
                        f'eval takes ~{eval_min:.1f} min but runs every {tcfg.get("eval_steps")} min '
                        '— most wall-clock time will be spent evaluating'
                    )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            R.fail(f'OOM during eval forward at batch={B_eval} — lower per_device_eval_batch_size')

    # ── DataLoader with the configured workers (base mode only) ─────────────
    if mode == 'base' and micro_s and ctx.get('train_files'):
        _dataloader_probe(cfg, R, ctx, micro_s)

    # Free before the compile probe rebuilds everything
    del model, opt, step
    torch.cuda.empty_cache()

    # ── torch.compile stability probe ────────────────────────────────────────
    if ctx.get('compile_requested') or args.check_compile:
        _compile_stability_probe(R, build_model, make_batch, tcfg, amp_dtype, mode, min(B, max_safe), args.compile_steps)
    else:
        R.info('torch_compile = false → compile stability probe skipped (force with --check-compile)')
    torch.cuda.empty_cache()


def _base_gpu_model_builder(cfg, ctx):
    def build():
        model_config, attn = build_model_config(cfg)
        kw = {'dtype': torch.float32}
        if attn:
            kw['attn_implementation'] = attn
        fp = cfg['training'].get('from_pretrained')
        if fp:
            model = AutoModelForCausalLM.from_pretrained(fp, config=model_config, **kw)
        else:
            model = AutoModelForCausalLM.from_config(model_config, **kw)
            if ctx.get('init_state'):
                model.load_state_dict(ctx['init_state'])
        if cfg['training'].get('gradient_checkpointing', False):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        return model.cuda()

    return build


def _base_batch_maker(ctx, cfg, seq):
    vocab = cfg['model']['vocab_size']
    files = ctx.get('train_files') or resolve_files(cfg['data'].get('train_files', []))
    mm = np.memmap(files[0], dtype=np.uint16, mode='r') if files else None

    def make(bsz, idx):
        need = bsz * seq
        start = idx * need
        if mm is not None and len(mm) >= start + need:
            arr = np.asarray(mm[start : start + need], dtype=np.int64).reshape(bsz, seq)
            t = torch.from_numpy(arr).clamp_(0, vocab - 1).cuda()
        else:
            g = torch.Generator().manual_seed(idx)
            t = torch.randint(0, vocab, (bsz, seq), generator=g).cuda()
        return {'input_ids': t, 'labels': t.clone()}

    return make


def _sft_gpu_model_builder(cfg, ctx):
    base_model = ctx.get('base_model')
    if not base_model:
        return None

    def build():
        kw = {'dtype': torch.bfloat16 if BF16_SUPPORTED else torch.float16}
        if attn := cfg['model'].get('attn_implementation'):
            kw['attn_implementation'] = attn
        model = AutoModelForCausalLM.from_pretrained(base_model, **kw)
        model.config.use_cache = False
        if cfg['training'].get('gradient_checkpointing', False):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        return model.cuda()

    return build


def _sft_batch_maker(ctx, cfg, seq):
    """Worst-case SFT batches: real samples rendered + padded to max_seq_length."""
    tokenizer = ctx.get('tokenizer')
    paths = ctx.get('sft_paths')
    if tokenizer is None or not paths:
        return None
    texts = []
    with open(paths[0], encoding='utf-8') as f:
        for line in f:
            if len(texts) >= 64:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'messages' in row:
                try:
                    texts.append(tokenizer.apply_chat_template(row['messages'], tokenize=False))
                except Exception:
                    continue
            elif 'text' in row:
                texts.append(row['text'])
    if not texts:
        return None
    enc = tokenizer(texts, truncation=True, max_length=seq, padding='max_length', return_tensors='pt')
    all_ids = enc.input_ids
    all_mask = enc.attention_mask

    def make(bsz, idx):
        rows = torch.arange(idx * bsz, (idx + 1) * bsz) % len(texts)
        ids = all_ids[rows].cuda()
        mask = all_mask[rows].cuda()
        labels = ids.clone()
        labels[mask == 0] = -100
        return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}

    return make


def _dataloader_probe(cfg, R, ctx, micro_s):
    tcfg = cfg['training']
    workers = tcfg.get('dataloader_num_workers', 4)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            dataset = load_binary_files(cfg['data']['train_files'], cfg['data']['max_seq_length'])
        kwargs = {
            'batch_size': tcfg.get('per_device_train_batch_size', 8),
            'num_workers': workers,
            'pin_memory': tcfg.get('dataloader_pin_memory', True),
            'collate_fn': default_data_collator,
        }
        if workers > 0:
            kwargs['prefetch_factor'] = tcfg.get('dataloader_prefetch_factor', 2)
        dl = DataLoader(dataset, **kwargs)
        it = iter(dl)
        t0 = time.perf_counter()
        next(it)
        first_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(4):
            next(it)
        per_batch = (time.perf_counter() - t0) / 4
        del it, dl
        R.ok(f'DataLoader with {workers} workers: first batch {first_s:.2f}s (worker spawn), then {per_batch * 1000:.1f} ms/batch')
        if per_batch > micro_s:
            R.warn(
                f'data loading ({per_batch * 1000:.0f} ms/batch) is slower than the GPU step '
                f'({micro_s * 1000:.0f} ms) — training will be data-bound; raise dataloader_num_workers'
            )
    except Exception as e:
        R.fail(
            f'DataLoader with num_workers={workers} failed: {type(e).__name__}: {e} '
            '(Python 3.14 forkserver pickling issue? Try dataloader_num_workers = 0)'
        )


def _compile_stability_probe(R, build_model, make_batch, tcfg, amp_dtype, mode, B, K):
    """Train eager vs torch.compile from IDENTICAL weights on IDENTICAL batches
    and compare the loss/grad-norm trajectories. This is the check that catches
    "training is green until torch_compile=true makes it explode"."""
    R.info(f'compile stability probe: {K} steps eager vs compiled, batch={B}, identical weights+batches, lr={tcfg.get("learning_rate")}')

    def run(use_compile):
        torch.manual_seed(1234)
        model = build_model()
        opt = _build_optimizer(tcfg, model, mode)
        scaler = torch.amp.GradScaler('cuda') if amp_dtype == torch.float16 else None
        run_model = torch.compile(model) if use_compile else model
        step = _train_step_fn(run_model, opt, amp_dtype, scaler, tcfg.get('max_grad_norm', 1.0))
        losses, gnorms = [], []
        t0 = time.perf_counter()
        first_s = None
        step_times = []
        for i in range(K):
            ts = time.perf_counter()
            loss, gn = step(make_batch(B, 100 + i))
            torch.cuda.synchronize()
            dt = time.perf_counter() - ts
            if first_s is None:
                first_s = dt
            else:
                step_times.append(dt)
            losses.append(loss)
            gnorms.append(gn)
        del model, opt, step, run_model
        torch.cuda.empty_cache()
        with contextlib.suppress(Exception):
            torch._dynamo.reset()
        return losses, gnorms, first_s, (statistics.median(step_times) if step_times else first_s), time.perf_counter() - t0

    try:
        e_loss, e_gn, _, e_step, _ = run(False)
    except Exception as e:
        R.fail(f'eager reference run crashed: {e}')
        return
    try:
        c_loss, c_gn, c_first, c_step, _ = run(True)
    except Exception as e:
        R.fail(f'torch.compile run CRASHED: {type(e).__name__}: {str(e).splitlines()[0][:120]} — set torch_compile = false')
        return

    print(f'      {"step":>4} {"eager loss":>11} {"compiled":>11} {"eager gnorm":>12} {"compiled":>11}')
    for i in range(K):
        print(f'      {i:>4} {e_loss[i]:>11.4f} {c_loss[i]:>11.4f} {e_gn[i]:>12.3f} {c_gn[i]:>11.3f}')
    R.info(
        f'compile overhead: first step {c_first:.0f}s, then {c_step * 1000:.0f} ms/step vs eager {e_step * 1000:.0f} ms/step '
        f'({(e_step / c_step - 1) * 100:+.0f}% speed)'
    )

    if e_loss[-1] > e_loss[0] + 0.5:
        R.warn(
            f'loss INCREASED during the probe even WITHOUT compile ({e_loss[0]:.2f} → {e_loss[-1]:.2f}) — '
            'the learning rate may be too high regardless of torch_compile'
        )

    bad_c = [v for v in c_loss + c_gn if not math.isfinite(v)]
    max_e_gn, max_c_gn = max(e_gn), max(c_gn)
    rel_final = abs(c_loss[-1] - e_loss[-1]) / max(abs(e_loss[-1]), 1e-8)
    rel_max = max(abs(c - e) / max(abs(e), 1e-8) for c, e in zip(c_loss, e_loss, strict=True))
    if bad_c:
        R.fail('torch.compile training produces NaN/Inf within the probe — set torch_compile = false')
    elif max_c_gn > 5 * max_e_gn and max_c_gn > 10:
        R.fail(
            f'compiled grad norms explode (max {max_c_gn:.1f} vs eager {max_e_gn:.1f}) '
            '— torch_compile is numerically unstable here; set torch_compile = false'
        )
    elif rel_final > 0.05 or rel_max > 0.15:
        R.warn(
            f'compiled loss trajectory drifts from eager (max step diff {rel_max * 100:.1f}%, final {rel_final * 100:.1f}%) — '
            'not proof of divergence, but watch grad_norm closely if you keep torch_compile = true'
        )
    else:
        R.ok(
            f'torch.compile matches eager training over {K} steps '
            f'(final loss {c_loss[-1]:.4f} vs {e_loss[-1]:.4f}, max drift {rel_max * 100:.1f}%)'
        )


# ============================================================================
# DATASET checks
# ============================================================================


def dataset_checks(cfg, mode, R, ctx, args):
    R.section('4. DATASET CHECKS' + ('' if args.deep else ' (sampled — use --deep for a full scan)'))
    _populate_ctx_quietly(cfg, mode, ctx)
    if mode == 'base':
        _dataset_checks_base(cfg, R, ctx, args)
    else:
        _dataset_checks_sft(cfg, R, ctx, args)


def _scan_binary(files, budget, eos_id, label, R):
    """Chunk-sampled scan over binary shards. Returns aggregate stats."""
    total_tokens = sum(os.path.getsize(f) // 2 for f in files)
    hist = np.zeros(65536, dtype=np.int64)
    doc_lens = []
    scanned = 0
    max_run_len, max_run_tok = 0, None

    for f in files:
        size = os.path.getsize(f)
        if size % 2 != 0:
            R.fail(f'{Path(f).name}: odd byte size {size} — not a valid uint16 token file')
            continue
        n = size // 2
        mm = np.memmap(f, dtype=np.uint16, mode='r')
        file_budget = n if budget is None else max(SCAN_CHUNK, int(budget * n / max(total_tokens, 1)))
        n_chunks = max(1, math.ceil(min(file_budget, n) / SCAN_CHUNK))
        starts = np.linspace(0, max(0, n - SCAN_CHUNK), n_chunks).astype(np.int64)
        for s in starts:
            chunk = np.asarray(mm[s : s + SCAN_CHUNK])
            scanned += len(chunk)
            hist += np.bincount(chunk, minlength=65536)
            rl, rt = longest_run(chunk)
            if rl > max_run_len:
                max_run_len, max_run_tok = rl, rt
            if eos_id is not None:
                eos_pos = np.flatnonzero(chunk == eos_id)
                if len(eos_pos) > 1:
                    doc_lens.append(np.diff(eos_pos))
        del mm

    doc_lens = np.concatenate(doc_lens) if doc_lens else np.array([], dtype=np.int64)
    return {
        'total_tokens': total_tokens,
        'scanned': scanned,
        'hist': hist,
        'doc_lens': doc_lens,
        'max_run': (max_run_len, max_run_tok),
        'label': label,
    }


def _report_binary_scan(stats, vocab_size, eos_id, seq, tokenizer, R):
    label = stats['label']
    hist = stats['hist']
    scanned = stats['scanned']
    pct = 100 * scanned / max(stats['total_tokens'], 1)
    R.info(f'{label}: scanned {scanned / 1e6:.0f}M of {stats["total_tokens"] / 1e6:.0f}M tokens ({pct:.0f}%)')

    nz = np.flatnonzero(hist)
    max_id, min_id = int(nz.max()), int(nz.min())
    if max_id >= vocab_size:
        n_bad = int(hist[vocab_size:].sum())
        R.fail(
            f'{label}: token IDs out of range! max={max_id} ≥ vocab_size={vocab_size} '
            f'({n_bad:,} bad tokens in sample) — embedding lookups WILL crash'
        )
    else:
        R.ok(f'{label}: token IDs in range [{min_id}, {max_id}] < vocab {vocab_size}')

    # An unknown eos_id means the tokenizer never loaded (see CONFIG CHECKS).
    # Reporting "no EOS tokens" here would be a vacuous finding — the scan had
    # no id to look for — so say the check was skipped and let the real failure
    # upstream carry the signal.
    if eos_id is None:
        R.info(f'{label}: EOS check skipped — tokenizer unavailable, no eos_token_id to scan for')
        eos_count = -1  # -1 = not checked, distinct from 0 = checked and absent
    else:
        eos_count = int(hist[eos_id]) if eos_id < 65536 else 0

    if eos_count == 0:
        R.warn(f'{label}: no EOS tokens (id={eos_id}) in the sample — no document boundaries')
    elif eos_count > 0:
        density = scanned / eos_count
        R.ok(f'{label}: EOS every ~{density:,.0f} tokens ({eos_count:,} in sample)')
        dl = stats['doc_lens']
        if len(dl):
            R.info(
                f'document lengths (sampled): median {int(np.median(dl)):,}, p10 {int(np.percentile(dl, 10)):,}, '
                f'p90 {int(np.percentile(dl, 90)):,}, max {int(dl.max()):,} tokens'
            )
            tiny = int((dl < 16).sum())
            if tiny > 0.05 * len(dl):
                R.warn(f'{label}: {100 * tiny / len(dl):.1f}% of documents are < 16 tokens (noise/junk?)')
            short = int((dl < seq).sum())
            R.info(f'{100 * short / len(dl):.0f}% of documents are shorter than one sequence ({seq}) — they get chunk-packed together')

    run_len, run_tok = stats['max_run']
    if run_len > 256:
        tok_repr = repr(tokenizer.decode([run_tok]))[:30] if tokenizer else str(run_tok)
        R.warn(f'{label}: longest identical-token run is {run_len:,} × token {run_tok} ({tok_repr}) — possible corruption or filler')

    used = int((hist[:vocab_size] > 0).sum())
    R.info(
        f'vocab coverage: {used:,} of {vocab_size:,} token IDs seen ({100 * used / vocab_size:.1f}%)'
        + (' — sampled scan undercounts rare tokens' if pct < 99 else '')
    )
    top = np.argsort(hist)[::-1][:8]
    if tokenizer:
        tops = ', '.join(f'{repr(tokenizer.decode([int(t)]))}:{100 * hist[t] / scanned:.1f}%' for t in top)
        R.info(f'top tokens: {tops}')
    top1_share = hist[top[0]] / scanned
    if top1_share > 0.10:
        R.warn(f'{label}: single token {int(top[0])} is {100 * top1_share:.0f}% of all data — degenerate distribution?')


def _dataset_checks_base(cfg, R, ctx, args):
    tokenizer = ctx.get('tokenizer')
    vocab_size = cfg['model']['vocab_size']
    seq = cfg['data']['max_seq_length']
    eos_id = tokenizer.eos_token_id if tokenizer else None

    train_files = ctx.get('train_files') or resolve_files(cfg['data'].get('train_files', []))
    valid_files = ctx.get('valid_files') or resolve_files(cfg['data'].get('valid_files', []))
    if not train_files:
        R.fail('no training files to scan')
        return

    stats = _scan_binary(train_files, None if args.deep else TRAIN_SCAN_BUDGET, eos_id, 'TRAIN', R)
    _report_binary_scan(stats, vocab_size, eos_id, seq, tokenizer, R)

    # ── Tokens : params budget ───────────────────────────────────────────────
    n_params = ctx.get('num_params')
    if not n_params:
        try:
            n_params = count_params_from_config(build_model_config(cfg)[0])
        except Exception:
            n_params = None
    if n_params and ctx.get('sched'):
        total = stats['total_tokens'] * ctx['sched']['epochs']
        ratio = total / n_params
        if ratio < 10:
            R.warn(f'tokens:params ratio {ratio:.0f}× is below Chinchilla-optimal (~20×) — model may underfit')
        else:
            R.ok(f'training budget: {total / 1e9:.1f}B tokens / {n_params / 1e6:.0f}M params = {ratio:.0f}× (Chinchilla-optimal ≈ 20×)')

    if valid_files:
        vstats = _scan_binary(valid_files, None if args.deep else VALID_SCAN_BUDGET, eos_id, 'VALID', R)
        _report_binary_scan(vstats, vocab_size, eos_id, seq, tokenizer, R)

        # train/valid contamination (cheap head-hash comparison)
        def head_hash(f):
            with open(f, 'rb') as fh:
                return hashlib.sha1(fh.read(1024 * 1024)).hexdigest()

        vhashes = {head_hash(f) for f in valid_files}
        dupes = [f for f in train_files if head_hash(f) in vhashes]
        if dupes:
            R.fail(f'validation data overlaps training data (identical file heads): {[Path(d).name for d in dupes]}')
        else:
            R.ok('validation files do not duplicate training files (head-hash check)')

    # ── Human eyeball check ──────────────────────────────────────────────────
    if tokenizer:
        mm = np.memmap(train_files[0], dtype=np.uint16, mode='r')
        rng = np.random.RandomState(42)
        start = int(rng.randint(0, max(1, len(mm) - 300)))
        text = tokenizer.decode(np.asarray(mm[start : start + 200]).tolist())
        snippet = text.replace('\n', ' ↵ ')[:300]
        R.info(f'random decoded sample @token {start:,}: "{snippet}..."')
        if '�' in text:
            R.warn('decoded sample contains U+FFFD replacement characters — tokenizer/data mismatch?')


def _dataset_checks_sft(cfg, R, ctx, args):
    tokenizer = ctx.get('tokenizer')
    paths = ctx.get('sft_paths')
    seq = cfg['data'].get('max_seq_length', 2048)
    if not paths:
        R.info('no local dataset_path — skipping JSONL scan')
        return

    row_cap = None if args.deep else 50_000
    n = 0
    blank_lines = 0
    parse_errors = 0
    schema_errors = []
    role_problems = 0
    empty_content = 0
    not_assistant_last = 0
    seen_hashes = set()
    dupes = 0
    samples_for_len = []
    valid_roles = {'system', 'user', 'assistant', 'tool'}

    for p in paths:
        with open(p, encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                if row_cap and n >= row_cap:
                    break
                n += 1
                if not line.strip():
                    blank_lines += 1
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
                    if parse_errors <= 3:
                        schema_errors.append(f'{Path(p).name}:{line_no} invalid JSON')
                    continue
                h = hashlib.sha1(line.strip().encode()).hexdigest()
                if h in seen_hashes:
                    dupes += 1
                seen_hashes.add(h)
                msgs = row.get('messages')
                if msgs is None:
                    if 'text' not in row:
                        schema_errors.append(f'{Path(p).name}:{line_no} has neither "messages" nor "text"')
                    elif len(samples_for_len) < 1500:
                        samples_for_len.append(row['text'])
                    continue
                roles = [m.get('role') for m in msgs]
                if any(r not in valid_roles for r in roles) or not msgs:
                    role_problems += 1
                if any(not str(m.get('content', '')).strip() for m in msgs):
                    empty_content += 1
                if roles and roles[-1] != 'assistant':
                    not_assistant_last += 1
                if len(samples_for_len) < 1500:
                    samples_for_len.append(msgs)

    scanned_note = f'{n:,} rows' + ('' if args.deep or n < (row_cap or 0) else f' (first {row_cap:,}; use --deep for all)')
    if parse_errors:
        R.fail(f'{parse_errors} unparseable JSON lines out of {scanned_note} (e.g. {schema_errors[:3]})')
    else:
        R.ok(f'all {scanned_note} parse as JSON')
    if blank_lines:
        R.warn(f'{blank_lines} blank line(s) in the JSONL (usually tolerated, but worth cleaning up)')
    if schema_errors and not parse_errors:
        R.fail(f'{len(schema_errors)} rows missing "messages"/"text": {schema_errors[:3]}')
    if role_problems:
        R.fail(f'{role_problems} rows have invalid/missing roles (allowed: {sorted(valid_roles)})')
    else:
        R.ok('all roles valid')
    if empty_content:
        R.warn(f'{empty_content} rows contain empty message content')
    if not_assistant_last:
        R.warn(f'{not_assistant_last} rows do not END with an assistant turn — they contribute no (or truncated) training signal')
    else:
        R.ok('every conversation ends with an assistant turn')
    if dupes:
        R.warn(f'{dupes} exact duplicate rows ({100 * dupes / max(n, 1):.1f}%)')
    else:
        R.ok('no exact duplicate rows')

    # ── Token length distribution (the truncation check) ─────────────────────
    if tokenizer and samples_for_len:
        lengths = []
        for s in samples_for_len:
            try:
                text = s if isinstance(s, str) else tokenizer.apply_chat_template(s, tokenize=False)
                lengths.append(len(tokenizer(text).input_ids))
            except Exception:
                continue
        if lengths:
            arr = np.array(lengths)
            over = int((arr > seq).sum())
            R.info(
                f'token lengths (n={len(arr)} sampled): median {int(np.median(arr))}, p90 {int(np.percentile(arr, 90))}, '
                f'p99 {int(np.percentile(arr, 99))}, max {int(arr.max())}'
            )
            if over:
                pct = 100 * over / len(arr)
                lvl = R.fail if pct > 25 else R.warn
                lvl(f'{pct:.1f}% of examples exceed max_seq_length={seq} and will be TRUNCATED (often cutting the assistant answer)')
            else:
                R.ok(f'no sampled example exceeds max_seq_length = {seq}')
            mean_len = float(arr.mean())
            waste = 100 * (1 - mean_len / seq)
            if not cfg['training'].get('packing', False) and waste > 60:
                R.info(
                    f'mean example is {mean_len:.0f} tokens vs max_length {seq} → ~{waste:.0f}% of each padded '
                    'batch is padding; consider packing = true (needs flash attention)'
                )


# ============================================================================
# DECODE checks (inference-side dry-run)
# ============================================================================
#
# Everything above this point measures TRAINING. None of it touches
# generate(), so a checkpoint can pass every other category and still be
# unusable at inference.
#
# Kept cheap by default: prefill 64 + 128 forced new tokens, cache on, batch 1.
# The expensive comparisons (cache off, which is quadratic, and the batch
# sweep) only run under --deep.

DECODE_PREFILL = 64
DECODE_NEW_TOKENS = 128
DECODE_DEEP_BATCHES = (1, 4, 16)


def _decode_model(cfg, mode):
    """Load the checkpoint the way INFERENCE would: eval mode, cache enabled."""
    dtype = torch.bfloat16 if BF16_SUPPORTED else torch.float16
    if mode == 'sft':
        src = cfg['model'].get('base_model')
        if not src:
            return None, None
        model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype)
        return model.cuda().eval(), src
    # Base mode: no trained weights yet. Throughput is an architecture
    # property, so a from_config model gives the right number; the loss-bearing
    # checks below are skipped because random weights say nothing about them.
    src = cfg['training'].get('from_pretrained')
    if src:
        model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype)
        return model.cuda().eval(), src
    model_config, _ = build_model_config(cfg)
    model = AutoModelForCausalLM.from_config(model_config, dtype=dtype)
    return model.cuda().eval(), None


def _time_decode(model, bsz, new_tokens, use_cache, vocab):
    """tok/s over `new_tokens` FORCED new tokens, warmed up first.

    min_new_tokens is not optional. A model that hits EOS after three tokens
    but is credited with the full requested length once measured 54x faster
    than it really was.
    """
    ids = torch.randint(0, min(vocab, 1000), (bsz, DECODE_PREFILL), device='cuda')
    gen = dict(do_sample=False, use_cache=use_cache, pad_token_id=0)
    with torch.no_grad():
        model.generate(ids, max_new_tokens=8, min_new_tokens=8, **gen)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, **gen)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    produced = (out.shape[1] - DECODE_PREFILL) * bsz
    return produced / dt, dt, torch.cuda.max_memory_allocated()


def _check_generation_config(model, tokenizer, R):
    """Stop-token and sampling sanity on the config that ships with the model."""
    gc_obj = getattr(model, 'generation_config', None)
    if gc_obj is None:
        R.warn('model has no generation_config — generate() will fall back to library defaults')
        return
    vocab = model.config.vocab_size

    eos = gc_obj.eos_token_id
    eos_list = [eos] if isinstance(eos, int) else list(eos or [])
    if not eos_list:
        R.fail('generation_config has no eos_token_id — generate() can only stop at max_length')
    else:
        # Duplicates are a provable no-op: EosTokenCriteria does
        # torch.isin(input_ids, eos_tensor), so a repeated id changes nothing.
        # Reported as info, not a warning — it only tells you the list was
        # hand-edited at some point.
        if len(eos_list) != len(set(eos_list)):
            dupes = sorted({t for t in eos_list if eos_list.count(t) > 1})
            R.info(f'eos_token_id {eos_list} repeats {dupes}; torch.isin ignores duplicates, so behaviour is unchanged')
        oob = [t for t in eos_list if not 0 <= t < vocab]
        if oob:
            R.fail(f'eos_token_id {oob} outside vocab_size {vocab}')
        if tokenizer is not None:
            named = {t: tokenizer.convert_ids_to_tokens(t) for t in set(eos_list) if 0 <= t < vocab}
            R.info(f'stop tokens: {", ".join(f"{t}={n!r}" for t, n in sorted(named.items()))}')
            tok_eos = getattr(tokenizer, 'eos_token_id', None)
            if tok_eos is not None and tok_eos not in eos_list:
                R.warn(f"tokenizer eos_token_id {tok_eos} ({tokenizer.eos_token!r}) is NOT in generation_config's {eos_list}")
            # A stop token the model never emits is inert, and if it ever DID
            # emit one, stopping is the behaviour you want. Worth surfacing so
            # nobody mistakes the list for something the model relies on, but
            # it is not a defect.
            inert = [(t, n) for t, n in named.items() if n and ('mask' in n.lower() or 'pad' in n.lower() or 'unk' in n.lower())]
            for t, n in inert:
                R.info(f'stop token {t} is {n!r}, a token the model is not trained to emit; inert, kept as a safety net')
        else:
            R.info(f'stop tokens: {eos_list}')

    if gc_obj.pad_token_id is None:
        R.warn('generation_config has no pad_token_id — batched generate() warns and falls back to eos')
    if getattr(gc_obj, 'max_length', None) == 20 and getattr(gc_obj, 'max_new_tokens', None) is None:
        R.warn('generation_config keeps the library default max_length = 20 — callers must pass max_new_tokens or output is cut off')
    if not gc_obj.do_sample:
        for k in ('temperature', 'top_p', 'top_k'):
            v = getattr(gc_obj, k, None)
            if v is not None and v not in (1.0, 1, 0, 50):
                R.warn(f'do_sample = false but {k} = {v} is set — transformers warns and ignores it')


def decode_checks(cfg, mode, R, ctx, args):
    R.section('5. DECODE CHECKS (inference dry-run)')
    if not torch.cuda.is_available():
        R.fail('CUDA not available — cannot dry-run decoding')
        return
    try:
        model, src = _decode_model(cfg, mode)
    except Exception as e:
        R.fail(f'could not load a model for decoding: {type(e).__name__}: {e}')
        return
    if model is None:
        R.info('no trained checkpoint to decode from (base run from scratch) — skipped')
        return
    # ctx['tokenizer'] is filled by config_checks. Load it here too so
    # `--only decode` still names its stop tokens instead of printing bare ids.
    tokenizer = ctx.get('tokenizer')
    if tokenizer is None and src:
        try:
            tokenizer = AutoTokenizer.from_pretrained(src)
        except Exception:
            tokenizer = None
    vocab = model.config.vocab_size
    R.info(f'decoding from {src or "a from_config model (random weights)"}')

    try:
        _check_generation_config(model, tokenizer, R)

        # ── The use_cache trap ───────────────────────────────────────────────
        # config.use_cache is what a plain from_pretrained().generate() honours.
        # It is false in every final/ in this repo, inherited from training.
        shipped = getattr(model.config, 'use_cache', True)
        if shipped is False:
            R.warn(
                'config.json ships "use_cache": false, inherited from the training config. '
                'A plain from_pretrained(...).generate() runs with no KV cache and recomputes the '
                'whole prefix every token. Affects speed only, never output, and the eval harness '
                'already overrides it (eval/helpers.py sets use_cache = True). Whether it costs '
                'anything is model-specific: run --deep to measure it instead of assuming.'
            )
        else:
            R.ok('config.json has use_cache enabled — generate() will use the KV cache')

        # ── Throughput, cache ON (the number that matters for serving) ───────
        tps, dt, peak = _time_decode(model, 1, DECODE_NEW_TOKENS, True, vocab)
        R.ok(
            f'decode throughput: {tps:,.0f} tok/s ({1000 / tps:.1f} ms/token) '
            f'at batch 1, greedy, {DECODE_NEW_TOKENS} forced new tokens, cache on ({dt:.1f}s)'
        )
        R.info(f'peak VRAM during decode: {gb(peak)}')

        # ── KV cache size at full context ────────────────────────────────────
        c = model.config
        n_kv = getattr(c, 'num_key_value_heads', None) or c.num_attention_heads
        head_dim = getattr(c, 'head_dim', None) or c.hidden_size // c.num_attention_heads
        bytes_per = 2  # bf16/fp16
        kv_full = 2 * c.num_hidden_layers * n_kv * head_dim * c.max_position_embeddings * bytes_per
        R.info(
            f'KV cache at full context ({c.max_position_embeddings} tokens): {kv_full / MB:.0f} MB per sequence '
            f'({n_kv} KV heads x {head_dim} head_dim x {c.num_hidden_layers} layers)'
        )

        if not args.deep:
            R.info('cache-off comparison and batch sweep skipped — rerun with --deep (adds ~30-60s)')
        else:
            # ── Cache OFF: quadratic, so this is the slow one ────────────────
            tps_off, dt_off, _ = _time_decode(model, 1, DECODE_NEW_TOKENS, False, vocab)
            speedup = tps / tps_off
            R.info(f'cache off: {tps_off:,.0f} tok/s ({dt_off:.1f}s) - the KV cache is worth {speedup:.2f}x here')
            if speedup < 1.2:
                # Measured on Llama-75M: 1.00x at 128 new tokens, still only
                # 1.08x at 900. Throughput sits at ~330 tok/s whatever the
                # length, which is kernel-launch latency, not attention math.
                # A cache saves recomputation the model was never spending
                # time on. Do not assume the textbook multiple at this size.
                R.info(
                    'under 1.2x: decoding is launch-latency bound at this size, not compute bound, '
                    'so the KV cache buys almost nothing and "use_cache": false is nearly free here'
                )
            # ── Batch scaling: where decode stops being memory-bound ─────────
            for b in DECODE_DEEP_BATCHES:
                try:
                    tps_b, _, peak_b = _time_decode(model, b, DECODE_NEW_TOKENS, True, vocab)
                    R.info(f'batch {b:>2}: {tps_b:>7,.0f} tok/s aggregate, {tps_b / b:>6,.0f} tok/s per sequence, peak {gb(peak_b)}')
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    R.warn(f'batch {b} OOMs during decode')
                    break
    except Exception as e:
        R.fail(f'decode probe crashed: {type(e).__name__}: {e}')
    finally:
        del model
        torch.cuda.empty_cache()


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description='Pre-flight doctor for base_train.py / fine_tune.py')
    parser.add_argument(
        'config', nargs='?', default=str(Path(__file__).with_name('config.toml')), help='TOML config (base or fine-tune; auto-detected)'
    )
    parser.add_argument('--skip', default='', help=f'comma-separated categories to skip {CATEGORIES}')
    parser.add_argument('--only', default='', help='comma-separated categories to run (overrides --skip)')
    parser.add_argument(
        '--deep',
        action='store_true',
        help='slow probes: scan the ENTIRE dataset, and add the decode cache-off comparison + batch sweep',
    )
    parser.add_argument('--steps', type=int, default=3, help='timed live training steps (default 3)')
    parser.add_argument('--compile-steps', type=int, default=8, help='steps for the eager-vs-compiled comparison (default 8)')
    parser.add_argument('--check-compile', action='store_true', help='run the compile probe even if torch_compile = false')
    args = parser.parse_args()

    if args.only:
        cats = [c.strip() for c in args.only.split(',')]
    else:
        skip = {c.strip() for c in args.skip.split(',') if c.strip()}
        cats = [c for c in CATEGORIES if c not in skip]
    unknown = [c for c in cats if c not in CATEGORIES]
    if unknown:
        print(f'Unknown categories: {unknown} (valid: {CATEGORIES})')
        return 2

    if not Path(args.config).exists():
        print(f'❌ Config file not found: {args.config}')
        return 1
    cfg = load_config(args.config)
    mode = 'sft' if cfg.get('model', {}).get('base_model') else 'base'

    transformers.utils.logging.disable_progress_bar()
    transformers.utils.logging.set_verbosity_error()

    print('═' * 72)
    print('  TRAINING PRE-FLIGHT DOCTOR')
    print('═' * 72)
    print(f'  Config    : {args.config}')
    print(f'  Mode      : {"supervised fine-tuning (fine_tune.py)" if mode == "sft" else "base pre-training (base_train.py)"}')
    print(f'  Stack     : torch {torch.__version__}, transformers {transformers.__version__}')
    if torch.cuda.is_available():
        print(f'  GPU       : {torch.cuda.get_device_name(0)} ({torch.cuda.mem_get_info()[1] / GB:.1f} GB)')
    else:
        print('  GPU       : NONE VISIBLE')
    print(f'  Categories: {", ".join(cats)}')

    t_start = time.time()
    R = Reporter()
    ctx = {}
    runners = {
        'config': config_checks,
        'model': model_checks,
        'training': training_checks,
        'dataset': dataset_checks,
        'decode': decode_checks,
    }
    for cat in CATEGORIES:
        if cat not in cats:
            continue
        try:
            runners[cat](cfg, mode, R, ctx, args)
        except Exception as e:
            R.fail(f'{cat} checks crashed unexpectedly: {type(e).__name__}: {e}')
            traceback.print_exc()

    code = R.summary()
    print(f'  (doctor finished in {time.time() - t_start:.0f}s)')
    return code


if __name__ == '__main__':
    sys.exit(main())

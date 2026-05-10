#!/usr/bin/env python3
"""
Quick test script to validate the training setup without actually training.
Checks that all components load correctly and runs dataset quality diagnostics.
"""

import sys
import traceback
from glob import glob
from pathlib import Path

import numpy as np
import torch
from base_train import load_binary_files, load_config
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)


def validate_binary_data(data, tokenizer, seq_length, label):
    """
    Run dataset quality checks on a binary token array.

    Returns True if data looks healthy, False if critical issues found.
    """
    vocab_size = len(tokenizer)
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id
    total_tokens = len(data)
    ok = True

    print(f'\n  {"─" * 60}')
    print(f'  {label}: {total_tokens:,} tokens')
    print(f'  {"─" * 60}')

    # ── Token ID range ──────────────────────────────────────────
    # Out-of-range IDs crash the embedding layer at training time.
    arr = np.array(data[:], dtype=np.uint16) if not isinstance(data, np.ndarray) else data
    max_id = int(arr.max())
    min_id = int(arr.min())
    if max_id >= vocab_size:
        print(f'  ❌ Token IDs out of range: max={max_id}, vocab_size={vocab_size}')
        bad_count = int(np.sum(arr >= vocab_size))
        print(f'     {bad_count:,} tokens have ID >= {vocab_size}')
        ok = False
    else:
        print(f'  ✓ Token IDs in range [0, {vocab_size}): min={min_id}, max={max_id}')

    # ── BOS/EOS presence and pairing ────────────────────────────
    # These markers are how the dataset finds document boundaries.
    eos_count = int(np.sum(arr == eos_id))
    bos_count = int(np.sum(arr == bos_id))

    if eos_count == 0:
        print(f'  ⚠️  No EOS tokens (id={eos_id}) found — cannot detect document boundaries')
        print('     All data will be treated as one giant document')
        ok = False  # not fatal but bad
    else:
        print(f'  ✓ EOS tokens: {eos_count:,}  (id={eos_id})')

    if bos_count == 0:
        print(f'  ⚠️  No BOS tokens (id={bos_id}) found')
    else:
        print(f'  ✓ BOS tokens: {bos_count:,}  (id={bos_id})')

    if bos_count > 0 and eos_count > 0 and bos_count != eos_count:
        print(f'  ⚠️  BOS/EOS count mismatch: {bos_count} BOS vs {eos_count} EOS')
        print('     This may indicate corrupted or partially tokenized data')

    # ── Document length distribution ────────────────────────────
    if eos_count > 0:
        eos_positions = np.where(arr == eos_id)[0]
        doc_starts = np.concatenate([[0], eos_positions[:-1] + 1]).astype(np.int64)
        doc_ends = (eos_positions + 1).astype(np.int64)
        doc_lengths = doc_ends - doc_starts

        num_docs = len(doc_lengths)
        print(f'\n  Documents: {num_docs:,}')
        print(f'    Min length:    {int(doc_lengths.min()):>12,} tokens')
        print(f'    Max length:    {int(doc_lengths.max()):>12,} tokens')
        print(f'    Median length: {int(np.median(doc_lengths)):>12,} tokens')
        print(f'    Mean length:   {int(np.mean(doc_lengths)):>12,} tokens')

        tiny_docs = int(np.sum(doc_lengths < 16))
        small_docs = int(np.sum(doc_lengths < seq_length))
        large_docs = int(np.sum(doc_lengths >= seq_length))
        huge_docs = int(np.sum(doc_lengths > seq_length * 100))

        if tiny_docs > 0:
            print(f'    ⚠️  {tiny_docs:,} documents < 16 tokens (may be noise)')
        print(
            f'    < seq_length ({seq_length}):  {small_docs:,} documents '
            f'({100 * small_docs / num_docs:.1f}%) — each becomes 1 padded sequence'
        )
        print(
            f'    ≥ seq_length ({seq_length}):  {large_docs:,} documents '
            f'({100 * large_docs / num_docs:.1f}%) — chunked into multiple sequences'
        )
        if huge_docs > 0:
            print(f'    > 100× seq_length:  {huge_docs:,} documents (very long)')
    else:
        print('\n  Documents: 1 (no EOS boundaries detected)')

    # ── Build dataset and measure padding overhead ──────────────
    print(f'\n  Building dataset (seq_length={seq_length})...')
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    dataset = load_binary_files(
        # We already have the data loaded, but load_binary_files needs file paths.
        # Skip this if we can't re-load — use the dataset from the caller instead.
        None,
        seq_length,
        eos_token_id=eos_id,
        pad_token_id=pad_token_id,
        _preloaded_data=data,
    )

    num_seqs = len(dataset)
    total_positions = num_seqs * seq_length
    padded_positions = 0
    boundary_violations = 0
    sample_count = min(num_seqs, 2000)  # check up to 2000 sequences for speed

    for i in range(sample_count):
        s = dataset[i]
        labs = s['labels']
        ids = s['input_ids']
        pad_count = int((labs == -1).sum())
        padded_positions += pad_count

        # Check that EOS only appears at the end of content, never in the middle.
        # A mid-sequence EOS means we're crossing a document boundary.
        content_len = seq_length - pad_count
        if content_len > 0:
            content = ids[:content_len]
            eos_mask = content == eos_id
            if eos_mask.sum() > 0:
                eos_positions_in_seq = torch.where(eos_mask)[0]
                # EOS at the last content position is fine (end of doc)
                if eos_positions_in_seq[-1] != content_len - 1 or len(eos_positions_in_seq) > 1:
                    boundary_violations += 1

    # Extrapolate padding to full dataset
    if sample_count < num_seqs:
        padded_positions = int(padded_positions * num_seqs / sample_count)
        note = f' (estimated from {sample_count:,} samples)'
    else:
        note = ''

    useful_positions = total_positions - padded_positions
    pad_pct = 100 * padded_positions / total_positions if total_positions > 0 else 0

    padded_seqs = sum(1 for i in range(sample_count) if int((dataset[i]['labels'] == -1).sum()) > 0)
    if sample_count < num_seqs:
        padded_seqs = int(padded_seqs * num_seqs / sample_count)

    print(f'\n  Sequences:        {num_seqs:,}')
    print(f'  Total positions:  {total_positions:,}')
    print(f'  Useful tokens:    {useful_positions:,}')
    print(f'  Padded positions: {padded_positions:,} ({pad_pct:.1f}% wasted){note}')
    print(f'  Padded sequences: {padded_seqs:,} / {num_seqs:,} ({100 * padded_seqs / num_seqs:.1f}%)')

    if pad_pct > 10:
        print(f'  ⚠️  High padding overhead ({pad_pct:.1f}%). Consider a shorter seq_length or longer documents.')
    elif pad_pct > 0:
        print('  ✓ Padding overhead is low')

    # ── Boundary integrity ──────────────────────────────────────
    if boundary_violations > 0:
        print(f'  ❌ {boundary_violations} sequences cross document boundaries!')
        ok = False
    else:
        checked_label = f' (checked {sample_count:,})' if sample_count < num_seqs else ''
        print(f'  ✓ No cross-document sequences{checked_label}')

    return ok, dataset, useful_positions


def test_setup():
    """Test the training setup."""

    print('\n' + '=' * 80)
    print('TESTING TRAINING SETUP')
    print('=' * 80 + '\n')

    # 1. Test config loading
    print('[1/6] Testing config loading...')
    config_path = 'training/config.toml'

    if not Path(config_path).exists():
        print(f'❌ Config file not found: {config_path}')
        return False

    cfg = load_config(config_path)
    print(f'✓ Config loaded from {config_path}')
    print(
        f'  Model: {cfg["model"]["num_hidden_layers"]} layers, '
        f'{cfg["model"]["hidden_size"]} hidden, '
        f'{cfg["model"]["num_attention_heads"]} heads'
    )

    # Test tokenizer
    print('\n[2/6] Testing tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(cfg['data']['tokenizer'], use_fast=True)
    # Ensure tokenizer has a pad token (required for DataCollator)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f'✓ Tokenizer loaded: {cfg["data"]["tokenizer"]}')
    print(f'  Vocab size: {len(tokenizer)}')

    # Quick test encoding
    test_text = 'Hello, world!'
    encoded = tokenizer(test_text)
    print(f'  Test encoding: {encoded}')
    decoded = tokenizer.decode(encoded['input_ids'], clean_up_tokenization_spaces=False)
    print(f'  Test decoding: {decoded}')

    # Test model creation
    print('\n[3/6] Testing model creation...')
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

    model = AutoModelForCausalLM.from_config(model_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'✓ Model created: {model.__class__.__name__}')
    print(f'  Architecture: {model_config.model_type}')
    print(f'  Parameters: {num_params:,} ({num_params / 1e6:.2f}M)')

    # Test data loading and dataset diagnostics
    print('\n[4/6] Validating datasets...')
    seq_length = cfg['data']['max_seq_length']
    train_files = cfg['data']['train_files']
    valid_files = cfg['data']['valid_files']

    # Check if data files exist
    if isinstance(train_files, str):
        train_exists = Path(train_files).exists()
    else:
        train_exists = any(Path(f).exists() for f in train_files)

    if not train_exists:
        print(f'⚠️  Training data not found: {train_files}')
        print('   Run: python training/create_dummy_data.py')
        return False

    def _resolve_files(pattern):
        if isinstance(pattern, str):
            return sorted(glob(pattern)) if ('*' in pattern or '?' in pattern) else [pattern]
        files = []
        for p in pattern:
            files.extend(sorted(glob(p)) if ('*' in p or '?' in p) else [p])
        return files

    # ── Train data ──
    train_data_files = _resolve_files(train_files)
    memmaps = [np.memmap(f, dtype=np.uint16, mode='r') for f in train_data_files]
    if len(memmaps) == 1:
        from base_train import ConcatMemmap

        train_data = memmaps[0]
    else:
        from base_train import ConcatMemmap

        train_data = ConcatMemmap(memmaps)

    train_ok, train_dataset, train_useful = validate_binary_data(train_data, tokenizer, seq_length, 'TRAIN')

    # ── Valid data ──
    if isinstance(valid_files, str):
        valid_exists = Path(valid_files).exists()
    else:
        valid_exists = any(Path(f).exists() for f in valid_files)

    if valid_exists:
        valid_data_files = _resolve_files(valid_files)
        vmemmaps = [np.memmap(f, dtype=np.uint16, mode='r') for f in valid_data_files]
        valid_data = vmemmaps[0] if len(vmemmaps) == 1 else ConcatMemmap(vmemmaps)
        validate_binary_data(valid_data, tokenizer, seq_length, 'VALID')
    else:
        print(f'\n  ⚠️  Validation data not found: {valid_files}')

    # ── Data / parameter ratio ──────────────────────────────────
    epochs = cfg['training'].get('num_train_epochs', 1)
    total_train_tokens = train_useful * epochs
    ratio = total_train_tokens / num_params if num_params > 0 else 0
    print(f'\n  {"─" * 60}')
    print('  TRAINING BUDGET')
    print(f'  {"─" * 60}')
    print(f'  Useful train tokens/epoch: {train_useful:,}')
    print(f'  Epochs:                    {epochs}')
    print(f'  Total train tokens:        {total_train_tokens:,}')
    print(f'  Model parameters:          {num_params:,}')
    print(f'  Tokens / params ratio:     {ratio:.1f}×')
    if ratio < 10:
        print(f'  ⚠️  Low data:param ratio ({ratio:.1f}×). Model may underfit.')
        print('     Chinchilla-optimal is ~20×. Consider more data or a smaller model.')
    elif ratio > 200:
        print(f'  ⚠️  Very high data:param ratio ({ratio:.1f}×). Model may be too small for this data.')
    else:
        print('  ✓ Data:param ratio looks reasonable')

    if not train_ok:
        print('\n❌ Dataset validation found issues — check warnings above')
        return False

    print('\n✓ Dataset validation passed')

    # Test data collator
    print('\n[5/6] Testing data collator...')
    batch = [train_dataset[i] for i in range(2)]
    collated = default_data_collator(batch)
    print('✓ Data collator (default) successful')

    # Test forward pass
    print('\n[6/6] Testing forward pass...')

    with torch.no_grad():
        outputs = model(**collated)

    print('✓ Forward pass successful')
    print(f'  Output shape: {outputs.logits.shape}')
    print(f'  Loss: {outputs.loss.item():.4f}')

    # Summary
    print('\n' + '=' * 80)
    print('ALL TESTS PASSED ✓')
    print('=' * 80)
    print('\nYou can start training with:')
    print('  python training/base_train.py\n')

    return True


if __name__ == '__main__':
    try:
        success = test_setup()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'\n❌ ERROR: {e}')
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
Quick test script to validate the training setup without actually training.
This checks that all components load correctly.
"""

import sys
import traceback
from pathlib import Path

import torch
from base_train import load_binary_files, load_config
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


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

    # 2. Test tokenizer
    print('\n[2/6] Testing tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(cfg['data']['tokenizer'], use_fast=True)
    # Ensure tokenizer has a pad token (required for DataCollator)
    tokenizer.pad_token = tokenizer.eos_token
    print(f'✓ Tokenizer loaded: {cfg["data"]["tokenizer"]}')
    print(f'  Vocab size: {len(tokenizer)}')

    # Quick test encoding
    test_text = 'Hello, world!'
    encoded = tokenizer(test_text)
    print(f'  Test encoding: {encoded}')
    decoded = tokenizer.decode(encoded['input_ids'], clean_up_tokenization_spaces=False)
    print(f'  Test decoding: {decoded}')

    # 3. Test model creation
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

    # 4. Test data loading
    print('\n[4/6] Testing data loading...')
    train_files = cfg['data']['train_files']

    # Check if data files exist
    if isinstance(train_files, str):
        train_exists = Path(train_files).exists()
    else:
        train_exists = any(Path(f).exists() for f in train_files)

    if not train_exists:
        print(f'⚠️  Training data not found: {train_files}')
        print('   Run: python training/create_dummy_data.py')
        return False

    train_dataset = load_binary_files(train_files, cfg['data']['max_seq_length'])
    print('✓ Training data loaded')
    print(f'  Sequences: {len(train_dataset):,}')

    # 5. Test data collator
    print('\n[5/6] Testing data collator...')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    print('✓ Data collator creation successful')

    # 6. Test forward pass
    print('\n[6/6] Testing forward pass...')
    batch = [train_dataset[i] for i in range(2)]
    collated = data_collator(batch)

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

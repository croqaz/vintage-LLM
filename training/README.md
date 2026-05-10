# Pre-training

This directory contains everything needed to pre-train a model from scratch using HuggingFace Transformers.

## Quick Start

1. **Prepare your data** (tokenize to binary format):
   ```sh
   # Your data should be pre-tokenized as:
   # - training/train.bin (uint16 numpy array)
   # - training/valid.bin (uint16 numpy array)
   ```

2. **Configure training** (edit `training/config.toml`):
   ```sh
   # Adjust model size, batch size, learning rate, etc.
   vim training/config.toml
   ```

3. **Start training**:
   ```sh
   python training/base_train.py
   ```

4. **Resume from checkpoint** (automatic):
   ```sh
   # Just run the script again - it auto-detects checkpoints
   python training/base_train.py
   ```

## File Structure

```
training/
├── base_train.py          # Main training script
├── config.toml            # Training configuration
├── train.bin              # Pre-tokenized training data (you provide)
├── valid.bin              # Pre-tokenized validation data (you provide)
├── checkpoints/           # Auto-created during training
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── final/            # Final trained model
```

## Configuration (`config.toml`)

### Model Architecture

Example of Pythia-31M params:

```toml
[model]
vocab_size = 32768         # Must match tokenizer
hidden_size = 256          # Embedding dimension
num_hidden_layers = 6      # Number of transformer layers
num_attention_heads = 8    # Must divide hidden_size evenly
intermediate_size = 1024   # FFN size (typically 4x hidden_size)
max_position_embeddings = 2048
```

**Scaling the model:**
- 14M params: `hidden_size=128, layers=6, heads=4, ffn=512`
- 31M params: `hidden_size=256, layers=6, heads=8, ffn=1024`
- 70M params: `hidden_size=512, layers=6, heads=8, ffn=2048`
- 160M params: `hidden_size=768, layers=12, heads=12, ffn=3072`
- 410M params: `hidden_size=1024, layers=24, heads=16, ffn=4096`

### Data Files

```toml
[data]
# Single file
train_files = ["training/train.bin"]

# Multiple files
train_files = ["training/train_0.bin", "training/train_1.bin"]

# Glob pattern
train_files = ["training/train_*.bin"]

# Tokenizer must match the one used to create .bin files
tokenizer = "EleutherAI/pythia-31m"
```

### Training Hyperparameters

```toml
[training]
# Effective batch size = per_device * gradient_accumulation * num_gpus
per_device_train_batch_size = 8
gradient_accumulation_steps = 4

learning_rate = 6e-4       # Start with 6e-4 for small models
weight_decay = 0.1
warmup_ratio = 0.05        # 5% warmup

# Precision (choose based on GPU)
bf16 = true                # Ampere+ (RTX 3090, A100)
fp16 = false               # Older GPUs (V100, T4)
```

### Checkpointing & Evaluation

```toml
[training]
save_steps = 500           # Save checkpoint every 500 steps
eval_steps = 500           # Evaluate every 500 steps
save_total_limit = 3       # Keep only 3 most recent checkpoints
```

## Data Format

### Binary Files (.bin)

The training script expects pre-tokenized data in numpy binary format:

```py
import numpy as np
from transformers import AutoTokenizer

# Tokenize your text
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-31m")
text = "Your training data here..."
tokens = tokenizer.encode(text)

# Save as uint16 binary
tokens_array = np.array(tokens, dtype=np.uint16)
tokens_array.tofile("training/train.bin")
```

**Format requirements:**
- Data type: `uint16` (supports vocab sizes up to 65,536)
- Shape: Flat 1D array of token IDs
- The script will chunk this into sequences of `max_seq_length`

## Training Output

### Console Output

```
[TRAIN] Epoch 1.00/3 | Step 10/3000 (0.3%) | Loss 10.5234 | LR 1.20e-04
[TRAIN] Epoch 1.02/3 | Step 20/3000 (0.7%) | Loss 9.8765 | LR 2.40e-04

================================================================================
[VALIDATION] Step 500
================================================================================
  Loss:           8.2345
  Perplexity:     3756.23
  Runtime:        12.34s
  Samples/sec:    82.96
  Steps/sec:      5.19
================================================================================
```

### Checkpoints

Each checkpoint contains:
- `model.safetensors` - Model weights
- `config.json` - Model configuration
- `optimizer.pt` - Optimizer state
- `scheduler.pt` - Learning rate scheduler state
- `trainer_state.json` - Training state (step, epoch, etc.)

### Final Model

After training, the final model is saved to `training/checkpoints/final/`:
- `model.safetensors` - Model weights
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer
- `tokenizer_config.json` - Tokenizer config
- `training_config.toml` - Copy of training config

**Load the trained model:**

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("training/checkpoints/final")
tokenizer = AutoTokenizer.from_pretrained("training/checkpoints/final")

# Generate text
inputs = tokenizer("Hello, I am a", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## Distributed Training

The script automatically supports distributed training via HuggingFace Accelerate.

### Single GPU
```sh
python training/base_train.py
```

### Multi-GPU (single machine)
```sh
accelerate launch training/base_train.py
```

### Multi-node (configure with accelerate config)
```sh
accelerate config  # Run once to configure
accelerate launch training/base_train.py
```

## Next Steps

After training completes:

1. **Convert to GGUF** (for llama.cpp):
   ```sh
   python llama.cpp/convert_hf_to_gguf.py training/checkpoints/final
   ```

2. **Quantize** (for deployment):
   ```sh
   python llama.cpp/quantize training/checkpoints/final/model.gguf Q4_K_M
   ```

3. **Inference**:
   ```sh
   llama.cpp/llama-cli -m model.gguf -p "Your prompt here"
   ```

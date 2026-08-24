# Tiny LLM Training

Everything needed to pre-train, fine-tune, and evaluate a small causal language model with HuggingFace Transformers. The typical pipeline is:

```
split_dataset.py → tokenize_dataset.py → doctor.py → base_train.py
                → fine_tune.py → generate.py / vibe_check.py → evaluate.py / evaluate2.py / evaluate3.py
```

## Data preparation

- **`split_dataset.py`** — Splits JSONL (`{"text": ...}`) or plain-text files into `<name>-train` / `<name>-valid` sets, appending the tokenizer's EOS to each document. Tiny files (<1000 chars) get only a `-train` output.
- **`tokenize_dataset.py`** — Tokenizes `.txt` / `.jsonl` / `.parquet` files into sharded uint16 binary files (`train_0000.bin`, ...) that `base_train.py` consumes. Expects input already EOS-terminated by `split_dataset.py`. Shards are capped at 1 GiB.

## Training

- **`base_train.py`** — Pre-trains a model from scratch on the `.bin` files using the HF Trainer, driven by `config.toml` (architecture, data files, hyperparameters, checkpointing). Memmaps data, auto-resumes from the last checkpoint, and supports multi-GPU via `accelerate launch`.
- **`fine_tune.py`** — Supervised fine-tuning of a `base_train.py` checkpoint using TRL's SFTTrainer, driven by `fine_tune_config.toml`. Supports full-parameter or LoRA training, and JSONL data in conversational (`messages`) or pre-formatted (`text`) form.
- **`doctor.py`** — Preflight check before training: verifies that model/tokenizer/config load, and runs quality diagnostics on the `.bin` data (token IDs in vocab range, EOS presence, etc.) without training anything.

## Inference

- **`generate.py`** — Generates a completion for a single prompt from a checkpoint (latest in `./checkpoints` by default), with the usual sampling knobs (temperature, top-p/k, repetition penalty) and optional `--chat` templating.
- **`vibe_check.py`** — Runs a fixed set of built-in prompts through a checkpoint for a quick qualitative smell test. Same checkpoint resolution and sampling options as `generate.py`.

## Evaluation

- **`evaluate.py`** — Checkpoint inspection: answers "*what* is this checkpoint, and is it still vintage?". Reports architecture/params/size, training lineage (base vs SFT, source model, LR, context surgery), period-fidelity probes, and text-degeneracy hygiene. Not a quality metric — use `evaluate2.py` for that.
- **`evaluate2.py`** — Answers "*is it baked?*": scores checkpoints in bits-per-byte on 200 held-out period documents (`eval_data/`), places the result on a measured reference ladder from untrained noise to best-in-class, and produces a plain-language verdict. Point it at one checkpoint or a folder of them for a training curve + best pick.
- **`evaluate3.py`** — Answers "*which experiment is best?*": recursively compares models with deterministic byte-normalised held-out loss, paired document-bootstrap confidence intervals, practical-equivalence ties, and wall-clock budget groups. It also runs the 28 offline `sample_autocomplete.sh` prompts, saves all prose for human review, reports only mechanical degeneration warnings, and caches raw per-document results for resumable runs. Example: `python evaluate3.py autoresearch autoresearch2 MODELS --out eval3_results/all.json`.

## Configs

- **`config.toml`** — Pre-training config for `base_train.py`: model architecture, data files, tokenizer, hyperparameters, checkpoint/eval cadence.
- **`fine_tune_config.toml`** — Fine-tuning config for `fine_tune.py`: base checkpoint, method (`full`/`lora`), datasets, hyperparameters.

## Quick start

```sh
python split_dataset.py data/*.jsonl
python tokenize_dataset.py data/*-train.jsonl --output train.bin
python tokenize_dataset.py data/*-valid.jsonl --output valid.bin
python doctor.py                    # sanity-check the setup
python base_train.py                # or: accelerate launch base_train.py
python vibe_check.py                # eyeball the output
python evaluate2.py                 # measure how well it trained
```

Most inference/eval scripts default to the latest checkpoint in `./checkpoints`; pass `--checkpoint PATH` or `--checkpoints-dir DIR` to override.

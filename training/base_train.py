#!/usr/bin/env python3
"""
LLM Pre-training Script.
Trains a causal language model from scratch using HuggingFace Transformers Trainer.
"""

import argparse
import glob
import math
import os
import shutil
import signal
import sys
import time
import tomllib
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator
from huggingface_hub import sync_bucket
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


class ShardedTokenArray:
    """
    Read-only virtual concatenation of per-file token shards.
    Presents the shards as a single array (len + contiguous slicing) while
    reads keep going through the OS page cache — unlike np.concatenate,
    which would copy every shard into anonymous RAM (and OOM on large corpora).

    Holds file paths rather than open memmaps: pickling a np.memmap
    serializes the WHOLE underlying file, and since Python 3.14 the default
    multiprocessing start method on Linux is 'forkserver', which pickles the
    dataset to every DataLoader worker. Each process instead re-opens its own
    memmaps lazily on first read.
    """

    def __init__(self, files: List[str], dtype=np.uint16):
        self.files = [str(f) for f in files]
        self.dtype = np.dtype(dtype)
        lengths = [os.path.getsize(f) // self.dtype.itemsize for f in self.files]
        self.offsets = np.cumsum([0] + lengths)
        self._memmaps = None  # opened lazily, once per process

    @property
    def memmaps(self) -> List[np.ndarray]:
        if self._memmaps is None:
            self._memmaps = [np.memmap(f, dtype=self.dtype, mode='r') for f in self.files]
        return self._memmaps

    def __getstate__(self) -> Dict:
        return {**self.__dict__, '_memmaps': None}

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, key: slice) -> np.ndarray:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError('ShardedTokenArray only supports contiguous slices')
        start = key.start or 0
        stop = len(self) if key.stop is None else min(key.stop, len(self))

        shard = int(np.searchsorted(self.offsets, start, side='right')) - 1
        shard_start = start - int(self.offsets[shard])
        # Fast path: the slice lives inside one shard (zero-copy memmap view)
        if stop <= self.offsets[shard + 1]:
            return self.memmaps[shard][shard_start : stop - int(self.offsets[shard])]

        # Slow path (rare): the slice spans a shard boundary
        parts = []
        pos = start
        while pos < stop:
            take = min(stop, int(self.offsets[shard + 1])) - pos
            base = pos - int(self.offsets[shard])
            parts.append(self.memmaps[shard][base : base + take])
            pos += take
            shard += 1
        return np.concatenate(parts)


class BinaryTokenDataset(Dataset):
    """
    Dataset for pre-tokenized binary data.
    Each sample is a fixed-length sequence of token IDs.
    Uses memmap directly to avoid loading entire files into RAM.
    Supports a per-epoch random offset so chunk boundaries vary across epochs.
    """

    def __init__(self, data: np.ndarray, seq_length: int, base_seed: int = 0, doc_masking: bool = False, eos_id: Optional[int] = None):
        """
        Args:
            data: numpy memmap (or array) of tokens (uint16)
            seq_length: sequence length for each sample
            doc_masking: emit `position_ids` that restart at 0 after every EOS.
                Transformers reads them, detects the document boundaries via
                `find_packed_sequence_indices`, and ANDs an intra-document mask
                into the causal mask -- so a token never attends into the
                previous document. RoPE positions restart too, since the same
                `position_ids` feed the rotary embedding. Supported on sdpa,
                flex_attention and flash_attention_2 (FA2 takes the varlen
                path instead of a mask tensor). Costs <0.5% step time.
            eos_id: token id that terminates a document. Required when
                doc_masking is on.
        """
        self.data = data
        self.seq_length = seq_length
        self.base_seed = base_seed
        self.doc_masking = doc_masking
        self.eos_id = eos_id
        if doc_masking and eos_id is None:
            raise ValueError('doc_masking requires eos_id')
        self.offset = 0  # random offset applied per epoch

        # Compute a stable sequence count that won't shrink when any per-epoch
        # offset in [0, seq_length) is applied.  The worst-case offset is
        # seq_length-1, so we reserve that many tokens from the tail.
        # This loses at most one sequence vs. the naive count - negligible at scale.
        self.num_sequences = max(0, (len(self.data) - (seq_length - 1)) // seq_length)

        if self.num_sequences == 0:
            raise ValueError(f'Data too short: {len(self.data)} tokens < {seq_length} seq_length')

        # Identity mapping by default (no shuffle until set_epoch is called)
        self._index_map = np.arange(self.num_sequences)

        print(f'  → Created dataset: {self.num_sequences:,} sequences of length {seq_length}')

    def set_epoch(self, epoch: int):
        """Set a per-epoch random offset and shuffle index order.
        num_sequences is intentionally NOT updated here - it was fixed at init
        to a conservative value that is valid for any possible offset, so the
        Trainer's max_steps (computed once from len(dataset)) stays accurate
        across all epochs.
        """
        rng = np.random.RandomState(seed=epoch)
        self.offset = rng.randint(0, self.seq_length)
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

        sample = {
            'input_ids': tokens,
            'labels': tokens,
        }
        if self.doc_masking:
            sample['position_ids'] = self._position_ids(tokens)
        return sample

    def _position_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        """Positions counted from the start of each document, not the window.

        EOS terminates the document it belongs to, so the token AFTER an EOS is
        position 0 of the next one.

        Windows are cut at arbitrary offsets, so the leading fragment of a
        window usually starts mid-document. It is given positions from 0 as if
        it were a document start -- the true offset would require scanning back
        past the window edge. That mislabels the RoPE positions of ~1 fragment
        per window; see autoresearch/attention/_probe/README.md.
        """
        pos = torch.arange(self.seq_length, dtype=torch.long)
        starts = torch.nonzero(tokens == self.eos_id, as_tuple=False).flatten() + 1
        starts = starts[starts < self.seq_length]
        if starts.numel() == 0:
            return pos
        # last_start[i] = index of the most recent document start at or before i
        last_start = torch.zeros(self.seq_length, dtype=torch.long)
        last_start[starts] = starts
        last_start = torch.cummax(last_start, 0).values
        return pos - last_start


def load_binary_files(
    file_pattern: Union[str, List[str]], seq_length: int, base_seed: int = 0, doc_masking: bool = False, eos_id: Optional[int] = None
) -> BinaryTokenDataset:
    """
    Load binary files from a list of paths or glob pattern.

    Args:
        file_pattern: str (glob pattern) or list of str (file paths)
        seq_length: sequence length for each sample
        base_seed: base seed for random number generator

    Returns:
        BinaryTokenDataset instance.
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

    for f in files:
        if not Path(f).exists():
            raise FileNotFoundError(f'File not found: {f}')

    # Virtual concatenation over lazily-opened memmaps: nothing is copied
    # into RAM here, and DataLoader workers only receive the file paths.
    data = ShardedTokenArray(files)
    for f, tokens in zip(files, np.diff(data.offsets)):
        print(f'  → Mapped {tokens:,} tokens from {Path(f).name}')
    print(f'Total tokens: {len(data):,}')

    return BinaryTokenDataset(data, seq_length, base_seed, doc_masking=doc_masking, eos_id=eos_id)


# ============================================================================
# Custom Callbacks
# ============================================================================


class DetailedLoggingCallback(TrainerCallback):
    """Log detailed training metrics per batch."""

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None
        self.current_step = 0
        self.instability_detected = False
        # Rolling window for the finite-blow-up detector (see on_log).
        self.loss_history = []
        self.best_loss_median = None
        self.sustained_strikes = 0

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

            # Only process training logs (skip eval-only logs)
            if loss is None or lr is None:
                return

            progress = (step / total_steps * 100) if total_steps > 0 else 0
            grad_norm = logs.get('grad_norm')
            grad_str = f'grad_norm={grad_norm:.4f}' if grad_norm is not None else ''

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

            # --- NaN / Inf detection ---
            loss_bad = math.isnan(loss) or math.isinf(loss)
            grad_bad = grad_norm is not None and (math.isnan(grad_norm) or math.isinf(grad_norm))
            loss_zero = loss == 0.0 and step > 1  # loss=0 after first step is suspicious

            # --- finite blow-up / sustained-degradation detection ---
            # The NaN/Inf tests above cannot see a run destroying itself with
            # large but FINITE numbers. long/kv1_1epoch went from loss 3.21 to a
            # SUSTAINED 6.8 and then to 3232, with grad_norm reaching 2.5e7, over
            # ~800 steps -- and this callback let every one of them through
            # because nothing was ever NaN. 26 GPU-hours were then spent on an
            # already-destroyed model. Two rules close that gap:
            #
            #   blow-up   : one logged loss >5x the rolling median (and >10)
            #   sustained : the rolling median itself rises >1.75x above the best
            #               median ever seen, twice in a row
            #
            # Both are deliberately loose enough to ignore recoverable spikes --
            # the same run survived an isolated loss of 11.96 and carried on, and
            # that would not have tripped either rule.
            self.loss_history.append(loss)
            if len(self.loss_history) > 20:
                self.loss_history.pop(0)
            blowup = sustained = False
            if len(self.loss_history) == 20 and not (loss_bad or loss_zero):
                med = sorted(self.loss_history)[10]
                warm = getattr(args, 'warmup_steps', 0) or 0
                if step > warm + 500:
                    if self.best_loss_median is None or med < self.best_loss_median:
                        self.best_loss_median = med
                    blowup = loss > max(10.0, 5.0 * med)
                    if med > 1.75 * self.best_loss_median:
                        self.sustained_strikes += 1
                        sustained = self.sustained_strikes >= 2
                    else:
                        self.sustained_strikes = 0
                    if blowup or sustained:
                        kind = 'BLOW-UP' if blowup else 'SUSTAINED DEGRADATION'
                        print(f'\n[INSTABILITY] {kind}: loss {loss:.4f}, rolling median {med:.4f}, best median {self.best_loss_median:.4f}')

            if loss_bad or grad_bad or loss_zero or blowup or sustained:
                print('\n' + '!' * 80)
                print('TRAINING HALTED - NUMERICAL INSTABILITY DETECTED !!')
                print('!' * 80)
                print(f'  Step:      {step}')
                print(f'  Loss:      {loss}  {"(NaN/Inf!)" if loss_bad else "(ZERO!)" if loss_zero else ""}')
                print(f'  Grad norm: {grad_norm}  {"(NaN/Inf!)" if grad_bad else ""}')
                print(f'  Full logs: {logs}')
                print()
                print('Likely causes:')
                print('  1. Learning rate too high - try reducing by 2-5x')
                print('  2. bf16 overflow - try fp32 or check for extreme values in data')
                print('  3. Bad data batch - check training data for corrupted sequences')
                print('  4. Gradient explosion - lower max_grad_norm or learning_rate')
                print('!' * 80 + '\n')
                self.instability_detected = True
                control.should_training_stop = True
                # A time-based save scheduled for this same step would capture
                # the diverged weights - and a later resume would silently
                # continue from them. Cancel it; the previous checkpoint stays.
                control.should_save = False


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
            print(f'  Epoch:          {state.epoch:.2f}')
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


class TimeIntervalCallback(TrainerCallback):
    """
    Wall-clock (minutes) based save/eval scheduling + clean time-limited stop.

    This is the officially-supported transformers extension point (see the
    TrainerCallback docs): instead of modifying or monkey-patching the
    transformers package (whose built-in `IntervalStrategy` only understands
    no/steps/epoch), we hook `on_step_end` and set `control.should_save` /
    `control.should_evaluate` / `control.should_training_stop` based on the
    elapsed wall-clock time in minutes. This mirrors exactly how the built-in
    `DefaultFlowCallback` schedules by step/epoch — just by time.
    """

    def __init__(self, save_minutes=None, eval_minutes=None, max_train_minutes=None):
        self.save_minutes = save_minutes
        self.eval_minutes = eval_minutes
        self.max_train_minutes = max_train_minutes
        self._train_start = None
        self._last_save = None
        self._last_eval = None

    def on_train_begin(self, args, state, control, **kwargs):
        now = time.time()
        self._train_start = now
        self._last_save = now
        self._last_eval = now

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        total_min = (now - self._train_start) / 60.0

        # Cleanly stop after max_train_minutes, guaranteeing a final checkpoint.
        if self.max_train_minutes is not None and total_min >= self.max_train_minutes:
            control.should_training_stop = True
            control.should_save = True
            if state.is_world_process_zero:
                print(
                    f'[TIME] Reached max_train_minutes={self.max_train_minutes} after {total_min:.1f} min. '
                    f'Stopping training cleanly and saving final checkpoint at step {state.global_step}.'
                )
            return control

        # Wall-clock save interval (minutes).
        if self.save_minutes and (now - self._last_save) / 60.0 >= self.save_minutes:
            control.should_save = True
            self._last_save = now

        # Wall-clock eval interval (minutes).
        if self.eval_minutes and (now - self._last_eval) / 60.0 >= self.eval_minutes:
            control.should_evaluate = True
            self._last_eval = now

        return control


class GracefulStopCallback(TrainerCallback):
    """
    Make Ctrl+C (and SIGTERM) a first-class way to pause training.

    First signal: finish the current optimizer step, save a full resume
    checkpoint (weights + optimizer + LR scheduler + RNG), then stop through
    the normal shutdown path (final model save included). Running the script
    again resumes from that exact step, losing nothing.
    Second signal: abort immediately (KeyboardInterrupt).
    """

    def __init__(self):
        self.stop_requested = False

    def install(self):
        def _handler(signum, frame):
            if self.stop_requested:
                raise KeyboardInterrupt
            self.stop_requested = True
            print(
                f'\n[STOP] {signal.Signals(signum).name} received: finishing the current step, '
                'saving a resume checkpoint, then stopping cleanly. Signal again to abort immediately.'
            )

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_requested:
            if state.is_world_process_zero:
                print(f'[STOP] Saving checkpoint at step {state.global_step} and stopping.')
            control.should_save = True
            control.should_training_stop = True
        return control


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


class S3UploadCallback(TrainerCallback):
    """Upload checkpoints to HuggingFace S3 bucket at configurable intervals."""

    def __init__(self, bucket: str, upload_steps: int, model_id: str):
        # Normalize bucket: strip hf://buckets/ prefix if provided
        self.bucket = bucket.removeprefix('hf://buckets/').strip('/')
        self.upload_steps = upload_steps
        self.model_id = model_id

    def _upload(self, local_path: str, remote_suffix: str):
        """Upload a local directory to the S3 bucket. Never raises."""
        dest = f'hf://buckets/{self.bucket}/{remote_suffix}'
        try:
            print(f'[S3] Uploading {local_path} → {dest}')
            sync_bucket(local_path, dest)
            print(f'[S3] ✓ Upload complete: {dest}')
        except Exception as e:
            print(f'[S3] ✗ Upload failed: {e}')

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        if state.global_step % self.upload_steps != 0:
            return

        checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
        self._upload(checkpoint_path, f'{self.model_id}-step-{state.global_step}')


# ============================================================================
# Configuration Loading
# ============================================================================


def _format_optim_args(value: Union[str, Dict, None]) -> Optional[str]:
    """Render `optim_args` as the "k=v,k=v" string transformers expects.

    A TOML table is the friendly form; a plain string is passed through so an
    already-formatted value still works. Ints are kept integral because
    transformers casts t_alpha / t_beta3 with int(), which rejects "4000.0".
    """
    if value is None or value == '':
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise ValueError(f'optim_args must be a table or a string, got {type(value).__name__}')

    def fmt(v):
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    return ','.join(f'{k}={fmt(v)}' for k, v in value.items())


def make_model_id(model_type: str, num_params: int) -> str:
    """Create a model slug like 'llama-340m' or 'llama-1.1b'."""
    if num_params >= 1e9:
        size = f'{num_params / 1e9:.1f}'.rstrip('0').rstrip('.') + 'b'
    else:
        size = f'{num_params // 1_000_000}m'
    return f'{model_type}-{size}'


def _resolve_path(value: Union[str, List[str]], base: Path) -> Union[str, List[str]]:
    """Resolve a (possibly relative) config path against a base directory."""
    if isinstance(value, list):
        return [_resolve_path(v, base) for v in value]
    path = Path(value).expanduser()
    return str(path) if path.is_absolute() else os.path.normpath(base / path)


# All config keys that hold filesystem paths. They are resolved once, at config
# load time, relative to the config file's directory - so training behaves the
# same no matter which directory the script is launched from.
# 'tokenizer' and 'from_pretrained' may instead hold HuggingFace hub ids;
# values that don't point at an existing local path are left untouched.
_PATH_KEYS = (
    # base_train.py config
    ('data', 'train_files', False),
    ('data', 'valid_files', False),
    ('data', 'tokenizer', True),
    ('training', 'from_pretrained', True),
    ('training', 'output_dir', False),
    ('training', 'final_model_dir', False),
    # fine_tune.py config (shares this loader)
    ('model', 'base_model', True),
    ('data', 'dataset_path', False),
)


def load_config(config_path: str) -> Dict:
    """Load configuration from TOML file.

    If a 'model_config' key is present, the [model] section is loaded
    from that external file instead of inline config.

    All relative paths (data files, tokenizer, output dirs, model_config)
    are resolved against the config file's directory, per _PATH_KEYS.
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    base = Path(config_path).resolve().parent

    if model_config_path := config.pop('model_config', None):
        resolved = Path(_resolve_path(model_config_path, base))
        if not resolved.exists():
            raise FileNotFoundError(f'Model config not found: {resolved}')
        with open(resolved, 'rb') as f:
            model_cfg = tomllib.load(f)
        # An inline [model] section overrides individual keys from the external
        # file, so experiments can share a model.toml and vary a single knob
        # (e.g. attn_implementation) without duplicating the architecture.
        model_cfg.update(config.get('model', {}))
        config['model'] = model_cfg

    for section, key, maybe_hub_id in _PATH_KEYS:
        value = config.get(section, {}).get(key)
        if not value:
            continue
        if maybe_hub_id and not (value.startswith(('.', '~', '/')) or (base / value).exists()):
            continue  # HuggingFace hub id, not a local path
        config[section][key] = _resolve_path(value, base)

    return config


def validate_config(cfg: Dict, accelerator: Accelerator):
    """Validate configuration and check for common issues."""
    accelerator.print('\n' + '=' * 80)
    accelerator.print('CONFIGURATION VALIDATION')
    accelerator.print('=' * 80)

    # Check model configuration
    model_cfg = cfg['model']
    required_model_keys = ['model_type', 'vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size']
    for key in required_model_keys:
        if key not in model_cfg:
            raise ValueError(f'Missing required model config key: {key}')

    # Check data files exist
    train_files = cfg['data']['train_files']
    valid_files = cfg['data']['valid_files']

    # Check if any files exist
    if isinstance(train_files, str) and not glob.glob(train_files):
        raise FileNotFoundError(f'No training files found: {train_files}')
    if isinstance(valid_files, str) and not glob.glob(valid_files):
        raise FileNotFoundError(f'No validation files found: {valid_files}')

    # Check if CUDA is available for GPU training
    if not torch.cuda.is_available():
        accelerator.print('⚠️ WARNING: CUDA not available, training on CPU will be slow')

    # Check precision settings
    if cfg['training'].get('bf16', False) and not BF16_SUPPORTED:
        accelerator.print('⚠️ WARNING: bf16 requested but not supported on this GPU!')

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

    parser = argparse.ArgumentParser(description='LLM Pre-training Script')
    parser.add_argument(
        '--cfg',
        type=str,
        default=str(Path(__file__).with_name('config.toml')),
        help='Path to TOML config file (default: config.toml next to this script)',
    )
    args = parser.parse_args()

    config_path = args.cfg
    if not Path(config_path).exists():
        print(f'ERROR: Config file not found at {config_path}')
        print('Please create a config file or specify one with --cfg')
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
    # Validate configuration
    # ========================================================================

    validate_config(cfg, accelerator)

    # ========================================================================
    # Set random seeds for reproducibility
    # ========================================================================

    seed = cfg['training'].get('seed', 0)
    if seed:
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
    accelerator.print(f'  BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})')
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
    weights_dtype = model_kwargs.pop('dtype', 'float32')
    model_kwargs['use_cache'] = False
    model_config = AutoConfig.for_model(model_type=model_type, **model_kwargs)

    # Verify vocab size matches tokenizer
    if model_config.vocab_size != len(tokenizer):
        accelerator.print(f'⚠️ WARNING: Model vocab_size ({model_config.vocab_size}) != tokenizer vocab_size ({len(tokenizer)})')

    # Default keeps master weights in fp32: the Trainer's bf16 mode autocasts
    # the compute, while optimizer updates stay in full precision. Pure-16-bit
    # weights make AdamW updates quantize away (8-bit mantissa, eps=1e-8
    # unrepresentable) and risk NaNs. Overridable per-experiment via
    # `dtype = "bfloat16"` etc. in the config's [model] section.
    model_init_kwargs = {'dtype': getattr(torch, weights_dtype)}
    if weights_dtype != 'float32':
        accelerator.print(f'  ⚠️ Model weights dtype: {weights_dtype} — no fp32 master copy, optimizer state in {weights_dtype}')
    if attn_implementation:
        model_init_kwargs['attn_implementation'] = attn_implementation
        accelerator.print(f'  Attention implementation: {attn_implementation}')

    # Initialize model: load from existing weights or create fresh
    from_pretrained = cfg['training'].get('from_pretrained')
    if from_pretrained:
        accelerator.print(f'  Loading weights from: {from_pretrained}')
        model = AutoModelForCausalLM.from_pretrained(from_pretrained, config=model_config, **model_init_kwargs)
    else:
        model = AutoModelForCausalLM.from_config(model_config, **model_init_kwargs)

    # Optional fp8 training (torchao). Replaces the Linear layers with
    # Float8Linear: master weights stay fp32 and autocast still runs bf16, but
    # the matmul inputs are cast to float8_e4m3 and run on the fp8 tensor cores.
    # This is a THROUGHPUT change, not a precision-of-record change -- unlike
    # `dtype = "bfloat16"`, the fp32 master copy is kept.
    #
    # Verified on this box (RTX 4090, sm_89): torch._scaled_mm works and
    # torchao converts cleanly, even though torchao's fast paths target sm_90.
    # lm_head is excluded: with tie_word_embeddings the output matrix IS the
    # input embedding, and swapping it for a Float8Linear breaks the tie.
    # NB: `train_cfg` is not bound until later in this function -- read the
    # section straight off `cfg` here.
    if cfg['training'].get('fp8', False):
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        def _fp8_filter(mod, fqn: str) -> bool:
            if 'lm_head' in fqn:
                return False
            # fp8 matmuls need both dims divisible by 16
            return all(d % 16 == 0 for d in (mod.in_features, mod.out_features))

        # round_scales_to_power_of_2 is REQUIRED here, not a tuning knob.
        # Without it, fp8 + torch.compile produces grad_norm=NaN on the very
        # first step: the loss is finite (10.56, a normal init value) but 48
        # weight grads are NaN -- exactly q_proj, o_proj and down_proj in all
        # 16 layers, i.e. every Linear whose out_features equals hidden_size.
        # Eager fp8 is fine (40 clean steps, loss 7.09 -> 6.95), and both sdpa
        # and flash-attn2 fail identically, so it is the compiled backward, not
        # the attention backend. A power-of-2 scale makes scale/unscale exactly
        # invertible in floating point, so the reciprocal carries no rounding
        # error into the e5m2 grad_output cast. Measured on this box: 48 NaN
        # grads -> 0, grad_norm nan -> 2.47.
        # (torch._inductor.config.emulate_precision_casts only got 48 -> 16.)
        # See autoresearch/precision/_probe/README.md.
        fp8_cfg = Float8LinearConfig(round_scales_to_power_of_2=True)
        convert_to_float8_training(model, module_filter_fn=_fp8_filter, config=fp8_cfg)
        n_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        accelerator.print(f'  fp8 training: ON via torchao ({n_fp8} Float8 modules, lm_head excluded, pow2 scales)')

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

    model_id = make_model_id(model_config.model_type, num_params)
    accelerator.print(f'  Model ID: {model_id}')
    accelerator.print()

    # ========================================================================
    # Load datasets
    # ========================================================================

    accelerator.print('Loading datasets...')
    seq_length = cfg['data']['max_seq_length']

    # Intra-document masking. Documents are packed back-to-back into the token
    # stream and windows are cut at arbitrary offsets, so a window normally
    # straddles a document boundary and a token can attend into the previous,
    # unrelated document. Turning this on emits `position_ids` that restart at
    # each EOS; Transformers turns those into an intra-document attention mask
    # (and per-document RoPE positions) with no model changes.
    doc_masking = cfg['data'].get('doc_masking', False)
    eos_id = tokenizer.eos_token_id if doc_masking else None
    if doc_masking:
        if eos_id is None:
            raise ValueError('doc_masking is enabled but the tokenizer has no eos_token_id')
        accelerator.print(f'  Intra-document masking: ON (eos_token_id={eos_id})')

    accelerator.print('\n[TRAINING DATA]')
    train_dataset = load_binary_files(
        cfg['data']['train_files'],
        seq_length,
        base_seed=seed,
        doc_masking=doc_masking,
        eos_id=eos_id,
    )

    accelerator.print('\n[VALIDATION DATA]')
    eval_dataset = load_binary_files(
        cfg['data']['valid_files'],
        seq_length,
        base_seed=seed,
        doc_masking=doc_masking,
        eos_id=eos_id,
    )

    # Optionally evaluate on a fixed random subset: each evaluation is a full
    # pass over eval_dataset, which for large validation sets takes far longer
    # than a typical eval interval.
    max_eval_samples = cfg['training'].get('max_eval_samples', 0)
    if max_eval_samples and len(eval_dataset) > max_eval_samples:
        rng = np.random.RandomState(seed)
        subset_idx = rng.choice(len(eval_dataset), size=max_eval_samples, replace=False)
        eval_dataset = torch.utils.data.Subset(eval_dataset, subset_idx.tolist())
        accelerator.print(f'  → Evaluating on a random subset of {max_eval_samples:,} sequences')
    accelerator.print()

    # ========================================================================
    # Setup TrainingArguments
    # ========================================================================

    accelerator.print('Setting up training arguments...')

    output_dir = cfg['training']['output_dir']
    train_cfg = cfg['training']

    # ------------------------------------------------------------------------
    # Strategy handling: the transformers IntervalStrategy only understands
    # no/steps/epoch. We map the config's "minutes" strategies onto the official
    # TrainerCallback mechanism (TimeIntervalCallback) and disable the built-in
    # step-based scheduling so they don't double-trigger. max_train_minutes is a
    # clean, graceful time limit that works regardless of save/eval strategy.
    # ------------------------------------------------------------------------
    save_strategy = train_cfg.get('save_strategy', 'steps')
    eval_strategy = train_cfg.get('eval_strategy', 'steps')
    max_train_minutes = train_cfg.get('max_train_minutes')

    time_callback = None
    if save_strategy == 'minutes' or eval_strategy == 'minutes' or max_train_minutes:
        time_callback = TimeIntervalCallback(
            save_minutes=train_cfg['save_steps'] if save_strategy == 'minutes' else None,
            eval_minutes=train_cfg['eval_steps'] if eval_strategy == 'minutes' else None,
            max_train_minutes=max_train_minutes or None,
        )
        # Delegate minutes-based saving/eval to the callback; the built-in
        # DefaultFlowCallback only understands no/steps/epoch.
        if save_strategy == 'minutes':
            save_strategy = 'no'
        if eval_strategy == 'minutes':
            eval_strategy = 'no'

    # ------------------------------------------------------------------------
    # Optimizers not in transformers' OptimizerNames (e.g. Gefen) are passed to
    # the Trainer via optimizer_cls_and_kwargs; TrainingArguments.optim then
    # holds an unused placeholder. Trainer still builds its usual decay/no-decay
    # param groups, so weight_decay behaves the same as for built-in optimizers.
    # ------------------------------------------------------------------------
    optim_name = train_cfg.get('optim', 'adamw_torch_fused')
    optimizer_cls_and_kwargs = None
    if optim_name.startswith('gefen'):
        from gefen import Gefen  # pip install gefen-x (imports as `gefen`)

        optimizer_cls_and_kwargs = (
            Gefen,
            {
                'lr': train_cfg.get('learning_rate', 5e-5),
                'betas': (train_cfg.get('adam_beta1', 0.9), train_cfg.get('adam_beta2', 0.95)),
                'eps': train_cfg.get('adam_epsilon', 1e-8),
                'weight_decay': train_cfg.get('weight_decay', 0.1),
                # fused kernels are CUDA-only (no ROCm); False skips the JIT attempt
                'fused': train_cfg.get('gefen_fused', False),
            },
        )
        optim_name = 'adamw_torch'  # placeholder, overridden by optimizer_cls_and_kwargs
    elif optim_name == 'normuon':
        # NorMuon (Muon plus a per-row second-moment normalizer on the
        # orthogonalized update, rescaled to preserve Muon's update norm --
        # arXiv:2510.05491). Vendored in ./NorMuon as a bare source tree; only
        # the SingleDevice* classes are usable here, the others call into
        # torch.distributed and this box trains on one GPU.
        #
        # MERGE NOTE (2026-09-04): base_train.py carried TWO `normuon` branches
        # in the same if/elif chain, so the second was unreachable dead code.
        # The DEAD one is what produced every published arm in
        # autoresearch2/optimizer/norMuon_lr_* (its log line "N hidden tensors
        # on the Muon path" appears in those train.logs), so its NUMERICS are
        # authoritative and are kept verbatim below. The live one contributed
        # the engineering (requires_grad filter, accelerator.print, the
        # applied-decay invariant) but had a serious defect: it read the Muon
        # group's LR from `normuon_muon_lr` (default 0.02) and NOT from
        # `learning_rate`, so an LR sweep driven by `learning_rate` would have
        # swept only the auxiliary AdamW group.
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'NorMuon'))
        from normuon import SingleDeviceNorMuonWithAuxAdam

        # Route parameters with MuonQ's EXACT rule so that a NorMuon-vs-MuonQ
        # comparison differs only in the update rule, not in which tensors each
        # rule touches: >=2D and not an embedding/head -> Muon; the rest -> aux
        # AdamW. Newton-Schulz orthogonalisation is only meaningful for weight
        # MATRICES, so embeddings, norms and biases must go to Adam.
        # `tie_word_embeddings = true` means lm_head IS embed_tokens, so it is
        # seen once and lands in the Adam group -- which is what we want.
        _EXCLUDE = ('embeddings', 'embed_tokens', 'wte', 'lm_head', 'wpe')
        muon_p, aux_p = [], []
        for _n, _p in model.named_parameters():
            if not _p.requires_grad:
                continue
            is_hidden = _p.ndim >= 2 and not any(e in _n for e in _EXCLUDE)
            (muon_p if is_hidden else aux_p).append(_p)

        # ONE lr drives both groups, matching MuonQ, whose AdamW backup reads
        # group['lr'] rather than a separate adamw_lr. The aux betas/eps below
        # are MuonQ's defaults (0.95, 0.95)/1e-8, NOT NorMuon's own
        # (0.9, 0.95)/1e-10, for the same reason -- keep the comparison to the
        # update rule alone.
        _lr = train_cfg.get('learning_rate', 8e-3)
        _wd = train_cfg.get('weight_decay', 0.1)
        _aux_lr = train_cfg.get('normuon_aux_lr', _lr)

        # OPT-IN applied-decay invariant (from the old live branch). Decoupled
        # decay applies lr * wd per step, so sweeping the LR silently sweeps the
        # regularisation with it. Setting `normuon_lrwd` pins lr * wd to a
        # constant per group instead. ABSENT BY DEFAULT: the published arms used
        # a plain shared weight_decay, and this must stay reproducible.
        _lrwd = train_cfg.get('normuon_lrwd')
        if _lrwd is not None:
            _muon_wd, _aux_wd = _lrwd / _lr, _lrwd / _aux_lr
        else:
            _muon_wd = _aux_wd = _wd

        # SingleDeviceNorMuonWithAuxAdam asserts an EXACT key set per group, so
        # these dicts must carry precisely these keys and nothing else.
        param_groups = [
            dict(
                params=muon_p,
                use_muon=True,
                lr=_lr,
                weight_decay=_muon_wd,
                momentum=train_cfg.get('muon_momentum', 0.95),
                beta2=train_cfg.get('normuon_beta2', 0.95),
            ),
            dict(
                params=aux_p,
                use_muon=False,
                lr=_aux_lr,
                weight_decay=_aux_wd,
                betas=(
                    train_cfg.get('normuon_adamw_beta1', 0.95),
                    train_cfg.get('normuon_adamw_beta2', 0.95),
                ),
                eps=train_cfg.get('normuon_adamw_eps', 1e-8),
            ),
        ]

        # --- the one MuonQ feature NorMuon was missing -------------------
        # MuonQ exposes `nesterov` (default False) and `ns_steps` (default 5).
        # normuon_update() accepts both, but SingleDeviceNorMuonWithAuxAdam.step()
        # never forwards them, pinning them at nesterov=True / ns_steps=5 --
        # the "KNOWN RESIDUAL DIFFERENCE ... not removable without editing
        # vendored upstream" recorded in autoresearch2/optimizer/REPORT.md.
        # A subclass forwards them without touching the vendored file.
        # The defaults below reproduce upstream exactly, so unless a config
        # overrides one of these keys the vendored class is used UNCHANGED and
        # this run stays bit-comparable with the published arms.
        _nesterov = train_cfg.get('normuon_nesterov', True)
        _ns_steps = train_cfg.get('normuon_ns_steps', 5)
        _optim_cls = SingleDeviceNorMuonWithAuxAdam
        if (_nesterov, _ns_steps) != (True, 5):
            from normuon import adam_update, normuon_update

            class _NorMuonTunable(SingleDeviceNorMuonWithAuxAdam):
                """
                MIRRORS SingleDeviceNorMuonWithAuxAdam.step() (NorMuon/normuon.py)
                and differs ONLY by forwarding nesterov/ns_steps. Re-sync this if
                the vendored normuon.py is ever updated.
                """

                @torch.no_grad()
                def step(self, closure=None):
                    loss = None
                    if closure is not None:
                        with torch.enable_grad():
                            loss = closure()
                    for group in self.param_groups:
                        for p in group['params']:
                            had_grad = p.grad is not None
                            if not had_grad:
                                p.grad = torch.zeros_like(p)
                            state = self.state[p]
                            if group['use_muon']:
                                if len(state) == 0:
                                    state['momentum_buffer'] = torch.zeros_like(p)
                                    state['second_momentum_buffer'] = torch.zeros_like(p[..., 0:1])
                                update = normuon_update(
                                    p.grad,
                                    state['momentum_buffer'],
                                    state['second_momentum_buffer'],
                                    beta=group['momentum'],
                                    beta2=group['beta2'],
                                    ns_steps=_ns_steps,
                                    nesterov=_nesterov,
                                ).reshape(p.shape)
                            else:
                                if len(state) == 0:
                                    state['exp_avg'] = torch.zeros_like(p)
                                    state['exp_avg_sq'] = torch.zeros_like(p)
                                    state['step'] = 0
                                state['step'] += 1
                                update = adam_update(
                                    p.grad,
                                    state['exp_avg'],
                                    state['exp_avg_sq'],
                                    state['step'],
                                    group['betas'],
                                    group['eps'],
                                )
                            if group['weight_decay'] and had_grad:
                                p.mul_(1 - group['lr'] * group['weight_decay'])
                            p.add_(update, alpha=-group['lr'])
                    return loss

            _optim_cls = _NorMuonTunable

        accelerator.print(
            f'  NorMuon: {len(muon_p)} hidden tensors via Muon '
            f'(lr {_lr:g}, wd {_muon_wd:.4g}, momentum {param_groups[0]["momentum"]:g}, '
            f'beta2 {param_groups[0]["beta2"]:g}, nesterov {_nesterov}, ns_steps {_ns_steps}), '
            f'{len(aux_p)} tensors via aux AdamW '
            f'(lr {_aux_lr:g}, wd {_aux_wd:.4g}, betas {param_groups[1]["betas"]}, '
            f'eps {param_groups[1]["eps"]:g})'
        )
        optimizer_cls_and_kwargs = (_optim_cls, {'params': param_groups})
        optim_name = 'adamw_torch'  # placeholder, overridden by optimizer_cls_and_kwargs
    elif optim_name == 'muonq':
        # MuonQ (4-bit quantized Muon) lives in ./MuonQ as a bare source tree,
        # importable only as the package `src` rooted there.
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'MuonQ'))
        from src.optim.muonq import MuonQ

        optimizer_cls_and_kwargs = (
            MuonQ,
            {
                # Trainer pops 'params' and passes it positionally; MuonQ needs
                # (name, param) pairs to route 2D hidden weights to Muon and
                # embeddings/head/norms to its internal AdamW backup.
                'params': list(model.named_parameters()),
                'lr': train_cfg.get('learning_rate', 1e-3),
                'weight_decay': train_cfg.get('weight_decay', 0.1),
                'momentum': train_cfg.get('muon_momentum', 0.95),
                'nesterov': train_cfg.get('muon_nesterov', False),
                'ns_steps': train_cfg.get('muon_ns_steps', 5),
                'polar_method': train_cfg.get('muon_polar_method', 'Keller'),
                # 4-bit state quantization knobs (defaults = the repo's muonq recipe)
                'qbit': train_cfg.get('muonq_qbit', 4),
                'gran': train_cfg.get('muonq_gran', 'tensor'),
                'compand': train_cfg.get('muonq_compand', True),
                'norm': train_cfg.get('muonq_norm', True),
                'rank': train_cfg.get('muonq_rank', 16),
            },
        )
        optim_name = 'adamw_torch'  # placeholder, overridden by optimizer_cls_and_kwargs
    training_args = TrainingArguments(
        output_dir=output_dir,
        # Training duration
        num_train_epochs=train_cfg['num_train_epochs'],
        # -1 = train for full epochs; setting this also fixes the LR-scheduler
        # horizon (otherwise decay spans the full epoch and a 1h run never anneals)
        max_steps=train_cfg.get('max_steps', -1),
        # Batch sizes
        per_device_train_batch_size=train_cfg.get('per_device_train_batch_size', 8),
        per_device_eval_batch_size=train_cfg.get('per_device_eval_batch_size', 8),
        gradient_accumulation_steps=train_cfg.get('gradient_accumulation_steps', 1),
        # Optimizer with tuned defaults
        optim=optim_name,
        # Extra optimizer hyperparameters that have no dedicated TrainingArguments
        # field. transformers parses this as "k=v,k=v" and each optimizer factory
        # reads the keys it knows. Required for anything richer than Adam's two
        # betas - e.g. AdEMAMix needs beta3 / alpha / t_alpha / t_beta3, and
        # without them it silently runs on defaults (beta3=0.9999, alpha=5, no
        # warmup), which is wrong for short runs. Accepts a TOML table or a
        # ready-made string:
        #   optim_args = { beta3 = 0.999, alpha = 5.0, t_beta3 = 4000 }
        optim_args=_format_optim_args(train_cfg.get('optim_args')),
        learning_rate=train_cfg.get('learning_rate', 5e-5),
        weight_decay=train_cfg.get('weight_decay', 0.1),
        adam_beta1=train_cfg.get('adam_beta1', 0.9),
        adam_beta2=train_cfg.get('adam_beta2', 0.95),
        adam_epsilon=train_cfg.get('adam_epsilon', 1e-8),
        max_grad_norm=train_cfg.get('max_grad_norm', 1.0),
        # Learning rate scheduler
        lr_scheduler_type=train_cfg.get('lr_scheduler_type', 'cosine_with_min_lr'),
        lr_scheduler_kwargs=train_cfg.get('lr_scheduler_kwargs', {'min_lr_rate': 0.05}),
        # required by metric-driven schedulers (greedy / reduce_lr_on_plateau):
        # they step on this eval metric instead of per optimizer step
        metric_for_best_model=train_cfg.get('metric_for_best_model', None),
        warmup_steps=train_cfg.get('warmup_steps', 200),
        # Resume behaviour. HF's default replays the sampler by fetching and
        # discarding every batch already consumed. On this corpus that costs
        # ~1,350 samples/s, so resuming a 23k-step run burns ~73 min before the
        # first new step. Set true to start from a fresh sampler draw instead --
        # only sound when the seed is also changed, otherwise the run replays
        # the batches from step 0. See llama-75/anneal-1h-step23098/config.toml.
        ignore_data_skip=train_cfg.get('ignore_data_skip', False),
        # Precision
        bf16=train_cfg.get('bf16', BF16_SUPPORTED),
        fp16=train_cfg.get('fp16', False),
        # Performance
        torch_compile=train_cfg.get('torch_compile', True),
        # Inductor compile mode. Default (None) is a balanced compile. "max-autotune"
        # benchmarks kernel variants at compile time -- it can raise steady-state
        # throughput but the one-off compile cost grows a lot, which matters here
        # because the budget is wall-clock. calibrate_steps.py measures the first
        # step separately, so a refit absorbs it correctly.
        torch_compile_mode=train_cfg.get('torch_compile_mode', None),
        gradient_checkpointing=train_cfg.get('gradient_checkpointing', False),
        gradient_checkpointing_kwargs={'use_reentrant': False},
        neftune_noise_alpha=train_cfg.get('neftune_noise_alpha', 0.0),
        # Checkpointing
        save_strategy=save_strategy,
        save_steps=train_cfg.get('save_steps', 500),
        save_total_limit=train_cfg.get('save_total_limit', 3),
        # Evaluation
        eval_strategy=eval_strategy,
        eval_steps=train_cfg.get('eval_steps', 500),
        # Logging
        logging_strategy=train_cfg.get('logging_strategy', 'steps'),
        logging_steps=train_cfg.get('logging_steps', 10),
        logging_first_step=train_cfg.get('logging_first_step', True),
        report_to=train_cfg.get('report_to', 'none'),  # e.g. ["tensorboard"] (requires the package)
        # Performance
        dataloader_num_workers=train_cfg.get('dataloader_num_workers', 4),
        dataloader_prefetch_factor=train_cfg.get('dataloader_prefetch_factor', 2),
        dataloader_pin_memory=train_cfg.get('dataloader_pin_memory', True),
        # dataloader_persistent_workers intentionally disabled by default
        # Reproducibility
        seed=seed,
        # data_seed controls the training sampler's shuffle order. Override it
        # (e.g. data_seed = 43) when continuing pretraining via from_pretrained,
        # otherwise the run replays the exact same data order the base model
        # already saw, while seed keeps the eval subset comparable across runs.
        data_seed=train_cfg.get('data_seed', seed),
    )

    accelerator.print('✓ Training arguments configured')
    effective_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes

    # ------------------------------------------------------------------------
    # WSD schedule coverage guard. When num_stable_steps is set explicitly,
    # transformers' get_wsd_schedule IGNORES the Trainer's num_training_steps
    # and silently pins the LR at min_lr_ratio for every step past
    # warmup + stable + decay (documented in its docstring, never enforced).
    # An 8h run once trained 75% of its steps at LR=0 this way. Fail fast.
    # ------------------------------------------------------------------------
    if train_cfg.get('lr_scheduler_type') == 'warmup_stable_decay':
        skw = train_cfg.get('lr_scheduler_kwargs', {}) or {}
        wsd_warmup = training_args.warmup_steps
        wsd_stable = skw.get('num_stable_steps')
        wsd_decay = skw.get('num_decay_steps', 0)
        steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
        if training_args.max_steps > 0:
            horizon, horizon_src = training_args.max_steps, 'max_steps'
        else:
            horizon = math.ceil(steps_per_epoch * training_args.num_train_epochs)
            horizon_src = f'{training_args.num_train_epochs} epoch(s) x {steps_per_epoch:,} steps'
        if wsd_stable is not None:
            covered = wsd_warmup + wsd_stable + wsd_decay
            if covered != horizon:
                sug_w = round(horizon * 0.263)
                sug_d = round(horizon * 0.20)
                raise ValueError(
                    f'WSD schedule covers {covered} steps (warmup {wsd_warmup} + stable {wsd_stable} '
                    f'+ decay {wsd_decay}) but the run horizon is {horizon} steps ({horizon_src}). '
                    f'Steps beyond the schedule would train at LR = min_lr_ratio '
                    f'({skw.get("min_lr_ratio", 0)}); a shorter horizon would cut the decay phase. '
                    f'Fix: set max_steps = {covered}, or scale the schedule to the horizon, e.g. '
                    f'warmup_steps = {sug_w}, num_stable_steps = {horizon - sug_w - sug_d}, '
                    f'num_decay_steps = {sug_d}.'
                )
        elif wsd_warmup + wsd_decay > horizon:
            raise ValueError(
                f'WSD warmup ({wsd_warmup}) + decay ({wsd_decay}) exceed the run horizon '
                f'({horizon} steps, {horizon_src}); the auto-fitted stable phase would be negative.'
            )

    accelerator.print(f'  Effective batch size: {effective_batch}')
    accelerator.print(f'  Epochs: {training_args.num_train_epochs}')
    accelerator.print(f'  Learning rate: {training_args.learning_rate}')
    accelerator.print(f'  Warmup steps: {training_args.warmup_steps}')
    accel_save_s = train_cfg.get('save_strategy', 'steps')
    accel_eval_s = train_cfg.get('eval_strategy', 'steps')
    accel_save_v = time_callback.save_minutes if time_callback and accel_save_s == 'minutes' else training_args.save_steps
    accel_eval_v = time_callback.eval_minutes if time_callback and accel_eval_s == 'minutes' else training_args.eval_steps
    accelerator.print(f'  Save strategy: {accel_save_s} ({accel_save_v} {"min" if accel_save_s == "minutes" else "steps"})')
    accelerator.print(f'  Eval strategy: {accel_eval_s} ({accel_eval_v} {"min" if accel_eval_s == "minutes" else "steps"})')
    accelerator.print(f'  Max train minutes: {max_train_minutes or "unlimited"}')
    accelerator.print()

    # ========================================================================
    # Create Trainer
    # ========================================================================

    accelerator.print('Creating Trainer...')

    logging_callback = DetailedLoggingCallback()
    graceful_stop = GracefulStopCallback()
    graceful_stop.install()
    callbacks = [
        logging_callback,
        DetailedEvaluationCallback(),
        EpochOffsetCallback(train_dataset),
        graceful_stop,
    ]

    # Optional: wall-clock (minutes) based scheduling + time-limited stop
    if time_callback is not None:
        callbacks.append(time_callback)

    # Optional: S3 checkpoint upload
    s3_cfg = cfg.get('s3')
    s3_callback = None
    if s3_cfg and s3_cfg.get('bucket') and s3_cfg.get('upload_steps'):
        s3_callback = S3UploadCallback(
            bucket=s3_cfg['bucket'],
            upload_steps=s3_cfg['upload_steps'],
            model_id=model_id,
        )
        callbacks.append(s3_callback)
        accelerator.print(
            f'✓ S3 upload enabled: every {s3_cfg["upload_steps"]} steps → hf://buckets/{s3_callback.bucket}/{model_id}-step-*'
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=callbacks,
        optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
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
        # Refuse to resume from a checkpoint of a different architecture:
        # resuming would crash on weight shapes, and save_total_limit rotation
        # would start deleting the other experiment's checkpoints.
        ckpt_config = AutoConfig.from_pretrained(last_checkpoint)
        arch_keys = ('model_type', 'vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size')
        mismatched = [k for k in arch_keys if getattr(ckpt_config, k, None) != getattr(model_config, k, None)]
        if mismatched:
            raise ValueError(
                f'output_dir ({output_dir}) contains a checkpoint from a different model: {last_checkpoint} '
                f'(mismatched: {", ".join(mismatched)}). '
                f'Point output_dir at a fresh directory, or move the old checkpoints away.'
            )
        accelerator.print(f'Resuming from checkpoint: {last_checkpoint}')
    else:
        accelerator.print('No checkpoint found, training from scratch.')

    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)

    except KeyboardInterrupt:
        # Only reached on a *second* Ctrl+C (hard abort): no checkpoint was
        # written for the aborted step. The latest periodic/graceful checkpoint
        # in output_dir is still valid and will be auto-resumed next run.
        accelerator.print('\n' + '=' * 80)
        accelerator.print('TRAINING ABORTED (hard interrupt)')
        accelerator.print('=' * 80)
        accelerator.print(f'Progress since the last checkpoint in {output_dir} is lost.')
        accelerator.print('Run the script again to resume from that checkpoint.')
        accelerator.print('=' * 80 + '\n')
        sys.exit(130)

    except Exception as e:
        accelerator.print('\n' + '=' * 80)
        accelerator.print('TRAINING ERROR')
        accelerator.print('=' * 80)
        accelerator.print(f'Error: {e}')
        accelerator.print('=' * 80 + '\n')
        raise

    training_time = time.time() - start_time

    # Numerical instability halts must not masquerade as a successful run:
    # don't save the diverged weights as the final model, exit non-zero.
    if logging_callback.instability_detected:
        accelerator.print('\n' + '=' * 80)
        accelerator.print('TRAINING HALTED BY INSTABILITY DETECTOR')
        accelerator.print('=' * 80)
        accelerator.print(f'Training time: {training_time / 60:.2f} minutes')
        accelerator.print('The diverged weights were NOT saved as the final model.')
        accelerator.print(f'Last good checkpoint (if any) is in: {output_dir}')
        accelerator.print('=' * 80 + '\n')
        sys.exit(2)

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
        Path(final_model_dir, 'training_args.bin').unlink(missing_ok=True)

        # Also save the training config for reference
        shutil.copy(config_path, os.path.join(final_model_dir, 'training_config.toml'))

        accelerator.print(f'✓ Final model saved to {final_model_dir}')
        accelerator.print('  - Model weights: model.safetensors')
        accelerator.print('  - Model config: config.json')

        # Upload final model to S3 if configured
        if s3_callback is not None:
            s3_callback._upload(final_model_dir, f'{model_id}-final')

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

"""Shared helpers for the merged evaluator: device/dtype selection, checkpoint
and tokenizer discovery, model loading/freeing, training-lineage sniffing,
evaluation-data loading and small formatting utilities.

Everything here is infrastructure; metric MATH lives in metrics.py.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import nanochat_models as nanochat

LOG2E = math.log2(math.e)

# ---------------------------------------------------------------------------
# Where things live.
#
# This package must keep working if it is MOVED (e.g. training/eval -> project
# root) and if its data is moved INTO it (eval_data/ next to the code). So
# nothing is hard-wired to one parent: each location is searched in order and
# the first that exists wins.
#
# Order matters. Package-local comes FIRST so a self-contained copy of eval/
# that carries its own eval_data/ is preferred over an unrelated directory that
# happens to sit beside it.
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent


def _first_existing(name: str, default_parent: Path) -> Path:
    for parent in (PACKAGE_DIR, PROJECT_DIR, PROJECT_DIR.parent, Path.cwd()):
        candidate = parent / name
        if candidate.is_dir():
            return candidate
    return default_parent / name  # nothing found: fall back, callers report it


EVAL_DATA = _first_existing('eval_data', PROJECT_DIR)

# The default held-out prose set. Piston-n-Prose since 2026-09-10: every model
# we still train comes from the Piston corpus and the v5 tokenizer, and 152 of
# the 153 Sprocket held-out documents turned up inside the Piston training data,
# so scoring a current model on Sprocket is contaminated as well as off-corpus.
# Sprocket-n-Say stays on disk to re-score the retired Sprocket-era models.
DEFAULT_HELDOUT = EVAL_DATA / 'heldout-Piston-n-Prose.jsonl'

# Kept SEPARATE from DEFAULT_HELDOUT on purpose. BPB_LADDER in metrics.py was
# hand-anchored against Sprocket scores, so the composite is uncalibrated for
# any other set - including the new default. Tying the warning to the default
# would have silently switched off a note that is still true.
LADDER_CALIBRATION_HELDOUT = EVAL_DATA / 'heldout-Sprocket-n-Say.jsonl'

# Results are WRITTEN, so an existing directory is preferred but a missing one
# is created next to the package rather than in some unrelated parent.
DEFAULT_RESULTS_DIR = _first_existing('eval_results', PROJECT_DIR)


# ============================================================================
# Formatting / printing
# ============================================================================


def banner(title: str) -> None:
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def fmt(v, nd: int = 4) -> str:
    """Human cell: em-dash for None/NaN, fixed decimals for floats."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return '—'
    return f'{v:.{nd}f}' if isinstance(v, float) else str(v)


def human_tokens(n: float | None) -> str:
    if n is None:
        return 'unknown'
    for unit, div in (('T', 1e12), ('B', 1e9), ('M', 1e6)):
        if n >= div:
            return f'{n / div:.1f}{unit}'
    return f'{n:.0f}'


def json_safe(obj):
    """Represent unavailable non-finite numbers as standard JSON null."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def atomic_json(path: Path, obj) -> None:
    """Write JSON atomically so an interrupted run never leaves a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(json_safe(obj), indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def stable_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_for(item_id: str) -> str:
    """Stable, approximately even A/B diagnostic split independent of file order."""
    return 'A' if int(stable_hash(item_id.encode())[:8], 16) % 2 == 0 else 'B'


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


# ============================================================================
# Device / dtype
# ============================================================================


def select_device(name: str) -> torch.device:
    if name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(name)


def select_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != 'auto':
        return getattr(torch, name)
    return torch.bfloat16 if device.type == 'cuda' else torch.float32


# ============================================================================
# Checkpoint / tokenizer discovery (union of the three scripts' conventions)
# ============================================================================


def has_hf_weights(path: Path) -> bool:
    return any(path.glob('*.safetensors')) or any(path.glob('pytorch_model*.bin'))


def step_of(p: Path) -> float:
    """Sort key: numeric suffix if present ('checkpoint-42' -> 42), else last."""
    m = re.search(r'(\d+)$', p.name)
    return int(m.group(1)) if m else 10**12


@dataclass
class Targets:
    checkpoints: list[Path]
    skipped: list[dict[str, str]]


def resolve_targets(targets: list[Path], include_checkpoints: bool = False) -> Targets:
    """Resolve a checkpoint dir, a folder of checkpoints, or a recursive tree.

    Accepted targets:
      * a checkpoint directory (contains config.json)
      * a folder of checkpoint-* directories (sorted by step)
      * an experiment tree (evaluates `final/` exports, skips stray configs)
      * `checkpoint-N` subdirs are skipped unless include_checkpoints is set
    """
    found: dict[str, Path] = {}
    skipped: list[dict[str, str]] = []
    for target in targets:
        target = target.resolve()
        if not target.exists():
            skipped.append({'path': str(target), 'reason': 'path does not exist'})
            continue

        # nanochat checkpoints carry no config.json, so they are discovered by
        # their own (model_N.pt, meta_N.json) pair instead.
        if nanochat.is_nanochat_checkpoint(target):
            candidates = [target]
        elif (target / 'config.json').is_file():
            candidates = [target]
        elif (target / 'final' / 'config.json').is_file():
            candidates = [target / 'final']
        else:
            candidates = sorted({p.parent for p in target.rglob('config.json')})
            candidates += sorted({p.parent for p in target.glob('*/meta_*.json') if nanochat.is_nanochat_checkpoint(p.parent)})

        filtered = 0
        for path in candidates:
            if not include_checkpoints and any(re.fullmatch(r'checkpoint-\d+', part) for part in path.parts):
                # Explicit targets bypass the recursive checkpoint filter.
                if path != target:
                    filtered += 1
                    continue
            # In experiment trees prefer the final export over stray configs.
            if path.name != 'final' and target not in (path, path.parent) and (path.parent / 'final').is_dir():
                continue
            if not has_hf_weights(path) and not nanochat.is_nanochat_checkpoint(path):
                if (path / 'model.pt').exists():
                    skipped.append({'path': str(path), 'reason': 'custom model.pt format is not loadable by AutoModelForCausalLM'})
                continue
            found[str(path)] = path

        # Discovered-but-filtered checkpoints are reported once per target
        # rather than one line per save, so a tree with 200 checkpoints stays
        # readable -- but never disappears silently.
        if filtered:
            skipped.append(
                {
                    'path': str(target),
                    'reason': f'{filtered} checkpoint-* dir(s) under this target were not '
                    f'evaluated; pass --include-checkpoints to score them',
                }
            )

    # Group by containing directory, then by STEP: str() alone would order
    # checkpoint-100 before checkpoint-2 and scramble a training sweep.
    checkpoints = sorted(found.values(), key=lambda p: (str(p.parent), step_of(p), p.name))
    return Targets(checkpoints=checkpoints, skipped=skipped)


def resolve_tokenizer(ckpt: Path, explicit: Path | None = None, checkpoints_dir: Path | None = None) -> Path:
    if nanochat.is_nanochat_checkpoint(ckpt):
        return nanochat.resolve_nanochat_tokenizer(ckpt, explicit)
    if explicit is not None:
        return explicit.resolve()
    candidates = [ckpt]
    if checkpoints_dir is not None:
        candidates += [checkpoints_dir / 'tokenizers', checkpoints_dir, checkpoints_dir.parent / 'tokenizers', checkpoints_dir.parent]
    else:
        candidates += [ckpt.parent / 'tokenizers', ckpt.parent, ckpt.parent.parent / 'tokenizers', ckpt.parent.parent]
    # A fast tokenizer.json is the common case, but a SentencePiece model with
    # only tokenizer.model and tokenizer_config.json is still loadable (MonadGPT
    # ships one), so accept any directory AutoTokenizer can actually open.
    marker_files = ('tokenizer.json', 'tokenizer.model', 'vocab.json', 'spiece.model')
    for candidate in candidates:
        if any((candidate / name).is_file() for name in marker_files):
            return candidate
    raise FileNotFoundError(
        f'no tokenizer files ({", ".join(marker_files)}) found near {ckpt}; pass --tokenizer DIR'
    )


def model_fingerprint(checkpoint: Path, tok_path: Path) -> str:
    """Cheap cache identity: hashes of small semantic files + weight file stats."""
    h = hashlib.sha256()
    semantic_files = {checkpoint / 'config.json', checkpoint / 'generation_config.json'}
    semantic_files.update(
        tok_path / name
        for name in (
            'tokenizer.json',
            'tokenizer_config.json',
            'special_tokens_map.json',
            'added_tokens.json',
            'vocab.json',
            'chat_template.json',
        )
    )
    semantic_files.update(tok_path.glob('*.model'))
    semantic_files.update(tok_path.glob('*.txt'))
    semantic_files.update(tok_path.glob('*.jinja'))
    semantic_files.update(checkpoint.glob('*.py'))
    semantic_files.update(tok_path.glob('*.py'))
    # Training summaries are cached too; a newly saved trainer state invalidates them.
    semantic_files.update((checkpoint / 'trainer_state.json', checkpoint.parent / 'trainer_state.json'))
    for path in sorted(semantic_files):
        if path.is_file():
            h.update(str(path.resolve()).encode())
            h.update(file_hash(path).encode())
    for path in sorted(tok_path.glob('tokenizer.pkl')) + sorted(checkpoint.glob('meta_*.json')):
        h.update(str(path.resolve()).encode())
        h.update(file_hash(path).encode())
    weight_files = (
        sorted(checkpoint.glob('*.safetensors')) + sorted(checkpoint.glob('pytorch_model*.bin')) + sorted(checkpoint.glob('model_*.pt'))
    )
    for path in weight_files:
        stat = path.stat()
        h.update(path.name.encode())
        h.update(str(stat.st_size).encode())
        h.update(str(stat.st_mtime_ns).encode())
    return h.hexdigest()


# ============================================================================
# Model load / free
# ============================================================================


def needs_remote_code(checkpoint: Path) -> bool:
    """True when config.json declares an `auto_map`, i.e. the architecture is
    NOT one transformers ships and loading it EXECUTES python from the model
    directory.

    Deliberately narrow: remote code is enabled only for models that genuinely
    cannot load without it (e.g. MODELS/Talkie-1930-13b -> modeling_talkie.py),
    never as a blanket default.
    """
    try:
        with open(checkpoint / 'config.json', 'rb') as fh:
            return bool(json.load(fh).get('auto_map'))
    except Exception:
        return False


def weight_bytes(checkpoint: Path) -> int:
    """On-disk size of the weight shards - a good proxy for resident size when
    loading at the checkpoint's own dtype."""
    files = list(checkpoint.glob('*.safetensors')) + list(checkpoint.glob('pytorch_model*.bin')) + list(checkpoint.glob('model_*.pt'))
    return sum(f.stat().st_size for f in files)


def free_vram(device: torch.device) -> int:
    if device.type != 'cuda':
        return 0
    free, _total = torch.cuda.mem_get_info(device.index or 0)
    return int(free)


def plan_placement(checkpoint: Path, device: torch.device, headroom: float = 0.90) -> dict | None:
    """Return from_pretrained kwargs for CPU/disk offload, or None to load
    wholly onto `device`.

    A 7B in bf16 is ~14.5 GB and just fits a 16 GB card; a 13B is ~26.5 GB and
    cannot. Rather than OOM, hand the oversized part to accelerate.
    """
    if device.type != 'cuda':
        return None
    need = weight_bytes(checkpoint)
    budget = int(free_vram(device) * headroom)
    if need == 0 or need <= budget:
        return None
    print(
        f'  placement: weights ~{need / 2**30:.1f} GiB > {budget / 2**30:.1f} GiB usable VRAM '
        f'-> device_map=auto with CPU offload (SLOW; offloaded layers run on CPU)'
    )
    return {
        'device_map': 'auto',
        'max_memory': {(device.index or 0): budget, 'cpu': '48GiB'},
        'low_cpu_mem_usage': True,
    }


def load_model_and_tokenizer(checkpoint: Path, tok_dir: Path, device: torch.device, dtype: torch.dtype, load_8bit: bool = False):
    if nanochat.is_nanochat_checkpoint(Path(checkpoint)):
        if load_8bit:
            raise SystemExit('--load-8bit is bitsandbytes over transformers layers; nanochat models do not go through it')
        if dtype == torch.float32 and device.type == 'cuda':
            # These are published in fp32 and are 2.8B-3.3B parameters, which
            # is 11-13 GiB before activations. bf16 is what they were trained
            # and served in.
            print('  nanochat: --dtype auto/float32 on GPU would not fit; loading bfloat16')
            dtype = torch.bfloat16
        return nanochat.load_nanochat(Path(checkpoint), Path(tok_dir), device, dtype)

    remote = needs_remote_code(Path(checkpoint))
    if remote:
        print(f'  trust_remote_code=True — executing custom modeling code from {checkpoint}')
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, trust_remote_code=remote)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError('tokenizer has neither pad nor EOS token; cannot batch or pad')
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # correct batched generation

    kwargs = {'dtype': dtype, 'trust_remote_code': remote}
    if load_8bit:
        # Explicit, never automatic: int8 changes the measurement (~0.0025 bpb on
        # our 75M control) and every number produced under it must be labelled.
        from transformers import BitsAndBytesConfig

        need = weight_bytes(Path(checkpoint))
        print(f'  load_in_8bit=True — ~{need / 2**30:.1f} GiB bf16 -> ~{need / 2**31:.1f} GiB int8 (QUANTIZED, see report)')
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
        placement = {'device_map': {'': (device.index or 0)}}
    else:
        placement = plan_placement(Path(checkpoint), device)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs, **(placement or {}))
    model.config.use_cache = True
    if placement is None:
        model.to(device)  # accelerate already placed the sharded/offloaded/quantized case
    model.eval()
    return model, tokenizer


def free_model(model, device: torch.device) -> None:
    """Actually release the weights.

    `del model` only drops the local name; callers may still hold references.
    Moving parameters to the 'meta' device frees the real storage regardless.
    """
    with contextlib.suppress(Exception):  # freeing must never break a run
        # An accelerate-dispatched model (device_map/offload) must not be moved:
        # .to() fights the hooks and warns. Dropping the reference is enough,
        # since accelerate owns the offload buffers.
        if not getattr(model, 'hf_device_map', None):
            model.to('meta')
    del model
    import gc

    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def prefix_id(tokenizer, model) -> int | None:
    """BOS if available, else EOS - the prepended scoring prefix."""
    for value in (
        getattr(model.config, 'bos_token_id', None),
        tokenizer.bos_token_id,
        getattr(model.config, 'eos_token_id', None),
        tokenizer.eos_token_id,
    ):
        if isinstance(value, int) and value >= 0:
            return value
    return None


def model_context_limit(model, requested: int) -> int:
    candidates = [requested]
    for name in ('max_position_embeddings', 'n_positions', 'max_seq_len'):
        value = getattr(model.config, name, None)
        if isinstance(value, int) and 16 <= value < 10_000_000:
            candidates.append(value)
    return min(candidates)


# ============================================================================
# Training lineage from checkpoint metadata and training logs
# ============================================================================


def checkpoint_lineage(ckpt_dir: Path, n_params: int | None = None, n_params_no_embed: int | None = None) -> dict:
    """Everything we can sniff about the recipe from files inside the checkpoint.

    Returns a single dict covering: base-vs-SFT kind, base model, LR, context
    length, trainer step/epoch, tokens-seen estimate (from total_flos) and
    wall-clock training time / budget bucket (from log history + train.log).

    `n_params_no_embed` MUST be passed for an accurate tokens-seen estimate --
    see the total_flos note below. Without it the estimate is a lower bound.
    """

    if nanochat.is_nanochat_checkpoint(Path(ckpt_dir)):
        return nanochat.nanochat_lineage(Path(ckpt_dir))

    def flatten(d, out):
        for k, v in d.items():
            if isinstance(v, dict):
                flatten(v, out)
            else:
                out[k] = v
        return out

    info = {
        'kind': 'unknown',
        'base_model': None,
        'learning_rate': None,
        'context': None,
        'global_step': None,
        'epoch': None,
        'tokens_seen': None,
        'tokens_per_param': None,
        'tokens_lower_bound': False,
        'runtime_minutes': None,
        'runtime_segments_minutes': [],
        'budget': 'unknown',
        'note': '',
        # Recipe fields from the training configuration.
        'optim': None,
        'lr_scheduler': None,
        'warmup_steps': None,
        'stable_steps': None,
        'decay_steps': None,
        'max_steps': None,
        'max_train_minutes': None,
        'train_seq_length': None,
        'tokens_per_step': None,
        'tokens_seen_from_steps': None,
        'seed': None,
        'config_file': None,
        'warnings': [],
    }

    cfg_json = ckpt_dir / 'config.json'
    if cfg_json.exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            info['context'] = json.loads(cfg_json.read_text()).get('max_position_embeddings')

    ft = ckpt_dir / 'fine_tune_config.toml'
    if ft.exists():
        try:
            with open(ft, 'rb') as f:
                flat = flatten(tomllib.load(f), {})
            info['kind'] = 'sft'
            info['base_model'] = Path(str(flat.get('base_model', '?'))).name
            info['learning_rate'] = flat.get('learning_rate')
        except (tomllib.TOMLDecodeError, OSError):
            info['kind'] = 'sft'
    else:
        tc = next(
            (
                c
                for c in (
                    ckpt_dir / 'training_config.toml',
                    ckpt_dir.parent / 'training_config.toml',
                    ckpt_dir / 'config.toml',
                    ckpt_dir.parent / 'config.toml',
                )
                if c.is_file()
            ),
            None,
        )
        if tc is not None or any((ckpt_dir / n).exists() for n in ('trainer_state.json', 'trainer_state2.json')):
            info['kind'] = 'base'
        if tc is not None:
            try:
                with open(tc, 'rb') as f:
                    flat = flatten(tomllib.load(f), {})
            except (tomllib.TOMLDecodeError, OSError):
                flat = {}
            if flat:
                info['config_file'] = str(tc)
                info['learning_rate'] = flat.get('learning_rate')
                info['optim'] = flat.get('optim')
                info['lr_scheduler'] = flat.get('lr_scheduler_type')
                info['warmup_steps'] = flat.get('warmup_steps')
                info['stable_steps'] = flat.get('num_stable_steps')
                info['decay_steps'] = flat.get('num_decay_steps')
                info['max_steps'] = flat.get('max_steps')
                info['max_train_minutes'] = flat.get('max_train_minutes')
                info['train_seq_length'] = flat.get('max_seq_length')
                info['seed'] = flat.get('seed')
                bs = flat.get('per_device_train_batch_size')
                ga = flat.get('gradient_accumulation_steps', 1)
                sl = flat.get('max_seq_length')
                if bs and sl:
                    info['tokens_per_step'] = int(bs) * int(ga or 1) * int(sl)

    # trainer_state.json: step/epoch, tokens seen (via total_flos), runtime.
    # Only look one level up when ckpt_dir IS a checkpoint export; pointing the
    # eval at a RUN directory would otherwise pick up the sibling run's files
    # from the parent tree (e.g. stability/train.log for stability/nodecay_2h).
    is_export = ckpt_dir.name == 'final' or ckpt_dir.name.startswith('checkpoint-')
    search = [ckpt_dir] + ([ckpt_dir.parent] if is_export else [])
    state = None
    for path in (d / 'trainer_state.json' for d in search):
        if path.is_file():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                state = json.loads(path.read_text())
            if state is not None:
                info['_state_file'] = str(path)
                break
    if state:
        info['global_step'] = state.get('global_step')
        info['epoch'] = state.get('epoch')
        final_log = next((x for x in reversed(state.get('log_history', [])) if 'train_runtime' in x), {})
        runtime = final_log.get('train_runtime')
        # HF's train_runtime covers only the current process. A resumed run can
        # look like a two-hour model while its weights contain an earlier hour
        # as well; base_train.py appends explicit "Training time" lines per
        # completed segment - sum those when present.
        segment_minutes: list[float] = []
        train_log = next((d / 'train.log' for d in search if (d / 'train.log').is_file()), ckpt_dir / 'train.log')
        if train_log.is_file():
            with contextlib.suppress(Exception):
                segment_minutes = [
                    float(x) for x in re.findall(r'Training time:\s*([0-9]+(?:\.[0-9]+)?)\s*minutes', train_log.read_text(errors='replace'))
                ]
        flops = state.get('total_flos') or final_log.get('total_flos')
        # HF computes total_flos as `6 * tokens * num_parameters(exclude_embeddings=True)`
        # (transformers/trainer.py: Trainer.floating_point_ops). Dividing by TOTAL params
        # therefore under-reports tokens by exactly the embedding fraction -- for the 77M
        # vintage model that is 34% (49.56M non-embedding / 74.72M total = 0.6632).
        # Always divide by the same non-embedding count HF multiplied by.
        denom = n_params_no_embed or n_params
        if flops and denom:
            tokens = flops / (6.0 * denom)
            info['tokens_seen'] = tokens
            # Report token budget relative to the total parameter count.
            if n_params:
                info['tokens_per_param'] = tokens / n_params
            if not n_params_no_embed:
                info['tokens_lower_bound'] = True
                info['note'] = (
                    'tokens-seen divided by TOTAL params because the non-embedding count was '
                    'unavailable; HF measures total_flos against non-embedding params, so this '
                    'is an UNDER-estimate (lower bound)'
                )
        info['runtime_minutes'] = (
            sum(segment_minutes) if segment_minutes else (runtime / 60.0 if isinstance(runtime, (int, float)) else None)
        )
        info['runtime_segments_minutes'] = segment_minutes
        info['budget'] = budget_label(info['runtime_minutes'])

        # `max_train_minutes` is a PER-PROCESS budget: base_train.py restarts the
        # clock on resume, so an interrupted run can legitimately exceed it. The
        # summed segments are the true GPU time -- say so rather than look wrong.
        mtm = info.get('max_train_minutes')
        if mtm and info['runtime_minutes'] and info['runtime_minutes'] > float(mtm) * 1.05:
            info['warnings'].append(
                f'wall clock {info["runtime_minutes"]:.0f} min exceeds the configured max_train_minutes='
                f'{mtm}; the run was resumed {max(len(segment_minutes) - 1, 1)} time(s) and the budget '
                'restarts per process, so the total is real GPU time, not a reporting error'
            )

        # Independent cross-check of tokens seen: steps x tokens/step from the
        # config, versus total_flos. They should agree to well under 1%.
        if info.get('tokens_per_step') and info.get('global_step'):
            by_steps = info['global_step'] * info['tokens_per_step']
            info['tokens_seen_from_steps'] = by_steps
            if info['tokens_seen'] and abs(by_steps - info['tokens_seen']) / by_steps > 0.02:
                info['warnings'].append(
                    f'tokens-seen disagree: {info["tokens_seen"] / 1e6:.1f}M from total_flos vs '
                    f'{by_steps / 1e6:.1f}M from steps x tokens/step'
                )

        # config.json max_position_embeddings is the ARCHITECTURE window; the
        # sequence length actually trained on lives in the training config.
        if info.get('train_seq_length') and info.get('context') and info['train_seq_length'] != info['context']:
            info['warnings'].append(f'trained at seq length {info["train_seq_length"]} but the config window is {info["context"]}')

        m = re.search(r'(\d+)$', ckpt_dir.name)
        if m and info['global_step'] and int(m.group(1)) != info['global_step']:
            # A restart without --resume zeroes the trainer counters, so
            # total_flos covers only the newest segment: a LOWER BOUND.
            info['tokens_lower_bound'] = True
            info['note'] = (
                f'folder name says step {m.group(1)} but trainer_state says {info["global_step"]} - '
                'training was restarted; the tokens-seen estimate covers only the latest segment (lower bound)'
            )
    if not info['note']:
        if state is None:
            info['note'] = 'no trainer_state.json found - tokens seen unknown'
    return info


def training_curve(ckpt_dir: Path) -> dict:
    """Logged trainer losses and gradient norms; validation protocol is run-specific."""
    out: dict = {
        'eval_curve': [],
        'train_curve': [],
        'eval_loss_nats_per_token': None,
        'eval_step': None,
        'eval_ppl': None,
        'train_loss_nats_per_token': None,
        'grad_norm_max': None,
        'grad_norm_min': None,
        'grad_norm_mean': None,
        'grad_norm_nonfinite': None,
        'grad_norm_count': 0,
    }
    state_file = None
    candidates = [ckpt_dir / 'trainer_state.json']
    if ckpt_dir.name == 'final' or ckpt_dir.name.startswith('checkpoint-'):
        candidates.append(ckpt_dir.parent / 'trainer_state.json')
    for cand in candidates:
        if cand.is_file():
            state_file = cand
            break
    if state_file is None:
        return out
    try:
        hist = json.loads(state_file.read_text()).get('log_history', [])
    except Exception:
        return out

    out['eval_curve'] = [[h['step'], h['eval_loss']] for h in hist if 'eval_loss' in h]
    out['train_curve'] = [[h['step'], h['loss']] for h in hist if 'loss' in h]
    gn = [h['grad_norm'] for h in hist if isinstance(h.get('grad_norm'), (int, float))]
    out['grad_norm_count'] = len(gn)
    if out['eval_curve']:
        out['eval_step'], out['eval_loss_nats_per_token'] = out['eval_curve'][-1]
        with contextlib.suppress(OverflowError):
            out['eval_ppl'] = math.exp(out['eval_loss_nats_per_token'])
    if out['train_curve']:
        out['train_loss_nats_per_token'] = out['train_curve'][-1][1]
    if gn:
        finite = [g for g in gn if g == g and abs(g) != float('inf')]
        out['grad_norm_nonfinite'] = len(gn) - len(finite)
        if finite:
            out['grad_norm_max'] = max(finite)
            out['grad_norm_min'] = min(finite)
            out['grad_norm_mean'] = sum(finite) / len(finite)
    return out


def budget_label(minutes: float | None) -> str:
    if minutes is None:
        return 'unknown'
    if minutes <= 40:
        return '~0.5h'
    if minutes <= 75:
        return '~1h'
    if minutes <= 150:
        return '~2h'
    return '>2h'


def model_slug(model_type: str | None, n_params: int) -> str:
    """Filesystem-safe identity from architecture + param count, e.g.
    'llama77M' or 'llama1.1B'. Mirrors base_train.make_model_id()."""
    if n_params >= 1e9:
        size = f'{n_params / 1e9:.1f}'.rstrip('0').rstrip('.') + 'B'
    else:
        size = f'{round(n_params / 1e6)}M'
    return f'{model_type or "model"}{size}'


def peek_model_identity(checkpoint: Path) -> tuple[str | None, int]:
    """(model_type, parameter count) read from config.json WITHOUT loading
    weights, for output filenames."""
    if nanochat.is_nanochat_checkpoint(Path(checkpoint)):
        return nanochat.nanochat_identity(Path(checkpoint))
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        remote = needs_remote_code(Path(checkpoint))
        cfg = AutoConfig.from_pretrained(checkpoint, trust_remote_code=remote)
        with torch.device('meta'):
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=remote)
        n = sum(p.numel() for p in model.parameters())
        del model
        return getattr(cfg, 'model_type', None), n
    except Exception:
        return None, 0


def provenance_line(info: dict) -> str:
    """One-line human-readable form of checkpoint_lineage()."""
    ctx = f', ctx {info["context"]}' if info.get('context') else ''
    step = ''
    if info.get('global_step') is not None:
        step = f', trainer step {info["global_step"]}'
        if info.get('epoch') is not None:
            step += f' (epoch {info["epoch"]:.2f})'
    if info['kind'] == 'sft':
        lr = info.get('learning_rate')
        lr_s = f' @ lr={lr}' if lr is not None else ''
        return f'SFT from {info.get("base_model") or "?"}{lr_s}{ctx}{step}'
    if info['kind'] == 'base':
        recipe = ''
        if info.get('optim') or info.get('learning_rate') is not None:
            bits = [str(info['optim'])] if info.get('optim') else []
            if info.get('learning_rate') is not None:
                bits.append(f'@ lr={info["learning_rate"]:g}')
            recipe = ' (' + ' '.join(bits) + ')'
        return f'base pretraining{recipe}{ctx}{step}'
    return f'(unknown recipe{ctx}{step})'


# ============================================================================
# Evaluation data loading (held-out prose + chat pairs, with stable A/B split)
# ============================================================================


@dataclass(frozen=True)
class TextItem:
    item_id: str
    text: str
    split: str


@dataclass(frozen=True)
class ChatItem:
    item_id: str
    context: str
    target: str
    split: str


def load_text_items(path: Path, limit: int) -> list[TextItem]:
    out: list[TextItem] = []
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if len(out) >= limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get('text')
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'{path}:{line_no}: expected a non-empty "text" string')
            item_id = stable_hash(text.encode('utf-8'))[:16]
            out.append(TextItem(item_id, text, split_for(item_id)))
    if not out:
        raise ValueError(f'no usable documents in {path}')
    return out


def load_chat_items(path: Path, limit: int) -> list[ChatItem]:
    out: list[ChatItem] = []
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if len(out) >= limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            context, target = obj.get('context'), obj.get('target')
            if not isinstance(context, str) or not isinstance(target, str) or not target:
                raise ValueError(f'{path}:{line_no}: expected "context" and non-empty "target" strings')
            item_id = stable_hash((context + '\0' + target).encode('utf-8'))[:16]
            out.append(ChatItem(item_id, context, target, split_for(item_id)))
    return out


def die(message: str) -> None:
    print(f'error: {message}', file=sys.stderr)
    sys.exit(1)

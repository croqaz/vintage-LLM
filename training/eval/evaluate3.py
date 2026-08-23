#!/usr/bin/env python3
"""Reliable model selection for the vintage-LLM experiments.

Unlike evaluate.py (checkpoint/vintage diagnostics) and evaluate2.py (an
absolute "bake" report), this program answers a narrower question:

    Which of these training experiments is best on held-out period prose?

The decision is based on deterministic, byte-normalised held-out likelihood.
It retains the per-document measurements so comparisons can use paired
bootstrap confidence intervals.  There is deliberately no weighted composite:
small logic quizzes and sampled generations are useful diagnostics, but proved
too noisy to choose among nearby one-hour runs.

Examples
--------
  python evaluate3.py autoresearch autoresearch2 --out eval3_results/all.json
  python evaluate3.py MODELS --out eval3_results/models.json
  python evaluate3.py path/to/final --docs 20 --no-chat

The JSON is a resumable cache.  Re-running the same command skips checkpoints
whose weights, tokenizer, data, and scoring settings have not changed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DATA = SCRIPT_DIR / 'eval_data'
VERSION = 2
LOG2E = math.log2(math.e)

# The exact human-review battery used by sample_autocomplete.sh.  Keeping it
# here makes an eval3 run self-contained and preserves line-for-line comparison
# with the samples that informed the experiment reports.
AUTOCOMPLETE_PROMPTS = [
    'Let them eat brioche,',
    'Elementary, my dear Watson,',
    'Alas, poor Yorick! I knew him,',
    'The love of money is the root of all',
    'Put your trust in God, my boys, and keep your',
    'I disapprove of what you say, but I will defend to the death your',
    'It was a cold morning in November when the carriage arrived at',
    '"You cannot mean it," she said, lowering her voice so that',
    'My dearest brother, I write to you from Lisbon, where the',
    'To prepare a proper broth for an invalid, first take',
    'The steam engine differs from the water-wheel chiefly in that',
    'LONDON, Tuesday. — The House of Commons yesterday debated',
    'On the cultivation of apple orchards in northern climates, the farmer must',
    'The old lighthouse keeper climbed the stairs slowly, remembering',
    'Among the curiosities exhibited at the fair was a mechanical',
    'The physician examined the patient and concluded that the fever',
    'A legal dispute involving an individual was presented, where',
    'We approach the close of our survey of the life and works of',
    'As our approach drew nearer, the scattered villages and humble enclosures',
    'This question, as well as the manner of its resolution,',
    'The English forces were preparing for an offensive on',
    'It was this rigorous self-discipline, this dedication to the unseen',
    'As the afternoon wore on, and the wind settled into a steady',
    'In the midst of this tempest of violence, a figure, whom we shall name',
    'Hark, gentle reader, and lend thine ear to a matter of profound import,',
    "The speaker then addressed the prior assertion that Her Majesty's Government should",
    'Let the farmers be warned that this poisonous plant is not to be confused with the edible fruit,',
    'Many the gay straw-rides to the Lake; frequent and long the walks through',
]

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


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


def stable_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_for(item_id: str) -> str:
    """Stable, approximately even diagnostic split independent of file order."""
    return 'A' if int(stable_hash(item_id.encode())[:8], 16) % 2 == 0 else 'B'


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


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
                raise ValueError(f"{path}:{line_no}: expected a non-empty 'text' string")
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
                raise ValueError(f"{path}:{line_no}: expected 'context' and non-empty 'target' strings")
            item_id = stable_hash((context + '\0' + target).encode('utf-8'))[:16]
            out.append(ChatItem(item_id, context, target, split_for(item_id)))
    return out


def has_hf_weights(path: Path) -> bool:
    return any(path.glob('*.safetensors')) or any(path.glob('pytorch_model*.bin'))


def resolve_targets(targets: list[Path], include_checkpoints: bool = False) -> tuple[list[Path], list[dict[str, str]]]:
    """Resolve checkpoint, experiment, or recursive collection paths."""
    found: dict[str, Path] = {}
    skipped: list[dict[str, str]] = []
    for target in targets:
        target = target.resolve()
        if not target.exists():
            skipped.append({'path': str(target), 'reason': 'path does not exist'})
            continue

        candidates: list[Path]
        if (target / 'config.json').is_file():
            candidates = [target]
        elif (target / 'final' / 'config.json').is_file():
            candidates = [target / 'final']
        else:
            candidates = sorted({p.parent for p in target.rglob('config.json')})

        for custom_weight in target.rglob('model.pt') if target.is_dir() else []:
            custom_dir = custom_weight.parent
            if not (custom_dir / 'config.json').is_file():
                skipped.append({'path': str(custom_dir), 'reason': 'custom model.pt format is not loadable by AutoModelForCausalLM'})

        for path in candidates:
            if not include_checkpoints and any(re.fullmatch(r'checkpoint-\d+', part) for part in path.parts):
                continue
            # In experiment trees, evaluate final exports rather than stray
            # configs.  Direct children in MODELS/ need not be named "final".
            if path.name != 'final' and target not in (path, path.parent) and (path.parent / 'final').is_dir():
                continue
            if not has_hf_weights(path):
                if (path / 'model.pt').exists():
                    skipped.append({'path': str(path), 'reason': 'custom model.pt format is not loadable by AutoModelForCausalLM'})
                continue
            found[str(path)] = path
    return list(found.values()), skipped


def tokenizer_path(checkpoint: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    for candidate in (
        checkpoint,
        checkpoint.parent / 'tokenizers',
        checkpoint.parent,
        checkpoint.parent.parent / 'tokenizers',
        checkpoint.parent.parent,
    ):
        if (candidate / 'tokenizer.json').is_file():
            return candidate
    raise FileNotFoundError(f'no tokenizer.json found near {checkpoint}; pass --tokenizer')


def model_fingerprint(checkpoint: Path, tok_path: Path) -> str:
    """Cheap cache identity: metadata plus hashes of small semantic files."""
    h = hashlib.sha256()
    for path in sorted([checkpoint / 'config.json', tok_path / 'tokenizer.json']):
        h.update(str(path.resolve()).encode())
        h.update(file_hash(path).encode())
    weight_files = sorted(checkpoint.glob('*.safetensors')) + sorted(checkpoint.glob('pytorch_model*.bin'))
    for path in weight_files:
        stat = path.stat()
        h.update(path.name.encode())
        h.update(str(stat.st_size).encode())
        h.update(str(stat.st_mtime_ns).encode())
    return h.hexdigest()


def choose_device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def choose_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != 'auto':
        return getattr(torch, name)
    return torch.bfloat16 if device.type == 'cuda' else torch.float32


def prefix_id(tokenizer, model) -> int | None:
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


@torch.inference_mode()
def score_texts(tokenizer, model, items: list[TextItem], max_tokens: int, progress_every: int = 25) -> list[dict[str, Any]]:
    """Return document clusters containing additive bits and byte counts."""
    records: list[dict[str, Any]] = []
    pre = prefix_id(tokenizer, model)
    limit = model_context_limit(model, max_tokens)
    content_budget = limit - (1 if pre is not None else 0)
    for i, item in enumerate(items, 1):
        ids = tokenizer(item.text, add_special_tokens=False).input_ids[:content_budget]
        if len(ids) < 8:
            continue
        input_ids = ([pre] if pre is not None else []) + ids
        inp = torch.tensor([input_ids], dtype=torch.long, device=model.device)
        logits = model(input_ids=inp, use_cache=False).logits[0, :-1].float()
        labels = inp[0, 1:]
        nll = F.cross_entropy(logits, labels, reduction='none').cpu().numpy().astype(np.float64)
        scored_ids = ids if pre is not None else ids[1:]
        if len(nll) != len(scored_ids):
            raise RuntimeError('internal token/loss alignment error')

        decoded = tokenizer.decode(scored_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        nbytes = len(decoded.encode('utf-8'))
        cut = min(128, len(scored_ids))
        early_text = tokenizer.decode(scored_ids[:cut], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        late_text = tokenizer.decode(scored_ids[cut:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        records.append(
            {
                'id': item.item_id,
                'split': item.split,
                'bits': float(nll.sum() * LOG2E),
                'bytes': nbytes,
                'tokens': len(scored_ids),
                'early_bits': float(nll[:cut].sum() * LOG2E),
                'early_bytes': len(early_text.encode('utf-8')),
                'late_bits': float(nll[cut:].sum() * LOG2E),
                'late_bytes': len(late_text.encode('utf-8')),
            }
        )
        if progress_every and i % progress_every == 0:
            print(f' {i}/{len(items)}', end='', flush=True)
    return records


@torch.inference_mode()
def score_chat(tokenizer, model, items: list[ChatItem], max_tokens: int, progress_every: int = 50) -> list[dict[str, Any]]:
    """Score assistant targets conditionally; retained as a diagnostic only."""
    records: list[dict[str, Any]] = []
    pre = prefix_id(tokenizer, model)
    if pre is None:
        return records
    limit = model_context_limit(model, max_tokens)
    budget = limit - 1
    for i, item in enumerate(items, 1):
        full = item.context + item.target
        encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded.input_ids
        offsets = encoded.offset_mapping
        boundary = len(item.context)
        target_positions = [j for j, (start, end) in enumerate(offsets) if end > boundary]
        if not target_positions:
            continue
        if len(ids) > budget:
            drop = len(ids) - budget
            ids = ids[drop:]
            target_positions = [j - drop for j in target_positions if j >= drop]
        if not target_positions:
            continue
        inp = torch.tensor([[pre] + ids], dtype=torch.long, device=model.device)
        logits = model(input_ids=inp, use_cache=False).logits[0, :-1].float()
        labels = inp[0, 1:]
        nll = F.cross_entropy(logits, labels, reduction='none')
        selected = nll[target_positions].cpu().numpy().astype(np.float64)
        target_ids = [ids[j] for j in target_positions]
        decoded = tokenizer.decode(target_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        records.append(
            {
                'id': item.item_id,
                'split': item.split,
                'bits': float(selected.sum() * LOG2E),
                'bytes': len(decoded.encode('utf-8')),
                'tokens': len(target_positions),
            }
        )
        if progress_every and i % progress_every == 0:
            print(f' c{i}/{len(items)}', end='', flush=True)
    return records


def longest_loop(words: list[str], max_period: int = 12) -> int:
    """Longest immediately repeating word block; zero means no full repeat."""
    best = 0
    for period in range(1, min(max_period, len(words) // 2) + 1):
        run = 0
        for i in range(len(words) - period):
            if words[i] == words[i + period]:
                run += 1
                if run >= period:
                    best = max(best, run + period)
            else:
                run = 0
    return best


def generation_health(prompt: str, text: str) -> dict[str, Any]:
    """Mechanical failure indicators, not a semantic quality score."""
    words = [w.lower() for w in WORD_RE.findall(text)]
    bigrams = list(zip(words, words[1:], strict=False))
    trigrams = list(zip(words, words[1:], words[2:], strict=False))
    echo = sum(word in words[max(0, i - 4) : i] for i, word in enumerate(words)) / max(1, len(words))
    distinct2 = len(set(bigrams)) / max(1, len(bigrams))
    distinct3 = len(set(trigrams)) / max(1, len(trigrams))
    loop = longest_loop(words)
    punct_issues = abs(text.count('(') - text.count(')')) + text.count('"') % 2
    punct_issues += len(re.findall(r'[,;:]{2,}|\.{4,}|\s,', text))
    prompt_words = {w.lower() for w in WORD_RE.findall(prompt)}
    content = [w for w in words if len(w) > 3]
    prompt_copy = sum(w in prompt_words for w in content) / max(1, len(content))
    # A gate only for unmistakable surface collapse.  Diverse nonsense passes
    # this gate, by design; held-out BPB and human reading must judge it.
    degenerate = len(words) < 8 or distinct2 < 0.55 or echo > 0.30 or loop >= 12
    return {
        'words': len(words),
        'distinct_2': distinct2,
        'distinct_3': distinct3,
        'echo_rate': echo,
        'longest_loop_words': loop,
        'punct_issues': punct_issues,
        'prompt_copy_rate': prompt_copy,
        'degenerate': degenerate,
    }


@torch.inference_mode()
def run_autocomplete(tokenizer, model, args) -> dict[str, Any]:
    """Generate the sample_autocomplete.sh battery entirely offline."""
    prompts = AUTOCOMPLETE_PROMPTS[: args.generation_prompts]
    if not prompts or args.generation_mode == 'none':
        return {}
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError('autocomplete generation needs a pad or EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    old_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    torch.manual_seed(args.seed)
    if model.device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    rows = []
    started = time.time()
    for start in range(0, len(prompts), args.generation_batch_size):
        batch_prompts = prompts[start : start + args.generation_batch_size]
        encoded = tokenizer(batch_prompts, return_tensors='pt', padding=True, add_special_tokens=True).to(model.device)
        input_width = encoded.input_ids.shape[1]
        sampled = args.generation_mode == 'sample'
        generated = model.generate(
            **encoded,
            max_new_tokens=args.generation_tokens,
            do_sample=sampled,
            temperature=args.temperature if sampled else None,
            top_p=args.top_p if sampled else None,
            top_k=args.top_k if sampled else None,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        for prompt, sequence in zip(batch_prompts, generated, strict=False):
            continuation = tokenizer.decode(sequence[input_width:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            rows.append({'prompt': prompt, 'continuation': continuation, **generation_health(prompt, continuation)})
        print(f' g{min(start + len(batch_prompts), len(prompts))}/{len(prompts)}', end='', flush=True)
    tokenizer.padding_side = old_side
    return {
        'mode': args.generation_mode,
        'seed': args.seed,
        'tokens_per_prompt': args.generation_tokens,
        'repetition_penalty': 1.0,
        'samples': rows,
        'summary': {
            'n_prompts': len(rows),
            'degenerate_rate': sum(r['degenerate'] for r in rows) / max(1, len(rows)),
            'mean_distinct_2': float(np.mean([r['distinct_2'] for r in rows])),
            'mean_echo_rate': float(np.mean([r['echo_rate'] for r in rows])),
            'worst_loop_words': max((r['longest_loop_words'] for r in rows), default=0),
            'mean_words': float(np.mean([r['words'] for r in rows])),
            'total_punct_issues': sum(r['punct_issues'] for r in rows),
        },
        'seconds': round(time.time() - started, 2),
        'warning': 'Surface-failure diagnostics only; lexical diversity does not measure coherence or factuality.',
    }


def aggregate(records: list[dict[str, Any]], split: str | None = None, prefix: str = '') -> float:
    rows = [r for r in records if split is None or r['split'] == split]
    bits = sum(float(r[f'{prefix}bits']) for r in rows)
    nbytes = sum(int(r[f'{prefix}bytes']) for r in rows)
    return bits / nbytes if nbytes else float('nan')


def bootstrap_ci(records: list[dict[str, Any]], draws: int, confidence: float, seed: int) -> tuple[float, float]:
    bits = np.asarray([r['bits'] for r in records], dtype=np.float64)
    nbytes = np.asarray([r['bytes'] for r in records], dtype=np.float64)
    if not len(bits):
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(bits), size=(draws, len(bits)))
    values = bits[indices].sum(axis=1) / nbytes[indices].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return tuple(float(x) for x in np.quantile(values, [alpha, 1.0 - alpha]))


def paired_bootstrap(
    candidate: list[dict[str, Any]], leader: list[dict[str, Any]], draws: int, confidence: float, seed: int
) -> dict[str, float]:
    ca, le = {r['id']: r for r in candidate}, {r['id']: r for r in leader}
    ids = sorted(ca.keys() & le.keys())
    if not ids:
        return {'delta': float('nan'), 'low': float('nan'), 'high': float('nan'), 'p_beats': float('nan')}
    cb = np.asarray([ca[x]['bits'] for x in ids])
    cn = np.asarray([ca[x]['bytes'] for x in ids])
    lb = np.asarray([le[x]['bits'] for x in ids])
    ln = np.asarray([le[x]['bytes'] for x in ids])
    delta = cb.sum() / cn.sum() - lb.sum() / ln.sum()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ids), size=(draws, len(ids)))
    values = cb[indices].sum(1) / cn[indices].sum(1) - lb[indices].sum(1) / ln[indices].sum(1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return {'delta': float(delta), 'low': float(low), 'high': float(high), 'p_beats': float(np.mean(values < 0))}


def training_metadata(checkpoint: Path, n_params: int) -> dict[str, Any]:
    for path in (checkpoint / 'trainer_state.json', checkpoint.parent / 'trainer_state.json'):
        if not path.is_file():
            continue
        try:
            state = json.loads(path.read_text())
        except Exception:
            continue
        final_log = next((x for x in reversed(state.get('log_history', [])) if 'train_runtime' in x), {})
        runtime = final_log.get('train_runtime')
        # HF's train_runtime covers only the current process.  A resumed run
        # can therefore look like a two-hour model while its weights actually
        # contain an earlier hour as well.  base_train.py appends one explicit
        # "Training time" line per completed segment; sum those when present.
        segment_minutes: list[float] = []
        train_log = checkpoint.parent / 'train.log'
        if train_log.is_file():
            with contextlib.suppress(Exception):
                segment_minutes = [
                    float(x) for x in re.findall(r'Training time:\s*([0-9]+(?:\.[0-9]+)?)\s*minutes', train_log.read_text(errors='replace'))
                ]
        flops = state.get('total_flos') or final_log.get('total_flos')
        tokens = flops / (6.0 * n_params) if flops and n_params else None
        minutes = sum(segment_minutes) if segment_minutes else (runtime / 60.0 if isinstance(runtime, (int, float)) else None)
        return {
            'state_file': str(path),
            'global_step': state.get('global_step'),
            'runtime_minutes': minutes,
            'runtime_segments_minutes': segment_minutes,
            'tokens_seen_estimate': tokens,
            'budget': budget_label(minutes),
        }
    return {
        'state_file': None,
        'global_step': None,
        'runtime_minutes': None,
        'runtime_segments_minutes': [],
        'tokens_seen_estimate': None,
        'budget': 'unknown',
    }


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


def release_model(model, device: torch.device) -> None:
    with contextlib.suppress(Exception):
        model.to('meta')
    del model
    import gc

    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding='utf-8')
    os.replace(temporary, path)


def cache_settings(args, heldout_hash: str, chat_hash: str | None) -> dict[str, Any]:
    return {
        'version': VERSION,
        'heldout_sha256': heldout_hash,
        'chat_sha256': chat_hash,
        'docs': args.docs,
        'chat_docs': 0 if args.no_chat else args.chat_docs,
        'max_tokens': args.max_tokens,
        'chat_max_tokens': args.chat_max_tokens,
        'generation_mode': args.generation_mode,
        'generation_prompts': args.generation_prompts,
        'generation_tokens': args.generation_tokens,
        'generation_batch_size': args.generation_batch_size,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'seed': args.seed,
        'dtype': args.dtype,
    }


def evaluate_checkpoint(checkpoint: Path, args, heldout, chats, device, dtype) -> dict[str, Any]:
    tok_dir = tokenizer_path(checkpoint, args.tokenizer)
    fingerprint = model_fingerprint(checkpoint, tok_dir)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    if chats and not tokenizer.is_fast:
        raise RuntimeError('chat target alignment requires a fast tokenizer; use --no-chat for this model')
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=dtype)
    model.config.use_cache = False
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(' prose', end='', flush=True)
    prose = score_texts(tokenizer, model, heldout, args.max_tokens)
    print(' chat', end='', flush=True) if chats else None
    chat = score_chat(tokenizer, model, chats, args.chat_max_tokens) if chats else []
    print(' autocomplete', end='', flush=True) if args.generation_mode != 'none' else None
    autocomplete = run_autocomplete(tokenizer, model, args)
    result = {
        'checkpoint': str(checkpoint),
        'label': str(checkpoint.relative_to(SCRIPT_DIR)) if checkpoint.is_relative_to(SCRIPT_DIR) else str(checkpoint),
        'fingerprint': fingerprint,
        'tokenizer': str(tok_dir),
        'params': n_params,
        'training': training_metadata(checkpoint, n_params),
        'prose_records': prose,
        'chat_records': chat,
        'autocomplete': autocomplete,
        'seconds': round(time.time() - started, 2),
    }
    result['metrics'] = {
        'prose_bpb': aggregate(prose),
        'split_a_bpb': aggregate(prose, 'A'),
        'split_b_bpb': aggregate(prose, 'B'),
        'early_bpb': aggregate(prose, prefix='early_'),
        'late_bpb': aggregate(prose, prefix='late_'),
        'chat_bpb': aggregate(chat),
    }
    release_model(model, device)
    return result


def ranking(results: list[dict[str, Any]], args, budget: str | None = None) -> list[dict[str, Any]]:
    eligible = [r for r in results if budget is None or r['training']['budget'] == budget]
    eligible = [r for r in eligible if math.isfinite(r['metrics']['prose_bpb'])]
    if not eligible:
        return []
    eligible.sort(key=lambda r: r['metrics']['prose_bpb'])
    leader = eligible[0]
    epsilon = leader['metrics']['prose_bpb'] * args.equivalence
    rows = []
    for rank, result in enumerate(eligible, 1):
        ci = bootstrap_ci(result['prose_records'], args.bootstrap, args.confidence, args.seed)
        pair = paired_bootstrap(result['prose_records'], leader['prose_records'], args.bootstrap, args.confidence, args.seed + rank)
        rows.append(
            {
                'rank': rank,
                'label': result['label'],
                'checkpoint': result['checkpoint'],
                'bpb': result['metrics']['prose_bpb'],
                'ci_low': ci[0],
                'ci_high': ci[1],
                'delta_vs_leader': pair['delta'],
                'delta_ci_low': pair['low'],
                'delta_ci_high': pair['high'],
                'p_beats_leader': pair['p_beats'],
                'equivalent_to_leader': pair['low'] <= epsilon,
                'split_a_bpb': result['metrics']['split_a_bpb'],
                'split_b_bpb': result['metrics']['split_b_bpb'],
                'late_bpb': result['metrics']['late_bpb'],
                'chat_bpb': result['metrics']['chat_bpb'],
                'degenerate_rate': result.get('autocomplete', {}).get('summary', {}).get('degenerate_rate'),
                'distinct_2': result.get('autocomplete', {}).get('summary', {}).get('mean_distinct_2'),
                'echo_rate': result.get('autocomplete', {}).get('summary', {}).get('mean_echo_rate'),
                'params': result['params'],
                'budget': result['training']['budget'],
                'runtime_minutes': result['training']['runtime_minutes'],
                'steps': result['training']['global_step'],
            }
        )
    return rows


def chat_ranking(results: list[dict[str, Any]], args, budget: str | None = None) -> list[dict[str, Any]]:
    """Independent secondary ranking; never blended into prose BPB."""
    eligible = [r for r in results if budget is None or r['training']['budget'] == budget]
    eligible = [r for r in eligible if r['chat_records'] and math.isfinite(r['metrics']['chat_bpb'])]
    if not eligible:
        return []
    eligible.sort(key=lambda r: r['metrics']['chat_bpb'])
    leader = eligible[0]
    epsilon = leader['metrics']['chat_bpb'] * args.equivalence
    rows = []
    for rank, result in enumerate(eligible, 1):
        pair = paired_bootstrap(result['chat_records'], leader['chat_records'], args.bootstrap, args.confidence, args.seed + rank)
        rows.append(
            {
                'rank': rank,
                'label': result['label'],
                'chat_bpb': result['metrics']['chat_bpb'],
                'prose_bpb': result['metrics']['prose_bpb'],
                'delta_vs_leader': pair['delta'],
                'delta_ci_low': pair['low'],
                'delta_ci_high': pair['high'],
                'p_beats_leader': pair['p_beats'],
                'equivalent_to_leader': pair['low'] <= epsilon,
            }
        )
    return rows


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Observed non-dominated one-hour models on prose and chat BPB."""
    return [
        row
        for row in rows
        if not any(
            other['bpb'] <= row['bpb']
            and other['chat_bpb'] <= row['chat_bpb']
            and (other['bpb'] < row['bpb'] or other['chat_bpb'] < row['chat_bpb'])
            for other in rows
        )
    ]


def fnum(value: Any, places: int = 4) -> str:
    return '—' if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) else f'{value:.{places}f}'


def render_report(payload: dict[str, Any], args) -> str:
    all_rows = payload['rankings']['all']
    one_hour = payload['rankings']['one_hour']
    chat_rows = payload['rankings']['chat_one_hour']
    pareto_rows = payload['rankings']['pareto_one_hour']
    confidence_pct = args.confidence * 100
    lines = [
        '# evaluate3 model-selection report',
        '',
        '## Decision',
        '',
    ]
    if one_hour:
        tied = [r for r in one_hour if r['equivalent_to_leader']]
        lead = one_hour[0]
        if len(one_hour) == 1:
            lines.append(
                f'**Only one checkpoint has a recorded one-hour budget: `{lead["label"]}` at {lead["bpb"]:.5f} bits/byte.** '
                'No comparative winner can be inferred from a one-model set.'
            )
        elif len(tied) == 1:
            lines.append(
                f'**Best one-hour experiment: `{lead["label"]}` at {lead["bpb"]:.5f} bits/byte.** '
                f'Its lead exceeds both sampling uncertainty and the {args.equivalence:.2%} practical-equivalence band.'
            )
        else:
            names = ', '.join(f'`{r["label"]}`' for r in tied)
            lines.append(
                f'**Observed one-hour leader: `{lead["label"]}` at {lead["bpb"]:.5f} bits/byte, '
                f'but the reliable decision is a {len(tied)}-way tie:** {names}. These models cannot be separated beyond the '
                f'{args.equivalence:.2%} practical-equivalence band at {confidence_pct:.0f}% confidence.'
            )
    else:
        lines.append('No checkpoints with recorded one-hour training budgets were found.')
    if all_rows:
        lines.extend(
            [
                '',
                f'The unrestricted observed leader is `{all_rows[0]["label"]}` ({all_rows[0]["budget"]}); '
                'it is not used to choose a one-hour recipe when its budget differs.',
            ]
        )

    if chat_rows:
        chat_tied = [r for r in chat_rows if r['equivalent_to_leader']]
        chat_names = ', '.join(f'`{r["label"]}`' for r in chat_tied)
        lines.extend(
            [
                '',
                '## Objective trade-off',
                '',
                f'For conditional chat-style targets, the observed one-hour leader is `{chat_rows[0]["label"]}` at '
                f'{chat_rows[0]["chat_bpb"]:.5f} BPB. Its {len(chat_tied)}-model statistical/practical equivalence group is '
                f'{chat_names}.',
                '',
                'The observed prose/chat Pareto frontier (no model is better on both columns) is:',
                '',
                '| model | historical prose BPB | chat-target BPB |',
                '|---|---:|---:|',
            ]
        )
        for row in pareto_rows:
            lines.append(f'| `{row["label"]}` | {row["bpb"]:.5f} | {row["chat_bpb"]:.5f} |')
        lines.extend(
            [
                '',
                'This is why eval3 does not publish one weighted score: choosing raw base-model period prose favours the old WSD/cosine '
                'pair, while choosing easiest later adaptation to clean dialogue favours the autoresearch2 frontier.',
            ]
        )

    lines.extend(
        [
            '',
            '## Why this decision is more reliable',
            '',
            'The primary metric is deterministic held-out historical-prose **bits per UTF-8 byte** (lower is better), '
            'which remains comparable across tokenizers. Confidence intervals resample whole documents, and every comparison uses '
            'the same resampled document IDs (paired bootstrap). Split A/B and late-context loss are stability diagnostics. '
            'Conditional chat-target loss is reported separately and never folded into an arbitrary composite.',
            '',
            'Eval3 also runs the complete 28-prompt `sample_autocomplete.sh` battery offline and saves every continuation in the '
            'samples report. Its numbers only detect surface failures such as loops, local echo, or vocabulary collapse. They do '
            '**not** claim to measure meaning: on the 21 pre-existing sample files, a naive diversity-minus-repetition rank '
            'correlated about −0.38 with held-out quality because undertrained nonsense is often lexically diverse. Human reading '
            'remains the semantic prose test.',
            '',
            f'A challenger is called worse only when the lower end of its paired {confidence_pct:.0f}% interval is more than '
            f"{args.equivalence:.2%} of the leader's BPB above the leader. Otherwise the result is reported as a tie. This avoids "
            'manufacturing a winner from negligible differences.',
            '',
            'Important limit: these intervals measure uncertainty from the held-out document sample, not training-seed variance. '
            'Moreover, this repository has repeatedly used the same 200 documents for experiment selection, so the set is now a '
            'validation benchmark rather than a pristine final test. Confirm any recipe chosen here with multiple training seeds '
            'and a newly frozen, never-consulted test set before making a publication claim.',
            '',
            '## One-hour experiments',
            '',
            '| rank | model | prose BPB | paired Δ [CI] | split A / B | chat BPB | degenerate samples | steps | params | decision |',
            '|---:|---|---:|---:|---:|---:|---:|---:|---:|---|',
        ]
    )
    for row in one_hour:
        decision = 'leader' if row['rank'] == 1 else ('tie' if row['equivalent_to_leader'] else 'worse')
        degenerate = fnum(None if row['degenerate_rate'] is None else 100 * row['degenerate_rate'], 1)
        lines.append(
            f'| {row["rank"]} | `{row["label"]}` | {row["bpb"]:.5f} | {row["delta_vs_leader"]:+.5f} '
            f'[{row["delta_ci_low"]:+.5f}, {row["delta_ci_high"]:+.5f}] | {row["split_a_bpb"]:.4f} / '
            f'{row["split_b_bpb"]:.4f} | {fnum(row["chat_bpb"])} | {degenerate}% | {row["steps"] or "—"} | '
            f'{row["params"] / 1e6:.1f}M | **{decision}** |'
        )

    lines.extend(
        [
            '',
            '## All evaluated models (budget-unrestricted)',
            '',
            f'| rank | model | budget | prose BPB | {confidence_pct:.0f}% marginal CI | chat BPB |',
            '|---:|---|---:|---:|---:|---:|',
        ]
    )
    for row in all_rows:
        lines.append(
            f'| {row["rank"]} | `{row["label"]}` | {row["budget"]} | {row["bpb"]:.5f} | '
            f'[{row["ci_low"]:.5f}, {row["ci_high"]:.5f}] | {fnum(row["chat_bpb"])} |'
        )

    if payload.get('skipped') or payload.get('failures'):
        lines.extend(['', '## Skipped / failed', ''])
        for item in payload.get('skipped', []) + payload.get('failures', []):
            lines.append(f'- `{item.get("path")}` — {item.get("reason")}')
    lines.extend(
        [
            '',
            '## Reproducibility',
            '',
            f'- Held-out data SHA-256: `{payload["settings"]["heldout_sha256"]}` ({payload["settings"]["docs"]} documents)',
            f'- Bootstrap: {args.bootstrap:,} paired document resamples, seed {args.seed}',
            f'- Device/dtype: `{payload["environment"]["device"]}` / `{payload["environment"]["dtype"]}`',
            f'- PyTorch / Transformers: `{payload["environment"]["torch"]}` / `{payload["environment"]["transformers"]}`',
            '- Raw per-document bits and byte counts are retained in the adjacent JSON, so the ranking can be audited without '
            'loading the models again.',
            '',
        ]
    )
    return '\n'.join(lines)


def render_samples(results: list[dict[str, Any]]) -> str:
    lines = [
        '# evaluate3 offline autocomplete samples',
        '',
        'These use the prompts from `sample_autocomplete.sh`. Read them directly; the mechanical statistics beside each sample '
        'only flag surface collapse and are not semantic judgments.',
        '',
    ]
    for result in sorted(results, key=lambda r: r['metrics']['prose_bpb']):
        auto = result.get('autocomplete') or {}
        if not auto:
            continue
        summary = auto['summary']
        lines.extend(
            [
                f'## {result["label"]}',
                '',
                f'Prose BPB {result["metrics"]["prose_bpb"]:.5f}; mode {auto["mode"]}; seed {auto["seed"]}; degeneration flags '
                f'{summary["degenerate_rate"]:.1%}; mean distinct-2 {summary["mean_distinct_2"]:.3f}; '
                f'mean echo {summary["mean_echo_rate"]:.3f}.',
                '',
            ]
        )
        for sample in auto['samples']:
            flag = ' **[SURFACE FAILURE FLAG]**' if sample['degenerate'] else ''
            lines.extend(
                [
                    f'**PROMPT:** {sample["prompt"]}{flag}',
                    '',
                    sample['prompt'] + (' ' if sample['continuation'] else '') + sample['continuation'],
                    '',
                    f'_distinct-2 {sample["distinct_2"]:.3f}; echo {sample["echo_rate"]:.3f}; '
                    f'longest loop {sample["longest_loop_words"]} words_',
                    '',
                    '---',
                    '',
                ]
            )
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'targets', nargs='*', type=Path, default=[SCRIPT_DIR / 'checkpoints'], help='Checkpoint, experiment, or collection directories.'
    )
    parser.add_argument('--heldout', type=Path, default=EVAL_DATA / 'heldout.jsonl')
    parser.add_argument('--chat-data', type=Path, default=EVAL_DATA / 'chat_sample.jsonl')
    parser.add_argument('--tokenizer', type=Path, default=None, help='Force one tokenizer for every checkpoint.')
    parser.add_argument('--docs', type=int, default=200)
    parser.add_argument('--chat-docs', type=int, default=200)
    parser.add_argument('--max-tokens', type=int, default=1024, help='Maximum total model input length for prose.')
    parser.add_argument('--chat-max-tokens', type=int, default=768)
    parser.add_argument('--no-chat', action='store_true', help='Skip the secondary conditional chat-target diagnostic.')
    parser.add_argument(
        '--generation-mode',
        choices=('none', 'greedy', 'sample'),
        default='sample',
        help='Offline autocomplete mode; sample matches human review better, greedy is stricter for loops.',
    )
    parser.add_argument(
        '--generation-prompts', type=int, default=len(AUTOCOMPLETE_PROMPTS), help='Number of sample_autocomplete.sh prompts to use.'
    )
    parser.add_argument('--generation-tokens', type=int, default=80, help='New tokens per autocomplete prompt.')
    parser.add_argument('--generation-batch-size', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--top-k', type=int, default=25)
    parser.add_argument('--include-checkpoints', action='store_true', help='Include checkpoint-N directories during recursive discovery.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--dtype', choices=('auto', 'float32', 'float16', 'bfloat16'), default='auto')
    parser.add_argument('--threads', type=int, default=None, help='Set PyTorch CPU worker threads.')
    parser.add_argument('--bootstrap', type=int, default=10_000)
    parser.add_argument('--confidence', type=float, default=0.95)
    parser.add_argument('--equivalence', type=float, default=0.001, help='Relative BPB difference treated as practically equivalent.')
    parser.add_argument('--seed', type=int, default=20260822)
    parser.add_argument(
        '--out',
        type=Path,
        default=SCRIPT_DIR / 'eval3_results' / 'evaluate3.json',
        help='Resumable JSON output; Markdown is written beside it.',
    )
    parser.add_argument('--force', action='store_true', help='Ignore matching cached checkpoint results.')
    args = parser.parse_args()

    if args.docs < 2 or args.chat_docs < 0 or args.bootstrap < 100:
        parser.error('--docs must be >=2, --chat-docs >=0, and --bootstrap >=100')
    if not 0 <= args.generation_prompts <= len(AUTOCOMPLETE_PROMPTS) or args.generation_tokens < 1 or args.generation_batch_size < 1:
        parser.error('invalid autocomplete prompt/token/batch count')
    if not 0 < args.confidence < 1 or args.equivalence < 0:
        parser.error('--confidence must be in (0,1) and --equivalence must be non-negative')
    if args.threads:
        torch.set_num_threads(args.threads)

    heldout_path = args.heldout.resolve()
    chat_path = args.chat_data.resolve()
    heldout = load_text_items(heldout_path, args.docs)
    chats = [] if args.no_chat or args.chat_docs == 0 else load_chat_items(chat_path, args.chat_docs)
    settings = cache_settings(args, file_hash(heldout_path), None if not chats else file_hash(chat_path))
    checkpoints, skipped = resolve_targets(args.targets, args.include_checkpoints)
    if not checkpoints:
        sys.exit('error: no Hugging Face checkpoints with config.json and weights found')

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    if device.type == 'cpu' and dtype in (torch.float16, torch.bfloat16):
        print('warning: reduced precision on CPU may be unsupported or slower', file=sys.stderr)

    out = args.out.resolve()
    cached: dict[str, dict[str, Any]] = {}
    if out.is_file() and not args.force:
        try:
            old = json.loads(out.read_text())
            if old.get('settings') == settings:
                cached = {r['fingerprint']: r for r in old.get('results', [])}
        except Exception as exc:
            print(f'warning: could not read cache {out}: {exc}', file=sys.stderr)

    payload: dict[str, Any] = {
        'schema_version': VERSION,
        'settings': settings,
        'environment': {
            'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'python': platform.python_version(),
            'torch': torch.__version__,
            'transformers': __import__('transformers').__version__,
            'device': str(device),
            'dtype': str(dtype).removeprefix('torch.'),
        },
        'results': [],
        'skipped': skipped,
        'failures': [],
    }
    print(f'evaluate3: {len(checkpoints)} models, {len(heldout)} prose docs, {len(chats)} chat docs on {device}/{dtype}')
    for index, checkpoint in enumerate(checkpoints, 1):
        print(f'[{index}/{len(checkpoints)}] {checkpoint}', end='', flush=True)
        try:
            tok_dir = tokenizer_path(checkpoint, args.tokenizer)
            fingerprint = model_fingerprint(checkpoint, tok_dir)
            if fingerprint in cached:
                result = cached[fingerprint]
                # Provenance files are tiny and may expose resumed segments;
                # refresh them even when expensive model scores are cached.
                result['training'] = training_metadata(checkpoint, result['params'])
                print(' cached')
            else:
                result = evaluate_checkpoint(checkpoint, args, heldout, chats, device, dtype)
                print(f' -> {result["metrics"]["prose_bpb"]:.5f} BPB ({result["seconds"]:.1f}s)')
            payload['results'].append(result)
            atomic_json(out, payload)
        except KeyboardInterrupt:
            atomic_json(out, payload)
            raise
        except Exception as exc:
            print(f' FAILED: {exc}')
            payload['failures'].append({'path': str(checkpoint), 'reason': f'{type(exc).__name__}: {exc}'})
            atomic_json(out, payload)

    if not payload['results']:
        sys.exit('error: every checkpoint failed')
    payload['rankings'] = {
        'all': ranking(payload['results'], args),
        'one_hour': ranking(payload['results'], args, budget='~1h'),
        'two_hour': ranking(payload['results'], args, budget='~2h'),
        'long_run': ranking(payload['results'], args, budget='>2h'),
    }
    payload['rankings']['chat_one_hour'] = chat_ranking(payload['results'], args, budget='~1h')
    payload['rankings']['pareto_one_hour'] = pareto_front(payload['rankings']['one_hour'])
    atomic_json(out, payload)
    report = render_report(payload, args)
    md_path = out.with_suffix('.md')
    md_path.write_text(report, encoding='utf-8')
    samples_path = out.with_name(out.stem + '-samples.md')
    samples_path.write_text(render_samples(payload['results']), encoding='utf-8')
    print(f'\nJSON: {out}\nReport: {md_path}\nSamples: {samples_path}\n')
    print('\n'.join(report.splitlines()[:12]))


if __name__ == '__main__':
    main()

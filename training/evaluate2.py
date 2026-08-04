#!/usr/bin/env python3
"""
evaluate2.py - Answers the question: "is it baked?" for tiny vintage LLMs.

A self-contained evaluation suite, a metric that tries to measure how well
trained is a vintage model.
Point it at one checkpoint, or at a folder full of them:

  python evaluate2.py                            # latest in ./checkpoints
  python evaluate2.py path/to/checkpoint-22944   # one checkpoint
  python evaluate2.py path/to/Vintage1           # every checkpoint inside -> curve + best pick
  python evaluate2.py Vintage1 --judge TypeWriter/base-v2   # add the big-model judge

The report is written for a non-data-scientist: every number comes with a plain
sentence, the model is placed on an absolute reference ladder measured on real
models (untrained noise ... best-in-class ... golden standard), and the
verdict says outright whether the money spent so far bought a good model.

Zero required data files beyond the two small samples in eval_data/ (bundled):
  eval_data/heldout.jsonl      200 held-out period documents (the ladder anchors
                               below were measured on exactly these docs)
  eval_data/chat_sample.jsonl  200 (context, target) chat turns for the
                               fine-tunability score
Override with --heldout / --chat-data. Logic/trap/generation probes are embedded
in this file; --probes FILE swaps them for another era or domain.
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent if Path(__file__).parent != Path('.') else Path.cwd()
EVAL_DATA = SCRIPT_DIR / 'eval_data'
LOG2E = 1.4426950408889634

# ============================================================================
# REFERENCE LADDER - the "brutally honest" part.
#
# Every anchor below was MEASURED, with this same code path (same 200 held-out
# docs, max 1024 tokens/doc), on real models in this project (Aug 2026):
#
#   bpb    model                                       tokens seen
#   3.5    untrained 32k-vocab model (uniform guessing)      0
#   1.328  Vintage2 500M @ step 2000                      ~0.7B
#   1.192  Vintage2 500M final (0.9 epoch)                ~4.7B
#   1.099  Vintage1 341M best base checkpoint             (best sub-1B here)
#   1.270* TypeWriter 7B (* domain-shifted: different corpus, judged elsewhere)
#
# The bake score interpolates through these points. It is calibrated for tiny
# (<=1B) models on 19th-century English; a modern-web model would need its own
# anchors. 100 means "as good as the best sub-1B model we have ever measured on
# this data, plus a little headroom" - NOT "as good as a 7B".
# ============================================================================

BPB_LADDER = [  # (bits/byte on eval_data/heldout.jsonl, bake points)
    (3.50, 0),  # untrained: pure noise
    (2.00, 20),  # word-salad: real words, no sentences
    (1.50, 40),  # broken prose: sentences form, meaning drifts within a line
    (1.33, 55),  # early training (measured: 500M @ 0.7B tokens)
    (1.19, 70),  # solid but visibly undertrained (measured: 500M @ 4.7B tokens)
    (1.10, 85),  # best tiny model measured on this data (341M, Vintage1)
    (1.05, 92),  # a little beyond the best we have seen at this scale
    (0.95, 100),  # estimated ceiling for sub-1B on this corpus
]

LOGIC_BANDS = [  # (accuracy on the 40 built-in minimal pairs, label)
    (0.55, 'coin-flip - the model does not prefer sense over nonsense'),
    (0.65, 'weak - style without understanding'),
    (0.75, 'normal for a good sub-1B model (Vintage1 scores 0.70-0.78)'),
    (0.85, 'unusually strong for this size'),
    (1.01, '7B territory (TypeWriter scores 0.88-0.93)'),
]

CHAT_LADDER = [  # bits/byte on eval_data/chat_sample.jsonl -> chat-readiness pts
    (1.60, 0),
    (1.26, 40),
    (1.10, 65),
    (0.93, 85),
    (0.82, 95),
    (0.73, 100),
]

# Composite weights (rho vs training step measured on Vintage2's 11 checkpoints)
WEIGHTS = {
    'bpb': 0.50,  # rho -1.000  held-out bits/byte
    'logic': 0.25,  # rho +0.64 (margin)  prefers coherent over incoherent
    'chat': 0.15,  # rho -0.99  fine-tunability for chat
    'hygiene': 0.10,  # rho -0.45/-0.62  greedy looping + punctuation
}

CHINCHILLA_TOKENS_PER_PARAM = 20  # compute-optimal rule of thumb

# ============================================================================
# BUILT-IN PROBES (Victorian / 19th-century English).
# Swap with --probes FILE where FILE is a JSON object with any of the keys
# "logic_items", "trap_pairs", "gen_stems" in the same shapes as below.
# ============================================================================

# (category, context, coherent continuation, incoherent continuation)
LOGIC_ITEMS = [
    (
        'physical',
        'He set the kettle upon the fire, and after some minutes the water',
        ' began to boil, and steam issued from the spout.',
        ' began to freeze, and ice issued from the spout.',
    ),
    (
        'physical',
        'The stone was dropped from the top of the tower, and it',
        ' fell swiftly to the ground below.',
        ' rose swiftly to the clouds above.',
    ),
    (
        'physical',
        'She left the milk standing in the sun for two days, and when she returned it',
        ' had turned sour and was fit only to be thrown away.',
        ' had turned fresh and was sweeter than when she left it.',
    ),
    (
        'physical',
        'The blacksmith heated the iron in the forge until it',
        ' glowed red and could be beaten into shape.',
        ' grew cold and could be beaten into shape.',
    ),
    (
        'physical',
        'He held the candle to the paper, and the paper',
        ' caught fire and was quickly consumed.',
        ' grew damp and was quickly frozen.',
    ),
    (
        'physical',
        'The ship sprang a leak below the water-line, and the hold',
        ' began to fill with water, so the men worked the pumps.',
        ' began to fill with air, so the men worked the pumps.',
    ),
    (
        'physical',
        'A heavy frost fell in the night, and in the morning the pond',
        ' was covered with ice, and the children slid upon it.',
        ' was covered with dust, and the children swam in it.',
    ),
    (
        'physical',
        'He carried the lamp into the cellar, for without it the cellar was',
        ' too dark for him to see the steps.',
        ' too bright for him to see the steps.',
    ),
    (
        'causal',
        'The harvest failed for the second year together, and consequently the price of bread',
        ' rose so high that the poor could scarcely buy it.',
        ' fell so low that the poor bought more than they wished.',
    ),
    (
        'causal',
        'He had not slept for two nights, and therefore at the meeting he',
        ' could scarcely keep his eyes open.',
        ' was livelier than any man in the room.',
    ),
    (
        'causal',
        'The bridge had been carried away by the flood, so the travellers',
        ' were obliged to seek a ford some miles upstream.',
        ' crossed it without difficulty and continued their journey.',
    ),
    (
        'causal',
        'Since the letter was never posted, his brother',
        ' remained wholly ignorant of the matter.',
        ' replied to it by the following morning.',
    ),
    (
        'causal',
        'The physician found the wound to be badly inflamed, and he therefore',
        ' ordered it to be cleansed and dressed afresh.',
        ' pronounced the man in perfect health and dismissed him.',
    ),
    (
        'causal',
        'A long drought had parched the fields, and the farmers',
        ' looked anxiously for rain.',
        ' looked anxiously for a further want of rain.',
    ),
    (
        'causal',
        'He staked his whole fortune upon the venture, and when the ship was lost he',
        ' was reduced to absolute poverty.',
        ' found himself richer than he had ever been.',
    ),
    (
        'causal',
        'The window had been left open all night in December, and in the morning the room',
        ' was bitterly cold.',
        ' was uncommonly warm.',
    ),
    (
        'social',
        'Being invited to dine at a house of higher station, he was careful to',
        ' arrive punctually and dressed with propriety.',
        ' arrive some hours late and in his working clothes.',
    ),
    (
        'social',
        'A letter to a bishop should properly be addressed',
        ' to His Lordship, with the respect due to his office.',
        ' to my dear old fellow, with the familiarity due to a schoolmate.',
    ),
    (
        'social',
        'His neighbour having lost her husband that week, he thought it right to',
        ' send a letter of condolence and offer what help he could.',
        ' send a letter of congratulation and invite her to a ball.',
    ),
    (
        'social',
        'The young man wished to marry, and as a matter of duty he first',
        " sought the consent of the lady's father.",
        " sought the consent of the lady's coachman.",
    ),
    (
        'social',
        'Having given his word before witnesses, he held himself',
        ' bound in honour to perform it.',
        ' at perfect liberty to forget it entirely.',
    ),
    (
        'social',
        'The servant announced a visitor at an hour past midnight, which the household thought',
        ' a most inconvenient and irregular time to call.',
        ' the usual and proper hour for paying calls.',
    ),
    (
        'social',
        'He was called as a witness, and being sworn upon the book he was bound to',
        ' speak nothing but the truth.',
        ' say whatever best served his own interest.',
    ),
    (
        'quantity',
        'The infant was but three weeks old, and therefore he',
        ' could neither walk nor speak.',
        ' walked to the village and argued upon politics.',
    ),
    (
        'quantity',
        'The distance was upwards of two hundred miles, and travelling by coach it occupied',
        ' the better part of three days.',
        ' rather less than four minutes.',
    ),
    (
        'quantity',
        'He earned eighteen shillings in the week, out of which the rent alone was twelve; so there remained',
        ' but six shillings for all else.',
        ' but nine pounds for all else.',
    ),
    (
        'quantity',
        'The room measured twelve feet by ten, so it was',
        ' too small to seat a hundred persons.',
        ' large enough to seat a hundred persons with ease.',
    ),
    (
        'quantity',
        'The child was of ordinary growth for seven years, and stood',
        ' something under four feet in height.',
        ' something above nine feet in height.',
    ),
    (
        'quantity',
        'A gallon of the liquid was required, but he had brought only a pint, which was',
        ' far short of what was wanted.',
        ' a good deal more than was wanted.',
    ),
    (
        'coref',
        'The master struck the dog with his stick, and the poor creature',
        ' ran howling from the yard.',
        ' laid down the stick and apologised.',
    ),
    (
        'coref',
        'When the doctor came to the sick woman, he found that she',
        ' had grown much weaker since his last visit.',
        ' had grown much weaker since her last visit to himself.',
    ),
    (
        'coref',
        'The mother gave the child a shilling, and he ran at once to the shop and',
        ' spent it upon sweets.',
        ' received it from the shopkeeper as wages.',
    ),
    (
        'coref',
        'John lent his umbrella to Thomas, and it rained; so Thomas',
        ' was kept dry and John was drenched.',
        ' was drenched and John was kept dry by the umbrella he had lent away.',
    ),
    (
        'coref',
        'The clerk handed the ledger to his employer, who opened it and',
        ' began to examine the accounts.',
        " began to examine the clerk's handwriting upon his own hand.",
    ),
    (
        'physical',
        'The seed was sown in April, and by the end of the summer it',
        ' had grown into a tall plant bearing grain.',
        ' had grown into a tall plant bearing coal.',
    ),
    (
        'causal',
        'The fire had been left unguarded, and the sparks falling upon the thatch',
        ' set the roof alight.',
        ' extinguished the roof entirely.',
    ),
    (
        'social',
        'A gentleman in mourning for his father would properly appear',
        ' in black, and decline all gaiety for a season.',
        ' in bright colours, and open the dancing himself.',
    ),
    ('quantity', 'The tide rises and falls twice in the course of', ' a single day.', ' a single century.'),
    ('physical', 'He poured the water upon the quicklime, and it', ' grew hot and hissed.', ' grew cool and silent as before.'),
    (
        'causal',
        'The horse had cast a shoe upon the stony road, and so the rider',
        ' led him slowly to the nearest smith.',
        ' galloped him the faster for the remaining twenty miles.',
    ),
]

# (trap stem, trap phrase, matched control stem, control phrase)
# The trap phrase is post-1900 knowledge; the control is the same sentence shape
# with a period-legitimate word. shock = bits(trap) - bits(control): a period
# model should find the modern word much MORE expensive. Negative shock on a
# pair means modern text leaked into training.
TRAP_PAIRS = [
    (
        'The aeroplane, which now carries passengers across the Atlantic, was',
        'aeroplane',
        'The steam locomotive, which now carries passengers across the country, was',
        'steam locomotive',
    ),
    (
        'The doctor prescribed a course of penicillin, and within',
        'penicillin',
        'The doctor prescribed a course of quinine, and within',
        'quinine',
    ),
    (
        'Every evening the family gathers before the television set to watch',
        'television set',
        'Every evening the family gathers before the fire to hear',
        'fire',
    ),
    (
        'The computer in the corner of the laboratory calculated the result in',
        'computer',
        'The telegraph in the corner of the office transmitted the message in',
        'telegraph',
    ),
    (
        'After the atomic bomb fell upon the city, the survivors',
        'atomic bomb',
        'After the cannonade fell upon the city, the survivors',
        'cannonade',
    ),
    (
        'She telephoned him from her motor car, using the wireless in her',
        'motor car',
        'She telegraphed him from her carriage, using the wire in her',
        'carriage',
    ),
]

# Completion stems for the generation probe (base models continue text; they do
# not answer questions). Deliberately spread across registers.
GEN_STEMS = [
    'LONDON, Tuesday. — The committee appointed to inquire into the condition of the',
    'A melancholy accident occurred on Thursday last at the works of Messrs. Harding and',
    'Brethren, the text which I have chosen for our consideration this morning is taken from',
    'March 12th. — Rose early, the frost being very sharp upon the windows. After breakfast I',
    'My dear Sister, — It is with no small shame that I take up my pen after so long a',
    'The prisoner, a labourer of some five-and-thirty years, was indicted for having',
    'The experiment was repeated with a coil of finer wire, and the deflection of the needle',
    'To make a plain seed cake. — Take one pound of flour, well dried before the fire, and',
    'The inn stood at the meeting of four roads, and it was there, upon a night of driving rain, that',
    'MANCHESTER, a city and municipal borough in the county of Lancaster, situated upon the',
    'COTTON. The plant from which this important article of commerce is obtained is',
    'The gardener who would have early peas must, in the first week of February,',
]


# ============================================================================
# Model loading / checkpoint resolution (mirrors evaluate.py conventions)
# ============================================================================


def resolve_targets(target: Path) -> list[Path]:
    """One checkpoint dir -> [it]; a folder of checkpoints -> all, step-sorted."""
    if (target / 'config.json').exists():
        return [target]

    def step_of(p):
        m = re.search(r'(\d+)$', p.name)
        return int(m.group(1)) if m else 10**12  # 'final' etc. sort last

    ckpts = sorted((p for p in target.iterdir() if p.is_dir() and (p / 'config.json').exists()), key=step_of)
    if not ckpts:
        sys.exit(f'error: {target} is neither a checkpoint (no config.json) nor a folder containing checkpoints.')
    return ckpts


def resolve_tokenizer(ckpt: Path, cli: Path | None) -> Path:
    if cli:
        return cli
    for cand in (ckpt, ckpt.parent / 'tokenizers', ckpt.parent, ckpt.parent.parent / 'tokenizers'):
        if (cand / 'tokenizer.json').exists():
            return cand
    sys.exit(f'error: no tokenizer.json found near {ckpt}. Pass --tokenizer DIR.')


def load_model(ckpt: Path, tok_dir: Path, device, dtype):
    tok = AutoTokenizer.from_pretrained(tok_dir)
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=dtype)
    model.config.use_cache = True
    return tok, model.to(device).eval()


def free_model(model):
    """Release the weights' device memory even if callers still hold references."""
    try:
        model.to('meta')
    except Exception:
        pass
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def bos_id(tok, model):
    b = model.config.bos_token_id
    return b if b is not None else tok.bos_token_id


# ============================================================================
# Metrics (each returns plain floats; all validated, see module docstring)
# ============================================================================


@torch.no_grad()
def bits_per_byte(tok, model, texts, max_tokens=1024) -> float:
    """Held-out loss in bits per UTF-8 byte of the scored span. Lower is better.
    Byte- not token-normalised, so it is comparable across different tokenizers."""
    bos = bos_id(tok, model)
    total_bits, total_bytes = 0.0, 0
    for text in texts:
        ids = tok(text, add_special_tokens=False).input_ids[:max_tokens]
        if len(ids) < 8:
            continue
        nbytes = len(tok.decode(ids).encode('utf-8'))
        inp = torch.tensor([[bos] + ids], device=model.device)
        logits = model(inp).logits[:, :-1].float()
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), inp[:, 1:].reshape(-1), reduction='sum')
        total_bits += nll.item() * LOG2E
        total_bytes += nbytes
    return total_bits / total_bytes if total_bytes else float('nan')


@torch.no_grad()
def span_bits(tok, model, prefix_plus_span: str, prefix: str) -> float:
    """Bits/byte the model spends on the part of the text after `prefix`.
    The prefix is cut at its last non-space char so the span owns the leading
    space -- these tokenizers glue spaces onto the FOLLOWING word, and cutting
    mid-space makes the two tokenisations non-nested (a bug we hit in sniff/)."""
    bos = bos_id(tok, model)
    prefix = prefix.rstrip()
    span = prefix_plus_span[len(prefix) :]
    pre = tok(prefix, add_special_tokens=False).input_ids
    full = tok(prefix + span, add_special_tokens=False).input_ids
    if len(full) <= len(pre) or full[: len(pre)] != pre:
        return float('nan')
    inp = torch.tensor([[bos] + full], device=model.device)
    logits = model(inp).logits[:, :-1].float()
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), inp[:, 1:].reshape(-1), reduction='none')
    return nll[len(pre) :].sum().item() * LOG2E / len(span.encode('utf-8'))


def run_logic(tok, model, items) -> dict:
    """Forced choice between a coherent and an incoherent continuation.
    Chance = 0.50. Margin = mean bits/byte by which coherence is cheaper."""
    correct, margins, per_cat = 0, [], {}
    for cat, ctx, good, bad in items:
        gb = span_bits(tok, model, ctx.rstrip() + good, ctx)
        bb = span_bits(tok, model, ctx.rstrip() + bad, ctx)
        if math.isnan(gb) or math.isnan(bb):
            continue
        ok = gb < bb
        correct += ok
        margins.append(bb - gb)
        per_cat.setdefault(cat, []).append(ok)
    n = len(margins)
    return {
        'acc': correct / n if n else float('nan'),
        'margin': sum(margins) / n if n else float('nan'),
        'n': n,
        'per_category': {c: round(sum(v) / len(v), 3) for c, v in sorted(per_cat.items())},
    }


def run_traps(tok, model, pairs) -> dict:
    """Anachronism shock: bits on a post-1900 phrase minus a matched period
    phrase in the same sentence shape. Positive = period-bounded."""
    shocks, worst = [], None
    for tstem, tph, cstem, cph in pairs:
        ti = tstem.index(tph)
        ci = cstem.index(cph)
        tb = span_bits(tok, model, tstem[: ti + len(tph)], tstem[:ti])
        cb = span_bits(tok, model, cstem[: ci + len(cph)], cstem[:ci])
        if math.isnan(tb) or math.isnan(cb):
            continue
        s = tb - cb
        shocks.append(s)
        if worst is None or s < worst[1]:
            worst = (tph, s)
    return {
        'mean_shock': sum(shocks) / len(shocks) if shocks else float('nan'),
        'min_shock': min(shocks) if shocks else float('nan'),
        'n_leaked': sum(1 for s in shocks if s <= 0),
        'n': len(shocks),
        'worst_pair': worst[0] if worst else None,
    }


_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _loop_len(toks, max_period=12) -> int:
    """Length of the longest immediately-repeating block of words. 0 = clean."""
    best = 0
    for period in range(1, max_period + 1):
        run = 0
        for i in range(len(toks) - period):
            if toks[i] == toks[i + period]:
                run += 1
                if run >= period:
                    best = max(best, run + period)
            else:
                run = 0
    return best


def _punct_issues_p100(text) -> float:
    issues = abs(text.count('(') - text.count(')')) + text.count('"') % 2
    issues += len(re.findall(r'[,;:]{2,}|\.{4,}|\s,', text))
    return issues * 100.0 / max(1, len(_WORD.findall(text.lower())))


@torch.no_grad()
def run_generation(tok, model, stems, max_new=150, seed=0) -> dict:
    """Greedy + one sampled continuation per stem. Greedy exposes degeneracy
    (an undertrained model falls into loops); sampled feeds the judge and the
    examples section. Returns metrics plus the raw texts."""
    bos = bos_id(tok, model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = {'greedy': [], 'sampled': []}
    for mode in ('greedy', 'sampled'):
        for stem in stems:
            ids = torch.tensor([[bos] + tok(stem, add_special_tokens=False).input_ids], device=model.device)
            torch.manual_seed(seed)
            kw = dict(do_sample=False) if mode == 'greedy' else dict(do_sample=True, temperature=0.8, top_p=0.95)
            res = model.generate(ids, max_new_tokens=max_new, pad_token_id=pad, **kw)
            out[mode].append(tok.decode(res[0, ids.shape[1] :], skip_special_tokens=True))
    g_words = [_WORD.findall(t.lower()) for t in out['greedy']]
    return {
        'greedy_loop_len': sum(_loop_len(w) for w in g_words) / len(g_words),
        'punct_issues_p100': sum(_punct_issues_p100(t) for t in out['sampled']) / len(out['sampled']),
        'greedy_texts': out['greedy'],
        'sampled_texts': out['sampled'],
    }


@torch.no_grad()
def run_chat_fit(tok, model, pairs, max_tokens=768) -> float:
    """Bits/byte on the assistant side of real chat turns, given the context.
    Approximates how cheaply the base model could be fine-tuned into a chatbot."""
    bos = bos_id(tok, model)
    bits, nbytes = 0.0, 0
    for ctx, tgt in pairs:
        cid = tok(ctx, add_special_tokens=False).input_ids
        tid = tok(tgt, add_special_tokens=False).input_ids
        if len(tid) < 4:
            continue
        budget = max_tokens - min(len(tid), max_tokens // 2)
        cid = cid[-budget:]
        tid = tid[: max_tokens - len(cid)]
        inp = torch.tensor([[bos] + cid + tid], device=model.device)
        logits = model(inp).logits[:, :-1].float()
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), inp[:, 1:].reshape(-1), reduction='none')
        bits += nll[len(cid) :].sum().item() * LOG2E
        nbytes += len(tok.decode(tid).encode('utf-8'))
    return bits / nbytes if nbytes else float('nan')


@torch.no_grad()
def run_judge(judge_tok, judge_model, texts, real_texts, max_tokens=320) -> dict:
    """Big-model judge. The score is NOT "lower surprisal = better" (degenerate
    loops are cheap to predict; that ranking is provably backwards, rho +0.86
    vs training step). The score is |judge bpb on generations - judge bpb on
    REAL period text|: good text costs the judge the same as genuine prose."""
    real = bits_per_byte(judge_tok, judge_model, real_texts, max_tokens=max_tokens)
    gen = bits_per_byte(judge_tok, judge_model, [t for t in texts if len(t.split()) >= 20], max_tokens=max_tokens)
    return {'judge_bpb_real': real, 'judge_bpb_gen': gen, 'judge_dev': abs(gen - real)}


# ============================================================================
# Provenance: how much compute has this checkpoint actually eaten?
# ============================================================================


def training_provenance(ckpt: Path, n_params: int) -> dict:
    """Estimate tokens seen from trainer_state.json's total_flos (flops =~
    6 * params * tokens for a decoder-only transformer). Robust to unknown
    batch sizes; absent trainer_state -> unknown."""
    out = {'tokens_seen': None, 'tokens_per_param': None, 'global_step': None, 'epoch': None, 'note': ''}
    ts = ckpt / 'trainer_state.json'
    if not ts.exists():
        out['note'] = 'no trainer_state.json in the checkpoint - tokens seen unknown'
        return out
    try:
        d = json.loads(ts.read_text())
    except Exception:
        out['note'] = 'trainer_state.json unreadable'
        return out
    out['global_step'] = d.get('global_step')
    out['epoch'] = d.get('epoch')
    flos = d.get('total_flos') or 0
    if flos and n_params:
        tokens = flos / (6.0 * n_params)
        out['tokens_seen'] = tokens
        out['tokens_per_param'] = tokens / n_params
    m = re.search(r'(\d+)$', ckpt.name)
    if m and out['global_step'] and int(m.group(1)) != out['global_step']:
        # A restart without --resume keeps the weights but zeroes the trainer
        # counters, so total_flos covers only the newest segment: the token
        # estimate is a LOWER BOUND, and "undertrained by construction" claims
        # based on it would be unsafe.
        out['tokens_lower_bound'] = True
        out['note'] = (
            f'folder name says step {m.group(1)} but trainer_state says '
            f'{out["global_step"]} - training was restarted, so the '
            f'tokens-seen estimate covers only the latest segment '
            f'(treat it as a lower bound)'
        )
    return out


# ============================================================================
# Scoring + verdict
# ============================================================================


def interp(x: float, ladder) -> float:
    """Piecewise-linear map through measured (value, points) anchors. The ladder
    is descending in value (lower bpb = more points)."""
    if math.isnan(x):
        return float('nan')
    if x >= ladder[0][0]:
        return float(ladder[0][1])
    if x <= ladder[-1][0]:
        return float(ladder[-1][1])
    for (x1, y1), (x2, y2) in zip(ladder, ladder[1:]):
        if x2 <= x <= x1:
            return y1 + (y2 - y1) * (x1 - x) / (x1 - x2)
    return float('nan')


def logic_points(acc: float) -> float:
    if math.isnan(acc):
        return float('nan')
    return max(0.0, min(100.0, (acc - 0.5) / (0.92 - 0.5) * 100.0))


def hygiene_points(loop_len: float, punct: float) -> float:
    # loop 0 words = 100 pts, 80+ = 0. punct 0/100w = 100 pts, 1+ = 0.
    lp = max(0.0, 100.0 - loop_len * 1.25)
    pp = max(0.0, 100.0 - punct * 100.0)
    return 0.7 * lp + 0.3 * pp


def bake_score(parts: dict) -> float:
    total, wsum = 0.0, 0.0
    for k, w in WEIGHTS.items():
        v = parts.get(k)
        if v is not None and not math.isnan(v):
            total += w * v
            wsum += w
    return total / wsum if wsum else float('nan')


def verdict_text(score: float, prov: dict) -> tuple[str, str]:
    """(tier, blunt paragraph)."""
    if math.isnan(score):
        return 'UNSCORED', 'Not enough metrics ran to produce a verdict.'
    if score >= 85:
        tier, body = (
            'BAKED',
            (
                'This is about as good as a sub-1B model gets on this corpus. Further '
                'pretraining will buy very little; if you want a chatbot, fine-tune this '
                'checkpoint. If you want a *better* model, you need more parameters or '
                'better data, not more steps.'
            ),
        )
    elif score >= 70:
        tier, body = (
            'GOLDEN CRUST, SOFT MIDDLE',
            (
                'A solid base model, but measurably short of what this scale can reach. '
                'It writes in period style and mostly holds a sentence together, yet it '
                'still loses the thread of meaning. More clean tokens would still help.'
            ),
        )
    elif score >= 55:
        tier, body = (
            'HALF-BAKED',
            (
                'Clearly undertrained. The style is there but the substance is not: '
                'expect confident nonsense, topic drift and heavy looping under greedy '
                'decoding. Do not fine-tune this for chat yet - keep pretraining.'
            ),
        )
    elif score >= 35:
        tier, body = (
            'DOUGH',
            (
                'Structure is forming - real words, some grammar - but this is not a '
                'usable language model yet. It needs several times more training tokens.'
            ),
        )
    else:
        tier, body = (
            'RAW BATTER',
            (
                'Barely past random guessing. Either training has only just started, or '
                'something is broken (learning rate, data pipeline, tokenizer mismatch).'
            ),
        )
    tpp = prov.get('tokens_per_param')
    if tpp is not None and prov.get('tokens_lower_bound'):
        body += (
            f' Compute check: at least ~{tpp:.1f} tokens per parameter, but the '
            f'training was restarted so the true total is unknown (rule of '
            f'thumb for "fully fed": ~{CHINCHILLA_TOKENS_PER_PARAM}).'
        )
    elif tpp is not None:
        if tpp < CHINCHILLA_TOKENS_PER_PARAM * 0.75:
            body += (
                f' Compute check: it has seen ~{tpp:.1f} tokens per parameter; '
                f'the compute-optimal rule of thumb is ~{CHINCHILLA_TOKENS_PER_PARAM}. '
                f'It is undertrained *by construction* - the single cheapest '
                f'improvement is simply more tokens.'
            )
        elif tpp > CHINCHILLA_TOKENS_PER_PARAM * 3:
            body += (
                f' Compute check: ~{tpp:.0f} tokens per parameter is well past '
                f'compute-optimal; if quality has plateaued, more steps are now '
                f'wasted money compared with training a bigger model.'
            )
        else:
            body += (
                f' Compute check: ~{tpp:.1f} tokens per parameter - in the sensible range (rule of thumb: ~{CHINCHILLA_TOKENS_PER_PARAM}).'
            )
    return tier, body


def band_label(acc: float) -> str:
    for hi, label in LOGIC_BANDS:
        if acc < hi:
            return label
    return LOGIC_BANDS[-1][1]


# ============================================================================
# Per-checkpoint evaluation
# ============================================================================


def evaluate_one(ckpt: Path, tok_dir: Path, data, args, device, dtype) -> dict:
    t0 = time.time()
    tok, model = load_model(ckpt, tok_dir, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    r = {'checkpoint': str(ckpt), 'name': ckpt.name, 'params_m': round(n_params / 1e6, 1)}
    r['provenance'] = training_provenance(ckpt, n_params)

    r['bpb'] = bits_per_byte(tok, model, data['heldout'], max_tokens=args.max_tokens)
    r['logic'] = run_logic(tok, model, data['logic_items'])
    r['traps'] = run_traps(tok, model, data['trap_pairs'])
    r['chat_bpb'] = run_chat_fit(tok, model, data['chat'])
    r['gen'] = run_generation(tok, model, data['gen_stems'], max_new=args.gen_tokens, seed=args.seed)
    free_model(model)

    r['points'] = {
        'bpb': interp(r['bpb'], BPB_LADDER),
        'logic': logic_points(r['logic']['acc']),
        'chat': interp(r['chat_bpb'], CHAT_LADDER),
        'hygiene': hygiene_points(r['gen']['greedy_loop_len'], r['gen']['punct_issues_p100']),
    }
    r['bake_score'] = bake_score(r['points'])
    r['secs'] = round(time.time() - t0, 1)
    return r


# ============================================================================
# Report rendering
# ============================================================================


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '—'
    return f'{v:.{nd}f}' if isinstance(v, float) else str(v)


def human_tokens(n):
    if n is None:
        return 'unknown'
    for unit, div in (('T', 1e12), ('B', 1e9), ('M', 1e6)):
        if n >= div:
            return f'{n / div:.1f}{unit}'
    return f'{n:.0f}'


def ladder_figure(bpb: float) -> str:
    rows = [
        (3.50, 'untrained model (uniform noise)'),
        (2.00, 'word-salad'),
        (1.50, 'broken prose'),
        (1.33, '500M model, ~0.7B tokens (early training)'),
        (1.19, '500M model, ~4.7B tokens (undertrained but solid)'),
        (1.10, 'best sub-1B measured on this data (341M)'),
        (0.95, 'estimated sub-1B ceiling on this corpus'),
    ]
    out, placed = [], False
    for v, label in rows:
        if not placed and not math.isnan(bpb) and bpb >= v:
            out.append(f'  --> {bpb:.3f}  YOUR MODEL')
            placed = True
        out.append(f'      {v:.2f}   {label}')
    if not placed and not math.isnan(bpb):
        out.append(f'  --> {bpb:.3f}  YOUR MODEL (below every anchor - excellent)')
    return '\n'.join(out)


def render_single(r: dict, extra_notes: list[str]) -> str:
    p = r['points']
    prov = r['provenance']
    tier, blunt = verdict_text(r['bake_score'], prov)
    L = []
    L.append(f'# Evaluation: {r["name"]}')
    L.append('')
    L.append(f'- **Parameters:** {r["params_m"]:.0f}M')
    if prov['tokens_seen']:
        L.append(
            f'- **Training tokens seen:** ~{human_tokens(prov["tokens_seen"])} '
            f'({prov["tokens_per_param"]:.1f} per parameter; '
            f"rule of thumb for 'fully fed' is ~{CHINCHILLA_TOKENS_PER_PARAM})"
        )
    if prov['epoch']:
        L.append(f'- **Epochs:** {prov["epoch"]:.2f}')
    if prov['note']:
        L.append(f'- **Note:** {prov["note"]}')
    L.append('')
    L.append(f'## Verdict: {tier}  (bake score {r["bake_score"]:.0f}/100)')
    L.append('')
    L.append(blunt)
    L.append('')
    L.append('## Where it sits (held-out bits/byte, lower = better)')
    L.append('')
    L.append('```')
    L.append(ladder_figure(r['bpb']))
    L.append('```')
    L.append('')
    L.append('## Scores')
    L.append('')
    L.append('| what | raw value | points /100 | weight | plain English |')
    L.append('|---|---|---|---|---|')
    L.append(
        f'| Held-out loss | {fmt(r["bpb"], 4)} bits/byte | {fmt(p["bpb"], 0)} | {WEIGHTS["bpb"]} '
        f'| how cheaply it predicts period text it never saw — the single best training signal |'
    )
    lg = r['logic']
    L.append(
        f'| Logic | {fmt(lg["acc"], 3)} acc, margin {fmt(lg["margin"], 3)} | {fmt(p["logic"], 0)} | {WEIGHTS["logic"]} '
        f'| picks the *sensible* continuation over matched nonsense; 0.50 = coin-flip. {band_label(lg["acc"])} |'
    )
    L.append(
        f'| Chat readiness | {fmt(r["chat_bpb"], 4)} bits/byte | {fmt(p["chat"], 0)} | {WEIGHTS["chat"]} '
        f'| how cheap well-formed period dialogue already is — predicts fine-tuning ease |'
    )
    g = r['gen']
    L.append(
        f'| Hygiene | loop {fmt(g["greedy_loop_len"], 1)}w, punct {fmt(g["punct_issues_p100"], 2)}/100w | {fmt(p["hygiene"], 0)} | {WEIGHTS["hygiene"]} '
        f'| greedy-decoding loop length and broken punctuation |'
    )
    if 'judge' in r:
        j = r['judge']
        L.append(
            f'| Judge deviation | {fmt(j["judge_dev"], 3)} bits/byte | — | (info) '
            f'| distance from what real period prose costs the judge ({fmt(j["judge_bpb_real"], 3)}); 0 = indistinguishable |'
        )
    t = r['traps']
    L.append('')
    L.append('## Period boundary')
    L.append('')
    if t['n_leaked'] == 0:
        L.append(
            f'Clean. All {t["n"]} post-1900 trap words cost the model more than their '
            f'period twins (mean shock +{fmt(t["mean_shock"], 2)} bits/byte, weakest pair '
            f"'{t['worst_pair']}' at +{fmt(t['min_shock'], 2)}). No sign of modern text in training."
        )
    else:
        L.append(
            f'**LEAKAGE WARNING:** {t["n_leaked"]}/{t["n"]} trap words are *cheaper* than '
            f"their period twins (worst: '{t['worst_pair']}'). Modern text has probably "
            f'contaminated the training corpus.'
        )
    L.append('')
    L.append('## See for yourself (sampled, t=0.8)')
    L.append('')
    for stem, cont in list(zip(GEN_STEMS, r['gen'].get('sampled_texts', [])))[:3]:
        one = ' '.join(cont.split())[:400]
        L.append(f'> **{stem}** {one}')
        L.append('>')
    L.append('')
    for n in extra_notes:
        L.append(f'*{n}*')
    return '\n'.join(L)


def render_folder(results: list[dict], target: Path, extra_notes: list[str]) -> str:
    ok = [r for r in results if not math.isnan(r['bake_score'])]
    L = [f'# Evaluation: {target.name} ({len(results)} checkpoints)', '']
    best = None
    if ok:
        # Adjacent checkpoints differ by a couple of bake points from noise alone
        # (greedy loop length is the jumpy component). Crowning a single winner
        # over that would be false precision, so: collect everything within the
        # noise window of the top bake score, then break the tie with held-out
        # bpb - the one metric that tracks training perfectly (rho -1.00) - so
        # the headline never contradicts the loss-curve warning below it.
        top = max(r['bake_score'] for r in ok)
        pool = [r for r in ok if top - r['bake_score'] <= 2.5]
        best = min(pool, key=lambda r: r['bpb'] if not math.isnan(r['bpb']) else float('inf'))
        ties = [r['name'] for r in pool if r is not best]
        tier, blunt = verdict_text(best['bake_score'], best['provenance'])
        L.append(f'## Best checkpoint: `{best["name"]}` — {tier} (bake score {best["bake_score"]:.0f}/100)')
        L.append('')
        if ties:
            L.append(
                f'Statistically tied with: {", ".join(f"`{t}`" for t in ties)} (within noise of each other - any of them is a fine pick).'
            )
            L.append('')
        L.append(blunt)
        # Trend. Judged on held-out bpb only (rho -1.00), NOT on the bake score:
        # the composite includes noisier components (greedy loop length varies a
        # few points between adjacent checkpoints) and would cry "peaked!" over
        # what is actually a tie. 0.005 bits/byte is the noise floor we observed
        # between adjacent same-quality checkpoints.
        sr = [r for r in results if not math.isnan(r['bpb'])]
        if len(sr) >= 3:
            best_bpb = min(r['bpb'] for r in sr)
            gap_last = sr[-1]['bpb'] - best_bpb
            # slope over the last third of the run, where convergence would show
            anchor = sr[max(0, (2 * len(sr)) // 3 - 1)]
            drop = anchor['bpb'] - sr[-1]['bpb']
            L.append('')
            if gap_last > 0.005:
                peak = min(sr, key=lambda r: r['bpb'])
                L.append(
                    f'**Held-out loss peaked at `{peak["name"]}` and got WORSE afterwards** '
                    f'(+{gap_last:.3f} bits/byte by the end). If training changed after that '
                    f'point (fine-tuning, LR change, new data), that change cost general '
                    f'quality; the later checkpoints may still be better for their own '
                    f'purpose (e.g. chat), so check the chat column.'
                )
            elif drop > 0.004:
                L.append(
                    f'**Still improving when training stopped** (held-out loss fell '
                    f'{drop:.3f} bits/byte over the last third of the run). This is not '
                    f'done baking — more steps would very likely keep helping.'
                )
            elif drop > 0.001:
                L.append(
                    f'Gains have slowed but not stopped (held-out loss still fell '
                    f'{drop:.4f} bits/byte over the last third of the run). Whether more '
                    f'steps are worth the money depends on the tokens-per-param check above.'
                )
            else:
                L.append(
                    'Held-out loss is flat over the last third of the run — additional '
                    'steps at this scale and learning rate are buying nothing.'
                )
        L.append('')
    L.append('## All checkpoints')
    L.append('')
    L.append('| checkpoint | bake /100 | bpb held-out | logic acc | chat bpb | loop (w) | shock | tokens seen |')
    L.append('|---|---|---|---|---|---|---|---|')
    for r in results:
        L.append(
            f'| {r["name"]} | {fmt(r["bake_score"], 0)} | {fmt(r["bpb"], 4)} '
            f'| {fmt(r["logic"]["acc"], 3)} | {fmt(r["chat_bpb"], 4)} '
            f'| {fmt(r["gen"]["greedy_loop_len"], 0)} | +{fmt(r["traps"]["mean_shock"], 2)} '
            f'| {human_tokens(r["provenance"]["tokens_seen"])} |'
        )
    L.append('')
    if best:
        L.append('## Best checkpoint in detail')
        L.append('')
        L.append(render_single(best, []))
    for n in extra_notes:
        L.append(f'*{n}*')
    return '\n'.join(L)


# ============================================================================
# main
# ============================================================================


def main():
    ap = argparse.ArgumentParser(
        description='Honest "is it baked?" evaluation for tiny vintage LLMs.', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument(
        'target', nargs='?', default=str(SCRIPT_DIR / 'checkpoints'), help='A checkpoint dir, or a folder containing checkpoint dirs.'
    )
    ap.add_argument('--tokenizer', type=Path, default=None, help='Tokenizer dir (auto-detected from checkpoint/parent when omitted).')
    ap.add_argument(
        '--heldout',
        type=Path,
        default=EVAL_DATA / 'heldout.jsonl',
        help='Held-out .jsonl with {"text": ...} lines. The default is the bundled 200-doc sample the reference ladder was measured on.',
    )
    ap.add_argument(
        '--chat-data', type=Path, default=EVAL_DATA / 'chat_sample.jsonl', help='Chat pairs .jsonl with {"context","target"} lines.'
    )
    ap.add_argument(
        '--probes',
        type=Path,
        default=None,
        help='JSON file overriding the built-in Victorian probes (keys: logic_items, trap_pairs, gen_stems).',
    )
    ap.add_argument(
        '--judge',
        type=Path,
        default=None,
        help='Optional big period model dir for the coherence judge. Needs enough VRAM for it *after* the evaluated model is freed.',
    )
    ap.add_argument('--docs', type=int, default=200, help='Held-out docs to score.')
    ap.add_argument(
        '--max-tokens',
        type=int,
        default=1024,
        help='Max tokens scored per held-out doc. The ladder anchors were measured at 1024; change it and they shift slightly.',
    )
    ap.add_argument('--gen-tokens', type=int, default=150, help='New tokens per generation stem.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto', choices=('auto', 'cuda', 'cpu'))
    ap.add_argument('--dtype', default='auto', choices=('auto', 'bfloat16', 'float16', 'float32'))
    ap.add_argument(
        '--out',
        type=Path,
        default=None,
        help='Report path (.md). Default: eval_results/evaluate2-<name>.md '
        'next to this script. A .json with all raw numbers is written beside it.',
    )
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    dtype = (torch.bfloat16 if device.type == 'cuda' else torch.float32) if args.dtype == 'auto' else getattr(torch, args.dtype)

    target = Path(args.target)
    if not target.exists():
        sys.exit(f'error: {target} does not exist')
    ckpts = resolve_targets(target)

    # ---- data ---------------------------------------------------------------
    extra_notes = []
    data = {'logic_items': LOGIC_ITEMS, 'trap_pairs': TRAP_PAIRS, 'gen_stems': GEN_STEMS}
    if args.probes:
        loaded = json.loads(args.probes.read_text())
        for k in data:
            if k in loaded:
                data[k] = [tuple(x) for x in loaded[k]]
        extra_notes.append(
            f'Probes loaded from {args.probes} - the logic bands in this report were calibrated on the built-in Victorian set.'
        )
    if args.heldout.exists():
        data['heldout'] = [json.loads(l)['text'] for l in args.heldout.open(encoding='utf-8')][: args.docs]
        if args.heldout != EVAL_DATA / 'heldout.jsonl':
            extra_notes.append(
                'Custom held-out set: the reference ladder was measured on the '
                'bundled sample, so ladder placement is approximate. If your '
                'set overlaps the training data, bpb will flatter the model.'
            )
    else:
        data['heldout'] = []
        extra_notes.append('NO HELD-OUT DATA FOUND - the strongest metric was skipped. Pass --heldout FILE.jsonl.')
    if args.chat_data.exists():
        data['chat'] = [(d['context'], d['target']) for d in (json.loads(l) for l in args.chat_data.open(encoding='utf-8'))][: args.docs]
    else:
        data['chat'] = []
        extra_notes.append('No chat sample found - chat-readiness skipped.')

    # ---- evaluate -----------------------------------------------------------
    tok_dir = resolve_tokenizer(ckpts[0], args.tokenizer)
    print(f'evaluating {len(ckpts)} checkpoint(s) on {device} ({dtype})')
    print(f'tokenizer: {tok_dir}')
    results = []
    for ckpt in ckpts:
        print(f'  {ckpt.name} ...', end='', flush=True)
        r = evaluate_one(ckpt, resolve_tokenizer(ckpt, args.tokenizer), data, args, device, dtype)
        results.append(r)
        print(f' bake {fmt(r["bake_score"], 0)}/100  bpb {fmt(r["bpb"], 4)}  logic {fmt(r["logic"]["acc"], 2)}  ({r["secs"]}s)')

    # ---- optional judge (loaded once, after all evaluated models are freed) --
    if args.judge:
        print(f'loading judge {args.judge} ...')
        jtok, jmodel = load_model(Path(args.judge), resolve_tokenizer(Path(args.judge), None), device, dtype)
        for r in results:
            r['judge'] = run_judge(jtok, jmodel, r['gen']['sampled_texts'], data['heldout'][:100])
            print(f'  {r["name"]}: judge_dev {fmt(r["judge"]["judge_dev"], 3)}')
        free_model(jmodel)

    # ---- report -------------------------------------------------------------
    name = target.name if len(ckpts) > 1 else ckpts[0].name
    out_md = args.out or (SCRIPT_DIR / 'eval_results' / f'evaluate2-{name}.md')
    out_md.parent.mkdir(parents=True, exist_ok=True)
    report = render_folder(results, target, extra_notes) if len(ckpts) > 1 else render_single(results[0], extra_notes)
    out_md.write_text(report, encoding='utf-8')
    slim = []
    for r in results:  # keep the sampled texts (small, and lets users re-read
        c = {k: v for k, v in r.items() if k != 'gen'}  # them later); drop
        c['gen'] = {
            k: v
            for k, v in r['gen'].items()  # the greedy ones,
            if k != 'greedy_texts'
        }  # which are loops
        slim.append(c)
    out_md.with_suffix('.json').write_text(json.dumps(slim, indent=2), encoding='utf-8')
    print(f'\nreport: {out_md}\njson:   {out_md.with_suffix(".json")}\n')
    print(report)


if __name__ == '__main__':
    main()

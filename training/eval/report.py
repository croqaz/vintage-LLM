"""JSON -> Markdown report rendering for the merged evaluator.

This file is deliberately separate from everything else: it consumes the payload
JSON produced by __main__.py and nothing more, so the human-readable report can
be re-styled, re-ordered or extended at any time WITHOUT touching metric code.
To regenerate a report from an existing JSON:

    python -m eval --render-report path/to/results.json [-o report.md]

Sections map to the payload keys; every number comes from results[i].summary or
a detail block. The flat `summary` dict in the JSON is the machine interface
(stable key names for agents to grep); this file is the human interface.
"""

from __future__ import annotations

import math

from .helpers import fmt, human_tokens
from .metrics import (
    CHINCHILLA_TOKENS_PER_PARAM,
    REFERENCE_LADDER,
    WEIGHTS,
    band_label,
)


def _ladder_figure(bpb: float, _unused: str | None = None) -> str:
    """ASCII placement of the model on the reference ladder.

    Rows marked `measured` in REFERENCE_LADDER are real models scored through
    this same code path on the same held-out file, so the comparison is exact.
    `synthetic` rows are qualitative signposts. If the model being reported IS a
    reference model its own row is dropped -- detected by an exact bits/byte
    match, since a reference row is produced by this identical code path.
    """
    rows = sorted((r for r in REFERENCE_LADDER if abs(r[0] - bpb) > 5e-6), key=lambda r: -r[0])
    out, placed = [], False
    for v, label, kind in rows:
        if not placed and not math.isnan(bpb) and bpb >= v:
            out.append(f'  --> {bpb:.3f}  YOUR MODEL')
            placed = True
        mark = '  ' if kind == 'measured' else ' ~'
        out.append(f'     {mark}{v:.3f}  {label}')
    if not placed and not math.isnan(bpb):
        out.append(f'  --> {bpb:.3f}  YOUR MODEL (below every anchor - excellent)')
    out.append('')
    out.append('   (~ = qualitative signpost; the rest are real models measured')
    out.append('    through this same eval on the same 200 held-out documents)')
    return '\n'.join(out)


# ============================================================================
# Per-checkpoint sections
# ============================================================================


def render_info_section(r: dict) -> list[str]:
    info, lineage = r.get('info', {}), r.get('lineage', {})
    s = r.get('summary', {})
    L = ['## Model info and training lineage', '']
    L.append(f'- **Checkpoint:** `{r["checkpoint"]}`')
    L.append(
        f'- **Architecture:** {info.get("architecture")} ({info.get("model_type")}), '
        f'{fmt(r.get("params_millions"), 1)}M params, {fmt(info.get("disk_size_mb"), 0)} MB on disk'
    )
    L.append(
        f'- **Layers / hidden / heads:** {info.get("num_layers")} / {info.get("hidden_size")} / '
        f'{info.get("num_attention_heads")} (KV heads: {info.get("num_key_value_heads")})'
    )
    L.append(f'- **Vocab (config / tokenizer):** {info.get("vocab_size_config")} / {info.get("vocab_size_tokenizer")}')
    L.append(f'- **Context length:** {info.get("max_position_embeddings")}')
    L.append(f'- **Chat template:** {"yes" if info.get("has_chat_template") else "no"}')
    L.append(f'- **Training lineage:** {lineage.get("line", "(unknown)")}')
    if s.get('tokens_seen_estimate'):
        note = ' (lower bound - run was restarted)' if lineage.get('tokens_lower_bound') else ''
        L.append(
            f'- **Training compute:** ~{human_tokens(s["tokens_seen_estimate"])} tokens seen '
            f'({fmt(s.get("tokens_per_param"), 1)} per parameter{note}; rule of thumb for "fully fed" is '
            f'~{CHINCHILLA_TOKENS_PER_PARAM}), budget bucket {lineage.get("budget")}'
        )
    sched = lineage.get('lr_scheduler')
    if sched:
        shape = ''
        w, st, dc = lineage.get('warmup_steps'), lineage.get('stable_steps'), lineage.get('decay_steps')
        if w is not None or st is not None or dc is not None:
            shape = f' (warmup {w} / stable {st} / decay {dc})'
        L.append(f'- **Schedule:** {sched}{shape}, max_steps {lineage.get("max_steps")}')
    if lineage.get('tokens_per_step'):
        L.append(f'- **Tokens per optimizer step:** {lineage["tokens_per_step"]:,} (seq {lineage.get("train_seq_length")})')
    if lineage.get('runtime_minutes'):
        minutes = lineage['runtime_minutes']
        segs = lineage.get('runtime_segments_minutes') or []
        # A resumed run is several processes. Showing only the total invites the
        # reader to check it against max_train_minutes and conclude it is wrong.
        detail = ''
        if len(segs) > 1:
            detail = ' = ' + ' + '.join(f'{x:.1f}' for x in segs) + f' over {len(segs)} segments (resumed)'
        budget = lineage.get('max_train_minutes')
        cfg = f'; configured max_train_minutes={budget} per process' if budget else ''
        L.append(f'- **Recorded training time:** {fmt(minutes, 0)} minutes (~{fmt(minutes / 60, 1)} h){detail}{cfg}')
    L.append(f'- **Embedding mean norm:** {fmt(s.get("embedding_mean_norm"))} (descriptive)')
    L.append(
        f'- **Embedding mean cosine:** {fmt(s.get("embedding_mean_cosine"))} — grows with training from ~0 at random init;'
        ' near 0 does NOT mean healthy'
    )
    if info.get('vocab_size_tokenizer', 0) > info.get('vocab_size_config', 0):
        L.append('- **WARNING:** tokenizer vocab is larger than config vocab_size - checkpoint/tokenizer mismatch?')
    if lineage.get('note'):
        L.append(f'- **Note:** {lineage["note"]}')
    for w in lineage.get('warnings') or []:
        L.append(f'- **Check:** {w}')
    if lineage.get('config_file'):
        L.append(f'- **Recipe read from:** `{lineage["config_file"]}`')
    L.append('')
    return L


def render_period_probes_section(r: dict) -> list[str]:
    p = r.get('period_probes') or {}
    if not p:
        return []
    ov = p.get('overall', {})
    hist, mod = p.get('historical', {}), p.get('modern', {})
    ratio = p.get('modern_over_historical_ratio')
    L = [
        '## Period fidelity on fixed probe sentences',
        '',
        f'Scored {ov.get("num_tokens", "?")} tokens over fixed historical + modern sentences '
        '(small set: read the RATIO, not absolute noise-level numbers).',
        '',
        '| group | sentences | perplexity | bits/byte |',
        '|---|---:|---:|---:|',
    ]
    for name, g in (('historical', hist), ('modern', mod)):
        if g:
            bpb = f'{g["bits_per_byte"]:.3f}' if g.get('bits_per_byte') is not None else '—'
            L.append(f'| {name} | {g["num_sentences"]} | {g["perplexity"]:.2f} | {bpb} |')
    L.append('')
    if ratio is not None:
        verdict = (
            '> Comfortably period-biased, as intended for a ~1900 knowledge cutoff.'
            if ratio >= 1.5
            else (
                '> Barely above 1.0: weak period preference.' if ratio >= 1.0 else '> BELOW 1.0: modern text is easier - check the corpus.'
            )
        )
        L += [f'MODERN/HISTORICAL perplexity ratio: **{ratio:.2f}** (>1 = period text easier, desired).', '', verdict, '']
    ov = p.get('overall') or {}
    if ov:
        L += [
            'Token-level health over the same probes '
            f'(mean token prob {ov.get("mean_token_prob", float("nan")):.4f}, '
            f'low-confidence tokens {100 * ov.get("frac_low_confidence", float("nan")):.1f}%, '
            f'mean entropy {ov.get("mean_entropy_nats", float("nan")):.3f} nats). '
            'Low-confidence = predicted with <1% probability; high entropy = hedging.',
            '',
        ]
    worst = sorted(p.get('per_sentence', []), key=lambda x: -x['perplexity'])[:5]
    if worst:
        L += ['Most surprising probe sentences:', '']
        for st in worst:
            L.append(f'- [{st["label"]}] ppl {st["perplexity"]:.1f}: {st["sentence"][:70]}')
        L.append('')
    return L


def render_heldout_section(r: dict) -> list[str]:
    s = r.get('summary', {})
    if s.get('prose_bpb') is None or (isinstance(s.get('prose_bpb'), float) and math.isnan(s['prose_bpb'])):
        return []
    n_docs = r.get('n_prose_docs')
    L = [
        '## Held-out period prose (bits per UTF-8 byte, lower = better)',
        '',
        f'{n_docs} unseen documents; byte-normalised so it compares across tokenizers.',
        '',
        f'- **Prose BPB: {fmt(s.get("prose_bpb"), 5)}**'
        + (f'  (95% CI [{fmt(r.get("ci_low"), 5)}, {fmt(r.get("ci_high"), 5)}])' if r.get('ci_low') is not None else ''),
        f'- Split A/B: {fmt(s.get("prose_bpb_split_a"), 4)} / {fmt(s.get("prose_bpb_split_b"), 4)} (stability diagnostic)',
        f'- Early-context BPB {fmt(s.get("prose_bpb_early"), 4)} vs late-context {fmt(s.get("prose_bpb_late"), 4)} '
        '(late >> early means long-range coherence is weaker)',
    ]
    if s.get('chat_bpb') is not None and not (isinstance(s.get('chat_bpb'), float) and math.isnan(s['chat_bpb'])):
        L.append(
            f'- Conditional chat-target BPB: {fmt(s.get("chat_bpb"), 5)} over {r.get("n_chat_docs", 0)} chat turns '
            '(how cheap well-formed dialogue already is; predicts fine-tuning ease)'
        )
    L.append('')
    return L


def render_bake_section(r: dict) -> list[str]:
    points = r.get('points') or {}
    score = r.get('bake_score')
    verdict = r.get('verdict', {})
    logic, traps = r.get('logic') or {}, r.get('traps') or {}
    gen = (r.get('generation') or {}).get('greedy_summary') or {}
    sampled_summary = (r.get('generation') or {}).get('sampled_summary') or {}
    tier = verdict.get('tier')

    L = [f'## Bake score: {fmt(score, 0)}/100' + (f' — **{tier}**' if tier else ''), '']
    if verdict.get('text'):
        L += [verdict['text'], '']

    if any(not math.isnan(v) for v in points.values() if isinstance(v, float)) or points:
        L += [
            '| component | raw value | points /100 | weight | plain English |',
            '|---|---|---:|---:|---|',
        ]
        if points.get('bpb') is not None:
            L.append(
                f'| Held-out loss | {fmt(r.get("summary", {}).get("prose_bpb"), 4)} bits/byte | {fmt(points["bpb"], 0)} '
                f'| {WEIGHTS["bpb"]} | how cheaply it predicts period text it never saw — the best training signal |'
            )
        if points.get('logic') is not None:
            L.append(
                f'| Logic | {fmt(logic.get("acc"), 3)} acc, margin {fmt(logic.get("margin"), 3)} | {fmt(points["logic"], 0)} '
                f'| {WEIGHTS["logic"]} | picks the *sensible* continuation over matched nonsense; 0.50 = coin-flip. '
                f'{band_label(logic.get("acc", float("nan")))} |'
            )
        if points.get('chat') is not None:
            L.append(
                f'| Chat readiness | {fmt(r.get("summary", {}).get("chat_bpb"), 4)} bits/byte | {fmt(points["chat"], 0)} '
                f'| {WEIGHTS["chat"]} | how cheap well-formed period dialogue already is |'
            )
        if points.get('hygiene') is not None:
            L.append(
                f'| Hygiene | loop {fmt(gen.get("mean_loop_words"), 1)}w avg, punct '
                f'{fmt(sampled_summary.get("mean_punct_issues_p100"), 2)}/100w | {fmt(points["hygiene"], 0)} '
                f'| {WEIGHTS["hygiene"]} | greedy-decoding loop length and broken punctuation |'
            )
        L.append('')

    if r.get('summary', {}).get('prose_bpb') is not None:
        L += [
            'Where it sits (held-out bits/byte, lower = better):',
            '',
            '```',
            _ladder_figure(r['summary']['prose_bpb'], r.get('reference_name')),
            '```',
            '',
        ]

    if traps:
        leaked, n = traps.get('n_leaked', 0), traps.get('n', 0)
        if n:
            if leaked == 0:
                L.append(
                    f'Period boundary: clean. All {n} post-1900 trap words cost more than their period twins '
                    f"(mean shock +{fmt(traps.get('mean_shock'), 2)}, weakest pair '{traps.get('worst_pair')}' "
                    f'at +{fmt(traps.get("min_shock"), 2)} bits/byte). No sign of modern text in training.'
                )
            else:
                L.append(
                    f'**LEAKAGE WARNING:** {leaked}/{n} trap words are *cheaper* than their period twins '
                    f"(worst: '{traps.get('worst_pair')}'). Modern text has probably contaminated the corpus."
                )
            L.append('')
    return L


def render_generation_section(r: dict, max_examples: int = 6) -> list[str]:
    gen = r.get('generation') or {}
    if not gen:
        return []
    greedy_sum, sampled_sum = gen.get('greedy_summary') or {}, gen.get('sampled_summary') or {}
    if not greedy_sum and not sampled_sum:
        return []
    L = [
        '## Generation hygiene (all prompts generated once; metrics on those same texts)',
        '',
    ]
    if greedy_sum:
        L.append(
            f'- Greedy: mean longest repeating block **{fmt(greedy_sum.get("mean_loop_words"), 1)} words**, '
            f'worst single prompt {fmt(greedy_sum.get("worst_loop_words"), 0)} words '
            '(an undertrained model falls into loops under greedy decoding)'
        )
    if sampled_sum:
        L += [
            f'- Sampled (t={gen.get("temperature")}): distinct-2 {fmt(sampled_sum.get("mean_distinct_2"), 3)}, '
            f'echo rate {fmt(sampled_sum.get("mean_echo_rate"), 3)}, degenerate prompts '
            f'{fmt(100 * (sampled_sum.get("degenerate_rate") or 0), 1)}%, '
            f'prompt-copy rate {fmt(sampled_sum.get("mean_prompt_copy_rate"), 3)}',
            f'- Punctuation issues per 100 words (sampled): {fmt(sampled_sum.get("mean_punct_issues_p100"), 2)}',
        ]
    L.append('')
    if sampled_sum.get('degenerate_rate', 0) > 0.34 or sampled_sum.get('mean_echo_rate', 0) > 0.30:
        L += ['> High repetition/degeneracy across probes - typical of an undertrained checkpoint.', '']
    elif sampled_sum.get('worst_echo_rate', 0) > 0.45:
        L += ['> Average looks fine but at least one probe looped badly - read the flagged samples below.', '']

    samples = gen.get('sampled_samples') or []
    flags = [s for s in samples if s.get('degenerate')]
    if flags:
        L += [f'{len(flags)} of {len(samples)} sampled continuations are flagged as surface failures:', '']
    shown = flags[:max_examples] or samples[:3]
    for s in shown:
        flag = ' **[SURFACE FAILURE]**' if s.get('degenerate') else ''
        one = ' '.join((s.get('continuation') or '').split())[:400]
        L += [
            f'> **PROMPT{flag}:** {s["prompt"]}',
            '> ',
            f'> {s["prompt"]}{" " if one else ""}{one}',
            '> ',
            f'> _distinct-2 {fmt(s.get("distinct_2"), 3)}; echo {fmt(s.get("echo_rate"), 3)}; '
            f'longest loop {s.get("longest_loop_words")} words_',
            '',
        ]
    return L


def render_tokenizer_and_optim_section(r: dict) -> list[str]:
    """Tokenizer efficiency and optimisation health -- neither needs the GPU."""
    t = r.get('tokenizer_stats') or {}
    c = r.get('training_curve') or {}
    if not t and not c.get('eval_curve'):
        return []
    L = ['## Tokenizer efficiency and optimisation health', '']
    if t:
        L += [
            f'- **Bytes per token: {t.get("bytes_per_token", float("nan")):.4f}** on the scored '
            f'held-out set (vocab {t.get("vocab_size")}). HIGHER = the same token budget carries '
            'more text, so a model trained with this tokenizer sees more data per step. '
            'Compare this across tokenizers before blaming one for a quality difference.',
        ]
    if c.get('eval_curve'):
        nf = c.get('grad_norm_nonfinite') or 0
        L += [
            f'- Final eval loss **{c.get("final_eval_loss"):.4f}** (ppl {c.get("final_eval_ppl"):.2f}) '
            f'at step {c.get("final_eval_step")}; final train loss {fmt(c.get("final_train_loss"), 4)}.',
            f'- Gradient norm: max {fmt(c.get("grad_norm_max"), 2)}, mean {fmt(c.get("grad_norm_mean"), 2)}, '
            f'min {fmt(c.get("grad_norm_min"), 3)}, **non-finite {nf}**' + ('' if nf == 0 else '  <-- INSTABILITY') + '.',
            f'- Eval curve retained ({len(c["eval_curve"])} points) for step-matched comparison. '
            '`eval_steps` is in MINUTES, so runs of different speed evaluate at different steps; '
            'raw endpoints are NOT comparable and must be interpolated onto a common grid.',
        ]
    L.append('')
    return L


def render_sense_section(r: dict) -> list[str]:
    emb = r.get('embeddings') or {}
    shifts = emb.get('semantic_shift') or {}
    if not shifts:
        return []
    L = [
        '## Diachronic word-sense separation',
        '',
        'Cosine similarity of the SAME shifted word (gay, awful, python...) in a period vs a modern sentence.',
        'Lower = the two senses are represented differently (good); ~1.0 = treated identically.',
        '',
        f'Mean sense-shift similarity: **{fmt(emb.get("mean_shift_similarity"), 3)}**',
        '',
    ]
    for w, s in sorted(shifts.items(), key=lambda kv: kv[1]):
        flag = ' ← suspiciously identical' if s >= 0.999 else ''
        L.append(f'- {w}: {s:+.3f}{flag}')
    L.append('')
    return L


def render_checkpoint_detail(r: dict) -> list[str]:
    L = []
    for section in (
        render_info_section,
        render_period_probes_section,
        render_heldout_section,
        render_bake_section,
        render_generation_section,
        render_sense_section,
        render_tokenizer_and_optim_section,
    ):
        L += section(r)
    return L


# ============================================================================
# Whole-payload reports
# ============================================================================


def _compare_table(results: list[dict]) -> list[str]:
    L = [
        '## All checkpoints',
        '',
        '| model | bake /100 | tier | prose BPB | chat BPB | logic acc | trap shock | loop (w) | echo | tokens seen | lineage |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|',
    ]
    for r in results:
        s = r.get('summary', {})
        gen = (r.get('generation') or {}).get('greedy_summary') or {}
        sampled = (r.get('generation') or {}).get('sampled_summary') or {}
        L.append(
            f'| `{r["label"]}` | {fmt(r.get("bake_score"), 0)} | {r.get("verdict", {}).get("tier", "—")} '
            f'| {fmt(s.get("prose_bpb"), 4)} | {fmt(s.get("chat_bpb"), 4)} | {fmt(s.get("logic_acc"), 3)} '
            f'| +{fmt(s.get("trap_mean_shock"), 2)} | {fmt(gen.get("mean_loop_words"), 0)} '
            f'| {fmt(sampled.get("mean_echo_rate"), 2)} | {human_tokens(s.get("tokens_seen_estimate"))} '
            f'| {r.get("lineage", {}).get("line", "")} |'
        )
    L.append('')
    return L


def render_curve_comparison(payload: dict) -> list[str]:
    """Eval loss for every collected run on ONE common step grid."""
    cc = payload.get('curve_comparison') or {}
    grid, series = cc.get('grid'), cc.get('series')
    if not grid or not series:
        return []
    L = [
        '## Eval loss at matched steps',
        '',
        'Interpolated onto a common grid because `eval_steps` is in MINUTES: runs of different '
        "speed evaluate at different step numbers, and a faster run's raw endpoint flatters it.",
        '',
        '| model | ' + ' | '.join(str(g) for g in grid) + ' |',
        '|---' * (len(grid) + 1) + '|',
    ]
    for label in sorted(series, key=lambda k: series[k][-1] if series[k][-1] == series[k][-1] else 9e9):
        vals = series[label]
        L.append(f'| `{label}` | ' + ' | '.join('—' if v != v else f'{v:.4f}' for v in vals) + ' |')
    L.append('')
    return L


def render_report(payload: dict) -> str:
    results = payload.get('results', [])
    settings = payload.get('settings', {})
    env = payload.get('environment', {})
    rankings = payload.get('rankings', {})

    title_name = payload.get('target_label') or (results[0]['label'] if len(results) == 1 else f'{len(results)} checkpoints')
    L = [f'# Evaluation: {title_name}', '']

    # --- decision header for multi-checkpoint runs (eval3-style ranking) -----
    if len(results) > 1:
        ranked = rankings.get('prose') or []
        scored = [r for r in results if r.get('bake_score') is not None and not math.isnan(r['bake_score'])]
        if ranked:
            lead = ranked[0]
            tied = [row for row in ranked if row.get('equivalent_to_leader')]
            if len(tied) > 1:
                names = ', '.join(f'`{t["label"]}`' for t in tied)
                L.append(
                    f'**Leader: `{lead["label"]}` at {lead["bpb"]:.5f} BPB, statistically/practically tied with {names}** '
                    f'(paired bootstrap, {payload.get("bootstrap_confidence", 0.95):.0%}).'
                )
            else:
                L.append(f'**Best checkpoint by held-out loss: `{lead["label"]}` at {lead["bpb"]:.5f} BPB.**')
        if scored:
            best = max(scored, key=lambda r: r['bake_score'])
            L.append(f'Best bake score: `{best["label"]}` at {best["bake_score"]:.0f}/100 ({best["verdict"].get("tier", "")})')
        L.append('')
        L += _compare_table(results)
        L += render_curve_comparison(payload)

    # --- per-checkpoint sections ---------------------------------------------
    for r in results:
        if len(results) > 1:
            L.append(f'---\n\n# {r["label"]}\n')
        L += render_checkpoint_detail(r)

    # --- reproducibility ------------------------------------------------------
    L += [
        '## Reproducibility',
        '',
        f'- Seed: {settings.get("seed")}; generation modes: {", ".join(settings.get("generation_modes", []))}',
        f'- Held-out data SHA-256: `{settings.get("heldout_sha256", "?")}` ({settings.get("docs")} docs)',
        f'- Device/dtype: `{env.get("device")}` / `{env.get("dtype")}`; PyTorch {env.get("torch")}, Transformers {env.get("transformers")}',
        '- Raw per-document bit/byte counts are retained in the JSON, so rankings can be audited without reloading models.',
        '',
    ]
    if payload.get('failures'):
        L += ['## Failures', '']
        for item in payload['failures']:
            L.append(f'- `{item.get("path")}` — {item.get("reason")}')
        L.append('')
    return '\n'.join(L)

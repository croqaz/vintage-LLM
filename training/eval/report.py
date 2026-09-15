"""Render retained evaluator measurements, without inferring model readiness."""

from __future__ import annotations

from html import escape

from .helpers import fmt, human_tokens
from .leaderboard import entries_for
from .metrics import SURFACE_REASONS, WEIGHTS, is_finite
from .prompts import HISTORICAL_WORDS

# The leaderboard is measured, not recorded. Entries live in
# eval_data/leaderboard.json and each carries the held-out hash it was scored
# against; see leaderboard.py for why that replaced a hard-coded table here.
# The retired pre-v1 figures were scored on heldout-Sprocket-n-Say.jsonl and
# have been deleted. They were never comparable to anything in the current
# table; re-measure a model rather than looking for them.


def _literal(text) -> str:
    """Treat measured text as text, not Markdown or HTML report structure."""
    return escape(str(text), quote=False).translate(str.maketrans({c: '\\' + c for c in '\\`*_{}[]()#+-.!|'}))


def _cell(text) -> str:
    return ' '.join(_literal(text).split())


def _quote(text: str) -> list[str]:
    return ['> ' + _literal(line) for line in text.splitlines()] + ['']


def _spread(rows: list, limit: int) -> list:
    """Select by position, including endpoints, without sorting by performance."""
    if limit <= 0:
        return []
    if len(rows) <= limit:
        return rows[:]
    if limit == 1:
        return rows[:1]
    return [rows[i * (len(rows) - 1) // (limit - 1)] for i in range(limit)]


def _percent(value) -> str:
    return '—' if not is_finite(value) else f'{100 * value:.1f}%'


def _confidence(value) -> str:
    return 'unknown-confidence' if not is_finite(value) else f'{100 * value:g}%'


def _descending(value) -> tuple:
    """Sort finite measurements largest first, with missing/non-finite values last."""
    return (0, -value) if is_finite(value) else (1, 0)


def _training_time(minutes) -> str:
    if not is_finite(minutes) or minutes < 0:
        return '—'
    hours, remainder = divmod(round(minutes, 1), 60)
    return f'{int(hours)} h {remainder:.1f} min' if hours else f'{remainder:.1f} min'


def model_title(r: dict) -> str:
    """Compact architecture identity using only recorded model metadata."""
    info = r.get('info') or {}
    parts = [info.get('model_type') or 'Unknown model']
    if is_finite(r.get('params_millions')):
        parts.append(f'{fmt(r["params_millions"], 2)}M params')
    for key, label in (
        ('num_layers', 'depth'),
        ('hidden_size', 'width'),
        ('num_attention_heads', 'heads'),
        ('num_key_value_heads', 'KV'),
        ('max_position_embeddings', 'ctx'),
        ('vocab_size_config', 'vocab'),
    ):
        if info.get(key) is not None:
            parts.append(f'{label} {info[key]}')
    if info.get('tie_word_embeddings') is True:
        parts.append('tied embeddings')
    elif info.get('tie_word_embeddings') is False:
        parts.append('untied embeddings')
    return ' · '.join(parts)


def render_info_section(r: dict) -> list[str]:
    lineage, s = r.get('lineage') or {}, r.get('summary') or {}
    info = r.get('info') or {}
    template = {True: 'yes', False: 'no'}.get(info.get('has_chat_template'), '—')
    return [
        '## Model and training records',
        '',
        f'- Checkpoint: `{r["checkpoint"]}`',
        f'- Recorded training time: {_training_time(lineage.get("runtime_minutes"))}',
        f'- Recorded lineage: {lineage.get("line", "unknown")}',
        f'- Estimated training tokens (`tokens_seen_estimate`): {human_tokens(s.get("tokens_seen_estimate"))}; '
        f'per parameter: {fmt(s.get("tokens_per_param"), 1)}. Compute-derived estimate, not a training-sufficiency threshold.',
        f'- Model storage: {fmt(info.get("disk_size_mb"), 1)} MB; chat template available: {template}.',
        f'- Training schedule: {_cell(lineage.get("lr_scheduler") or "—")}; '
        f'warmup / stable / decay: {fmt(lineage.get("warmup_steps"), 0)} / '
        f'{fmt(lineage.get("stable_steps"), 0)} / {fmt(lineage.get("decay_steps"), 0)} steps; '
        f'max optimizer steps: {fmt(lineage.get("max_steps"), 0)}.',
        f'- Training batch: {fmt(lineage.get("tokens_per_step"), 0)} tokens/optimizer step; '
        f'sequence: {fmt(lineage.get("train_seq_length"), 0)} tokens; '
        f'configured time limit: {_training_time(lineage.get("max_train_minutes"))} per process.',
        f'- Recipe: {_cell(lineage.get("config_file") or "—")}.',
        '',
    ]


def render_heldout_section(r: dict) -> list[str]:
    s = r.get('summary') or {}
    if 'prose_bpb' not in s and 'chat_target_bpb' not in s:
        return []
    L = [
        '## Prose and chat-target loss',
        '',
        'BPB = total negative log-likelihood in bits / scored UTF-8 bytes. Lower means higher likelihood '
        'per byte on the scored text; compare matching documents, scored spans and context settings.',
        '',
        f'- `prose_bpb`: {fmt(s.get("prose_bpb"), 5)}'
        + (
            f' ({_confidence(s.get("prose_bpb_ci_confidence"))} CI [{fmt(s.get("prose_bpb_ci_low"), 5)}, {fmt(s.get("prose_bpb_ci_high"), 5)}])'
            if is_finite(s.get('prose_bpb_ci_low'))
            else ''
        ),
        f'- `prose_docs_scored`: {s.get("prose_docs_scored", "unknown")}; '
        f'`prose_docs_excluded`: {s.get("prose_docs_excluded", "unknown")}; '
        f'`prose_truncated_docs`: {s.get("prose_truncated_docs", "not recorded")}.',
        f'- `prose_bpb_split_a` / `prose_bpb_split_b`: {fmt(s.get("prose_bpb_split_a"))} / {fmt(s.get("prose_bpb_split_b"))}.',
        f'- `prose_bpb_early` / `prose_bpb_late`: {fmt(s.get("prose_bpb_early"))} / {fmt(s.get("prose_bpb_late"))} '
        f'over {s.get("prose_bpb_position_docs", "unknown")} documents holding both halves (`prose_bpb_position_docs`). '
        'First 128 scored tokens versus the remainder; different text positions, not a controlled long-context test.',
        f'- `chat_target_bpb`: {fmt(s.get("chat_target_bpb"), 5)}'
        + (
            f' ({_confidence(s.get("chat_target_bpb_ci_confidence"))} CI '
            f'[{fmt(s.get("chat_target_bpb_ci_low"), 5)}, {fmt(s.get("chat_target_bpb_ci_high"), 5)}])'
            if is_finite(s.get('chat_target_bpb_ci_low'))
            else ''
        )
        + f'; `chat_target_docs_scored`: {s.get("chat_target_docs_scored", "unknown")}. '
        'Teacher-forced target likelihood given context, not generated chat quality or fine-tuning readiness.',
        '',
    ]
    return L


def render_logic_section(r: dict) -> list[str]:
    s = r.get('summary') or {}
    logic = r.get('logic') or {}
    if not logic:
        return []
    L = [
        '## Fixed-choice logic',
        '',
        f'`logic_accuracy`: {_percent(s.get("logic_accuracy"))}; '
        f'`logic_items_correct`: {logic.get("items_correct", "unknown")} / '
        f'`logic_items_scored`: {logic.get("items_scored", "unknown")}; '
        f'`logic_items_skipped`: {logic.get("items_skipped", "unknown")}. '
        f'`logic_margin_bpb`: {fmt(s.get("logic_margin_bpb"))} bits/byte.',
        '',
        f'{_confidence(s.get("logic_accuracy_ci_confidence"))} Wilson interval on the accuracy '
        f'(`logic_accuracy_ci_low` / `logic_accuracy_ci_high`): '
        f'[{_percent(s.get("logic_accuracy_ci_low"))}, {_percent(s.get("logic_accuracy_ci_high"))}]. '
        'The set is small, so this interval is wide; a few items of difference between checkpoints is not a difference.',
        '',
        'Correct means the designated good continuation has lower BPB; ties are incorrect. '
        'Margin is bad-minus-good BPB. Random two-choice accuracy is 50%; '
        'these small fixed sets are not a general reasoning benchmark.',
        '',
    ]
    categories = logic.get('categories') or {}
    if categories:
        L += [
            '| category | accuracy | scored pairs | JSON accuracy key |',
            '|---|---:|---:|---|',
        ]
        for category, row in sorted(categories.items()):
            L.append(
                f'| {_cell(category)} | {_percent(row.get("accuracy"))} | {row.get("items_scored", "unknown")} '
                f'| `logic_category_{category}_accuracy` |'
            )
        L += ['']
    return L


def render_logic_examples(r: dict) -> list[str]:
    items = (r.get('logic') or {}).get('items') or []
    L = []
    if items:
        L += [
            '### Logic choices and BPBs',
            '',
            'Up to two correct and two incorrect pairs, spread across saved order; one skipped pair if present. '
            'These are scored alternatives, not generated answers. All pairs are in `logic.items`.',
            '',
        ]
        for outcome, label, limit in ((True, 'Correct', 2), (False, 'Incorrect', 2), (None, 'Skipped', 1)):
            for item in _spread([row for row in items if row.get('correct') is outcome], limit):
                tie = ' (tie)' if outcome is False and item.get('good_bpb') == item.get('bad_bpb') else ''
                L += [f'**{label}{tie} · {_literal(item["category"])}**', '', 'Context:', '', *_quote(item['context'])]
                L += [
                    '| designated choice | continuation | BPB |',
                    '|---|---|---:|',
                    f'| good | {_cell(item["good_continuation"])} | {fmt(item.get("good_bpb"))} |',
                    f'| bad | {_cell(item["bad_continuation"])} | {fmt(item.get("bad_bpb"))} |',
                    '',
                    f'`margin_bpb`: {fmt(item.get("margin_bpb"))}.',
                    '',
                ]
    return L


def render_probe_section(r: dict) -> list[str]:
    s = r.get('summary') or {}
    L = ['## Sentence and phrase likelihood', '']
    if 'trap_mean_delta_bpb' in s:
        L += [
            f'- `trap_mean_delta_bpb`: {fmt(s.get("trap_mean_delta_bpb"))} '
            f'(standard error {fmt(s.get("trap_mean_delta_stderr_bpb"))}, `trap_mean_delta_stderr_bpb`); '
            f'`trap_min_delta_bpb`: {fmt(s.get("trap_min_delta_bpb"))} bits/byte; '
            f'`trap_nonpositive_pairs`: {s.get("trap_nonpositive_pairs", "unknown")} / {s.get("trap_pairs_scored", "unknown")}.',
            f'  Smallest delta belongs to {_cell((r.get("traps") or {}).get("min_delta_phrase") or "—")} (`traps.min_delta_phrase`).',
            '  Modern-minus-period phrase BPB on fixed pairs. A nonpositive value is a phrase preference, not evidence of training-data leakage.',
        ]
    probes = r.get('period_probes') or {}
    if probes:
        L += ['', '| measurement (JSON key) | value |', '|---|---:|']
        for group in ('historical', 'modern', 'overall'):
            for metric in ('bpb', 'ppl'):
                key = f'probe_{group}_{metric}'
                label = 'BPB' if metric == 'bpb' else 'token perplexity'
                L.append(f'| {group.capitalize()} {label} (`{key}`) | {fmt(s.get(key))} |')
        for label, key, format_value in (
            ('Historical sentences', 'probe_historical_sentences_scored', lambda v: fmt(v, 0)),
            ('Modern sentences', 'probe_modern_sentences_scored', lambda v: fmt(v, 0)),
            ('Scored tokens', 'probe_overall_tokens_scored', lambda v: fmt(v, 0)),
            ('Mean probability of observed token', 'probe_overall_mean_token_prob', fmt),
            ('Observed tokens assigned <1% probability', 'probe_overall_token_prob_below_0_01_rate', _percent),
            ('Mean next-token entropy (nats)', 'probe_overall_mean_entropy_nats', fmt),
        ):
            L.append(f'| {label} (`{key}`) | {format_value(s.get(key))} |')
        ratio = s.get('probe_modern_historical_ppl_ratio')
        comparison = (
            f'The modern sentence set has {ratio:.2f} times the token perplexity of the historical set.'
            if is_finite(ratio)
            else 'The modern-to-historical token perplexity ratio is unavailable.'
        )
        L += [
            '',
            comparison + ' This is modern perplexity divided by historical perplexity '
            '(`probe_modern_historical_ppl_ratio`), not evidence of a knowledge cutoff. '
            'Token perplexity depends on tokenization.',
            '',
        ]
    return L + [''] if len(L) > 2 else []


def render_probe_examples(r: dict) -> list[str]:
    sentences = (r.get('period_probes') or {}).get('per_sentence') or []
    L = []
    if sentences:
        L += [
            f'### All {len(sentences)} recorded probe sentences',
            '',
            'Highest token perplexity first (most surprising to the model); ties keep saved order, missing values last. '
            'Lower BPB means higher text likelihood per scored byte, not factual correctness or understanding. '
            'Values are from `period_probes.per_sentence`.',
            '',
            '| group | sentence | token perplexity | BPB | scored bytes |',
            '|---|---|---:|---:|---:|',
        ]
        for row in sorted(sentences, key=lambda row: _descending(row.get('ppl'))):
            L.append(
                f'| {_cell(row.get("label", "unknown"))} | {_cell(row["sentence"])} '
                f'| {fmt(row.get("ppl"), 2)} | {fmt(row.get("bpb"))} | {row.get("scored_bytes", "unknown")} |'
            )
        L += ['']
    return L


def render_bake_section(r: dict) -> list[str]:
    s = r.get('summary') or {}
    L = [
        '## Experimental composite',
        '',
        f'`bake_score`: {fmt(s.get("bake_score"), 2)}/100; `bake_status`: {s.get("bake_status", "unknown")}; '
        f'`bake_weight_covered`: {_percent(s.get("bake_weight_covered"))}.',
        'Fixed hand-set transforms and weights, not a validated overall-quality scale or a recommendation to stop training. '
        'Partial scores renormalize the available weights and are excluded from full-score rankings.',
        '',
        '| component (JSON points key) | measured input | points /100 | weight |',
        '|---|---|---:|---:|',
    ]
    for key, label, raw in (
        ('bpb', 'Prose loss', f'`prose_bpb`: {fmt(s.get("prose_bpb"))} bits/byte'),
        ('logic', 'Fixed-choice logic', f'`logic_accuracy`: {_percent(s.get("logic_accuracy"))}'),
        ('chat', 'Chat-target loss', f'`chat_target_bpb`: {fmt(s.get("chat_target_bpb"))} bits/byte'),
        (
            'hygiene',
            'Generation surface checks',
            f'`greedy_mean_loop_words`: {fmt(s.get("greedy_mean_loop_words"), 1)} words; '
            f'`sampled_mean_punct_issues_p100`: {fmt(s.get("sampled_mean_punct_issues_p100"), 2)}/100 words',
        ),
    ):
        L.append(f'| {label} (`points_{key}`) | {raw} | {fmt(s.get("points_" + key), 2)} | {WEIGHTS[key]} |')
    return L + ['', *render_reference_section(r)]


def render_reference_section(r: dict) -> list[str]:
    """One leaderboard: this checkpoint among everything scored the same way.

    Rows come from eval_data/leaderboard.json and only those scored on the
    same held-out file are shown, so the column can be ranked honestly. A row
    that cannot be compared is counted in the caption, never listed beside one
    that can.
    """
    s = r.get('summary') or {}
    settings = r.get('evaluation_settings') or {}
    peers, hidden = entries_for(settings.get('heldout_sha256'))
    this_name = (r.get('model_label') or r.get('name') or '').strip()

    note = ''
    if hidden:
        note = (
            f' {hidden} further entr{"y is" if hidden == 1 else "ies are"} on file but scored against a '
            'different held-out set and therefore not shown; re-run them to add them.'
        )
    L = [
        '### Prose BPB leaderboard',
        '',
        'Every row below was scored by this code on the same held-out documents, so the column can be read '
        'as a ranking. Lower BPB means higher likelihood per byte, which is retention of the training '
        f'register, not overall capability.{note}',
        '',
        '| model | params (M) | prose BPB | hist | modern | modern/hist | notes |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]

    def ratio(hist, modern):
        if is_finite(hist) and is_finite(modern) and hist:
            return f'{modern / hist:.2f}'
        return '\u2014'

    rows = [
        (
            s.get('prose_bpb'),
            f'| **This checkpoint** | {fmt(r.get("params_millions"), 2)} | **{fmt(s.get("prose_bpb"), 5)}** | '
            f'{fmt(s.get("probe_historical_bpb"), 4)} | {fmt(s.get("probe_modern_bpb"), 4)} | '
            f'{ratio(s.get("probe_historical_bpb"), s.get("probe_modern_bpb"))} | this run |',
        ),
        (
            s.get('prose_uniform_token_bpb'),
            f'| Uniform-token baseline | \u2014 | {fmt(s.get("prose_uniform_token_bpb"), 5)} | \u2014 | \u2014 | \u2014 | '
            'calculated for this tokenizer and scored text |',
        ),
    ]
    for e in peers:
        if e.get('name') == this_name:
            continue  # the live measurement above is the same model
        bits = [b for b in (e.get('kind'), e.get('origin'), e.get('note')) if b]
        rows.append(
            (
                e.get('prose_bpb'),
                f'| {_cell(e.get("display") or e.get("name"))} | {fmt(e.get("params_millions"), 2)} | '
                f'{fmt(e.get("prose_bpb"), 5)} | {fmt(e.get("probe_historical_bpb"), 4)} | '
                f'{fmt(e.get("probe_modern_bpb"), 4)} | {ratio(e.get("probe_historical_bpb"), e.get("probe_modern_bpb"))} | '
                f'{_cell("; ".join(bits)) if bits else "measured " + str(e.get("measured", ""))} |',
            )
        )

    L.extend(line for _, line in sorted(rows, key=lambda row: _descending(row[0])))
    return L + [
        '',
        'modern/hist is the modern-probe BPB over the historical-probe BPB. Above 1 means the model finds '
        'period text easier than modern text, which is what a vintage model should do. At or below 1 the '
        'model is more at home in modern English whatever its prose BPB says, which is the signature of a '
        'fine-tune over a modern base rather than pre-modern pretraining.',
        '',
        '`prose_uniform_token_bpb` = log2(model vocabulary size) \u00d7 scored tokens / scored bytes; '
        'equal token probabilities, not an evaluated untrained network. Leaderboard rows do not affect the composite.',
        '',
    ]


def _completion(sample: dict, label: str) -> list[str]:
    status = sample.get('surface_failure')
    state = 'flagged' if status is True else 'unflagged' if status is False else 'not assessed'
    reasons = sample.get('surface_failure_reasons')
    flags = ', '.join(reasons) if reasons else 'none' if reasons == [] else 'not recorded'
    text = sample.get('continuation') or ''
    L = [f'**{label} continuation — {state}**', '']
    L += _quote(text[:1200]) if text else ['_(empty continuation)_', '']
    if len(text) > 1200:
        L += ['_Excerpt truncated at 1200 characters; the full continuation is in JSON._', '']
    L += [
        f'Flags: {_literal(flags)}; distinct-2: {fmt(sample.get("distinct_2"), 3)}; '
        f'echo rate: {_percent(sample.get("echo_rate"))}; longest loop: {fmt(sample.get("longest_loop_words"), 0)} words.',
        '',
    ]
    return L


def _generation_examples(gen: dict) -> list[str]:
    greedy, sampled = gen.get('greedy_samples') or [], gen.get('sampled_samples') or []
    if not greedy and not sampled:
        return []
    L = ['### Autocomplete examples', '']
    paired_prompts = []
    sampled_by_prompt = {row['prompt']: row for row in sampled}
    paired = [row for row in greedy if row['prompt'] in sampled_by_prompt][:2]
    if paired:
        L += [
            '#### Same prompts: greedy vs sampled',
            '',
            'First two shared prompts in saved order, selected independently of flags or scores. '
            'The prompts stay fixed when the prompt set/order is unchanged.',
            '',
        ]
        for row in paired:
            prompt = row['prompt']
            paired_prompts.append(prompt)
            L += ['**Prompt**', '', *_quote(prompt)]
            L += _completion(row, 'Greedy')
            L += _completion(sampled_by_prompt[prompt], 'Sampled')

    mode, samples = ('Sampled', sampled) if sampled else ('Greedy', greedy)
    flagged = sum(row.get('surface_failure') is True for row in samples)
    unflagged = sum(row.get('surface_failure') is False for row in samples)
    unknown = len(samples) - flagged - unflagged
    L += [
        f'{mode}: {flagged} of {len(samples)} saved continuations flagged; '
        f'{unflagged} unflagged; {unknown} not assessed. '
        'Unflagged does not mean coherent or correct. The galleries are selected subsets, not an estimate of typical quality.',
        '',
    ]
    for status, label, limit in ((False, 'Unflagged', 4), (True, 'Flagged', 2), (None, 'Not assessed', 2)):
        candidates = [row for row in samples if row.get('surface_failure') is status and row['prompt'] not in paired_prompts]
        shown = _spread(candidates, limit)
        if not shown:
            continue
        L += [
            f'#### {label} {mode.lower()} continuations ({len(shown)} examples)',
            '',
            'Evenly spaced through this group in saved order; prompts shown above are excluded. Not ranked by quality.',
            '',
        ]
        for row in shown:
            L += ['**Prompt**', '', *_quote(row['prompt']), *_completion(row, mode)]
    return L


def render_generation_section(r: dict) -> list[str]:
    gen = r.get('generation') or {}
    if not gen:
        return []
    modes = [mode for mode in ('greedy', 'sampled') if gen.get(mode + '_summary') or gen.get(mode + '_samples')]
    settings = r.get('evaluation_settings') or {}
    L = [
        '## Autocomplete and generation',
        '',
        f'Generation budget: {settings.get("generation_tokens", "not recorded")} new tokens; seed: {settings.get("seed", "not recorded")}.',
    ]
    if 'sampled' in modes:
        L += [
            f'Sampling: temperature {fmt(gen.get("temperature"), 2)}, top-p {fmt(gen.get("top_p"), 2)}, top-k {fmt(gen.get("top_k"), 0)}.',
        ]
    if modes:
        L += [
            '',
            'Table keys are JSON suffixes: prepend the column mode (for example, `sampled_surface_failure_rate`). '
            'Surface flags detect short text, repetition or formatting—not coherence; valid text can fail and nonsense can pass.',
            '',
            '| measurement (JSON suffix) | ' + ' | '.join(mode.capitalize() for mode in modes) + ' |',
            '|---|' + '---:|' * len(modes),
        ]
        measurements = (
            ('Completions', 'n_prompts', lambda v: fmt(v, 0)),
            ('Flagged completions', 'surface_failure_rate', _percent),
            ('Short-text or repetition flags', 'degenerate_rate', _percent),
            ('Mean length (words)', 'mean_words', lambda v: fmt(v, 1)),
            ('Distinct word bigrams (fraction)', 'mean_distinct_2', lambda v: fmt(v, 3)),
            ('Local repetition', 'mean_echo_rate', _percent),
            ('Mean longest loop (words)', 'mean_loop_words', lambda v: fmt(v, 1)),
            ('Worst longest loop (words)', 'worst_loop_words', lambda v: fmt(v, 0)),
            ('Shared word 4-grams (fraction)', 'shared_4gram_fraction', lambda v: fmt(v, 3)),
            ('Prompt-word overlap', 'mean_prompt_word_overlap_rate', _percent),
            ('Punctuation issues / 100 words', 'mean_punct_issues_p100', lambda v: fmt(v, 2)),
            ('Mean sentence length (words)', 'mean_sentence_words', lambda v: fmt(v, 1)),
        )
        for label, key, format_value in measurements:
            values = [format_value((gen.get(mode + '_summary') or {}).get(key)) for mode in modes]
            L.append(f'| {label} (`{key}`) | ' + ' | '.join(values) + ' |')
        L += [
            '',
            'Distinct-2 is unique/total word bigrams; echo counts words repeated within the previous four words. '
            'Shared 4-grams measure overlap across completions, not coherence. '
            'Compare matching prompts, sample counts, generation budgets and decoding settings.',
            '',
            '### Flag reasons',
            '',
            'Fraction of completions triggering each check. Reasons overlap; do not add their rates.',
            '',
            '| flag (JSON suffix) | ' + ' | '.join(mode.capitalize() for mode in modes) + ' |',
            '|---|' + '---:|' * len(modes),
        ]
        for reason in SURFACE_REASONS:
            values = [_percent((gen.get(mode + '_summary') or {}).get(reason + '_rate')) for mode in modes]
            L.append(f'| `{reason}_rate` | ' + ' | '.join(values) + ' |')
        L += ['']
    sweep = gen.get('temperature_sweep') or {}
    if sweep:
        L += [
            '### Surface flags vs sampling temperature',
            '',
            'Flag rates describe these checks at each temperature; they do not measure coherence.',
            '',
            '| temperature | surface failure rate | degenerate | back-matter-like | shared 4-gram fraction | mean loop (w) |',
            '|---|---:|---:|---:|---:|---:|',
        ]
        for temp in sorted(sweep, key=float):
            row = sweep[temp] or {}
            L.append(
                f'| t={temp} | {_percent(row.get("surface_failure_rate"))} '
                f'| {_percent(row.get("degenerate_rate"))} '
                f'| {_percent(row.get("back_matter_rate"))} '
                f'| {fmt(row.get("shared_4gram_fraction"), 3)} | {fmt(row.get("mean_loop_words"), 1)} |'
            )
        L.append('')
    return L


def render_training_section(r: dict) -> list[str]:
    curve = r.get('training_curve') or {}
    series = [(key, curve.get(key) or []) for key in ('eval_curve', 'train_curve')]
    L = [
        '## Recorded training progress',
        '',
        'Trainer-log loss in nats/token, not held-out BPB. Up to five evenly spaced entries per series, '
        'including first/latest, without interpolation; unused slots are —. '
        f'Retained entries: {len(series[0][1])} validation, {len(series[1][1])} training. These points do not establish convergence.',
        '',
        '| series (`training_curve`) | optimizer step | loss (nats/token) |',
        '|---|---:|---:|',
    ]
    for key, rows in series:
        selected = _spread(rows, 5)
        for step, loss in selected + [(None, None)] * (5 - len(selected)):
            L.append(f'| `{key}` | {fmt(step, 0)} | {fmt(loss, 4)} |')
    return L + ['']


def render_diagnostics_section(r: dict) -> list[str]:
    s = r.get('summary') or {}
    keys = [
        'tokenizer_bytes_per_token',
        'tokenizer_vocab_size',
        'trainer_eval_loss_nats_per_token',
        'trainer_eval_ppl',
        'trainer_eval_step',
        'trainer_train_loss_nats_per_token',
        'grad_norm_max',
        'grad_norm_mean',
        'grad_norm_min',
        'grad_norm_nonfinite',
        'embedding_mean_norm',
        'embedding_mean_cosine',
        'sense_shift_mean_cosine',
        'judge_reference_bpb',
        'judge_generated_bpb',
        'judge_absolute_bpb_difference',
    ]
    L = ['## Other recorded diagnostics', '', '| summary key | value |', '|---|---:|']
    L.extend(f'| `{key}` | {fmt(s.get(key))} |' for key in keys)
    L += [
        '',
        'Trainer losses come from training logs, not this held-out scorer. Cross-run loss comparisons require the same '
        'validation data, tokenizer and reduction; matching optimizer steps alone is insufficient. '
        'Embedding norms and cosine similarities describe these vectors, without an established good/bad direction. '
        'Tokenizer bytes/token describes this text sample, not training throughput. '
        'Judge BPB describes likelihood under the specified reference model, not a coherence rating.',
        '',
    ]
    L += [
        '### Paired-context word similarities',
        '',
        'The historical/modern cosine compares the same word across its two probe sentences; '
        'lower means less similar vectors, not established better sense understanding. '
        'Closest/farthest compare different words in their historical sentences, among these ten probes only—not vocabulary-wide synonyms/antonyms. '
        'Rows are alphabetical; each list shows up to three recorded matches, nearest/farthest first, with pairwise cosines in parentheses.',
        '',
        '| word | historical/modern cosine | closest probe words | farthest probe words |',
        '|---|---:|---|---|',
    ]
    pairwise = (r.get('embeddings') or {}).get('pairwise_historical') or {}
    for word in sorted(HISTORICAL_WORDS):
        key = f'sense_shift_{word}_cosine'
        similarities = []
        for other in HISTORICAL_WORDS:
            if other == word:
                continue
            # The saved matrix stores each unordered pair once.
            value = pairwise.get(f'{word}|{other}', pairwise.get(f'{other}|{word}'))
            if is_finite(value):
                similarities.append((other, value))
        matches = []
        for direction in (-1, 1):
            ordered = sorted(similarities, key=lambda row: (direction * row[1], row[0]))[:3]
            matches.append(', '.join(f'{_cell(other)} ({fmt(value, 3)})' for other, value in ordered) or '—')
        L.append(f'| {_cell(word)} | {fmt(s.get(key))} | {matches[0]} | {matches[1]} |')
    L.append('')
    return L


def render_checkpoint_measurements(r: dict) -> list[str]:
    L = []
    for section in (
        render_info_section,
        render_training_section,
        render_bake_section,
        render_diagnostics_section,
        render_heldout_section,
        render_logic_section,
        render_probe_section,
        render_generation_section,
    ):
        L += section(r)
    return L


def render_checkpoint_examples(r: dict) -> list[str]:
    """Keep variable-length measured text out of the measurement layout."""
    return render_logic_examples(r) + render_probe_examples(r) + _generation_examples(r.get('generation') or {})


def _compare_table(results: list[dict], prose_rankings: list[dict]) -> list[str]:
    comparisons = {row['label']: row for row in prose_rankings}
    L = [
        '## Checkpoint measurements',
        '',
        '| model | prose_bpb | comparison group | prose vs group leader | chat_target_bpb | chat vs group leader | logic_accuracy | sampled_surface_failure_rate | bake_score | bake_status |',
        '|---|---:|---|---|---:|---|---:|---:|---:|---|',
    ]
    for r in results:
        s, row = r.get('summary') or {}, comparisons.get(r['label'], {})
        L.append(
            f'| `{r["label"]}` | {fmt(s.get("prose_bpb"), 5)} | {row.get("comparison_group", "unavailable")} '
            f'| {row.get("comparison_to_leader", "unavailable")} | {fmt(s.get("chat_target_bpb"), 5)} '
            f'| {s.get("chat_target_comparison_to_leader", "unavailable")} '
            f'| {fmt(s.get("logic_accuracy"), 3)} | {_percent(s.get("sampled_surface_failure_rate"))} '
            f'| {fmt(s.get("bake_score"), 2)} | {s.get("bake_status", "unknown")} |'
        )
    return L + ['']


def render_report(payload: dict) -> str:
    """Format saved measurements without modifying the payload or recomputing scores."""
    results = payload.get('results', [])
    title = model_title(results[0]) if len(results) == 1 else f'Evaluation comparison: {len(results)} checkpoints'
    L = [
        f'# {title}',
        '',
        'Flat measurements: `results[].summary` in the JSON report. '
        'The JSON metric guide supplies units and comparison conditions; missing values are not zero.',
        '',
    ]
    ranked = (payload.get('rankings') or {}).get('prose') or []
    if len(results) > 1:
        L += [
            f'Comparisons use {_confidence(payload.get("bootstrap_confidence"))} paired confidence intervals. '
            'Equivalent requires the entire candidate-minus-leader interval inside the recorded ±BPB margin; '
            'better/worse requires it wholly beyond that margin. Otherwise the result is inconclusive (or unavailable). '
            'Intervals describe document resampling, not variation across training seeds.',
            '',
            'Rankings are within matching protocol/coverage groups, not across them. '
            'Groups require matching document IDs and exact scored-text hashes.',
            '',
        ]
        for row in ranked:
            if row.get('comparison_to_leader') == 'leader':
                L.append(
                    f'- Group `{row.get("comparison_group", "unavailable")}`: lowest measured prose BPB '
                    f'`{row["label"]}` = {fmt(row.get("bpb"), 5)}; equivalence margin '
                    f'±{fmt(row.get("equivalence_margin_bpb"), 5)} BPB.'
                )
        L += ['', *_compare_table(results, ranked)]
    for r in results:
        if len(results) > 1:
            L += ['---', '', f'# {model_title(r)}', '']
        L += render_checkpoint_measurements(r)
    if payload.get('failures'):
        L += ['## Failures', '']
        L.extend(f'- `{item.get("path")}` — {item.get("reason")}' for item in payload['failures'])
        L.append('')
    # All checkpoints' measurements precede any text examples, including in collections.
    for r in results:
        examples = render_checkpoint_examples(r)
        if examples:
            title = 'Text examples' if len(results) == 1 else f'Text examples: {_literal(r["label"])}'
            L += [f'## {title}', '', *examples]
    L += ['## Reproducibility', '']
    for r in results:
        settings, env = r.get('evaluation_settings') or {}, r.get('evaluation_environment') or {}
        template = {True: 'yes', False: 'no'}.get(settings.get('chat_template_enabled'), '—')
        if len(results) > 1:
            L += [f'### {_literal(r["label"])}', '']
        L += [
            f'- Seed: {fmt(settings.get("seed"), 0)}; device/dtype: '
            f'{_cell(env.get("device") or "—")} / {_cell(env.get("dtype") or "—")}; '
            f'Python {_cell(env.get("python") or "—")}, PyTorch {_cell(env.get("torch") or "—")}, '
            f'Transformers {_cell(env.get("transformers") or "—")}.',
            f'- Prose source SHA-256: `{settings.get("heldout_sha256") or "—"}`; '
            f'requested documents: {fmt(settings.get("docs"), 0)}; token limit: {fmt(settings.get("max_tokens"), 0)}.',
            f'- Chat source SHA-256: `{settings.get("chat_sha256") or "—"}`; '
            f'requested documents: {fmt(settings.get("chat_docs"), 0)}; token limit: {fmt(settings.get("chat_max_tokens"), 0)}.',
            f'- Generation chat template applied: {template}.',
            '',
        ]
    L += [
        '- Scoring protocols are recorded per result in `evaluation_settings` and `evaluation_environment`. '
        'Raw records allow recomputing aggregates and document bootstrap without reloading models.',
        '',
    ]
    return '\n'.join(L)

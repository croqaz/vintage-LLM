"""Machine-readable dictionary for every key in `result.summary`.

WHY THIS EXISTS: an agent reading one of these JSONs cold, on another machine,
with no access to this conversation, must be able to answer "which number do I
quote, and which way is good?" without reading the source. Key names alone
cannot carry direction AND unit AND comparability caveats without becoming
unusable, so the names stay short and this table carries the semantics.

Every entry:
  desc              one line, plain English
  unit              physical unit, or 'fraction' / 'count' / 'points'
  direction         lower_is_better | higher_is_better | neutral
  comparable_across what the number may be compared over. The critical field:
                    several metrics here are NOT valid across models.
  caveat            present only where a naive reading produces a WRONG answer
  range             for bounded metrics

Keep this in sync when adding a metric: a summary key with no entry is a bug,
and `python -m eval --audit-guide` fails on one.
"""

from __future__ import annotations

ANY = ['tokenizers', 'context_lengths', 'model_sizes', 'training_datasets']
SAME_DATA = ['same_training_dataset_only']

PRIMARY_METRIC = 'prose_bpb'

HOW_TO_READ = (
    'RANK MODELS ON `prose_bpb` (lower is better). It is byte-normalised and scored in a '
    'fixed 1024-token window, so it is the only headline number valid across different '
    'tokenizers, context lengths and model sizes. '
    'NEVER rank models on `final_eval_loss` unless they trained on the SAME dataset - it is '
    "computed on each run's own validation split. "
    'BEFORE comparing `bake_score`, check `bake_is_partial`: a partial score was renormalised '
    'over only the components that were measured and is inflated. '
    'Every key below is described in `metric_guide`, which carries the unit and the direction '
    '(higher/lower is better) for each one. `rankings.prose` in a --collect output already '
    'contains PAIRED bootstrap comparisons, which are ~30x tighter than the per-model CIs.'
)


def _m(desc, unit, direction, comparable=None, caveat=None, rng=None, primary=False):
    entry = {'desc': desc, 'unit': unit, 'direction': direction, 'comparable_across': comparable or ANY}
    if caveat:
        entry['caveat'] = caveat
    if rng:
        entry['range'] = rng
    if primary:
        entry['is_primary'] = True
    return entry


GUIDE: dict[str, dict] = {
    # ---- headline ----------------------------------------------------------
    'prose_bpb': _m(
        'Bits per UTF-8 byte on 200 held-out period documents. THE headline metric.',
        'bits/byte',
        'lower_is_better',
        primary=True,
    ),
    'prose_bpb_early': _m(
        'prose_bpb over the FIRST part of each document (little context available).',
        'bits/byte',
        'lower_is_better',
    ),
    'prose_bpb_late': _m(
        'prose_bpb over the LATER part of each document (full context available). '
        'The early-minus-late gap measures how much the model exploits long context.',
        'bits/byte',
        'lower_is_better',
    ),
    'prose_bpb_split_a': _m(
        'prose_bpb over a stable hash-based half of the documents. A DIAGNOSTIC only: '
        'A and B should be close; a large gap means the estimate is unstable.',
        'bits/byte',
        'lower_is_better',
        caveat='Not a train/test split. Do not report as a separate result.',
    ),
    'prose_bpb_split_b': _m(
        'The other diagnostic half. See prose_bpb_split_a.',
        'bits/byte',
        'lower_is_better',
        caveat='Not a train/test split. Do not report as a separate result.',
    ),
    'chat_bpb': _m(
        'Bits per byte of assistant TARGETS only, conditioned on the dialogue context. '
        'Predicts how cheaply the model can be fine-tuned for chat.',
        'bits/byte',
        'lower_is_better',
    ),
    'prose_nonfinite_docs': _m(
        'Documents whose scoring produced NaN/Inf and were EXCLUDED from prose_bpb.',
        'count',
        'lower_is_better',
        caveat='Non-zero means prose_bpb was computed over FEWER documents than n_prose_docs, '
        'so it is not strictly comparable with a clean run. Re-run the checkpoint; this '
        'has been observed to be transient.',
    ),
    'n_prose_docs': _m('Documents actually scored for prose_bpb.', 'count', 'neutral'),
    'n_chat_docs': _m('Chat turns actually scored for chat_bpb.', 'count', 'neutral'),
    # ---- composite ---------------------------------------------------------
    'bake_score': _m(
        'Weighted composite: prose_bpb 0.50, logic 0.25, chat 0.15, hygiene 0.10.',
        'points',
        'higher_is_better',
        rng=[0, 100],
        caveat='Check bake_is_partial FIRST. Weights were fitted to correlate with TRAINING '
        'STEP on our own checkpoints, so it tracks pretraining progress well and is a '
        'poor ranking for finished third-party models. Prefer prose_bpb.',
    ),
    'bake_is_partial': _m(
        'True when a bake component could not be measured (e.g. --gen-mode none skips hygiene).',
        'bool',
        'neutral',
        caveat='If true, bake_score is renormalised over the measured weight only and is '
        'INFLATED. It cannot be compared with a full score.',
    ),
    'bake_components_missing': _m(
        'Comma-separated bake components that were not measured, or null.',
        'string',
        'neutral',
    ),
    'bake_weight_covered': _m(
        'Fraction of the total bake weight that was actually measured. 1.0 = complete.',
        'fraction',
        'higher_is_better',
        rng=[0, 1],
    ),
    'verdict_tier': _m(
        'Qualitative bucket derived from bake_score (DOUGH < HALF-BAKED < GOLDEN CRUST < BAKED).',
        'string',
        'neutral',
    ),
    'points_bpb': _m(
        'prose_bpb mapped onto the 0-100 bake ladder.',
        'points',
        'higher_is_better',
        rng=[0, 100],
        caveat='NOT a bits/byte value. The raw measurement is prose_bpb.',
    ),
    'points_logic': _m('logic_acc mapped onto 0-100.', 'points', 'higher_is_better', rng=[0, 100]),
    'points_chat': _m('chat_bpb mapped onto 0-100.', 'points', 'higher_is_better', rng=[0, 100]),
    'points_hygiene': _m('Generation hygiene mapped onto 0-100.', 'points', 'higher_is_better', rng=[0, 100]),
    # ---- reasoning / period boundary ---------------------------------------
    'logic_acc': _m(
        'Forced-choice accuracy: picks the sensible continuation over matched nonsense. 0.50 = chance.',
        'fraction',
        'higher_is_better',
        rng=[0, 1],
        caveat='Small item set: 1 s.d. is about 7.9 percentage points. Differences under ~15pp are not meaningful.',
    ),
    'logic_margin': _m(
        'Mean log-probability margin between the sensible and nonsense continuation.',
        'nats',
        'higher_is_better',
    ),
    'trap_mean_shock': _m(
        'Mean extra cost of a POST-1900 word versus its period twin. A vintage model should find modern words expensive.',
        'bits/byte',
        'higher_is_better',
    ),
    'trap_min_shock': _m('The weakest trap pair - the period boundary is only as good as this.', 'bits/byte', 'higher_is_better'),
    'trap_n_leaked': _m(
        'Trap pairs where the POST-1900 word was NOT more expensive, i.e. modern text leaked into training. 0 is the expected value.',
        'count',
        'lower_is_better',
    ),
    'trap_n_pairs': _m('Trap pairs evaluated.', 'count', 'neutral'),
    # ---- period fidelity probes --------------------------------------------
    'probe_modern_over_historical_ratio': _m(
        'Perplexity on modern probe sentences divided by perplexity on historical ones. >1 means period text is easier, which is the goal.',
        'ratio',
        'higher_is_better',
        caveat='Only 20 fixed sentences. Read the RATIO; the absolute perplexities are noise-level.',
    ),
    'probe_historical_ppl': _m(
        'Perplexity on 10 fixed historical probe sentences.',
        'perplexity',
        'lower_is_better',
        caveat='Tiny sample. Use the ratio, not this.',
    ),
    'probe_modern_ppl': _m(
        'Perplexity on 10 fixed modern probe sentences.', 'perplexity', 'higher_is_better', caveat='Tiny sample. Use the ratio, not this.'
    ),
    'probe_mean_token_prob': _m(
        'Mean probability assigned to the correct token over the probes.', 'probability', 'higher_is_better', rng=[0, 1]
    ),
    'probe_frac_low_confidence': _m(
        'Fraction of probe tokens predicted with probability < 0.01.', 'fraction', 'lower_is_better', rng=[0, 1]
    ),
    'probe_mean_entropy_nats': _m('Mean predictive entropy over the probes; high = hedging.', 'nats', 'neutral'),
    # ---- representation ----------------------------------------------------
    'sense_shift_mean_cosine': _m(
        'Cosine similarity of the SAME shifted word (gay, awful, python...) in a period versus '
        'a modern sentence. LOW means the model represents the two senses differently, which is '
        'what a period model should do.',
        'cosine',
        'lower_is_better',
        rng=[-1, 1],
    ),
    'embedding_mean_norm': _m('Mean L2 norm of input embeddings. Descriptive only.', 'norm', 'neutral'),
    'embedding_mean_cosine': _m(
        'Mean pairwise cosine between sampled embeddings; grows from ~0 at init.',
        'cosine',
        'neutral',
        caveat='Near 0 does NOT mean healthy - an untrained model also scores ~0.',
    ),
    # ---- generation hygiene ------------------------------------------------
    'sampled_temperature': _m('Temperature used for the `sampled_*` metrics below.', 'temperature', 'neutral'),
    'sampled_unusable_rate': _m(
        'Fraction of sampled completions that are degenerate OR back matter - the closest '
        'proxy here for what a synth-data filter would DROP.',
        'fraction',
        'lower_is_better',
        rng=[0, 1],
    ),
    'sampled_degenerate_rate': _m(
        'Fraction of sampled completions failing a surface-collapse gate (too short, low distinct-2, high echo, or a long repeat loop).',
        'fraction',
        'lower_is_better',
        rng=[0, 1],
    ),
    'sampled_back_matter_rate': _m(
        'Fraction of completions that are index / catalogue / table-of-contents text rather '
        'than prose. Such text is lexically DIVERSE, so distinct-n, echo and loop detection are '
        'all blind to it.',
        'fraction',
        'lower_is_better',
        rng=[0, 1],
        caveat='Precise but not exhaustive: 0.00 means "none detected", not "none present". '
        'Calibrated at 48.7% recall with 0.00% false positives on 17,085 real completions.',
    ),
    'sampled_self_bleu_4': _m(
        "Fraction of each completion's 4-grams that also occur in ANOTHER completion. "
        'Mode-collapse detector: distinct-n only looks INSIDE one completion, so many '
        'near-identical completions each still score as diverse.',
        'fraction',
        'lower_is_better',
        rng=[0, 1],
    ),
    'sampled_mean_distinct_1': _m('Mean unique-word fraction within a completion.', 'fraction', 'higher_is_better', rng=[0, 1]),
    'sampled_mean_distinct_2': _m('Mean unique-bigram fraction within a completion.', 'fraction', 'higher_is_better', rng=[0, 1]),
    'sampled_mean_echo_rate': _m('Mean fraction of words repeated within the previous 4 words.', 'fraction', 'lower_is_better', rng=[0, 1]),
    'sampled_prompt_copy_rate': _m('Mean fraction of content words copied from the prompt.', 'fraction', 'lower_is_better', rng=[0, 1]),
    'sampled_punct_issues_p100': _m('Mechanical punctuation breakage per 100 words.', 'count/100 words', 'lower_is_better'),
    'sampled_mean_sentence_words': _m(
        'Mean sentence length in generated text. Very low values indicate list/catalogue output.',
        'words',
        'neutral',
    ),
    'sampled_mean_loop_words': _m(
        'Mean longest immediately-repeating word block under SAMPLED decoding. The temperature sweep reports this per temperature.',
        'words',
        'lower_is_better',
    ),
    'greedy_mean_loop_words': _m(
        'Mean longest immediately-repeating word block under GREEDY decoding. Greedy is the '
        'harshest loop test; an undertrained model loops badly here.',
        'words',
        'lower_is_better',
    ),
    'greedy_worst_loop_words': _m('Worst single prompt loop length under greedy decoding.', 'words', 'lower_is_better'),
    'greedy_self_bleu_4': _m('self_bleu_4 over greedy completions.', 'fraction', 'lower_is_better', rng=[0, 1]),
    'greedy_back_matter_rate': _m('back_matter_rate over greedy completions.', 'fraction', 'lower_is_better', rng=[0, 1]),
    # ---- tokenizer ---------------------------------------------------------
    'tokenizer_bytes_per_token': _m(
        'Mean UTF-8 bytes per token on the held-out text. HIGHER = more text per token budget '
        '= better compression. Check this before blaming a tokenizer for a quality gap.',
        'bytes/token',
        'higher_is_better',
    ),
    'tokenizer_vocab_size': _m('Tokenizer vocabulary size.', 'count', 'neutral'),
    # ---- training lineage --------------------------------------------------
    'final_eval_loss': _m(
        'Final eval loss recorded in trainer_state.json.',
        'nats/token',
        'lower_is_better',
        comparable=SAME_DATA,
        caveat="COMPUTED ON EACH RUN'S OWN VALIDATION SPLIT. Comparing this across models "
        'trained on different datasets is MEANINGLESS. Use prose_bpb instead. Also note '
        'eval cadence is in MINUTES, so runs of different speed evaluate at different '
        'step numbers.',
    ),
    'final_eval_ppl': _m(
        'exp(final_eval_loss).',
        'perplexity',
        'lower_is_better',
        comparable=SAME_DATA,
        caveat='Same own-validation-split caveat as final_eval_loss.',
    ),
    'final_eval_step': _m('Optimizer step of the last recorded eval.', 'count', 'neutral'),
    'final_train_loss': _m('Last recorded training loss.', 'nats/token', 'lower_is_better', comparable=SAME_DATA),
    'grad_norm_max': _m('Largest gradient norm seen in training. A stability signal.', 'norm', 'lower_is_better', comparable=SAME_DATA),
    'grad_norm_mean': _m('Mean gradient norm over training.', 'norm', 'neutral', comparable=SAME_DATA),
    'grad_norm_min': _m('Smallest gradient norm over training.', 'norm', 'neutral', comparable=SAME_DATA),
    'grad_norm_nonfinite': _m(
        'Count of NaN/Inf gradient norms. ANY non-zero value means the run was unstable.',
        'count',
        'lower_is_better',
    ),
    'tokens_seen_estimate': _m(
        'Training tokens, derived from total_flos / (6 * non-embedding params).',
        'tokens',
        'neutral',
        caveat='An ESTIMATE. If total_flos is absent it falls back to a lower bound and the lineage records tokens_lower_bound=true.',
    ),
    'tokens_per_param': _m(
        'tokens_seen_estimate divided by parameter count. The compute-optimal rule of thumb is '
        'about 20; well below that means undertrained by construction.',
        'tokens/param',
        'higher_is_better',
    ),
}


def sweep_entry(key: str) -> dict | None:
    """Describe a `sweep_t<T>_<metric>` key by delegating to its base metric."""
    if not key.startswith('sweep_t'):
        return None
    rest = key[len('sweep_t') :]
    temp, _, metric = rest.partition('_')
    base = GUIDE.get(f'sampled_{metric}')
    if base is None:
        return None
    out = dict(base)
    out['desc'] = f'[temperature {temp}] {base["desc"]}'
    return out


def describe(keys) -> dict[str, dict]:
    """Guide entries for exactly the keys present in a summary."""
    out = {}
    for key in keys:
        entry = GUIDE.get(key) or sweep_entry(key)
        if entry is not None:
            out[key] = entry
    return out


def undocumented(keys) -> list[str]:
    """Summary keys with no guide entry - a bug, not a warning."""
    return sorted(k for k in keys if k not in GUIDE and sweep_entry(k) is None)

"""Units and comparison conditions for every flat summary key; audit with --audit-guide."""

from __future__ import annotations

PRIMARY_METRIC = 'prose_bpb'
HOW_TO_READ = (
    'prose_bpb is sum(bits)/sum(scored UTF-8 bytes), not token perplexity. '
    'Compare matching documents, scored spans and context settings; rankings are within protocol groups. '
    'Trainer loss needs matching validation data, tokenizer and reduction. '
    'Surface flags, phrase preferences and embedding similarities are diagnostics, not usability/readiness/leakage tests. '
    'bake_score is an experimental composite; only complete, protocol-matched scores are ranked. '
    'Missing/non-finite measurements are null, never zero. See metric_guide for each key.'
)


def _m(desc, unit, direction='neutral', compare='same measurement protocol', caveat=None):
    entry = dict(desc=desc, unit=unit, direction=direction, comparable_across=[compare])
    if caveat:
        entry['caveat'] = caveat
    return entry


BPB_COMPARE = 'same documents, scored text spans and conditioning/window settings'
GEN_COMPARE = 'same ordered prompts, sample count, length budget, seed, template and decoding settings'
TRAIN_COMPARE = 'same validation/training data, tokenizer, loss reduction and logging convention'
GUIDE = {
    'prose_bpb': _m('Total finite scored bits / total scored UTF-8 bytes.', 'bits/byte', 'lower_is_better', BPB_COMPARE),
    'prose_uniform_token_bpb': _m(
        'log2(model config vocabulary size) * prose_scored_tokens / prose_scored_bytes, using the same finite document pool.',
        'bits/byte',
        compare=BPB_COMPARE,
        caveat='Calculated equal-probability token baseline, not a measured untrained network or a tokenizer-independent constant.',
    ),
    'prose_bpb_early': _m('BPB of the first 128 scored tokens per document.', 'bits/byte', compare=BPB_COMPARE),
    'prose_bpb_late': _m(
        'BPB after the first 128 scored tokens; different positions, not a controlled context-length test.',
        'bits/byte',
        compare=BPB_COMPARE,
    ),
    'chat_target_bpb': _m(
        'Target token NLL / decoded target bytes, conditioned on context. Boundary-crossing tokens are included; long inputs drop tokens from the left.',
        'bits/byte',
        'lower_is_better',
        BPB_COMPARE,
        'Teacher-forced likelihood, not chat quality or fine-tuning readiness.',
    ),
    'logic_accuracy': _m(
        'Fraction of scored pairs with good BPB < bad BPB; ties incorrect, random two-choice baseline 0.5.',
        'fraction',
        'higher_is_better',
        'same fixed-choice items and span scoring',
        'Small fixed set; report items_scored, not general reasoning ability.',
    ),
    'logic_accuracy_ci_low': _m(
        'Lower Wilson score bound on logic_accuracy.', 'fraction', compare='same fixed-choice items and span scoring'
    ),
    'logic_accuracy_ci_high': _m(
        'Upper Wilson score bound on logic_accuracy.', 'fraction', compare='same fixed-choice items and span scoring'
    ),
    'logic_accuracy_ci_confidence': _m('Confidence level of the Wilson interval on logic_accuracy; fixed at 0.95.', 'fraction'),
    'logic_margin_bpb': _m(
        'Mean bad-minus-good continuation BPB.', 'bits/byte', 'higher_is_better', 'same fixed-choice items and span scoring'
    ),
    'trap_mean_delta_bpb': _m(
        'Mean modern-minus-period phrase BPB over the fixed pairs.', 'bits/byte', caveat='Phrase preference, not a leakage detector.'
    ),
    'trap_mean_delta_stderr_bpb': _m(
        'Standard error of trap_mean_delta_bpb over the scored pairs; null with fewer than two.',
        'bits/byte',
        caveat='These sets have single-digit sample sizes; read the mean with this beside it.',
    ),
    'trap_min_delta_bpb': _m('Minimum modern-minus-period phrase BPB.', 'bits/byte'),
    'trap_nonpositive_pairs': _m('Scored pairs with modern phrase BPB <= period phrase BPB (includes ties).', 'count'),
    'probe_modern_historical_ppl_ratio': _m(
        'Modern token perplexity / historical token perplexity.',
        'ratio',
        compare='same tokenizer and fixed sentence sets',
        caveat='Not a knowledge-cutoff or contamination test.',
    ),
    'embedding_mean_norm': _m('Mean L2 norm of all input-token embedding vectors.', 'norm'),
    'embedding_mean_cosine': _m('Mean off-diagonal pairwise cosine among up to 512 seed-selected input embeddings.', 'cosine'),
    'sense_shift_mean_cosine': _m(
        'Mean cosine of the same words in paired historical/modern contexts.',
        'cosine',
        caveat='No established good/bad direction or proof of sense understanding.',
    ),
    'sense_historical_pairwise_mean_cosine': _m('Mean cosine between different probe words in historical contexts.', 'cosine'),
    'sense_historical_pairwise_std_cosine': _m('Population standard deviation of those pairwise cosines.', 'cosine'),
    'tokenizer_bytes_per_token': _m(
        'Total decoded UTF-8 bytes / tokens in sampled document prefixes, before model-context clipping.',
        'bytes/token',
        compare='same input documents and token budget',
        caveat='Not training throughput or model quality. May differ from prose-scored spans.',
    ),
    'tokenizer_vocab_size': _m('Tokenizer vocabulary size including added tokens.', 'count'),
    'tokenizer_tokens_scored': _m('Tokens counted in tokenizer prefix diagnostic.', 'tokens'),
    'tokenizer_bytes_scored': _m('Decoded bytes counted in tokenizer prefix diagnostic.', 'bytes'),
    'tokens_seen_estimate': _m(
        'Training-token estimate from FLOPs / (6 * non-embedding params), or recorded fallback.',
        'tokens',
        caveat='See lineage for method and lower-bound flags.',
    ),
    'tokens_per_param': _m('Estimated training tokens / total parameters; not a sufficiency threshold.', 'tokens/param'),
    'trainer_eval_loss_nats_per_token': _m('Last logged trainer validation cross-entropy.', 'nats/token', 'lower_is_better', TRAIN_COMPARE),
    'trainer_train_loss_nats_per_token': _m('Last logged trainer training loss.', 'nats/token', 'lower_is_better', TRAIN_COMPARE),
    'trainer_eval_ppl': _m('exp(trainer_eval_loss_nats_per_token).', 'perplexity', 'lower_is_better', TRAIN_COMPARE),
    'trainer_eval_step': _m('Optimizer step of the last logged validation loss.', 'count'),
    'grad_norm_nonfinite': _m('Count of logged non-finite gradient norms; null if no gradient norms were recorded.', 'count'),
    'grad_norm_count': _m('Number of logged gradient norms, including non-finite values.', 'count'),
    'bake_score': _m(
        'Experimental weighted component score: BPB .50, logic .25, chat-target .15, generation .10; renormalized over finite components.',
        'points',
        caveat='Fixed hand-set transforms; not a validated quality scale. Partial and full scores are not comparable.',
    ),
    'bake_is_partial': _m('At least one composite component is missing/non-finite.', 'bool'),
    'bake_components_missing': _m('Comma-separated missing composite components, or null when complete.', 'string'),
    'bake_weight_covered': _m('Sum of weights with finite component scores.', 'fraction'),
    'bake_status': _m('complete (all four finite), partial (some), or unscored (none).', 'string'),
    'sampled_temperature': _m('Temperature for the primary sampled pass.', 'temperature'),
    'judge_reference_bpb': _m('Reference-model BPB on up to 100 reference documents, prefix budget 320 tokens.', 'bits/byte'),
    'judge_generated_bpb': _m(
        'Reference-model BPB on sampled continuations with at least 20 whitespace-separated words, prefix budget 320 tokens.', 'bits/byte'
    ),
    'judge_absolute_bpb_difference': _m(
        'Absolute generated-minus-reference BPB difference under the judge.',
        'bits/byte',
        caveat='Neither a coherence score nor a semantic similarity measurement.',
    ),
}
GUIDE['prose_bpb']['is_primary'] = True
GUIDE['prose_bpb_position_docs'] = _m(
    'Documents contributing to BOTH prose_bpb_early and prose_bpb_late; documents with no second half are in neither.',
    'count',
)
for split in ('a', 'b'):
    GUIDE['prose_bpb_split_' + split] = _m(
        'BPB on content-hash diagnostic split ' + split.upper() + '; not a train/test split.', 'bits/byte', compare=BPB_COMPARE
    )
for prefix in ('prose', 'chat_target'):
    for suffix, desc, unit in (
        ('docs_requested', 'Loaded inputs requested for scoring.', 'count'),
        ('docs_scored', 'Positive-byte records with finite bit counts included in aggregate.', 'count'),
        (
            'docs_excluded',
            'Retained records excluded for invalid bits/bytes or zero bytes; does not include inputs skipped before recording.',
            'count',
        ),
        ('nonfinite_docs', 'Retained records with missing/NaN/Inf bit counts.', 'count'),
        ('scored_bytes', 'Total bytes included in aggregate.', 'bytes'),
        ('scored_tokens', 'Total predicted tokens in included records.', 'tokens'),
        ('truncated_docs', 'Retained records whose input was clipped to the effective context budget.', 'count'),
        (
            'docs_skipped',
            'Inputs dropped before any record existed: prose shorter than the scoring minimum, or chat items with no scoreable target token.',
            'count',
        ),
    ):
        GUIDE[prefix + '_' + suffix] = _m(desc, unit)
GUIDE['chat_input_truncated_docs'] = _m('Retained chat inputs clipped from the left to fit the context budget.', 'count')
GUIDE['chat_target_truncated_docs'] = _m('Retained chat records that lost target tokens through left truncation.', 'count')
for key in ('items_scored', 'items_correct', 'items_skipped'):
    GUIDE['logic_' + key] = _m('Fixed-choice ' + key.replace('_', ' ') + '.', 'count')
for key in ('pairs_scored', 'pairs_skipped'):
    GUIDE['trap_' + key] = _m('Fixed trap ' + key.replace('_', ' ') + '.', 'count')
for key in ('max', 'mean', 'min'):
    GUIDE['grad_norm_' + key] = _m(key.capitalize() + ' of finite logged gradient norms.', 'norm', compare=TRAIN_COMPARE)
for key in ('bpb', 'logic', 'chat', 'hygiene'):
    GUIDE['points_' + key] = _m('Fixed 0–100 transform for composite component ' + key + '; not a raw likelihood.', 'points')
for suffix, desc, unit in (
    ('bpb_ci_low', 'Lower document-bootstrap percentile bound on prose BPB.', 'bits/byte'),
    ('bpb_ci_high', 'Upper document-bootstrap percentile bound on prose BPB.', 'bits/byte'),
    ('bpb_ci_confidence', 'Confidence level used for prose and paired-delta intervals.', 'fraction'),
    ('delta_vs_leader_bpb', 'Candidate-minus-group-leader BPB.', 'bits/byte'),
    ('delta_ci_low_bpb', 'Lower paired-bootstrap percentile bound on candidate-minus-leader BPB.', 'bits/byte'),
    ('delta_ci_high_bpb', 'Upper paired-bootstrap percentile bound on candidate-minus-leader BPB.', 'bits/byte'),
    ('equivalence_margin_bpb', 'Absolute practical tolerance applied to the entire paired CI.', 'bits/byte'),
    ('paired_docs', 'Number of matching finite documents in paired bootstrap.', 'count'),
    (
        'bootstrap_fraction_lower_than_leader',
        'Fraction of paired bootstrap replicates with candidate BPB < leader BPB; NOT a posterior probability or p-value.',
        'fraction',
    ),
    ('comparison_group', 'Identity of matched scoring protocol and document coverage.', 'string'),
    ('comparison_leader', 'Lowest measured BPB model within this comparison group.', 'string'),
    (
        'comparison_to_leader',
        'leader, equivalent, better, worse, inconclusive, or unavailable, based on the whole paired interval and tolerance.',
        'string',
    ),
    ('comparison_unavailable_reason', 'Why this result could not be placed in a comparison group; null when it was.', 'string'),
):
    for prefix in ('prose', 'chat_target'):
        GUIDE[prefix + '_' + suffix] = _m(desc, unit, compare=BPB_COMPARE)

GENERATION = {
    'n_prompts': ('Number of continuations in this aggregate.', 'count'),
    'surface_failure_rate': ('Fraction with any short-text, repetition or back-matter-like flag.', 'fraction'),
    'degenerate_rate': ('Fraction with any short-text/repetition gate, excluding formatting-only flags.', 'fraction'),
    'back_matter_rate': ('Fraction flagged by back-matter-like formatting heuristics; valid short prose can trigger this.', 'fraction'),
    'shared_4gram_fraction': (
        'Mean fraction of each eligible completion’s unique lowercased word 4-grams also in another completion; first 120 completions, at least 4 words each, at least 2 eligible required.',
        'fraction',
    ),
    'mean_echo_rate': ('Mean fraction of words occurring among their previous 4 words.', 'fraction'),
    'worst_echo_rate': ('Maximum within-completion echo fraction.', 'fraction'),
    'worst_distinct_2': ('Minimum within-completion unique/total bigram fraction.', 'fraction'),
    'mean_loop_words': ('Mean longest contiguous periodic word run; periods 1–12, at least 2 repeats.', 'words'),
    'worst_loop_words': ('Maximum such periodic run over completions.', 'words'),
    'mean_punct_issues_p100': (
        'Mean per-completion punctuation-pattern count per 100 regex words (equal completion weight).',
        'count/100 words',
    ),
    'total_punct_issues': ('Total regex punctuation-pattern matches.', 'count'),
    'mean_prompt_word_overlap_rate': (
        'Mean fraction of continuation regex words longer than 3 letters found in the prompt (case-insensitive); overlap, not proof of copying.',
        'fraction',
    ),
    'mean_words': ('Mean count of regex English words per completion.', 'words'),
    'mean_digit_frac': ('Mean fraction of all characters (including whitespace) that are digits.', 'fraction'),
    'mean_caps_frac': ('Mean fraction of regex words that are all-uppercase and longer than 1 letter.', 'fraction'),
    'mean_sentence_words': ('Mean whitespace-separated words per punctuation-delimited sentence, averaged over completions.', 'words'),
}
for n in (1, 2, 3):
    GENERATION[f'mean_distinct_{n}'] = (f'Mean unique/total within-completion word {n}-gram fraction.', 'fraction')
for reason, desc in {
    'short_text': '<8 regex words',
    'low_bigram_diversity': 'distinct-2 <0.55',
    'local_repetition': 'echo fraction >0.30',
    'repeat_loop': 'periodic run >=12 words',
    'back_matter_like': 'back-matter formatting heuristic',
}.items():
    GENERATION[reason + '_rate'] = ('Fraction flagged for ' + desc + '; reasons may overlap.', 'fraction')
for mode in ('greedy', 'sampled'):
    for metric, (desc, unit) in GENERATION.items():
        GUIDE[f'{mode}_{metric}'] = _m(desc, unit, compare=GEN_COMPARE, caveat='Surface statistics can flag valid text and miss nonsense.')
PROBE = {
    'ppl': ('exp(mean token NLL in nats).', 'perplexity'),
    'bpb': ('Total predicted bits / decoded scored UTF-8 bytes.', 'bits/byte'),
    'tokens_scored': ('Number of predicted tokens.', 'tokens'),
    'sentences_scored': ('Number of fixed sentences.', 'count'),
    'mean_token_prob': ('Mean probability assigned to the observed target token.', 'probability'),
    'median_token_prob': ('Median target-token probability.', 'probability'),
    'p10_token_prob': ('10th percentile target-token probability.', 'probability'),
    'min_token_prob': ('Minimum target-token probability.', 'probability'),
    'token_prob_below_0_01_rate': ('Fraction of target tokens assigned probability <0.01.', 'fraction'),
    'mean_entropy_nats': ('Mean entropy of the next-token distribution.', 'nats'),
}
for group in ('historical', 'modern', 'overall'):
    for metric, (desc, unit) in PROBE.items():
        GUIDE[f'probe_{group}_{metric}'] = _m(desc, unit, compare='same fixed sentences, tokenizer and special-token handling')


def entry_for(key: str) -> dict | None:
    if key in GUIDE:
        return GUIDE[key]
    if key.startswith('sweep_t'):
        temp, _, metric = key[7:].partition('_')
        base = GUIDE.get('sampled_' + metric)
        if base:
            return {**base, 'desc': f'[temperature {temp}] {base["desc"]}'}
    if key.startswith('logic_category_'):
        if key.endswith('_accuracy'):
            return _m('Forced-choice accuracy for this category.', 'fraction')
        if key.endswith('_items_scored'):
            return _m('Scored fixed-choice items in this category.', 'count')
    if key.startswith('sense_shift_') and key.endswith('_cosine'):
        return _m('Cosine for this word in paired historical/modern contexts; no established quality direction.', 'cosine')
    return None


def describe(keys) -> dict[str, dict]:
    return {key: entry for key in keys if (entry := entry_for(key)) is not None}


def undocumented(keys) -> list[str]:
    return sorted(key for key in keys if entry_for(key) is None)

"""Flat summaries and composite scores, derived from current measurements."""

import math

from . import metrics as M


def summarize_result(result: dict) -> None:
    """Rebuild the summary from measurements; skipped suites contribute no values."""
    carried = ('prose_docs_requested', 'chat_target_docs_requested', 'prose_docs_skipped', 'chat_target_docs_skipped')
    s = {key: result[key] for key in carried if key in result}

    def add_scalars(prefix: str, block: dict) -> None:
        s.update({prefix + key: value for key, value in block.items() if isinstance(value, (int, float)) or value is None})

    for block, prefix in (('prose_records', 'prose'), ('chat_records', 'chat_target')):
        if block not in result:
            continue
        records = result[block]
        valid = M.finite_records(records)
        s[prefix + '_bpb'] = M.records_bpb(records)
        s[prefix + '_docs_scored'] = len(valid)
        s[prefix + '_docs_excluded'] = len(records) - len(valid)
        s[prefix + '_nonfinite_docs'] = M.count_nonfinite_records(records)
        s[prefix + '_scored_bytes'] = sum(r['bytes'] for r in valid)
        s[prefix + '_scored_tokens'] = sum(r['tokens'] for r in valid)
        truncation_key = 'prose_truncated_docs' if prefix == 'prose' else 'chat_input_truncated_docs'
        s[truncation_key] = sum(r['truncated'] for r in records)
        if prefix == 'prose':
            vocab = (result.get('info') or {}).get('vocab_size_config')
            # Equal probability over model output IDs, on the same finite scored pool.
            s['prose_uniform_token_bpb'] = (
                math.log2(vocab) * s['prose_scored_tokens'] / s['prose_scored_bytes']
                if M.is_finite(vocab) and vocab > 0 and s['prose_scored_bytes'] > 0
                else None
            )
            for split in ('A', 'B'):
                s['prose_bpb_split_' + split.lower()] = M.records_bpb(records, split)
            # Both positions must come from the SAME documents: a document with no
            # second half would otherwise sit in `early` and be missing from `late`.
            positional = [
                r for r in M.finite_records(records, 'early_') if M.is_finite(r.get('late_bits')) and (r.get('late_bytes') or 0) > 0
            ]
            for position in ('early', 'late'):
                s['prose_bpb_' + position] = M.records_bpb(positional, prefix=position + '_')
            s['prose_bpb_position_docs'] = len(positional)
        else:
            s['chat_target_truncated_docs'] = sum(r['target_truncated'] for r in records)

    logic = result.get('logic') or {}
    add_scalars('logic_', logic)
    for category, measurements in logic.get('categories', {}).items():
        add_scalars('logic_category_' + category + '_', measurements)
    add_scalars('trap_', result.get('traps') or {})

    probes = result.get('period_probes') or {}
    for group in ('historical', 'modern', 'overall'):
        add_scalars('probe_' + group + '_', probes.get(group) or {})
    if 'modern_historical_ppl_ratio' in probes:
        s['probe_modern_historical_ppl_ratio'] = probes['modern_historical_ppl_ratio']

    emb = result.get('embeddings') or {}
    add_scalars('sense_', emb)
    s.update({'sense_shift_' + word + '_cosine': value for word, value in emb.get('shift_cosines', {}).items()})
    add_scalars('embedding_', result.get('embedding_stats') or {})
    lineage = result.get('lineage') or {}
    s['tokens_seen_estimate'] = lineage.get('tokens_seen')
    s['tokens_per_param'] = lineage.get('tokens_per_param')
    add_scalars('tokenizer_', result.get('tokenizer_stats') or {})
    for key, value in (result.get('training_curve') or {}).items():
        if not isinstance(value, list):
            s[key if key.startswith('grad_norm_') else 'trainer_' + key] = value

    generation = result.get('generation') or {}
    if generation.get('sampled_summary'):
        s['sampled_temperature'] = generation['temperature']
    for mode in ('greedy', 'sampled'):
        add_scalars(mode + '_', generation.get(mode + '_summary') or {})
    for temp, aggregate in (generation.get('temperature_sweep') or {}).items():
        add_scalars('sweep_t' + temp + '_', aggregate)
    add_scalars('judge_', (result.get('judge') or {}).get('scores') or {})
    for prefix in ('prose', 'chat_target'):
        s.update({prefix + '_' + key: value for key, value in (result.get(prefix + '_comparison') or {}).items()})

    points = {}
    if M.is_finite(s.get('prose_bpb')):
        points['bpb'] = M.interp(s['prose_bpb'], M.BPB_LADDER)
    if M.is_finite(s.get('logic_accuracy')):
        points['logic'] = M.logic_points(s['logic_accuracy'])
    if M.is_finite(s.get('chat_target_bpb')):
        points['chat'] = M.interp(s['chat_target_bpb'], M.CHAT_LADDER)
    loop, punct = s.get('greedy_mean_loop_words'), s.get('sampled_mean_punct_issues_p100')
    if M.is_finite(loop) and M.is_finite(punct):
        points['hygiene'] = M.hygiene_points(loop, punct)
    missing = [key for key in M.WEIGHTS if key not in points]
    result.update(points=points, bake_score=M.bake_score(points), bake_components_missing=missing, bake_is_partial=bool(missing))
    s.update(
        bake_score=result['bake_score'],
        bake_components_missing=','.join(missing) or None,
        bake_is_partial=bool(missing),
        bake_weight_covered=sum(M.WEIGHTS[key] for key in points),
        bake_status='complete' if not missing else 'partial' if points else 'unscored',
    )
    s.update({'points_' + key: value for key, value in points.items()})
    result['summary'] = s

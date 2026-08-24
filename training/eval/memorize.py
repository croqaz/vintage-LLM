"""Verbatim-continuation memorization probe.

WHY: a model that trained on a corpus CONTAINING the held-out documents cannot
have its held-out loss read as clean generalisation. But "saw it once" is not
"memorised it", and the difference is measurable rather than arguable.

Capacity is usually the deciding factor. A model stores on the order of a couple
of bits per parameter, so a small model against a large corpus seen roughly once
has nowhere to put verbatim text. Two cheap signals point the same way before you
run anything: whether the early-vs-late context gap is unusual compared with a
model that did NOT see the data, and whether held-out loss sits where clean
scaling predicts for that parameter count.

METHOD: feed the first `prefix_tokens` of a held-out document, greedy-decode
`gen_tokens`, and compare against the document's TRUE continuation at the token
level. A model reciting from memory emits long exact spans; a model generalising
agrees for a few tokens by chance and then diverges.

ALWAYS RUN A CONTROL that did not train on these documents. Absolute span length
is not interpretable on its own -- natural prose is formulaic, and short
agreements happen constantly. Only the paired, per-document DIFFERENCE against
the control is evidence.

Memorisation shows up as a HEAVY RIGHT TAIL (a few documents recited at length),
not a small uniform shift. If the control is also the weaker model, expect a
modest positive delta on capability grounds alone.

CLI: see probe_memorization.py in this package.
"""

import numpy as np
import torch

from .helpers import TextItem, model_context_limit, prefix_id


def _longest_common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _longest_common_span(a: list[int], b: list[int]) -> int:
    """Longest contiguous token run appearing in both, anywhere.

    O(len(a) * len(b)) with a rolling row; both are <= a few hundred tokens.
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


@torch.no_grad()
def verbatim_probe(
    tokenizer,
    model,
    items: list[TextItem],
    prefix_tokens: int = 256,
    gen_tokens: int = 128,
    batch_size: int = 8,
    progress_every: int = 50,
) -> dict:
    """Greedy-continue each document from its own opening; score exact overlap.

    Returns aggregate stats plus per-document spans, so two models can be
    compared document-by-document (paired) rather than only on means.
    """
    pre = prefix_id(tokenizer, model)
    limit = model_context_limit(model, prefix_tokens + gen_tokens)
    n_prefix = min(prefix_tokens, max(16, limit - gen_tokens - 1))

    records: list[dict] = []
    pending: list[tuple[str, list[int], list[int]]] = []

    def flush(batch):
        if not batch:
            return
        width = max(len(p) for _, p, _ in batch)
        pad = tokenizer.pad_token_id or 0
        # LEFT pad so every row's generation starts at the same position.
        input_ids = torch.tensor([[pad] * (width - len(p)) + p for _, p, _ in batch], dtype=torch.long, device=model.device)
        attn = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for _, p, _ in batch], dtype=torch.long, device=model.device)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=gen_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad,
        )
        for (doc_id, _prefix, truth), row in zip(batch, out, strict=True):
            got = row[width:].tolist()
            records.append(
                {
                    'id': doc_id,
                    'prefix_match': _longest_common_prefix(got, truth),
                    'longest_span': _longest_common_span(got, truth),
                    'generated_tokens': len(got),
                    'truth_tokens': len(truth),
                }
            )

    for i, item in enumerate(items, 1):
        ids = tokenizer(item.text, add_special_tokens=False).input_ids
        if len(ids) < n_prefix + 16:
            continue
        prefix = ids[:n_prefix]
        truth = ids[n_prefix : n_prefix + gen_tokens]
        if pre is not None:
            prefix = [pre] + prefix
        pending.append((item.item_id, prefix, truth))
        if len(pending) >= batch_size:
            flush(pending)
            pending = []
        if progress_every and i % progress_every == 0:
            print(f' m{i}/{len(items)}', end='', flush=True)
    flush(pending)

    if not records:
        return {'n': 0}

    prefix_matches = np.array([r['prefix_match'] for r in records], dtype=float)
    spans = np.array([r['longest_span'] for r in records], dtype=float)
    return {
        'n': len(records),
        'prefix_tokens': n_prefix,
        'gen_tokens': gen_tokens,
        # Exact agreement from the very first generated token. The strongest
        # recall signal: a reciting model continues correctly, immediately.
        'mean_prefix_match': float(prefix_matches.mean()),
        'p95_prefix_match': float(np.percentile(prefix_matches, 95)),
        'max_prefix_match': float(prefix_matches.max()),
        # Longest exact run anywhere in the continuation.
        'mean_longest_span': float(spans.mean()),
        'p95_longest_span': float(np.percentile(spans, 95)),
        'max_longest_span': float(spans.max()),
        # Formulaic period prose produces short agreements constantly; runs this
        # long are not chance.
        'frac_span_ge_16': float((spans >= 16).mean()),
        'frac_span_ge_32': float((spans >= 32).mean()),
        'records': records,
    }


def compare_probes(seen: dict, control: dict) -> dict:
    """Paired comparison of a suspect model against a never-saw-it control.

    Absolute span lengths are uninterpretable (period prose is formulaic); the
    delta against a control is the actual evidence.
    """
    by_id = {r['id']: r for r in control.get('records', [])}
    pairs = [(r, by_id[r['id']]) for r in seen.get('records', []) if r['id'] in by_id]
    if not pairs:
        return {'n_paired': 0}
    d_span = np.array([a['longest_span'] - b['longest_span'] for a, b in pairs], dtype=float)
    d_prefix = np.array([a['prefix_match'] - b['prefix_match'] for a, b in pairs], dtype=float)
    return {
        'n_paired': len(pairs),
        'mean_delta_longest_span': float(d_span.mean()),
        'mean_delta_prefix_match': float(d_prefix.mean()),
        'frac_docs_seen_model_longer': float((d_span > 0).mean()),
        # A memorising model should show a heavy right tail, not a uniform shift.
        'p95_delta_longest_span': float(np.percentile(d_span, 95)),
        'max_delta_longest_span': float(d_span.max()),
    }

"""Verbatim-continuation memorization probe: CLI.

Answers "did this model MEMORISE the held-out set, or just learn the domain?"
for any two checkpoints, by comparing a SUSPECT (which may have trained on a
corpus containing the held-out documents) against a CONTROL (which did not).

    python -m eval.probe_memorization SUSPECT CONTROL
    python -m eval.probe_memorization runs/big/final runs/small/final --out probe.json

Both arguments are checkpoint directories, resolved the same way `python -m eval`
resolves them, so `path/to/run` works as well as `path/to/run/final`.

WHY A CONTROL IS MANDATORY: absolute overlap length means nothing on its own.
Natural prose is formulaic, so any two competent models agree with the reference
continuation for a few tokens by chance. Only the PAIRED, per-document
difference against a model that never saw the data is evidence.

WHAT TO LOOK FOR: memorisation produces a HEAVY RIGHT TAIL -- a handful of
documents recited at length -- not a small uniform shift. A large p95/max delta
with a near-zero mean is the signature. A uniform small shift usually just means
one model is better than the other, so compare the models' held-out BPB before
reading anything into it.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ''):  # allow running the file directly, not just -m
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from eval.helpers import (
        DEFAULT_RESULTS_DIR,
        EVAL_DATA,
        die,
        free_model,
        load_model_and_tokenizer,
        load_text_items,
        resolve_targets,
        resolve_tokenizer,
        select_device,
        select_dtype,
    )
    from eval.memorize import compare_probes, verbatim_probe
else:
    from .helpers import (
        DEFAULT_RESULTS_DIR,
        EVAL_DATA,
        die,
        free_model,
        load_model_and_tokenizer,
        load_text_items,
        resolve_targets,
        resolve_tokenizer,
        select_device,
        select_dtype,
    )
    from .memorize import compare_probes, verbatim_probe


def one_checkpoint(target: str) -> Path:
    """Resolve a user-supplied path to exactly one checkpoint directory."""
    found = resolve_targets([Path(target)]).checkpoints
    if not found:
        die(f'no loadable checkpoint found at {target}')
    if len(found) > 1:
        die(f'{target} resolves to {len(found)} checkpoints; point at exactly one')
    return found[0]


def run_one(label: str, checkpoint: Path, items, device, dtype, args) -> dict:
    print(f'\n=== {label}: {checkpoint} ===', flush=True)
    tok_dir = resolve_tokenizer(checkpoint, args.tokenizer)
    model, tokenizer = load_model_and_tokenizer(checkpoint, tok_dir, device, dtype)
    try:
        return verbatim_probe(
            tokenizer,
            model,
            items,
            prefix_tokens=args.prefix_tokens,
            gen_tokens=args.gen_tokens,
            batch_size=args.batch_size,
        )
    finally:
        free_model(model, device)
        print(' freed', flush=True)


def render(suspect_path, control_path, suspect, control, delta, args) -> str:
    no_signature = delta.get('mean_delta_longest_span', 0) < 1.0 and delta.get('p95_delta_longest_span', 0) < 4.0
    verdict = 'NO MEMORIZATION SIGNATURE' if no_signature else 'POSSIBLE RECALL - investigate'

    lines = [
        '# Memorization probe',
        '',
        f'**Verdict: {verdict}**',
        '',
        f'* suspect: `{suspect_path}`',
        f'* control: `{control_path}`',
        f'* held-out data: `{args.heldout}` ({suspect.get("n", 0)} documents scored)',
        f'* method: feed the first {suspect.get("prefix_tokens")} tokens of each document, '
        f'greedy-decode {suspect.get("gen_tokens")}, measure exact TOKEN overlap with the true continuation.',
        '',
        'A model reciting from memory emits long exact spans. Absolute lengths are NOT',
        'interpretable on their own -- natural prose is formulaic -- so every conclusion',
        'below rests on the difference against the control.',
        '',
        '| metric | suspect | control |',
        '|---|---:|---:|',
    ]
    for key, human in (
        ('mean_prefix_match', 'mean exact prefix match (tokens)'),
        ('p95_prefix_match', 'p95 exact prefix match'),
        ('max_prefix_match', 'max exact prefix match'),
        ('mean_longest_span', 'mean longest exact span'),
        ('p95_longest_span', 'p95 longest exact span'),
        ('max_longest_span', 'max longest exact span'),
        ('frac_span_ge_16', 'fraction of docs with a span >= 16 tokens'),
        ('frac_span_ge_32', 'fraction of docs with a span >= 32 tokens'),
    ):
        lines.append(f'| {human} | {suspect.get(key, float("nan")):.3f} | {control.get(key, float("nan")):.3f} |')

    lines += [
        '',
        '## Paired difference (suspect minus control, same documents)',
        '',
        f'* documents paired: {delta.get("n_paired")}',
        f'* mean delta, longest span: **{delta.get("mean_delta_longest_span", float("nan")):+.3f} tokens**',
        f'* mean delta, prefix match: {delta.get("mean_delta_prefix_match", float("nan")):+.3f} tokens',
        f'* documents where the suspect ran longer: {100 * delta.get("frac_docs_seen_model_longer", 0):.1f}%',
        f'* p95 delta: {delta.get("p95_delta_longest_span", float("nan")):+.1f}, '
        f'max delta: {delta.get("max_delta_longest_span", float("nan")):+.1f}',
        '',
        'Reading it: memorisation produces a HEAVY RIGHT TAIL (a few documents recited at',
        'length), not a small uniform shift. A large p95/max delta with a near-zero mean is',
        'the signature to worry about. If the control is also the WEAKER model, a modest',
        'positive delta is expected on capability grounds alone -- check both models held-out BPB before concluding anything.',
        '',
    ]
    return '\n'.join(lines) + '\n'


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description='Verbatim-continuation memorization probe: does SUSPECT recite the held-out set?',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('suspect', help='Checkpoint that MAY have trained on the held-out documents.')
    p.add_argument('control', help='Checkpoint that did NOT. Required: the delta is the evidence.')
    p.add_argument('--heldout', type=Path, default=EVAL_DATA / 'heldout-Sprocket-n-Say.jsonl')
    p.add_argument('--docs', type=int, default=200, help='Held-out documents to probe.')
    p.add_argument('--prefix-tokens', type=int, default=256, help='Context tokens fed before decoding.')
    p.add_argument('--gen-tokens', type=int, default=128, help='Tokens to greedy-decode and score.')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--tokenizer', type=Path, default=None, help='Force one tokenizer for both models.')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    p.add_argument('--dtype', choices=('auto', 'float32', 'float16', 'bfloat16'), default='auto')
    p.add_argument(
        '--out',
        type=Path,
        default=None,
        help='Output JSON path. Markdown is written beside it. Default: <results dir>/memorization-probe.json',
    )
    args = p.parse_args(argv)

    if not args.heldout.is_file():
        die(f'held-out file not found: {args.heldout} (pass --heldout PATH)')

    suspect_path = one_checkpoint(args.suspect)
    control_path = one_checkpoint(args.control)
    if suspect_path == control_path:
        die('suspect and control are the same checkpoint; the comparison would be meaningless')

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    items = load_text_items(args.heldout, limit=args.docs)
    print(f'{len(items)} held-out documents from {args.heldout} on {device}/{dtype}')

    suspect = run_one('SUSPECT', suspect_path, items, device, dtype, args)
    control = run_one('CONTROL', control_path, items, device, dtype, args)
    delta = compare_probes(suspect, control)

    out_json = (args.out or DEFAULT_RESULTS_DIR / 'memorization-probe.json').resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'suspect': {'checkpoint': str(suspect_path), **suspect},
        'control': {'checkpoint': str(control_path), **control},
        'comparison': delta,
        'settings': {
            'heldout': str(args.heldout),
            'docs': args.docs,
            'prefix_tokens': args.prefix_tokens,
            'gen_tokens': args.gen_tokens,
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    out_md = out_json.with_suffix('.md')
    out_md.write_text(render(suspect_path, control_path, suspect, control, delta, args), encoding='utf-8')

    print()
    print(json.dumps({k: v for k, v in delta.items()}, indent=2))
    print(f'\nJSON: {out_json}')
    print(f'MD:   {out_md}')


if __name__ == '__main__':
    with torch.no_grad():
        main()

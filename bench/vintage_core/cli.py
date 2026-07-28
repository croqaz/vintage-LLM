"""Command-line entry point for running Vintage CORE against an API model."""

import argparse
import json
import os
import sys
import time

from . import data
from .api import evaluate, resolve_modes
from .client import APIClient
from .local import LocalClient


def build_parser():
    p = argparse.ArgumentParser(
        prog='vintage-core',
        description='Run the Vintage CORE benchmark against any OpenAI-compatible API model.',
    )
    p.add_argument('--base-url', default=None, help='API base URL, e.g. http://localhost:1234/v1 or https://openrouter.ai/api/v1')
    p.add_argument('--model', default=None, help='Model name to send in requests (optional label when --local-path is used)')
    p.add_argument(
        '--local-path',
        default=None,
        help='Load a local HuggingFace checkpoint directory in-process (needs torch+transformers) instead of an API endpoint',
    )
    p.add_argument('--device', default=None, help='Torch device for --local-path (default: cuda if available, else cpu)')
    p.add_argument(
        '--max-context',
        type=int,
        default=4096,
        help='Max prompt tokens for --local-path; longer prompts are left-truncated (default: 4096)',
    )
    p.add_argument(
        '--api-key', default=os.environ.get('OPENAI_API_KEY'), help='Bearer token (defaults to $OPENAI_API_KEY; omit for local servers)'
    )
    p.add_argument(
        '--scoring',
        choices=['auto', 'generation', 'logprob'],
        default='auto',
        help='auto: faithful logprob scoring if the backend supports it, else generation',
    )
    p.add_argument(
        '--api', choices=['auto', 'chat', 'completions'], default='auto', help='Which generation endpoint to use (auto probes the backend)'
    )
    p.add_argument('--tasks', default=None, help='Comma-separated task labels to run (default: all 21)')
    p.add_argument('--max-per-task', type=int, default=-1, help='Cap examples per task (-1 = all)')
    p.add_argument('--concurrency', type=int, default=8, help='Concurrent in-flight requests')
    p.add_argument('--bundle-dir', default=data.DEFAULT_BUNDLE_DIR, help='Path to the data bundle (default: bundled ./data)')
    p.add_argument('--output', default=None, help='Write summary results JSON here')
    p.add_argument('--debug-file', default=None, help='Write every per-example record (prompt/output/pred/gold) as JSONL')
    p.add_argument(
        '--show-logprobs',
        action='store_true',
        help='Capture generated-token logprobs into the debug records (generation mode; requires backend logprob support)',
    )
    p.add_argument('--timeout', type=int, default=120)
    p.add_argument('--max-retries', type=int, default=5)
    p.add_argument('--extra-body', default=None, help='JSON object merged into every request payload')
    p.add_argument('--quiet', action='store_true', help='Suppress per-task progress')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    extra_body = json.loads(args.extra_body) if args.extra_body else None

    if args.local_path:
        model_name = args.model or os.path.basename(os.path.normpath(args.local_path))
        base_url = f'local://{args.local_path}'
        print(f'Loading local checkpoint {args.local_path} ...', file=sys.stderr)
        client = LocalClient(args.local_path, device=args.device, max_context=args.max_context)
        caps = client.probe()
        print(f'  device: {client.device} | capabilities: {caps}', file=sys.stderr)
    else:
        if not args.base_url or not args.model:
            print('ERROR: --base-url and --model are required unless --local-path is given', file=sys.stderr)
            return 2
        model_name, base_url = args.model, args.base_url
        client = APIClient(
            base_url, model_name, api_key=args.api_key, timeout=args.timeout, max_retries=args.max_retries, extra_body=extra_body
        )
        print(f'Probing backend at {base_url} (model={model_name}) ...', file=sys.stderr)
        caps = client.probe()
        print(f'  capabilities: {caps}', file=sys.stderr)
    try:
        mode, style, use_chat = resolve_modes(caps, scoring=args.scoring, api=args.api)
    except ValueError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2
    endpoint = f'logprob({style})' if mode == 'logprob' else ('chat' if use_chat else 'completions')
    print(f'  scoring mode: {mode} via {endpoint}\n', file=sys.stderr)

    tasks = [t.strip() for t in args.tasks.split(',')] if args.tasks else None
    t0 = time.time()
    try:
        res = evaluate(
            base_url,
            model_name,
            api_key=args.api_key,
            tasks=tasks,
            max_per_task=args.max_per_task,
            scoring=args.scoring,
            api=args.api,
            concurrency=args.concurrency,
            extra_body=extra_body,
            timeout=args.timeout,
            max_retries=args.max_retries,
            bundle_dir=args.bundle_dir,
            debug_file=args.debug_file,
            progress=not args.quiet,
            capture_logprobs=args.show_logprobs,
            client=client,
            caps=caps,
        )
    except ValueError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2
    elapsed = time.time() - t0

    results = res['results']
    centered_results = res['centered_results']
    unparsed = res.get('unparsed_rate', {})
    flagged = []
    print('\n' + '=' * 72)
    print(f'{"TASK":<38} {"ACC":>8} {"CENTERED":>10} {"NO-ANS%":>10}')
    print('-' * 72)
    for label in results:
        u = unparsed.get(label, 0.0)
        mark = '  <-- see note' if u >= 0.10 else ''
        if u >= 0.10:
            flagged.append((label, u))
        print(f'{label:<38} {results[label]:>8.4f} {centered_results[label]:>10.4f} {100 * u:>9.1f}%{mark}')
    print('-' * 72)
    print(f'{"CORE metric":<38} {"":>8} {res["core_metric"]:>10.4f}')
    print('=' * 72)
    print(f'({res["num_tasks"]} tasks, mode={mode}, {elapsed:.1f}s)')
    if flagged:
        print(
            '\nNote: NO-ANS% is the share of items where the model produced no '
            'parseable answer\n(a refusal or off-format reply, scored as wrong). '
            'A high value means the score\nreflects format-following, not '
            'knowledge — inspect --debug-file for these tasks:'
        )
        for label, u in flagged:
            print(f'  - {label}: {100 * u:.0f}% unanswered')

    errored = {k: v for k, v in res.get('error_rate', {}).items() if v > 0}
    if errored:
        samples = res.get('error_samples', {})
        print(
            '\nWARNING: some requests failed (API/transport errors, counted as '
            'wrong). These are\nnot model refusals — check connectivity / context '
            'limits / rate limits:'
        )
        for label, rate in errored.items():
            print(f'  - {label}: {100 * rate:.0f}% errored | e.g. {samples.get(label, "")[:140]}')
    if args.debug_file:
        print(f'Debug trace: {args.debug_file}', file=sys.stderr)

    if args.output:
        res['elapsed_seconds'] = elapsed
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2)
        print(f'Wrote {args.output}', file=sys.stderr)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""High-level programmatic API.

Import this from your own experiment scripts to score a model on Vintage CORE
(full suite or a subset) and get back a results dict you can compare across runs:

    from vintage_core import evaluate

    res = evaluate(
        base_url="http://localhost:8000/v1",
        model="my-experiment-v3",
        tasks=["arc_easy", "boolq", "squad"],   # or None for all 22 tasks
        max_per_task=200,                        # or -1 for all
    )
    print(res["core_metric"], res["results"])
"""

import json
import os

# Aliased: evaluate() takes a `scoring` parameter that would shadow the module.
from . import data, runner
from . import scoring as scoring_mod
from .client import APIClient


def resolve_modes(caps, scoring='auto', api='auto'):
    """Decide the scoring mode / endpoint from probed capabilities.
    Returns (mode, style, use_chat). Raises ValueError on an impossible request."""
    if scoring == 'logprob':
        if not caps.has_prompt_logprobs:
            raise ValueError("scoring='logprob' needs prompt logprobs (echo / prompt_logprobs); backend exposes none")
        mode, style = 'logprob', caps.logprob_style
    elif scoring == 'generation':
        mode, style = 'generation', None
    elif scoring == 'auto':
        if caps.has_prompt_logprobs:
            mode, style = 'logprob', caps.logprob_style
        else:
            mode, style = 'generation', None
    else:
        raise ValueError(f'unknown scoring: {scoring!r}')

    if api == 'auto':
        use_chat = caps.has_chat or not caps.has_completions
    elif api in ('chat', 'completions'):
        use_chat = api == 'chat'
    else:
        raise ValueError(f'unknown api: {api!r}')

    if mode == 'generation':
        if use_chat and not caps.has_chat:
            raise ValueError('chat endpoint requested/needed but unavailable')
        if not use_chat and not caps.has_completions:
            raise ValueError("completions endpoint needed but unavailable; try api='chat'")
    return mode, style, use_chat


def evaluate(
    base_url=None,
    model=None,
    api_key=None,
    local_path=None,
    device=None,
    max_context=4096,
    tasks=None,
    max_per_task=-1,
    scoring='auto',
    api='auto',
    concurrency=8,
    extra_body=None,
    timeout=120,
    max_retries=5,
    bundle_dir=data.DEFAULT_BUNDLE_DIR,
    debug_file=None,
    return_records=False,
    progress=False,
    capture_logprobs=False,
    client=None,
    caps=None,
):
    """Run Vintage CORE against an API model and return a results dict.

    Parameters mirror the CLI. ``tasks`` is a list of labels (or None = all 22).
    If ``debug_file`` is set, every per-example record (prompt, output,
    prediction, gold, correctness) is streamed there as JSONL. Set
    ``return_records=True`` to also get the records back in memory.
    ``capture_logprobs=True`` (generation mode only) asks the backend for
    generated-token logprobs and stores them on each record under ``logprobs``.

    Pass ``local_path`` (a HuggingFace checkpoint directory) instead of
    ``base_url``/``model`` to run a local model in-process — no API server
    needed (requires torch + transformers). ``device``/``max_context`` only
    apply in that case.

    Returns a dict with: model, base_url, scoring_mode, logprob_style, endpoint,
    core_metric, results (accuracy per task), centered_results, num_tasks,
    max_per_task, and — when requested — records.
    """
    if client is None:
        if local_path:
            from .local import LocalClient

            client = LocalClient(local_path, device=device, max_context=max_context)
            model = model or client.name
            base_url = base_url or f'local://{local_path}'
        else:
            if not base_url or not model:
                raise ValueError('evaluate() needs either base_url+model or local_path')
            client = APIClient(base_url, model, api_key=api_key, timeout=timeout, max_retries=max_retries, extra_body=extra_body)
    caps = caps or client.probe()
    mode, style, use_chat = resolve_modes(caps, scoring=scoring, api=api)
    endpoint = f'logprob({style})' if mode == 'logprob' else ('chat' if use_chat else 'completions')

    all_tasks = data.load_bundle(bundle_dir)
    if tasks:
        wanted = set(tasks)
        selected = [t for t in all_tasks if t.label in wanted]
        missing = wanted - {t.label for t in selected}
        if missing:
            raise ValueError(f'unknown task label(s): {sorted(missing)}')
    else:
        selected = all_tasks

    # Shout before spending any tokens if vintage_qa is about to be scored with
    # the exact-prefix fallback instead of ROUGE-L.
    scores_vintage_qa = any(t.label == 'vintage_qa' for t in selected)
    if scores_vintage_qa:
        scoring_mod.warn_if_rouge_missing()

    debug_fh = None
    if debug_file:
        os.makedirs(os.path.dirname(os.path.abspath(debug_file)), exist_ok=True)
        debug_fh = open(debug_file, 'w', encoding='utf-8')

    def writer(rec):
        debug_fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
        debug_fh.flush()

    results, centered_results, unparsed_rate = {}, {}, {}
    error_rate, error_samples, all_records = {}, {}, []
    try:
        for task in selected:
            acc, records = runner.evaluate_task(
                client,
                task,
                mode=mode,
                use_chat=use_chat,
                style=style,
                concurrency=concurrency,
                max_per_task=max_per_task,
                progress=progress,
                on_record=writer if debug_fh else None,
                capture_logprobs=capture_logprobs and mode == 'generation',
            )
            results[task.label] = acc
            centered_results[task.label] = runner.centered(acc, task.random_baseline)
            n = len(records)
            unanswered = sum(1 for r in records if not r.get('answered', True))
            errored = [r for r in records if r.get('error')]
            unparsed_rate[task.label] = unanswered / n if n else 0.0
            error_rate[task.label] = len(errored) / n if n else 0.0
            if errored:
                error_samples[task.label] = errored[0]['error']
            if return_records:
                all_records.extend(records)
    finally:
        if debug_fh:
            debug_fh.close()

    out = {
        'model': model,
        'base_url': base_url,
        'scoring_mode': mode,
        'logprob_style': style,
        'endpoint': endpoint,
        'core_metric': runner.core_metric(centered_results),
        'results': results,
        'centered_results': centered_results,
        'unparsed_rate': unparsed_rate,
        'error_rate': error_rate,
        'error_samples': error_samples,
        'num_tasks': len(selected),
        'max_per_task': max_per_task,
    }
    if scores_vintage_qa:
        # 'prefix-fallback-DEGRADED' here means rouge_score was missing and the
        # vintage_qa number is not comparable to a ROUGE-L run.
        out['vintage_qa_scoring'] = scoring_mod.ROUGE_SCORING_MODE
    if return_records:
        out['records'] = all_records
    return out

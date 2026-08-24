#!/usr/bin/env python3
"""Fit `max_steps` to a wall-clock budget so the LR schedule anneals fully.

Why this exists
---------------
With `max_steps` unset, the HF Trainer sizes the LR scheduler to a full epoch.
On this dataset that is ~150,000 optimizer steps, while a 1-hour experiment
reaches ~2,000. The cosine therefore covers ~1% of its horizon and the LR never
leaves its peak value -- `min_lr_rate` is never approached, and every
"cosine_with_min_lr" run is really a constant-LR run. `base_train.py` documents
this at the `max_steps` argument; this script computes the number to put there.

How it works
------------
It runs `base_train.py` itself on a short throwaway copy of the config, so the
timing reflects the real code path (the configured optimizer, attention
implementation, torch.compile, dataloader -- no reimplementation to drift out
of sync). From that probe it measures:

  * first-step cost    -- dynamo/inductor compile, paid once
  * steady s/step      -- the marginal cost of one optimizer step
  * eval pass duration -- from base_train's own "[VALIDATION] ... Runtime:" line

and then solves for the largest `max_steps` that fits the budget once
per-run overheads (evals, checkpoint saves) are subtracted.

Run it once per experiment, before the experiment:

    python calibrate_steps.py autoresearch/optimizer/adamw_torch/config.toml --write

`--write` edits `max_steps` (and, with --warmup-frac, `warmup_steps`) in place,
leaving a comment recording how the number was derived.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable

# tqdm progress line: "  0%|   | 137/150420 [02:19<64:28:18,  1.55s/it]"
# The denominator is captured because base_train.py also prints a SECOND tqdm
# bar for each eval pass, whose counter restarts at 0 with its own elapsed
# clock. Mixing the two timelines yields a negative step time, so only lines
# whose total matches the training horizon are kept.
TQDM_RE = re.compile(r'\|\s*(\d+)/(\d+)\s*\[(\d+):(\d+)(?::(\d+))?<')
# re.M is required: these anchors match indented lines inside the log body,
# not the start of the file.
EVAL_RUNTIME_RE = re.compile(r'^\s*Runtime:\s+([0-9.]+)s', re.M)
# base_train.py prints "  Total parameters: 147,874,560 (147.87M)"
PARAM_RE = re.compile(r'^\s*Total parameters:.*?\(([0-9.]+)M\)', re.M)

PROBE_TEMPLATE_KEYS = {
    'max_steps',
    'max_train_minutes',
    'num_train_epochs',
    'save_strategy',
    'save_steps',
    'save_total_limit',
    'eval_strategy',
    'eval_steps',
    'logging_strategy',
    'logging_steps',
    'output_dir',
    'final_model_dir',
    'warmup_steps',
}


def _elapsed(m):
    """Parse a tqdm [MM:SS<...] or [H:MM:SS<...] elapsed stamp into seconds."""
    a, b, c = m.group(3), m.group(4), m.group(5)
    return int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)


def parse_probe(log_text, total_steps, eval_every=None):
    """Extract (first_step_s, steady_s_per_step, eval_s, n_params_M) from a probe log.

    `total_steps` selects the training progress bar; eval passes print their own
    tqdm bar with an independent counter and clock.
    """
    text = log_text.replace('\r', '\n')
    marks = {}
    for m in TQDM_RE.finditer(text):
        if int(m.group(2)) != total_steps:
            continue  # eval bar (or some other bar), not the training timeline
        marks.setdefault(int(m.group(1)), _elapsed(m))

    if len(marks) < 5:
        raise RuntimeError(f'probe produced too few training progress marks ({len(marks)}) for a {total_steps}-step horizon; see the log')

    steps = sorted(marks)
    first_step_s = marks[steps[1]] - marks[steps[0]] if len(steps) > 1 else 0.0

    # Steady state: measure between the first post-compile step and the last
    # step, skipping the warmup region where compile/cudnn autotune still bleed
    # in. Requires no eval or save inside the window -- the probe schedules
    # both only at the very end, so the window is clean by construction.
    lo = steps[min(len(steps) - 2, 8)]
    hi = steps[-1]
    if eval_every:
        # Defensive: never let an eval pause fall inside the window, whatever
        # the probe config says. An eval costs ~10 s and silently inflates the
        # per-step time by ~11% on a 120-step probe.
        boundaries = [e for e in range(eval_every, hi + 1, eval_every) if lo < e < hi]
        if boundaries:
            hi = boundaries[0] - 1
    span_steps = hi - lo
    if span_steps < 3:
        raise RuntimeError(f'probe window too short ({span_steps} steps)')
    steady = (marks[hi] - marks[lo]) / span_steps
    if steady <= 0:
        raise RuntimeError(
            f'measured a non-positive step time ({steady:.3f} s/step) between '
            f'steps {lo} and {hi} -- the progress timeline is corrupt, refusing '
            f'to write a max_steps derived from it'
        )

    evals = [float(m.group(1)) for m in EVAL_RUNTIME_RE.finditer(text)]
    params = [float(m.group(1)) for m in PARAM_RE.finditer(text)]
    # (first eval, steady eval): the first pays eval-graph compile.
    eval_pair = None
    if evals:
        eval_pair = (evals[0], evals[-1] if len(evals) > 1 else evals[0])
    # tqdm reports elapsed time at 1-second resolution, so the step time carries
    # +/-1s spread over the measurement window. Report it: a window shorter than
    # ~60 s gives an error big enough to overshoot the wall-clock budget.
    quant_err = 1.0 / (span_steps * steady)
    return first_step_s, steady, eval_pair, (params[0] if params else None), quant_err


def make_probe_config(cfg_path, probe_steps, workdir):
    """Copy the config, stripping it down to `probe_steps` steps + one eval."""
    with open(cfg_path, 'rb') as fh:
        cfg = tomllib.load(fh)
    train = cfg.setdefault('training', {})

    train['max_steps'] = probe_steps
    train['num_train_epochs'] = 1
    train.pop('max_train_minutes', None)
    train['warmup_steps'] = probe_steps  # never anneal during the probe
    # A WSD config carries a num_decay_steps sized for the real horizon (~550),
    # which is far larger than probe_steps (~120). base_train.py's WSD coverage
    # guard would then refuse to start, because warmup + decay exceeds the probe
    # horizon and the auto-fitted stable phase would be negative. Zero the decay
    # for the probe: combined with warmup_steps = probe_steps above, this keeps
    # the probe at a flat peak LR, which is exactly what the probe wants -- it
    # measures step TIME, and the schedule does not affect step time.
    if train.get('lr_scheduler_type') == 'warmup_stable_decay':
        skw = dict(train.get('lr_scheduler_kwargs') or {})
        skw['num_decay_steps'] = 0
        skw.pop('num_stable_steps', None)
        train['lr_scheduler_kwargs'] = skw
    train['save_strategy'] = 'no'
    train['save_total_limit'] = 1
    train['eval_strategy'] = 'steps'
    # Exactly ONE eval, at the very last step, so the step-timing window is free
    # of eval pauses. This is load-bearing: an eval inside the window adds its
    # ~10 s to the step time (measured 0.781 -> 0.866 s/step, an 11% error that
    # would overrun the wall-clock budget). The single eval measured is the
    # first of the run and therefore the most expensive (it pays the eval-graph
    # compile); charging every eval at that rate overestimates overhead, which
    # is the safe direction -- the run finishes early rather than truncating.
    train['eval_steps'] = probe_steps
    train['logging_strategy'] = 'steps'
    train['logging_steps'] = max(1, probe_steps // 4)
    train['output_dir'] = str(workdir)
    train['final_model_dir'] = str(workdir / 'final')

    # Rewrite relative paths against the ORIGINAL config's directory, because
    # the probe config lives somewhere else.
    base = Path(cfg_path).resolve().parent

    def absolutize(value):
        if isinstance(value, list):
            return [absolutize(v) for v in value]
        p = Path(value).expanduser()
        return str(p) if p.is_absolute() else str((base / p).resolve())

    if 'model_config' in cfg:
        cfg['model_config'] = absolutize(cfg['model_config'])
    for section, key in (('data', 'train_files'), ('data', 'valid_files'), ('data', 'tokenizer'), ('training', 'from_pretrained')):
        if cfg.get(section, {}).get(key):
            cfg[section][key] = absolutize(cfg[section][key])

    probe_path = workdir / 'probe_config.toml'
    probe_path.write_text(_dump_toml(cfg))
    return probe_path


def _dump_toml(cfg):
    """Minimal TOML writer (tomllib is read-only and tomli-w may be absent)."""

    def fmt(v):
        if isinstance(v, bool):
            return 'true' if v else 'false'
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, list):
            return '[' + ', '.join(fmt(x) for x in v) + ']'
        if isinstance(v, dict):
            return '{ ' + ', '.join(f'{k} = {fmt(x)}' for k, x in v.items()) + ' }'
        return json.dumps(str(v))

    lines, sections = [], []
    for key, val in cfg.items():
        (sections if isinstance(val, dict) else lines).append((key, val))
    out = [f'{k} = {fmt(v)}' for k, v in lines]
    for name, body in sections:
        out.append(f'\n[{name}]')
        out.extend(f'{k} = {fmt(v)}' for k, v in body.items())
    return '\n'.join(out) + '\n'


def count_overhead_events(cfg, budget_minutes, steady_s, max_steps_guess):
    """How many eval/save pauses a full-length run will take."""
    train = cfg['training']

    def events(strategy_key, value_key):
        strategy = train.get(strategy_key, 'no')
        value = train.get(value_key, 0) or 0
        if strategy == 'no' or not value:
            return 0
        if strategy == 'minutes':
            return int(budget_minutes / value)
        if strategy == 'steps':
            return int(max_steps_guess / value)
        if strategy == 'epoch':
            return 1
        return 0

    return events('eval_strategy', 'eval_steps'), events('save_strategy', 'save_steps')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', help='experiment config.toml')
    ap.add_argument('--budget-minutes', type=float, default=None, help='target wall-clock (default: the config max_train_minutes)')
    ap.add_argument(
        '--probe-steps',
        type=int,
        default=120,
        help='optimizer steps to time (default 120). tqdm reports elapsed '
        'time to the nearest second, so the timing window must span '
        'well over a minute for ~1%% accuracy; 40 steps gave 4%% error '
        'and would have overrun a 60-minute budget.',
    )
    ap.add_argument(
        '--safety',
        type=float,
        default=0.97,
        help='fraction of the budget to fill, leaving slack so the time guard never truncates the schedule (default 0.97)',
    )
    ap.add_argument('--warmup-frac', type=float, default=None, help='also set warmup_steps to this fraction of max_steps')
    ap.add_argument('--write', action='store_true', help='write max_steps into the config')
    ap.add_argument('--keep-probe', action='store_true', help='keep the probe workdir')
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    with open(cfg_path, 'rb') as fh:
        cfg = tomllib.load(fh)
    train = cfg['training']

    budget = args.budget_minutes or train.get('max_train_minutes')
    if not budget:
        raise SystemExit('no budget: pass --budget-minutes or set max_train_minutes')

    workdir = Path(tempfile.mkdtemp(prefix='calib_', dir='/tmp'))
    try:
        probe_cfg = make_probe_config(cfg_path, args.probe_steps, workdir)
        print(
            f'Probing {cfg_path.name}: {args.probe_steps} steps '
            f'(grad_accum={train.get("gradient_accumulation_steps", 1)}, '
            f'bsz={train.get("per_device_train_batch_size", 8)}, '
            f'compile={train.get("torch_compile", False)})',
            flush=True,
        )

        env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
        t0 = time.time()
        proc = subprocess.run(
            [PYTHON, str(HERE / 'base_train.py'), '--cfg', str(probe_cfg)],
            cwd=str(HERE),
            capture_output=True,
            text=True,
            env=env,
        )
        probe_wall = time.time() - t0
        log_text = proc.stdout + proc.stderr
        (workdir / 'probe.log').write_text(log_text)
        if proc.returncode != 0:
            print(log_text[-4000:], file=sys.stderr)
            raise SystemExit(f'probe failed (rc={proc.returncode}); log: {workdir}/probe.log')

        first_s, steady_s, eval_pair, params_m, quant_err = parse_probe(log_text, args.probe_steps, eval_every=args.probe_steps)
        if eval_pair is None:
            raise SystemExit(
                'probe produced no "[VALIDATION] ... Runtime:" line, so eval cost '
                f'is unknown; inspect the log ({workdir}/probe.log) with --keep-probe'
            )
    finally:
        if not args.keep_probe:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f'probe workdir kept: {workdir}')

    # Checkpoint = fp32 weights + grads-free optimizer state (2 moments) ~= 12 B/param.
    save_s = 1.0 + (params_m * 12e6 / 1.5e9) if params_m else 3.0
    first_eval_s, steady_eval_s = eval_pair

    # Bias the step time UP by the tqdm quantization error. Overshooting the
    # budget truncates the cosine and invalidates the run; undershooting only
    # costs a little training time. The asymmetry is deliberate.
    steady_used = steady_s * (1.0 + quant_err)

    def eval_cost(n):
        return first_eval_s + max(0, n - 1) * steady_eval_s

    # Two-pass: overhead counts can depend on max_steps (eval_strategy="steps").
    max_steps = int((budget * 60 * args.safety - first_s) / steady_used)
    for _ in range(6):
        n_eval, n_save = count_overhead_events(cfg, budget, steady_used, max_steps)
        usable = budget * 60 * args.safety - first_s - eval_cost(n_eval) - n_save * save_s
        new = max(1, int(usable / steady_used))
        if new == max_steps:
            break
        max_steps = new
    n_eval, n_save = count_overhead_events(cfg, budget, steady_used, max_steps)

    ga = train.get('gradient_accumulation_steps', 1)
    bsz = train.get('per_device_train_batch_size', 8)
    seq = cfg.get('data', {}).get('max_seq_length', 1024)
    tokens = max_steps * ga * bsz * seq
    predicted = (first_s + eval_cost(n_eval) + n_save * save_s + max_steps * steady_used) / 60

    print(f'\n  probe wall-clock     : {probe_wall / 60:.1f} min')
    print(f'  model                : {params_m:.1f}M params' if params_m else '')
    print(f'  first step (compile) : {first_s:.0f} s')
    print(f'  steady state         : {steady_s:.3f} s/optimizer-step (+/-{quant_err:.1%}, {ga * bsz * seq / steady_s:,.0f} tokens/s)')
    if quant_err > 0.02:
        print(
            f'  ! timing window is short ({quant_err:.1%} error). Raise --probe-steps '
            f'to at least {int(args.probe_steps * quant_err / 0.01)} for 1% accuracy.'
        )
    print(f'  budgeted at          : {steady_used:.3f} s/step (biased up by the error)')
    print(f'  eval pass            : {first_eval_s:.1f} s first, {steady_eval_s:.1f} s after x {n_eval} = {eval_cost(n_eval):.0f} s')
    print(f'  checkpoint save      : ~{save_s:.1f} s x {n_save} = {n_save * save_s:.0f} s')
    print(f'\n  budget               : {budget:g} min (filling {args.safety:.0%})')
    print(f'  => max_steps         : {max_steps:,}')
    print(f'     predicted run     : {predicted:.1f} min')
    print(f'     tokens seen       : {tokens / 1e6:,.0f}M ({ga * bsz * seq:,} per step)')
    if args.warmup_frac:
        warmup = max(1, round(max_steps * args.warmup_frac))
        print(f'  => warmup_steps      : {warmup} ({args.warmup_frac:.0%} of max_steps)')

    if args.write:
        text = cfg_path.read_text()
        stamp = (
            f'# max_steps fitted by calibrate_steps.py: {steady_s:.3f} s/step measured, '
            f'{budget:g} min budget -> ~{predicted:.1f} min predicted'
        )
        text = re.sub(r'^\s*#\s*max_steps fitted by calibrate_steps\.py.*\n', '', text, flags=re.M)
        if re.search(r'^\s*max_steps\s*=', text, flags=re.M):
            text = re.sub(r'^\s*max_steps\s*=.*$', f'{stamp}\nmax_steps = {max_steps}', text, count=1, flags=re.M)
        else:
            text = re.sub(r'^(\s*num_train_epochs\s*=.*)$', rf'\1\n{stamp}\nmax_steps = {max_steps}', text, count=1, flags=re.M)
        if args.warmup_frac:
            warmup = max(1, round(max_steps * args.warmup_frac))
            text = re.sub(r'^\s*warmup_steps\s*=.*$', f'warmup_steps = {warmup}', text, count=1, flags=re.M)
        cfg_path.write_text(text)
        print(f'\n  ✓ wrote max_steps = {max_steps} into {cfg_path}')
    else:
        print('\n  (dry run -- pass --write to update the config)')


if __name__ == '__main__':
    main()

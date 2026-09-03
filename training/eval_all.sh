#!/usr/bin/env bash
# Evaluate EVERY model in autoresearch/ and autoresearch2/ with the unified eval/ harness.
# One invocation per model so the JSON + MD land in that model's own final/ folder
# (eval-<arch><params>.json), which is what makes the numbers greppable per-experiment.
#
#   setsid nohup ./eval_all.sh >/dev/null 2>&1 & disown
#
# Re-runs are cheap: results are cached by model fingerprint + settings hash. Pass --force
# through EVAL_ARGS to recompute. TO STOP: kill the process GROUP, never a bare pkill -f.
set -u
cd /home/cro/Dev/vintage-LLM/training
PY=/home/cro/Dev/vintage-LLM/.venv/bin/python
LOG=eval_all.log
# `:-` would treat EVAL_ARGS="" as unset; `-` lets an explicit empty string mean
# "no extra args", i.e. reuse the cache instead of recomputing.
EVAL_ARGS=${EVAL_ARGS---force}
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

mapfile -t MODELS < <($PY - <<'PYEOF'
import sys; sys.path.insert(0, '/home/cro/Dev/vintage-LLM/training')
from pathlib import Path
from eval.helpers import resolve_targets
for c in resolve_targets([Path('autoresearch'), Path('autoresearch2')]).checkpoints:
    print(c.relative_to(Path.cwd()))
PYEOF
)

say "EVAL ALL START — ${#MODELS[@]} models — args: $EVAL_ARGS"
OK=0; FAIL=0
for i in "${!MODELS[@]}"; do
  M=${MODELS[$i]}
  if $PY -m eval "$M" $EVAL_ARGS >/dev/null 2>>"$LOG"; then
    OK=$((OK+1))
    say "[$((i+1))/${#MODELS[@]}] OK   $M"
  else
    FAIL=$((FAIL+1))
    say "[$((i+1))/${#MODELS[@]}] FAIL $M"
  fi
done
say "EVAL ALL DONE — $OK ok, $FAIL failed"

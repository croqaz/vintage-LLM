"""The standing table of models scored with THIS protocol, on THIS held-out set.

Every evaluation report opens with a leaderboard, because the first thing
anyone wants to know is roughly where a checkpoint sits. That only works if
the table is one table. It used to be two: the reference points were hard
coded in report.py from a 2026-08-23 run on `heldout-Sprocket-n-Say.jsonl`,
and when the default held-out moved to `heldout-Piston-n-Prose.jsonl` the
renderer correctly noticed the mismatch and split them into a comparable half
and an incomparable half. Correct, and useless to read.

The fix is to stop hard coding recorded numbers and keep measured ones. This
module owns `eval_data/leaderboard.json`: entries written from real eval runs,
each carrying the held-out hash it was scored against. The report shows the
entries whose hash matches the current run and says how many it hid. Nothing
that cannot be ranked is ever placed in a ranked column.

Refresh it after evaluating new models:

    python -m eval --update-leaderboard MODELS Llama-141M/final-anneal-3h

Curated fields (`display`, `kind`, `origin`, `note`) survive a refresh; the
measurements are overwritten from the eval JSON every time.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .helpers import EVAL_DATA, atomic_json

LEADERBOARD = EVAL_DATA / 'leaderboard.json'

# Measurements copied out of a result's summary. Everything here is scored on
# the same documents by the same code, so it can share a table.
MEASURED = (
    'prose_bpb',
    'probe_historical_bpb',
    'probe_modern_bpb',
    'chat_target_bpb',
    'logic_accuracy',
)
# Set by hand in the JSON and preserved across refreshes.
CURATED = ('display', 'kind', 'origin', 'note')


def load(path: Path | None = None) -> dict:
    path = Path(path or LEADERBOARD)
    if not path.is_file():
        return {'schema': 1, 'entries': []}
    with open(path, 'rb') as fh:
        return json.load(fh)


def entries_for(heldout_sha256: str | None, path: Path | None = None) -> tuple[list[dict], int]:
    """(entries scored on this held-out file, number hidden because they were not).

    A caller that ignores the second value and prints the first is still
    correct. That is the point: the incomparable rows cannot leak into a
    ranking by accident.
    """
    data = load(path)
    rows = data.get('entries', [])
    if not heldout_sha256:
        return [], len(rows)
    keep = [e for e in rows if e.get('heldout_sha256') == heldout_sha256]
    return keep, len(rows) - len(keep)


def register(results: list[dict], path: Path | None = None) -> tuple[int, int]:
    """Merge eval results into the leaderboard. Returns (added, updated).

    Keyed on the display name rather than the checkpoint path, so re-running a
    model from a moved directory updates its row instead of duplicating it.
    """
    path = Path(path or LEADERBOARD)
    data = load(path)
    existing = {e['name']: e for e in data.get('entries', [])}
    added = updated = 0

    for r in results:
        summary = r.get('summary') or {}
        settings = r.get('evaluation_settings') or r.get('settings') or {}
        if summary.get('prose_bpb') is None:
            continue
        name = _name_for(r)
        entry = {
            'name': name,
            'params_millions': r.get('params_millions'),
            'heldout_sha256': settings.get('heldout_sha256'),
            'schema': settings.get('schema'),
            'measured': date.today().isoformat(),
            'source': r.get('source_file') or r.get('checkpoint'),
            **{k: summary.get(k) for k in MEASURED},
        }
        previous = existing.get(name)
        if previous:
            entry.update({k: previous[k] for k in CURATED if k in previous})
            updated += 1
        else:
            added += 1
        existing[name] = entry

    data['schema'] = 1
    data['updated'] = date.today().isoformat()
    data['entries'] = sorted(existing.values(), key=lambda e: (e.get('prose_bpb') is None, e.get('prose_bpb')))
    atomic_json(path, data)
    return added, updated


# Directory names that describe where weights live rather than which model
# they are. Publishers use them freely: Bartholomew-sft keeps its weights in
# checkpoints/, our runs use sft_checkpoints/final. A leaderboard row called
# "checkpoints" is useless, so these are skipped when naming.
_PLUMBING_DIRS = ('final', 'final_merged', 'sft_checkpoints', 'checkpoints', 'base_checkpoints', 'chat_checkpoints')


def name_for_checkpoint(path) -> str:
    """A readable, stable model name from a checkpoint path.

    Shared with chat_eval so one model carries one name everywhere. Without it
    a checkpoint under Bartholomew-sft/checkpoints/ is reported as
    "checkpoints" in one table and "Bartholomew-sft" in another.
    """
    checkpoint = Path(path)
    name = checkpoint.name
    if name in _PLUMBING_DIRS:
        # .../Bartholomew-sft/checkpoints           -> Bartholomew-sft
        # .../attnonly-r16/sft_checkpoints/final    -> attnonly-r16
        parts = [p for p in checkpoint.parts if p not in _PLUMBING_DIRS]
        name = parts[-1] if parts else name
    return name


def _name_for(r: dict) -> str:
    return name_for_checkpoint(r.get('checkpoint') or r.get('label') or 'model')

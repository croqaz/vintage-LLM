#!/usr/bin/env bash
# Generate a standard battery of autocomplete samples from a checkpoint.
# Usage: ./sample_autocomplete.sh <checkpoint-dir> [output.md]
# The prompt set is fixed so sample files from different models compare line-for-line
set -u
PY=~/Dev/vintage-LLM/.venv/bin/python
TR=~/Dev/vintage-LLM/training

CKPT=${1:?usage: sample_autocomplete.sh <checkpoint-dir> [output.md]}
OUT=${2:-autocomplete_samples.md}

echo "# Autocomplete samples — $CKPT ($(date '+%F %H:%M'))" > "$OUT"
while IFS= read -r p; do
  printf '\n---\n**PROMPT:** %s\n\n' "$p" >> "$OUT"
  $PY "$TR/generate.py" --checkpoint "$CKPT" --tokens 130 "$p" 2>/dev/null | tail -n +4 >> "$OUT"
done <<'EOF'
Let them eat brioche,
Elementary, my dear Watson,
Alas, poor Yorick! I knew him,
The love of money is the root of all
Put your trust in God, my boys, and keep your
I disapprove of what you say, but I will defend to the death your
It was a cold morning in November when the carriage arrived at
"You cannot mean it," she said, lowering her voice so that
My dearest brother, I write to you from Lisbon, where the
To prepare a proper broth for an invalid, first take
The steam engine differs from the water-wheel chiefly in that
LONDON, Tuesday. — The House of Commons yesterday debated
On the cultivation of apple orchards in northern climates, the farmer must
The old lighthouse keeper climbed the stairs slowly, remembering
Among the curiosities exhibited at the fair was a mechanical
The physician examined the patient and concluded that the fever
A legal dispute involving an individual was presented, where
We approach the close of our survey of the life and works of
As our approach drew nearer, the scattered villages and humble enclosures
This question, as well as the manner of its resolution,
The English forces were preparing for an offensive on
It was this rigorous self-discipline, this dedication to the unseen
As the afternoon wore on, and the wind settled into a steady
In the midst of this tempest of violence, a figure, whom we shall name
Hark, gentle reader, and lend thine ear to a matter of profound import,
The speaker then addressed the prior assertion that Her Majesty's Government should
Let the farmers be warned that this poisonous plant is not to be confused with the edible fruit,
Many the gay straw-rides to the Lake; frequent and long the walks through
EOF
echo "wrote $OUT"

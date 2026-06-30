# pre1900-filter

A small, self-contained tool that estimates **how likely a piece of text is to
be pre-1900** (written before the year 1900) and lets you **keep or drop** texts
against a cut-off. It was built to curate clean training/eval data for a
"vintage" language model with a knowledge cut-off of **1899** — i.e. to throw
out anything that sounds modern or mentions things that didn't exist yet.

**No machine learning libraries needed to run it.** Just Python 3 (standard
library only) and the bundled `data/` folder. No internet, no GPU, no install.

---

## 1. What's in here

```
pre1900-filter/
├── detect.py          # THE FILTER: scores text, keeps/drops it. Run this.
├── banned_terms.py    # A REUSABLE checker: "does this text contain anything
│                      #   post-1900?" Use it standalone in other projects.
├── data/              # the dictionaries + word lists the tools read
│   ├── old.json         word frequencies from ~14k pre-1900 books
│   ├── modern.json      word frequencies from modern text (stories, web, etc.)
│   ├── 17c.json 18c.json 19c.json   per-century word lists (for century guess)
│   ├── calib.json       the trained weights that turn scores into a probability
│   ├── banned.txt       BLOCKLIST: post-1900 concepts/brands/people/tech
│   └── allowed.txt      ALLOWLIST: terms to never flag (your escape hatch)
├── test/              # positive + negative tests for both tools
└── README.md          # this file
```

There are **two programs**, used for two different jobs:

| You want to…                                              | Use            |
|-----------------------------------------------------------|----------------|
| Score / filter texts by how pre-1900 they sound           | `detect.py`    |
| Just ask "does this text mention anything post-1900?"     | `banned_terms.py` |

---

## 2. How it decides (plain English)

`detect.py` runs **two checks** and a text must pass **both**:

1. **The anachronism veto (hard rule).** If the text contains a banned word/
   phrase (`data/banned.txt` — things like *internet, television, World War,
   Einstein, movie, social media*) **or an explicit post-1900 year** (1900–2099,
   e.g. *"in 1906"*, *"the year 2016"*), the score is slammed to ~0.03 and the
   text is dropped. Pre-1900 years like *1815* or *1066* are fine.
   *(This exact check is what `banned_terms.py` exposes on its own.)*

2. **The modernity score (soft rule).** Every word is weighed by how much more
   common it is in modern writing than in pre-1900 books. Lots of modern-leaning
   words → low score. The result is calibrated to a probability between 0 and 1.

The final number, **`p_pre1900`**, is that probability (capped near 0 if the
veto fired). You compare it to a **threshold** (default 0.75) to keep or drop.

---

## 3. Quick start

```bash
# type/echo a single text (reads STDIN)
echo "What a beautiful day!"            | python detect.py
echo "I posted a selfie online in 2016" | python detect.py

# a plain text file (scored as one document)
python detect.py mybook.txt

# a JSONL file (one JSON object per line; scores the "text" field of each)
python detect.py data.jsonl --field text

# CLEAN A DATASET: write only the kept records to a new file
python detect.py data.jsonl --filter > kept.jsonl
```

Reading one text prints a line like:

```
KEEP 0.99 [19c]  'The cavalry advanced at dawn, their muskets gleaming...'
DROP 0.00 hits=['2016', 'selfie']  'I posted a selfie online in 2016'
```

`KEEP`/`DROP` = the decision, the number = `p_pre1900`, `[19c]` = century guess
(only with `--century`), `hits=[...]` = which banned terms/years tripped the veto.

---

## 4. Every flag explained

| Flag | Default | What it does |
|------|---------|--------------|
| `--threshold F` | `0.75` | **The main dial.** Keep a text only if `p_pre1900 ≥ F`. Higher = stricter/purer (drops more). Lower = more permissive. |
| `--old-prior F` | `0.0` | Nudges **short** texts toward "pre-1900". A 4-word line carries little evidence; raise this (try `1`–`3`) so neutral short text isn't dropped. Has almost no effect on long texts. |
| `--style-weight F` | `1.0` | How much **modern phrasing** counts. `1` = full (modern-sounding prose is dropped). `0` = ignore writing style entirely. |
| `--marker-weight F` | `1.0` | How much **modern vocabulary** counts (words like *okay, mom, backpack* that aren't "future concepts", just modern). `1` = full, `0` = ignore. |
| `--min-english F` | `0.0` | Drop texts where fewer than fraction `F` of words are recognized English (kills foreign-language / gibberish docs). Try `0.6`. |
| `--min-tokens N` | `0` | Drop texts shorter than `N` words (kills near-empty docs). |
| `--field NAME` | `text` | Which JSON field to read from a `.jsonl` input. |
| `--filter` | off | Output **only the kept records**, as JSONL. Use this to clean a dataset. |
| `--century` | off | Add a best-guess century (`17c`/`18c`/`19c`) to kept items. Low confidence — a nice-to-have, not reliable. |
| `--json` | off | Force JSONL output (with all the detail fields) even for a single text. |

### The three "what counts as modern?" knobs, explained together

`--threshold`, `--style-weight`, and `--marker-weight` work together. The veto
(banned terms + post-1900 years) **always** applies regardless of these.

- **`--style-weight` = modern *phrasing*** (sentence rhythm/word choice that just
  *feels* recent). This grows with text length, so for long modern text it's
  decisive and a threshold can't undo it — you must lower this weight.
- **`--marker-weight` = modern *vocabulary*** (specific modern words).
- Setting **both to 0** gives a **"concepts-only" filter**: keep everything
  unless it actually *names* a post-1900 thing. Useful when you don't care that
  the writing is modern, only that the *content* is timeless.

```
                    drops modern STYLE?   drops modern WORDS?   drops post-1900 CONCEPTS?
default (1, 1)            yes                   yes                     yes
style 0, marker 1         no                    yes                     yes
style 0, marker 0         no                    no                      yes   (concepts-only)
```

---

## 5. Recommended presets

| Goal | Command |
|------|---------|
| **Strict purity** — drop anything that sounds modern | `python detect.py in.jsonl --filter --threshold 0.85` |
| **Balanced** (the default) | `python detect.py in.jsonl --filter` |
| **Short questions / one-liners** — don't over-drop short text | `python detect.py in.jsonl --filter --old-prior 2` |
| **Keep modern-written but timeless content** (drop modern *words*) | `python detect.py in.jsonl --filter --style-weight 0 --threshold 0.6` |
| **Concepts-only** — keep unless it names a post-1900 thing | `python detect.py in.jsonl --filter --style-weight 0 --marker-weight 0 --threshold 0.6` |
| **Clean a messy web dataset** (drop foreign + near-empty) | `python detect.py in.jsonl --filter --min-english 0.6 --min-tokens 50` |
| **Just look, don't filter** | `python detect.py in.jsonl --json > scored.jsonl` |

Rule of thumb: the **veto + `--threshold`** control purity; **`--old-prior`**
saves short texts; **`--style-weight` / `--marker-weight`** decide whether
"modern" means modern *writing* or only modern *knowledge*.

---

## 6. The `banned_terms.py` tool (use it anywhere)

A standalone, dependency-free checker for "does this text contain anything that
couldn't exist before 1900?" — i.e. a banned term (`data/banned.txt`) or a
post-1900 year. Copy `banned_terms.py` + `data/banned.txt` + `data/allowed.txt`
into any project.

**Command line:**
```bash
python banned_terms.py "We watched a movie in 2016."   # -> ANACHRONISTIC | hits: ['2016', 'movie']
python banned_terms.py "He sent a telegraph in 1885."  # -> CLEAN | hits: []
echo "some text" | python banned_terms.py              # reads STDIN
python banned_terms.py --quiet file.txt                # no output; exit 0=clean, 1=flagged
```

**As a Python library:**
```python
from banned_terms import contains_anachronism, find_anachronisms

contains_anachronism("I love my smartphone")   # -> True
contains_anachronism("a quiet country morning") # -> False
find_anachronisms("Built in 1906, near the airport")  # -> ['1906', 'airport']

# ignore years, only check the term list:
contains_anachronism("Built in 1906.", check_years=False)  # -> False
```

---

## 7. Editing the word lists

Both lists live in `data/` and take effect immediately (no rebuild).

- **`data/banned.txt`** — one term or phrase per line, `#` for comments. A hit
  drops the text, so keep it **high-precision** (prefer distinctive words and
  multi-word phrases). A short phrase also matches longer mentions
  (`world war` catches `world war ii`; `nuclear` catches `nuclear reactor`).
- **`data/allowed.txt`** — terms here are **never** flagged, even if they appear
  in `banned.txt`. Your escape hatch: if the filter wrongly drops legitimate
  pre-1900 text on some word, add that word here.

⚠️ A few banned terms have innocent old meanings and *will* drop some genuine
old text (e.g. `plastic` as in "plastic arts", `rocket` as a pre-1900 firework,
`jet` as in "jet-black"). That's an intentional trade — we'd rather drop a
little good text than let modern text leak in. Remove them from `banned.txt`
(or add to `allowed.txt`) if that matters to you.

---

## 8. Output fields (JSONL mode)

```json
{"id": "data.jsonl:2", "p_pre1900": 0.03, "n_tokens": 9, "english_frac": 1.0,
 "banned_hits": ["1906"], "features": {"sum_lr": -3.1, "gated_sum": 0.0, "arch_count": 0.0},
 "keep": false}
```
- `p_pre1900` — the probability (0–1). Your keep/drop number.
- `banned_hits` — which banned terms / post-1900 years fired the veto (empty if none).
- `english_frac` — fraction of words recognized as English.
- `features` — internals, for debugging why something scored as it did.

---

## 9. Tests

```bash
python test/test_banned_terms.py   # anachronism detection (positive + negative)
python test/test_detect.py         # scoring keep/drop + knob behaviour
```
Each prints a pass count and exits non-zero on any failure.

---

## 10. Known limitations (be honest with yourself)

- **Very short inputs** (a few words) carry little evidence; "What a beautiful
  day!" lands around 0.60 by default. Use `--old-prior` to keep such lines.
- **Modern writing about timeless topics** can read as old to the word model if
  it names no post-1900 thing — the veto only catches *named* anachronisms, not
  modern *style*, when you've set the weights low.
- **It does not understand meaning.** It's word statistics + a banned list, not
  comprehension. A clever modern pastiche of old prose can slip through.
- **Century guess is rough** — treat `17c/18c/19c` as a hint, not a fact.
- **Spelling isn't used** (many old books were re-typeset with modern spelling),
  so the signal is vocabulary, not orthography.

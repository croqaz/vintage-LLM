# Tokenizer benchmarks

Comparison of four BPE tokenizers on the pre-1900 corpus (`dataset-text1..4.txt`, ~21 GB raw text):

| tokenizer | path | vocab size | notes |
|-----------|------|-----------:|-------|
| violet | `violet/` | 50,277 | HF zakarth/violet-1b4-chat (Pythia/GPT-NeoX-style vocab) |
| v2 | `t-v2/` | 32,700 | trained April 2026 on an earlier corpus, same script/config |
| v3 | `t-v3/` | 32,752 | trained April 2026 on an earlier corpus, same script/config |
| v4 | `t-v4/` | 32,768 | current: trained by `gen_tok.py` on the cleaned corpus; exactly 2¹⁵ tokens |

v4 is trained on `dataset-clean1..4.txt`, produced by the cleaning pass in `gen_tok.py`: character runs of 4+ collapse to 3 (`--------` → `---`), mixed ruler runs of `-=_~+*#.` collapse to 3× the dominant character (`+-----+-----+` → `---`), and repeated punctuation units collapse to one (`?!?!?!` → `?!`). This eliminated all 31 repeated-pattern junk tokens the uncleaned training produced. The cleaning rules are locked in by `tests/test_gen_tok.py`.

All benchmarks run on samples of the **original uncleaned** files, so v4 gets no advantage from its own preprocessing.

Every metric is reported in two variants:

- **bare** - the word encoded exactly as extracted (`profound`), the literal SmolLM procedure;
- **Ġ-prefix** - the word encoded with a leading space (`Ġprofound`), which is how nearly every word occurs in running text under a ByteLevel pre-tokenizer.

## 1. Fertility

**Method** (SmolLM): words are extracted with `nltk.word_tokenize`; each word is encoded on its own with `add_special_tokens=False`; fertility is the mean number of tokens per word instance. Lower is better. The sample is deterministic: 64 chunks of 64 KB from random newline-aligned offsets of each raw dataset file, seeded with `SEED = 42` per file path - 3,349,844 word instances (111,589 unique). The implementation encodes each unique word once and weights by its count, which is mathematically identical to the per-instance mean. Reproduce with `python3 benchmark_tok.py` (metric math unit-tested in `tests/test_benchmark_tok.py`).

| dataset | variant | violet | v2 | v3 | v4 |
|---------|---------|-------:|---:|---:|---:|
| dataset-text1.txt | bare | 1.3104 | 1.2539 | **1.2402** | 1.2736 |
| | Ġ-prefix | 1.1584 | 1.1510 | **1.1434** | 1.1503 |
| dataset-text2.txt | bare | 1.2876 | 1.2358 | 1.2418 | **1.2343** |
| | Ġ-prefix | 1.1773 | 1.1481 | 1.1582 | **1.1286** |
| dataset-text3.txt | bare | 1.3649 | 1.2921 | **1.2757** | 1.3226 |
| | Ġ-prefix | 1.1129 | 1.1155 | 1.1024 | **1.0746** |
| dataset-text4.txt | bare | 1.2831 | 1.2131 | **1.2050** | 1.2464 |
| | Ġ-prefix | 1.1129 | 1.1015 | **1.0972** | 1.1013 |
| **OVERALL** | bare | 1.3098 | 1.2476 | **1.2400** | 1.2673 |
| | Ġ-prefix | 1.1421 | 1.1301 | 1.1268 | **1.1154** |

v4 has the lowest running-text (Ġ-prefix) fertility overall, beating even violet's 54%-larger vocab. v3 leads the bare variant; a word-level vocab diff shows its entire edge sits in space-less word forms (`profound`, `observe`, …) plus the NLTK quote artifact `''` - its April corpus was hard-wrapped, so line-initial bare forms were frequent enough to earn tokens.

## 2. Proportion of continued words

**Method**: same word sample and encoding as section 1; the metric is the share of word instances split into 2 or more tokens ("continued" into subwords). Lower is better. Produced by the same `python3 benchmark_tok.py` run.

| dataset | variant | violet | v2 | v3 | v4 |
|---------|---------|-------:|---:|---:|---:|
| dataset-text1.txt | bare | 22.49% | 18.25% | **16.95%** | 19.50% |
| | Ġ-prefix | 10.63% | 10.16% | **9.61%** | 10.03% |
| dataset-text2.txt | bare | 21.28% | **17.36%** | 17.57% | 17.60% |
| | Ġ-prefix | 12.96% | 10.82% | 11.09% | **9.30%** |
| dataset-text3.txt | bare | 28.56% | 23.39% | **21.92%** | 25.42% |
| | Ġ-prefix | 8.79% | 9.19% | 7.95% | **5.67%** |
| dataset-text4.txt | bare | 21.47% | 16.29% | **15.49%** | 18.67% |
| | Ġ-prefix | 8.01% | 7.32% | **6.91%** | 7.21% |
| **OVERALL** | bare | 23.27% | 18.68% | **17.87%** | 20.11% |
| | Ġ-prefix | 10.21% | 9.42% | 8.98% | **8.16%** |

Same as fertility: in running-text form v4 splits the fewest words overall, with the largest margin on the noisiest file (`dataset-text3`, OCR'd newspapers: 5.67% vs v3's 7.95%).

## 3. Single-token coverage of the top corpus words

**Method**: `words.json` is a word → occurrence-count dictionary (3,966,551 entries) built from a much larger dataset (HF croqaz/vintage-words). The top 100,000 words by count are encoded per tokenizer with `add_special_tokens=False`; a word counts as covered when it encodes to exactly **one** token. Reported at the top-1k/10k/32k/100k cutoffs (unweighted share of words) and as **weighted** coverage over the full top-100k, where each word is weighted by its corpus count - approximating the share of running-text word occurrences that cost a single token. Higher is better. Reproduce with `python3 benchmark_top_words.py`.

**Ġ-prefix:**

| tokenizer | top 1k | top 10k | top 32k | top 100k | weighted |
|---|---:|---:|---:|---:|---:|
| violet | 95.70% | 75.81% | 38.73% | 15.47% | 88.36% |
| v2 | **96.60%** | **88.36%** | 40.24% | 13.85% | **90.51%** |
| v3 | 96.40% | 85.23% | 40.57% | 14.00% | 89.93% |
| v4 | 96.20% | 85.23% | **44.07%** | **15.52%** | 90.14% |

**Bare:**

| tokenizer | top 1k | top 10k | top 32k | top 100k | weighted |
|---|---:|---:|---:|---:|---:|
| violet | 79.80% | 33.03% | 16.64% | 7.77% | 75.14% |
| v2 | **96.00%** | 40.71% | **17.82%** | 7.66% | **81.52%** |
| v3 | 95.60% | **40.77%** | 16.71% | 7.12% | 81.06% |
| v4 | 94.20% | 31.35% | 13.54% | 5.97% | 77.82% |

All in-house tokenizers cover ~96% of the top-1k words as single tokens in running-text form. v2 packs mid-frequency words hardest (best top-10k and weighted score); v4 has the widest deep coverage (best top-32k and top-100k, ahead of violet despite a 35% smaller vocab). Note the ceiling: a 32,768-token vocab can hold at most ~32k single-token words, so top-100k coverage cannot exceed ~33% for the in-house tokenizers.

## Environment

Python 3.13, `tokenizers` 0.22.2, `transformers` 5.12.0, `nltk` 3.10.0 (requires the `punkt_tab`. Tests: `python3 -m pytest tests/`. Retrain v4 from scratch with `python3 gen_tok.py` (delete `dataset-clean*.txt` first to force re-cleaning).

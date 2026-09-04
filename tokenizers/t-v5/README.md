# Tokenizer comparison

| tokenizer | path | vocab size | notes |
|---|---|---:|---|
| v2 | `t-v2` | 32,700 | in-house, April 2026, earlier corpus |
| v3 | `t-v3` | 32,752 | in-house, April 2026, earlier corpus |
| v4 | `t-v4` | 32,768 | in-house, cleaned corpus (datasets 1-4) |
| v5 | `t-v5` | 32,768 | in-house: regex-free plain ByteLevel BPE - number, charset, and top-word guarantees enforced by data preparation and merge surgery, warranted against a case-preserving curated dictionary; runs id-exact in HF tokenizers, Tokie, and Gigatoken; fully reproducible from raw sources (`t-v5/`) |
| violet | `violet` | 50,277 | external, Pythia/GPT-NeoX-style |
| chatterbox | `chatterbox/tokenizer.json` | 32,768 | external Victorian BPE, plain ByteLevel pre-tokenizer |
| typeWriter | `typeWriter/tokenizer.json` | 32,000 | external vintage BPE, Llama-3-style conventions |
| talkie | `talkie/tokenizer.json` | 65,540 | external vintage BPE, Llama-3-style conventions |
| SmolLM2 | `SmolLM/SmolLM2-tokenizer.json` | 49,152 | external general-purpose |
| SmolLM3 | `SmolLM/SmolLM3-tokenizer.json` | 128,256 | external general-purpose |
| GCT | `GCTokenizer/tokenizer.json` | 33,024 | corpus-free consensus of 6 frontier tokenizers; no whitespace tokens |

## 1. How we test

Four data sources feed the tables below; each section notes which one it uses.

**A. The raw-text sample** - used by Compression (§2: corpus-level cost and bits per byte), Fertility (§3), and Proportion of continued words (§4). The six evaluation domains are `dataset-text1..4.txt`, all seven raw `train-*.jsonl` shards, and all seven raw `fine-synth-*.jsonl.xz` shards. We draw **128 chunks of 64 KB per domain**: each shard in the two grouped domains contributes an equal one-seventh share. Plain text and uncompressed JSONL use pseudo-random byte offsets trimmed to whole records; XZ input uses an equally sized fixed prefix because random seeking would require a full decompression pass on the HDD (the shard seed indices are randomized). The offsets are seeded per file (`SEED = 42`), so every run and every tokenizer sees the byte-identical 49.0M-character sample. The sample comes only from the **original files, never any `*-clean*` file**, so the tokenizer is always scored on the original text that will be supplied to the LLM. The six domains contribute equally (~8 MB each), rather than being dominated by the largest source.

For the word-level metrics, a Unicode lexical-word matcher extracts **8,246,186 word instances (180,063 unique)** from that sample. It keeps English contractions and possessives intact (`wasn't`, `Darcy's`) and does not turn quote glyphs into fake words. Each word is encoded in isolation with `add_special_tokens=False`, in two variants: **bare** (the word exactly as extracted) and **Ġ-prefix** (with a leading space - a useful whitespace-context diagnostic, though corpus compression below is the context-faithful headline). Each unique word is encoded once and weighted by its count, which is mathematically identical to the per-instance mean. Fertility charges every token produced; continued-words counts word *fragmentation* only - a standalone leading-space token (as GCTokenizer emits, having no whitespace tokens at all) is charged in fertility but is not a "continuation" of the word. The corpus-level cost table encodes the same 49.0M-character sample verbatim, start to finish - spaces, punctuation, newlines, numbers included - which makes it the single most decision-relevant number.

**B. The word-frequency dictionary** - used by Single-token coverage (§5), Morphological coherence (§10), and Plural duplication (§10b). `words-uncased/words-cased.json` is the **case-preserving curated list** built by `words-uncased/make_cased_words.py`: 65,536 words kept exactly as real text writes them (`Jesus`, `London`, and `the` are separate entries with their real frequencies), counted over four training-data sources and scaled by dataset size so no single source skews the distribution. Every word passed a spell gate (aspell en_GB/en_US in its exact case - `Jesus` is accepted, `jesus` and OCR junk like `Tlie` are not) or a two-source consensus anchored to the curated period list (keeps legitimate period spellings such as `connexion` and `thou`); words shorter than 3 letters are excluded. Coverage therefore measures how well a tokenizer fits words as they actually occur - earlier reports scored a lowercase-folded dictionary, so their §5/§10 columns are not comparable. Independent of sample A.

**C. The vocabularies themselves** - used by the three scan sections (§6–§8). No text sample at all: every ordinary token of each tokenizer is decoded to its surface string and classified. Special/added tokens and byte-fallback entries (`<0xNN>`, partial-UTF-8 fragments) are skipped, since they are structural rather than learned.

**D. The bigram-frequency dictionary** - used by Bigram cost (§5b). `bigrams/bigrams.json` holds the top 100,000 adjacent lowercase word pairs with their aggregate occurrence counts, merged by `bigrams/merge_bigrams.py` from the per-source counts in `bigrams/*-bigrams.json`, which were measured on the actual training data. Independent of sample A. We test the top 10,000 pairs by count.

Result tables are sorted best-first (by the OVERALL / headline column), except §7 where no ordering is meaningful.

## 2. Compression

### 2.1 Corpus-level cost

**What:** this table counts how many tokens each tokenizer needs to write out the same 49M-character text sample - lower is better. **Why:** the model pays for every token, in training time and in answer speed, so fewer tokens means the same text costs less. **How we measured:** we encoded sample A (see §1) from start to finish - spaces, punctuation, and numbers included - and counted the tokens; chars/token is the sample length divided by that count.

| tokenizer | corpus tokens | chars/token |
|---|---:|---:|
| talkie | **10,571,006** | **4.640** |
| SmolLM3 | 10,902,499 | 4.499 |
| v5 | 10,910,380 | 4.496 |
| v4 | 10,932,606 | 4.487 |
| v3 | 11,112,931 | 4.414 |
| v2 | 11,177,577 | 4.388 |
| typeWriter | 11,216,024 | 4.373 |
| SmolLM2 | 11,229,874 | 4.368 |
| chatterbox | 11,256,976 | 4.357 |
| violet | 11,287,511 | 4.345 |
| GCT | 21,436,888 | 2.288 |

### 2.2 Budget-fair compression (bits per byte)

**What:** this table scores each tokenizer as a text compressor - plain text costs 8.0 bits per byte, and **bits/byte** is how many bits the tokenizer spends per byte of the original text (lower is better). **Why:** a bigger vocabulary shortens the token list "for free" in §2.1, but here every token costs log2(vocab size) bits - 15 bits for a 32,768 vocabulary, 16 bits for 65,536 - so tokenizers of different sizes compete fairly, and extra vocabulary must earn its keep. **How we measured:** we multiplied the §2.1 token count by the bit cost of one token and divided by the sample size in bytes; **entropy bpb** repeats this with each token priced by how often it really occurs (the compression a model with a perfectly learned output layer would reach; **unigram ppl** is the same information shown as an effective alphabet size), and **vocab used** is the share of tokens that appeared at least once - every unused token is a dead row in the model, so 100% is the ideal.

| tokenizer | bits/byte | entropy bpb | unigram ppl | vocab used |
|---|---:|---:|---:|---:|
| talkie | 3.4420 | **2.2865** | 1,583 | 85.5% |
| v5 | **3.3305** | 2.3054 | 1,336 | **99.3%** |
| v4 | 3.3373 | 2.3131 | 1,348 | **99.3%** |
| v3 | 3.3922 | 2.3514 | 1,349 | 98.6% |
| SmolLM3 | 3.7649 | 2.3539 | 1,562 | 39.7% |
| v2 | 3.4114 | 2.3661 | 1,353 | 98.2% |
| typeWriter | 3.4160 | 2.3705 | 1,338 | 97.3% |
| SmolLM2 | 3.5617 | 2.3727 | 1,335 | 83.2% |
| chatterbox | 3.4363 | 2.3765 | 1,327 | 97.7% |
| violet | 3.5875 | 2.3933 | 1,369 | 77.1% |
| GCT | 6.5487 | 3.1411 | **147** | 56.8% |

## 3. Fertility

**What:** this table shows the average number of tokens each tokenizer needs for one word - lower is better. **Why:** a good tokenizer keeps common words whole, and every extra token per word makes every text longer for the model. **How we measured:** we took all 8.2M word instances from sample A (see §1) and encoded each unique word alone; the first table puts a leading space before the word (how words appear inside a sentence), the second encodes the bare word.

**Ġ-prefix (running text):**

| tokenizer | text1 | text2 | text3 | text4 | train | fine-synth | OVERALL |
|---|---:|---:|---:|---:|---:|---:|---:|
| talkie | **1.0730** | **1.1387** | 1.0687 | **1.0576** | **1.0571** | 1.0692 | **1.0776** |
| v5 | 1.1046 | 1.1648 | **1.0632** | 1.0825 | 1.0829 | **1.0636** | 1.0943 |
| v4 | 1.1090 | 1.1454 | 1.0830 | 1.0912 | 1.0919 | 1.0835 | 1.1011 |
| typeWriter | 1.0999 | 1.1942 | 1.1038 | 1.0828 | 1.0832 | 1.1049 | 1.1116 |
| chatterbox | 1.1071 | 1.1988 | 1.1138 | 1.0909 | 1.0910 | 1.1154 | 1.1195 |
| v3 | 1.1084 | 1.1983 | 1.1145 | 1.0936 | 1.0936 | 1.1160 | 1.1208 |
| SmolLM3 | 1.1166 | 1.1993 | 1.1061 | 1.0998 | 1.0994 | 1.1066 | 1.1216 |
| SmolLM2 | 1.1268 | 1.2148 | 1.0960 | 1.1017 | 1.1008 | 1.0965 | 1.1234 |
| v2 | 1.1182 | 1.1892 | 1.1255 | 1.0973 | 1.0972 | 1.1267 | 1.1257 |
| violet | 1.1379 | 1.2307 | 1.1302 | 1.1164 | 1.1162 | 1.1306 | 1.1439 |
| GCT | 2.3498 | 2.3861 | 2.4710 | 2.3437 | 2.3440 | 2.4732 | 2.3928 |

**Bare (SmolLM-literal):**

| tokenizer | text1 | text2 | text3 | text4 | train | fine-synth | OVERALL |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3 | **1.2304** | 1.3190 | **1.3199** | **1.2231** | **1.2240** | **1.3224** | **1.2719** |
| v2 | 1.2417 | **1.3068** | 1.3310 | 1.2276 | 1.2278 | 1.3332 | 1.2767 |
| v4 | 1.2685 | 1.3115 | 1.3724 | 1.2658 | 1.2674 | 1.3752 | 1.3086 |
| SmolLM3 | 1.3032 | 1.3591 | 1.4019 | 1.2967 | 1.2967 | 1.4031 | 1.3420 |
| violet | 1.3246 | 1.3811 | 1.4233 | 1.3153 | 1.3153 | 1.4250 | 1.3626 |
| talkie | 1.3189 | 1.3538 | 1.4605 | 1.3198 | 1.3197 | 1.4631 | 1.3705 |
| SmolLM2 | 1.3362 | 1.3942 | 1.4286 | 1.3226 | 1.3218 | 1.4300 | 1.3709 |
| v5 | 1.3275 | 1.3708 | 1.4441 | 1.3257 | 1.3270 | 1.4474 | 1.3720 |
| GCT | 1.3498 | 1.3861 | 1.4710 | 1.3437 | 1.3440 | 1.4732 | 1.3928 |
| typeWriter | 1.5297 | 1.5852 | 1.7139 | 1.5474 | 1.5482 | 1.7178 | 1.6043 |
| chatterbox | 1.5617 | 1.6206 | 1.7366 | 1.5762 | 1.5760 | 1.7407 | 1.6327 |

## 4. Proportion of continued words

**What:** this table shows the share of words that get broken into 2 or more tokens - lower is better. **Why:** a word cut into pieces is harder for the model to learn than a word kept whole, because the meaning must be assembled from fragments. **How we measured:** we encoded every word of sample A (see §1) alone and counted the words that produced more than one token; a stand-alone leading-space token does not count as a break.

**Ġ-prefix (running text):**

| tokenizer | text1 | text2 | text3 | text4 | train | fine-synth | OVERALL |
|---|---:|---:|---:|---:|---:|---:|---:|
| talkie | **5.29%** | 10.07% | 5.50% | **4.25%** | **4.22%** | 5.51% | **5.81%** |
| v5 | 6.59% | 11.02% | **4.38%** | 5.38% | 5.40% | **4.39%** | 6.23% |
| v4 | 7.04% | **9.77%** | 6.00% | 6.11% | 6.14% | 6.04% | 6.87% |
| typeWriter | 6.42% | 13.09% | 7.62% | 5.64% | 5.67% | 7.68% | 7.68% |
| chatterbox | 6.76% | 13.36% | 8.19% | 6.01% | 6.01% | 8.26% | 8.09% |
| v3 | 7.24% | 13.67% | 8.63% | 6.43% | 6.43% | 8.72% | 8.51% |
| SmolLM2 | 8.48% | 14.84% | 7.11% | 6.98% | 6.92% | 7.10% | 8.60% |
| v2 | 7.79% | 13.19% | 9.48% | 6.78% | 6.77% | 9.56% | 8.91% |
| SmolLM3 | 8.55% | 14.30% | 8.45% | 7.38% | 7.35% | 8.45% | 9.09% |
| violet | 9.45% | 16.13% | 9.88% | 8.17% | 8.15% | 9.89% | 10.28% |
| GCT | 25.64% | 28.22% | 34.72% | 25.49% | 25.52% | 34.85% | 28.94% |

**Bare (SmolLM-literal):**

| tokenizer | text1 | text2 | text3 | text4 | train | fine-synth | OVERALL |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3 | **16.89%** | 22.71% | **25.20%** | **16.85%** | **16.92%** | **25.35%** | **20.54%** |
| v2 | 17.60% | **22.14%** | 25.94% | 17.24% | 17.27% | 26.08% | 20.92% |
| v4 | 19.68% | 22.89% | 29.05% | 20.06% | 20.16% | 29.22% | 23.37% |
| SmolLM3 | 23.43% | 26.52% | 31.76% | 23.03% | 23.08% | 31.86% | 26.49% |
| violet | 24.21% | 27.52% | 32.79% | 23.96% | 24.00% | 32.88% | 27.43% |
| v5 | 23.90% | 26.59% | 33.91% | 24.35% | 24.41% | 34.07% | 27.72% |
| SmolLM2 | 24.85% | 28.14% | 32.92% | 24.30% | 24.30% | 33.01% | 27.80% |
| GCT | 25.64% | 28.22% | 34.72% | 25.49% | 25.52% | 34.85% | 28.94% |
| talkie | 25.35% | 26.88% | 36.76% | 25.73% | 25.72% | 36.91% | 29.39% |
| typeWriter | 36.66% | 38.92% | 47.43% | 37.88% | 37.94% | 47.60% | 40.91% |
| chatterbox | 39.22% | 41.80% | 49.49% | 40.23% | 40.22% | 49.66% | 43.29% |

## 5. Single-token coverage of the top corpus words

**What:** this table shows the share of the most common dictionary words that each tokenizer stores as exactly one token - higher is better. **Why:** the top words appear millions of times each, so giving each one its own token saves the most space and gives the model one stable unit to learn per word. **How we measured:** we encoded the top-N words of the curated dictionary (source B, see §1) and counted the ones that came back as a single token; **weighted** weighs each word by how often it appears in the corpus.

**Ġ-prefix:**

| tokenizer | top 1k | top 2k | top 10k | top 32k | top 65k | weighted |
|---|---:|---:|---:|---:|---:|---:|
| talkie | 99.70% | **99.70%** | 97.85% | **74.20%** | **45.00%** | **95.96%** |
| v5 | **99.80%** | 99.50% | 98.42% | 58.86% | 30.09% | 94.58% |
| v4 | 99.70% | 99.35% | **98.59%** | 52.11% | 26.66% | 93.78% |
| typeWriter | 99.60% | 99.55% | 94.55% | 56.42% | 29.36% | 93.44% |
| chatterbox | 99.50% | 99.00% | 93.15% | 55.43% | 29.24% | 92.94% |
| v3 | 99.60% | 99.05% | 93.29% | 49.09% | 25.37% | 92.30% |
| SmolLM2 | 99.40% | 98.50% | 87.89% | 55.38% | 34.14% | 91.85% |
| v2 | 99.50% | 98.95% | 92.87% | 46.12% | 23.94% | 91.78% |
| SmolLM3 | 99.50% | 98.15% | 86.17% | 55.70% | 35.66% | 91.57% |
| violet | 99.30% | 97.55% | 82.62% | 47.39% | 28.28% | 89.85% |
| GCT | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

**Bare:**

| tokenizer | top 1k | top 2k | top 10k | top 32k | top 65k | weighted |
|---|---:|---:|---:|---:|---:|---:|
| v3 | 97.40% | **92.50%** | **39.99%** | 14.95% | 8.05% | **77.31%** |
| v2 | **97.50%** | 92.20% | 38.85% | 14.89% | 8.08% | 76.88% |
| v4 | 96.90% | 86.90% | 31.92% | 12.08% | 6.52% | 73.96% |
| SmolLM3 | 76.00% | 64.90% | 36.26% | **20.39%** | **13.28%** | 70.00% |
| violet | 77.00% | 63.50% | 31.51% | 15.82% | 9.62% | 68.66% |
| v5 | 89.20% | 67.00% | 23.54% | 9.11% | 4.97% | 68.45% |
| SmolLM2 | 73.70% | 62.10% | 32.20% | 16.13% | 9.65% | 68.03% |
| talkie | 76.60% | 57.60% | 28.11% | 15.20% | 9.14% | 67.26% |
| GCT | 71.10% | 58.90% | 31.34% | 17.54% | 11.45% | 66.94% |
| typeWriter | 40.40% | 26.95% | 10.10% | 4.51% | 2.63% | 51.42% |
| chatterbox | 39.40% | 28.35% | 11.07% | 4.99% | 2.92% | 48.43% |

## 5b. Token cost of the top corpus bigrams

**What:** this table shows how many tokens each tokenizer needs for the most common two-word pairs, like " of the". **Why:** 2 tokens is the natural floor (one per word), so a pair that costs 3 or more means the tokenizer breaks even the most frequent text - a lower **weighted mean** is better. **How we measured:** we encoded the top pairs of the bigram dictionary (source D, see §1) with a leading space, bucketed each pair by its token count, and weighed the mean by each pair's corpus count.

**Top 1,000 bigrams:**

| tokenizer | 2 tokens | 3 tokens | ≥4 tokens | weighted mean |
|---|---:|---:|---:|---:|
| SmolLM3 | **99.70%** | 0.30% | **0.00%** | **2.002** |
| talkie | 99.30% | 0.70% | **0.00%** | 2.004 |
| v2 | 99.30% | 0.60% | 0.10% | 2.005 |
| SmolLM2 | 99.00% | 1.00% | **0.00%** | 2.006 |
| v3 | 98.90% | 1.00% | 0.10% | 2.007 |
| v4 | 98.90% | 1.00% | 0.10% | 2.007 |
| v5 | 98.90% | 1.00% | 0.10% | 2.007 |
| chatterbox | 98.90% | 1.00% | 0.10% | 2.007 |
| typeWriter | 98.90% | 1.00% | 0.10% | 2.007 |
| violet | 98.80% | 1.10% | 0.10% | 2.008 |
| GCT | 0.00% | **0.00%** | 100.00% | 4.039 |

**Top 10,000 bigrams:**

| tokenizer | 2 tokens | 3 tokens | ≥4 tokens | weighted mean |
|---|---:|---:|---:|---:|
| talkie | **98.25%** | 1.67% | **0.08%** | **2.011** |
| SmolLM3 | 97.54% | 2.27% | 0.19% | 2.012 |
| v2 | 97.40% | 2.05% | 0.55% | 2.018 |
| SmolLM2 | 96.75% | 2.86% | 0.39% | 2.020 |
| v5 | 97.00% | 2.48% | 0.52% | 2.020 |
| v4 | 96.95% | 2.54% | 0.51% | 2.020 |
| v3 | 97.04% | 2.35% | 0.61% | 2.021 |
| typeWriter | 97.20% | 2.04% | 0.76% | 2.021 |
| chatterbox | 97.02% | 2.20% | 0.78% | 2.022 |
| violet | 96.06% | 3.34% | 0.60% | 2.025 |
| GCT | 0.00% | **0.00%** | 100.00% | 4.168 |


## 6. Ornamental ASCII artifacts

**What:** this table counts vocabulary tokens that are pure ASCII decoration - repeated punctuation like `----` or `***` from page rulers and table borders. **Why:** every decoration token wastes a vocabulary slot that a real word could use, so lower is better (a run of exactly 3 is tolerated by design in the in-house line). **How we measured:** we decoded every learned token of each tokenizer (source C, see §1) and matched it against punctuation-run patterns; each cell is the count and its share of that vocabulary.

| tokenizer | run=3 | run≥4 | mixed≥4 |
|---|---:|---:|---:|
| v5 | 8 / 0.025% | **0 / 0.000%** | 5 / 0.015% |
| chatterbox | 2 / 0.006% | 10 / 0.031% | **0 / 0.000%** |
| v4 | 12 / 0.037% | **0 / 0.000%** | 13 / 0.040% |
| typeWriter | 2 / 0.006% | 15 / 0.048% | **0 / 0.000%** |
| v2 | 8 / 0.025% | 27 / 0.083% | 3 / 0.009% |
| v3 | 12 / 0.037% | 56 / 0.17% | 16 / 0.049% |
| talkie | 41 / 0.063% | 78 / 0.12% | 12 / 0.018% |
| SmolLM2 | 44 / 0.090% | 100 / 0.20% | 57 / 0.12% |
| GCT | 32 / 0.098% | 66 / 0.20% | 92 / 0.28% |
| violet | 79 / 0.16% | 188 / 0.38% | 302 / 0.60% |
| SmolLM3 | 236 / 0.18% | 424 / 0.33% | 1132 / 0.89% |

## 7. Numeric sequence tokens

**What:** this table shows how much of each vocabulary is spent on number tokens, bucketed by the longest digit run inside the token. **Why:** it exposes each tokenizer's number policy - whole-year tokens (the 4-digits column) are efficient for history-dense text, while splitting numbers into small chunks is better for arithmetic, so there is no single best value. **How we measured:** we decoded every learned token (source C, see §1) and bucketed the ones that contain 2, 3, 4, or more consecutive digits.

| tokenizer | 2-digits | 3-digits | 4-digits | digits++ |
|---|---:|---:|---:|---:|
| v2 | 194 / 0.60% | 441 / 1.36% | 158 / 0.49% | 0 / 0.000% |
| v3 | 191 / 0.59% | 497 / 1.53% | 176 / 0.54% | 0 / 0.000% |
| v4 | 182 / 0.56% | 348 / 1.07% | 115 / 0.35% | 0 / 0.000% |
| v5 | 200 / 0.62% | 0 / 0.000% | 0 / 0.000% | 0 / 0.000% |
| violet | 200 / 0.40% | 1422 / 2.84% | 370 / 0.74% | 24 / 0.048% |
| chatterbox | 189 / 0.58% | 615 / 1.89% | 276 / 0.85% | 0 / 0.000% |
| typeWriter | 100 / 0.32% | 308 / 0.98% | 0 / 0.000% | 0 / 0.000% |
| talkie | 100 / 0.15% | 1000 / 1.53% | 0 / 0.000% | 0 / 0.000% |
| SmolLM2 | 0 / 0.000% | 0 / 0.000% | 0 / 0.000% | 0 / 0.000% |
| SmolLM3 | 100 / 0.078% | 1000 / 0.78% | 0 / 0.000% | 0 / 0.000% |
| GCT | 0 / 0.000% | 0 / 0.000% | 0 / 0.000% | 0 / 0.000% |

**Top 3 smallest (vocab share spent on numbers):** 1. SmolLM2 - 0.00% · 2. GCT - 0.00% · 3. v5 - 0.62%
**Top 3 biggest:** 1. violet - 4.03% · 2. chatterbox - 3.31% · 3. v3 - 2.66%

## 8. Character-set plausibility (OCR debris and out-of-domain scripts)

**What:** this table counts tokens that contain characters you would not expect in English text printed before 1900 - scanner (OCR) debris, modern symbols, and foreign scripts. **Why:** for a vintage-English model such tokens are wasted vocabulary slots and a sign of dirty training data, so lower is better. **How we measured:** we decoded every learned token (source C, see §1) and checked each character against an allowlist (ASCII, period typography, extended Latin, Greek); the last column shows the most common offending characters.

| tokenizer | improbable | dominant offending characters |
|---|---:|---:|
| v5 | **9 / 0.028%** | § • « ½ » |
| typeWriter | 14 / 0.044% | • ■ » « § © ½ ™ |
| v4 | 38 / 0.12% | § ― │ ¦ ▪ · ¶ ½ ❞ ─ ʒ ❝ • ´ ¹ ╌ ℥ » י ₂ ‖ U+0097 |
| chatterbox | 41 / 0.13% | • ■ § « » с и н в о а е © т ъ ™ р л ♦ м у ы п я |
| v2 | 42 / 0.13% | ─ ― ┼ ═ │ ½ ¹ © U+00A0 · ⁄ § ₂ ┴ ₄ ² י ³ ¾ ا ¼ ╤ |
| v3 | 76 / 0.23% | ─ ― ┼ ═ ½ § » │ © U+00A0 · ´ • « ■ ¹ ¼ ¶ ♦ ₂ ⁄ י ¾ ו |
| SmolLM2 | 321 / 0.66% | о а н е и т р ا л • с ─ к м µ ل € █ в д · ® © − |
| talkie | 426 / 0.65% | « » • ■ ꝛ © § ® ¢ ¥ ♦ ⸗ · ▼ ™ € U+200B ± ½ ו ˙ ל ا ► |
| violet | 1316 / 2.63% | о а е т и с н р л в п д к у м U+00A0 ا я ь г ы б い з |
| GCT | 3912 / 11.94% | ¸ º ¯ ½ § ¹ » ¾ ± ¼ ´ µ ³ ¿ ¤ ¨ ¥ ¦ ® ¡ ² ª ¬ © |
| SmolLM3 | 21140 / 16.55% | о а е и р т ا н с в л д к м ر п у ل م ن و ی і ت |

## 9. Context fit (whole documents)

**What:** this table shows the share of complete documents that do not fit inside common context windows of 512, 1024, and 2048 tokens - lower is better. **Why:** a document longer than the model's context window gets cut or packed during training, and both hurt quality. **How we measured:** we encoded 5,146 whole documents (the `text` records sampled from the raw JSONL shards; the plain-text sources have no document boundaries and are excluded) and counted the ones longer than each window.

| tokenizer | documents | >512 | >1024 | >2048 |
|---|---:|---:|---:|---:|
| talkie | **5,146** | **52.99%** | **9.76%** | **0.68%** |
| v5 | **5,146** | 54.14% | 13.21% | **0.68%** |
| SmolLM3 | **5,146** | 53.83% | 14.36% | 0.70% |
| v4 | **5,146** | 54.59% | 14.52% | 0.70% |
| typeWriter | **5,146** | 54.78% | 15.31% | 0.70% |
| v3 | **5,146** | 54.96% | 15.43% | 0.70% |
| SmolLM2 | **5,146** | 54.57% | 15.51% | 0.74% |
| chatterbox | **5,146** | 55.29% | 15.70% | 0.74% |
| v2 | **5,146** | 55.31% | 15.95% | 0.72% |
| violet | **5,146** | 55.11% | 16.63% | 0.74% |
| GCT | **5,146** | 89.43% | 54.88% | 15.99% |

## 10. Morphological family coherence (source B)

**What:** this table shows how often related word forms - `walk`, `walks`, `walked`, `walking` - start with the same first token; higher is better. **Why:** when all forms of a word share one stem token, the model learns the word once instead of once per spelling. **How we measured:** for up to 2,000 frequent base words from `words-uncased/words-cased.json` we built families from the regular endings found in the dictionary (`-s -es -ed -ing -'s -ly -er -est`), encoded every member with a leading space, and averaged the share of members whose first token equals the family's most common first token.

| tokenizer | coherence |
|---|---:|
| GCT | **100.00%** |
| v2 | 57.98% |
| violet | 57.94% |
| v3 | 57.04% |
| chatterbox | 56.76% |
| SmolLM3 | 56.47% |
| v4 | 56.40% |
| SmolLM2 | 56.26% |
| v5 | 55.46% |
| typeWriter | 54.10% |
| talkie | 52.29% |

## 10b. Singular/plural vocabulary duplication (source B)

**What:** this table shows how each tokenizer stores singular/plural pairs - **both 1-token** (`canon` and `canons` each own a token), **stem+suffix** (the plural starts with the singular's own token), or **divergent** (the plural starts with a different token, so the model must learn the same word's spelling twice). **Why:** the storage choice is a real trade - dedicated plural tokens cost vocabulary slots (**plural slots**), removing them would make texts longer (**cost to free**, extra tokens per 1,000 running words) - and only **divergent** is clearly bad (lower is better). **How we measured:** we encoded 2,000 singular/plural pairs from `words-uncased/words-cased.json` (regular `-s/-es/-ies` plurals, both forms in the dictionary, leading space added) and bucketed each pair by how the plural tokenizes against its singular.

| tokenizer | both 1-token | stem+suffix | divergent | plural slots | cost to free (tok/1k words) |
|---|---:|---:|---:|---:|---:|
| talkie | 81.30% | 3.65% | 14.95% | 1,626 | 46.23 |
| v5 | 74.75% | 6.35% | 18.90% | 1,495 | 46.10 |
| SmolLM2 | 71.50% | 4.60% | 22.30% | 1,432 | 44.34 |
| v4 | 71.45% | 6.95% | 21.50% | 1,430 | 45.90 |
| SmolLM3 | 69.65% | 4.90% | 23.40% | 1,393 | 43.75 |
| typeWriter | 67.30% | 7.45% | 24.70% | 1,347 | 44.67 |
| chatterbox | 65.95% | 7.75% | 25.55% | 1,320 | 44.48 |
| v3 | 64.20% | 7.15% | 28.15% | 1,285 | 44.41 |
| v2 | 62.90% | 7.30% | 29.30% | 1,259 | 44.37 |
| violet | 61.75% | 6.45% | 29.00% | 1,237 | 42.23 |
| GCT | 0.00% | 0.00% | **0.00%** | 0 | 0.00 |

## 11. Quoted-dialogue cost (sample A)

**What:** this table shows how well each tokenizer compresses quoted dialogue compared with the surrounding prose - the **index** is dialogue chars/token divided by whole-sample chars/token, and higher is better. **Why:** dialogue dominates pre-1900 fiction, so a tokenizer that fragments quoted speech pays that cost on a large share of the corpus. **How we measured:** we extracted all `“…”` spans from sample A (3,363 spans, 152,044 chars), encoded them alone, and compared their chars-per-token with the §2.1 whole-sample number.

| tokenizer | dialogue chars/tok | overall chars/tok | index |
|---|---:|---:|---:|
| talkie | **3.799** | **4.640** | **0.819** |
| SmolLM3 | 3.672 | 4.499 | 0.816 |
| v4 | 3.624 | 4.487 | 0.808 |
| violet | 3.492 | 4.345 | 0.804 |
| v5 | 3.594 | 4.496 | 0.799 |
| SmolLM2 | 3.464 | 4.368 | 0.793 |
| GCT | 1.711 | 2.288 | 0.748 |
| v3 | 3.105 | 4.414 | 0.704 |
| v2 | 2.879 | 4.388 | 0.656 |
| chatterbox | 2.843 | 4.357 | 0.652 |
| typeWriter | 2.843 | 4.373 | 0.650 |

## 12. Encoding throughput and deployment compatibility

**What:** these three tables show how fast each library encodes text (MB/s, higher is better) and whether it produces exactly the same token ids as the HuggingFace reference (**parity**). **Why:** production tokenizes with Tokie or Gigatoken, not with HF - so a **`LOAD FAILED`** or **`IDS DIFFER`** row disqualifies that tokenizer for that library, because shipping it would feed the model a different tokenization than the one this report scored. **How we measured:** every library encoded the same sample A (49.1 MB) - HF was timed during the §2.1 encode itself - and parity requires identical ids on the first domain plus an equal total token count.

**HuggingFace tokenizers (reference):**

| tokenizer | MB/s | Mtokens/s |
|---|---:|---:|
| talkie | **12.0** | **2.6** |
| SmolLM3 | 11.3 | 2.5 |
| SmolLM2 | 9.4 | 2.1 |
| v5 | 9.3 | 2.1 |
| chatterbox | 9.2 | 2.1 |
| v4 | 9.2 | 2.1 |
| typeWriter | 9.1 | 2.1 |
| v3 | 9.0 | 2.0 |
| v2 | 8.9 | 2.0 |
| violet | 8.2 | 1.9 |
| GCT | 2.6 | 1.1 |

**Tokie:**

| tokenizer | MB/s | vs HF | parity |
|---|---:|---:|---:|
| v5 | **558.2** | **60.3×** | OK |
| chatterbox | 537.3 | 58.1× | **IDS DIFFER - 11,257,498 tokens vs HF 11,256,976** |
| SmolLM2 | 533.4 | 57.0× | OK |
| v4 | 530.8 | 57.5× | OK |
| v3 | 522.9 | 57.9× | OK |
| typeWriter | 502.9 | 55.2× | OK |
| talkie | 429.1 | 35.8× | OK |
| v2 | 425.0 | 47.7× | OK |
| SmolLM3 | 413.2 | 36.5× | OK |
| violet | 199.6 | 24.4× | OK |
| GCT | - | - | **LOAD FAILED - Invalid format: vocab should be object** |

**Gigatoken:**

| tokenizer | MB/s | vs HF | parity |
|---|---:|---:|---:|
| typeWriter | **1610.3** | **176.7×** | OK |
| talkie | 1543.1 | 128.9× | OK |
| SmolLM3 | 1289.1 | 114.0× | OK |
| violet | 1028.7 | 125.7× | OK |
| v5 | 605.4 | 65.4× | OK |
| v2 | - | - | **LOAD FAILED - Byte remapping failed: no single-byte vocab entry for byte 0x00** |
| v3 | - | - | **LOAD FAILED - Byte remapping failed: no single-byte vocab entry for byte 0x00** |
| v4 | - | - | **LOAD FAILED - Byte remapping failed: no single-byte vocab entry for byte 0x00** |
| chatterbox | - | - | **LOAD FAILED - Byte remapping failed: no single-byte vocab entry for byte 0x00** |
| SmolLM2 | - | - | **LOAD FAILED - Byte remapping failed: no single-byte vocab entry for byte 0x04** |
| GCT | - | - | **LOAD FAILED - Failed to parse tokenizer JSON: missing field `model` at line 33032 column 13** |

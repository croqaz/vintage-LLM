#!/usr/bin/env python3
"""
Text Sharding Tool
Splits text files into ~500-word chunks at paragraph boundaries,
with sentence-level fallback for overly long paragraphs.

Algorithm:
  1. Split text into paragraphs with exact char positions
  2. Greedily accumulate paragraphs until word count crosses into target range
  3. Find best cut (paragraph boundary closest to 500 within [300,700])
  4. Emit shard: text = exact original span, pos = start offset, words = count
  5. Resume from next paragraph

For paragraphs >700 words that can't be split at paragraph boundaries,
fall back to sentence-level splitting within that paragraph.

Usage: python3 shard.py [--dir DIR] [--target N] [--tolerance N]
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict

from ftfy import TextFixerConfig, fix_text

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

fixer = TextFixerConfig(
    unescape_html=False,
    remove_terminal_escapes=True,
    fix_encoding=True,
    restore_byte_a0=True,
    replace_lossy_sequences=True,
    decode_inconsistent_utf8=True,
    fix_c1_controls=True,
    fix_latin_ligatures=True,
    fix_character_width=True,
    uncurl_quotes=False,
    fix_line_breaks=True,
    fix_surrogates=True,
    remove_control_chars=True,
)

BAD_CHAR_RANGE = '[%s]' % ''.join(
    [
        '\x00-\x08',
        '\x0b',
        '\x0c',
        '\x0e-\x1f',
        '\x7f-\x9f',
        '\x81',
        '\x90-\x9f',
        '\xa0',
        '\xad',
        '\u0604',
        '\u0cce',
        '\u1311',
        '\u2003',
        '\u200b',
        '\u200e',
        '\u2063',
        '\ue000',
        '\ue002',
        '\ue03f',
        '\ue084',
        '\ue096',
        '\ue0ab',
        '\u200b-\u200f',
        '\ud800-\udfff',
        '\ufdd0-\ufdef',
        '\ufeff',
        '\uffbf',
        '\ufffd',
        '\N{HANGUL FILLER}',
        '\N{HANGUL CHOSEONG FILLER}',
        '\N{HANGUL JUNGSEONG FILLER}',
    ]
    + [chr(65534 + 65536 * x + y) for x in range(17) for y in range(2)]
)
BAD_CHARS_RE = re.compile(BAD_CHAR_RANGE)

TARGET_WORDS = 500
TOLERANCE = 200
# These are recomputed in main() based on args

FILES = [
    ('Shakespeare', 'Shakespeare.txt'),
    ('John-Milton', 'John-Milton.txt'),
    ('La-Fontaine', 'La-Fontaine.txt'),
    ('John-Bunyan', 'John-Bunyan.txt'),
]

# Abbreviations that should NOT trigger sentence splits
ABBREVIATIONS = {
    'mr',
    'mrs',
    'ms',
    'dr',
    'prof',
    'rev',
    'hon',
    'st',
    'sr',
    'jr',
    'esq',
    'capt',
    'col',
    'gen',
    'lt',
    'maj',
    'sgt',
    'corp',
    'gov',
    'sen',
    'rep',
    'pres',
    'sec',
    'dept',
    'univ',
    'co',
    'inc',
    'ltd',
    'bros',
    'vs',
    'etc',
    'viz',
    'al',
    'ibid',
    'op',
    'cit',
    'vol',
    'no',
    'nos',
    'p',
    'pp',
    'ch',
    'fig',
    'eq',
    'ed',
    'eds',
    'trans',
    'cf',
    'e.g',
    'i.e',
    'a.m',
    'p.m',
    'a.d',
    'b.c',
    'c',
    'ca',
    'approx',
    'est',
    'min',
    'max',
    'misc',
    'ph.d',
    'm.d',
    'b.a',
    'm.a',
    'll.d',
    'd.d',
    'esqr',
    'messrs',
    'mme',
    'mlle',
    'i',
    'ii',
    'iii',
    'iv',
    'v',
    'vi',
    'vii',
    'viii',
    'ix',
    'x',
    'xi',
    'xii',
    'xiii',
    'xiv',
    'xv',
    'xvi',
    'xvii',
    'xviii',
    'xix',
    'xx',
}


# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────


def read_file(path):
    """Read file, strip BOM, normalize line endings, return raw text."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    if text.startswith('\ufeff'):
        text = text[1:]
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = fix_text(text, config=fixer)
    text = re.sub(BAD_CHARS_RE, '', text)
    return text


def count_words(text):
    """Count whitespace-delimited tokens."""
    return len(text.split())


def split_paragraphs(text):
    """
    Split text into paragraphs separated by one or more blank lines.
    Returns list of dicts: {text, start, end}
    where start/end are char offsets in the original text.
    The 'end' of each paragraph includes the trailing separator,
    so paragraphs are contiguous with no gaps.
    Whitespace-only paragraphs are preserved to maintain position continuity.
    """
    paragraphs = []
    # Pattern: blank line = newline, optional whitespace (spaces/tabs), newline
    pattern = re.compile(r'\n[ \t]*\n')
    pos = 0

    for match in pattern.finditer(text):
        separator_start = match.start()
        separator_end = match.end()

        # The paragraph text is from pos to separator start (exclusive)
        para_text = text[pos:separator_start]
        # Keep ALL paragraphs, even whitespace-only, to maintain contiguous positions
        paragraphs.append(
            {
                'text': para_text,
                'start': pos,
                # Include the separator in this paragraph's span
                'end': separator_end,
            }
        )
        pos = separator_end

    # Last segment (no trailing separator)
    if pos < len(text):
        remaining = text[pos:]
        paragraphs.append(
            {
                'text': remaining,
                'start': pos,
                'end': len(text),
            }
        )

    return paragraphs


def tokenize_sentences(text):
    """
    Split text into sentences.
    Returns list of dicts: {text, start, end}
    where start/end are offsets *within the given text*.
    The 'end' of each sentence includes trailing whitespace up to
    the start of the next sentence, so sentences are contiguous.
    """
    if not text.strip():
        return [{'text': text, 'start': 0, 'end': len(text)}]

    # First pass: find all sentence boundary positions
    boundaries = []  # (end_pos_inclusive, next_start_pos)
    n = len(text)
    i = 0

    while i < n:
        ch = text[i]

        if ch in '.!?':
            j = i + 1
            # Skip whitespace after punctuation
            while j < n and text[j] in ' \t\n':
                j += 1

            next_ch = text[j] if j < n else ''
            is_boundary = False

            if ch == '.':
                # Check for abbreviation
                word_start = i - 1
                while word_start >= 0 and text[word_start].isalpha():
                    word_start -= 1
                word_start += 1
                word = text[word_start:i].lower()

                if word in ABBREVIATIONS:
                    i = j
                    continue

                # Check if followed by digit (numbered list item)
                if next_ch.isdigit():
                    i = j
                    continue

            # Boundary if followed by uppercase or quote or end of text
            if next_ch and (next_ch.isupper() or next_ch in '"\'(') or next_ch == '' and ch in '.!?':
                is_boundary = True

            if is_boundary:
                boundaries.append((i, j))  # sentence ends at i (inclusive), next starts at j
                i = j
                continue

        i += 1

    # Build sentence dicts from boundaries
    sentences = []
    prev_end = 0

    for end_pos, next_start in boundaries:
        sent_text = text[prev_end:next_start]  # include whitespace after punctuation
        sentences.append(
            {
                'text': sent_text,
                'start': prev_end,
                'end': next_start,
            }
        )
        prev_end = next_start

    # Emit trailing text
    if prev_end < n:
        remaining_text = text[prev_end:]
        if remaining_text.strip():
            sentences.append(
                {
                    'text': remaining_text,
                    'start': prev_end,
                    'end': n,
                }
            )

    if not sentences:
        sentences = [{'text': text, 'start': 0, 'end': n}]

    return sentences


# ──────────────────────────────────────────────
# Core algorithm
# ──────────────────────────────────────────────


def find_best_split(cumulative_counts, target, min_wc, max_wc):
    """
    Given a list of cumulative word counts for accumulated units
    (paragraphs or sentences), find the best cut index.

    Returns: (cut_index, is_oversized_first_unit)
      cut_index: index into the list (exclusive) — units[:cut_index] form the shard
      is_oversized_first_unit: True if the first unit alone exceeds max_wc
    """
    n = len(cumulative_counts)
    first_wc = cumulative_counts[0]

    # If even the first unit is oversized, signal sentence fallback
    if first_wc > max_wc and n > 1:
        return 1, True

    # If only one unit and it's oversized
    if n == 1 and first_wc > max_wc:
        return 1, True

    # If we never reached min_wc, return all
    if cumulative_counts[-1] < min_wc:
        return n, False

    # Otherwise, find the index closest to target within [min_wc, max_wc]
    best_idx = None
    best_score = float('inf')

    for idx, cw in enumerate(cumulative_counts):
        if min_wc <= cw <= max_wc:
            score = abs(cw - target)
            # Slight preference for overshooting (> target) vs undershooting
            if cw < target:
                score += 5
            if score < best_score:
                best_score = score
                best_idx = idx + 1  # exclusive index

    if best_idx is not None:
        return best_idx, False

    # No candidate in range, take closest to max_wc without going too far over
    best_idx = None
    best_dist = float('inf')
    for idx, cw in enumerate(cumulative_counts):
        # Prefer cuts at or under max_wc + 100 (soft limit)
        if cw <= max_wc + 100:
            dist = abs(cw - target)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx + 1
        elif cw <= max_wc + 300:
            dist = abs(cw - target) + 50  # penalty
            if dist < best_dist:
                best_dist = dist
                best_idx = idx + 1

    if best_idx is not None:
        return best_idx, False

    return n, False


def shard_file(filepath, target=TARGET_WORDS):
    """
    Main sharding function.
    Returns list of shard dicts: {text, words, pos}
    """
    text = read_file(filepath)
    min_wc = target - TOLERANCE
    max_wc = target + TOLERANCE

    # Step 1: Get all paragraphs with positions
    paragraphs = split_paragraphs(text)
    total_paras = len(paragraphs)

    shards = []
    cursor = 0  # index into paragraphs list

    while cursor < total_paras:
        # --- Accumulate paragraphs ---
        accumulated_paras = []
        cumulative_wc = []
        running_wc = 0

        while cursor < total_paras and running_wc < max_wc:
            para = paragraphs[cursor]
            para_wc = count_words(para['text'])
            accumulated_paras.append(para)
            running_wc += para_wc
            cumulative_wc.append(running_wc)
            cursor += 1

            if running_wc >= min_wc:
                # Peek at next paragraph to see if adding it gets closer
                if cursor < total_paras:
                    next_para = paragraphs[cursor]
                    next_wc = count_words(next_para['text'])
                    if running_wc + next_wc <= max_wc:
                        continue  # add the next one too
                    elif abs(running_wc + next_wc - target) < abs(running_wc - target):
                        # Adding next gets closer to 500 even if it exceeds 700
                        continue
                break

        # --- Find best cut ---
        cut_idx, is_oversized = find_best_split(cumulative_wc, target, min_wc, max_wc)

        # --- Handle oversized single paragraph ---
        if is_oversized and cut_idx == 1:
            oversized_para = accumulated_paras[0]
            para_text = oversized_para['text']
            para_start = oversized_para['start']

            # Sentence-level splitting
            sentences = tokenize_sentences(para_text)

            # Convert sentence offsets to absolute (in original text)
            abs_sentences = []
            for sent in sentences:
                abs_sentences.append(
                    {
                        'text': sent['text'],
                        'start': para_start + sent['start'],
                        'end': para_start + sent['end'],
                    }
                )
            # Ensure the last sentence's end covers the full paragraph span
            # (including the trailing separator that the paragraph owned)
            if abs_sentences:
                abs_sentences[-1]['end'] = oversized_para['end']

            # Accumulate ALL sentences (don't stop early — we need them all
            # so remaining sentences aren't lost)
            sent_acc = []
            sent_cum = []
            sent_running = 0
            for sent in abs_sentences:
                sw = count_words(sent['text'])
                sent_acc.append(sent)
                sent_running += sw
                sent_cum.append(sent_running)

            sent_cut, sent_oversized = find_best_split(sent_cum, target, min_wc, max_wc)

            # If even a single sentence is >700, just take that one sentence
            if sent_oversized and sent_cut == 1:
                sent_cut = 1
            elif sent_cut == 0:
                sent_cut = len(sent_acc)

            use_sents = sent_acc[:sent_cut]
            remaining_sents = sent_acc[sent_cut:]

            # Emit shard from consumed sentences
            if use_sents:
                shard_start = use_sents[0]['start']
                shard_end = use_sents[-1]['end']
                shard_text = text[shard_start:shard_end]
                shard_wc = count_words(shard_text)

                shards.append(
                    {
                        'text': shard_text,
                        'words': shard_wc,
                        'pos': shard_start,
                    }
                )

            # Inject remaining sentences as synthetic paragraphs into the main list.
            # The oversized paragraph occupied paragraphs[consumed_idx].
            # cursor already advanced past it (cursor = consumed_idx + 1).
            consumed_idx = cursor - 1

            if remaining_sents:
                synthetic_paras = []
                for rs in remaining_sents:
                    synthetic_paras.append(
                        {
                            'text': rs['text'],
                            'start': rs['start'],
                            'end': rs['end'],
                        }
                    )
                # Replace the consumed oversized paragraph with remaining sentences
                paragraphs = paragraphs[:consumed_idx] + synthetic_paras + paragraphs[cursor:]
                cursor = consumed_idx  # re-process the first remaining sentence
            else:
                # No remaining sentences — remove the consumed paragraph entirely
                paragraphs = paragraphs[:consumed_idx] + paragraphs[cursor:]
                cursor = consumed_idx  # will advance to next paragraph

            total_paras = len(paragraphs)
            continue  # next iteration of outer loop

        # --- Emit shard from paragraphs ---
        use_paras = accumulated_paras[:cut_idx]
        unconsumed = accumulated_paras[cut_idx:]

        if not use_paras:
            # Shouldn't happen, but safety
            if unconsumed:
                cursor -= len(unconsumed)
            continue

        shard_start = use_paras[0]['start']
        shard_end = use_paras[-1]['end']
        shard_text = text[shard_start:shard_end]
        shard_wc = count_words(shard_text)

        shards.append(
            {
                'text': shard_text,
                'words': shard_wc,
                'pos': shard_start,
            }
        )

        # --- Backtrack for unconsumed paragraphs ---
        if unconsumed:
            backtrack = len(unconsumed)
            cursor -= backtrack

    return shards


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────


def validate_shards(shards, original_text, filename):
    """Verify integrity of shards against original text."""
    errors = []

    if not shards:
        return ['No shards produced']

    # Check first shard
    if shards[0]['pos'] != 0:
        errors.append(f'First shard pos={shards[0]["pos"]}, expected 0')

    # Check continuity and content
    for i, shard in enumerate(shards):
        expected_span = original_text[shard['pos'] : shard['pos'] + len(shard['text'])]
        if expected_span != shard['text']:
            errors.append(f'Shard {i}: text mismatch at pos {shard["pos"]} (len shard={len(shard["text"])}, len span={len(expected_span)})')

        actual_wc = count_words(shard['text'])
        if shard['words'] != actual_wc:
            errors.append(f'Shard {i}: stored words={shard["words"]}, actual={actual_wc}')

        if i + 1 < len(shards):
            this_end = shard['pos'] + len(shard['text'])
            next_start = shards[i + 1]['pos']
            if this_end != next_start:
                gap_text = original_text[this_end:next_start]
                errors.append(
                    f'Gap at shard {i}: end={this_end}, next_pos={next_start}, '
                    f'gap_len={next_start - this_end}, '
                    f'gap_repr={repr(gap_text[:120])}'
                )

    # Check last shard
    last = shards[-1]
    last_end = last['pos'] + len(last['text'])
    remaining = original_text[last_end:]
    if remaining.strip():
        errors.append(f'Last shard ends at {last_end}, file len={len(original_text)}, uncovered non-whitespace: {repr(remaining[:200])}')

    # Word count sanity
    total_shard_words = sum(s['words'] for s in shards)
    total_original_words = count_words(original_text)
    # Small differences are expected since paragraph separators affect tokenization
    # Allow up to ~2% difference or max 50 words
    diff = abs(total_shard_words - total_original_words)
    if diff > max(total_original_words * 0.02, 50):
        errors.append(f'Total words mismatch: shards={total_shard_words}, original={total_original_words}, diff={diff}')

    return errors


# ──────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────


def print_report(shards, filename):
    """Print sharding statistics."""
    wc_list = [s['words'] for s in shards]
    print(f'\n  {"─" * 50}')
    print(f'  File: {filename}')
    print(f'  Total shards: {len(shards)}')
    print(
        f'  Word counts: min={min(wc_list)}, max={max(wc_list)}, '
        f'mean={sum(wc_list) / len(wc_list):.1f}, '
        f'median={sorted(wc_list)[len(wc_list) // 2]}'
    )
    print(f'  Total words across shards: {sum(wc_list):,}')

    buckets = OrderedDict(
        [
            ('<300', 0),
            ('300-400', 0),
            ('400-450', 0),
            ('450-500', 0),
            ('500-550', 0),
            ('550-600', 0),
            ('600-700', 0),
            ('>700', 0),
        ]
    )
    for wc in wc_list:
        if wc < 300:
            buckets['<300'] += 1
        elif wc < 400:
            buckets['300-400'] += 1
        elif wc < 450:
            buckets['400-450'] += 1
        elif wc <= 500:
            buckets['450-500'] += 1
        elif wc <= 550:
            buckets['500-550'] += 1
        elif wc <= 600:
            buckets['550-600'] += 1
        elif wc <= 700:
            buckets['600-700'] += 1
        else:
            buckets['>700'] += 1

    print('  Distribution:')
    n = max(len(shards), 1)
    for k, v in buckets.items():
        bar = '█' * (v * 40 // n)
        print(f'    {k:>8}: {v:>6} ({100 * v / n:5.1f}%) {bar}')


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='Shard text files into ~500-word chunks')
    parser.add_argument('--dir', type=str, default='.', help='Directory containing input .txt files')
    parser.add_argument('--target', type=int, default=500, help='Target word count per shard')
    parser.add_argument('--tolerance', type=int, default=200, help='+/- tolerance for word count')
    args = parser.parse_args()

    global TARGET_WORDS, TOLERANCE
    TARGET_WORDS = args.target
    TOLERANCE = args.tolerance
    min_wc = TARGET_WORDS - TOLERANCE
    max_wc = TARGET_WORDS + TOLERANCE

    print('=' * 60)
    print('TEXT SHARDING TOOL')
    print(f'Target: {TARGET_WORDS} words, Tolerance: +/-{TOLERANCE}')
    print(f'Range: [{min_wc}, {max_wc}]')
    print('=' * 60)

    all_errors = {}
    all_shards_info = []

    for name, filename in FILES:
        filepath = os.path.join(args.dir, filename)
        if not os.path.exists(filepath):
            print(f'\n  SKIP: {filepath} not found')
            continue

        print(f'\n  Processing: {filename} ...')

        original_text = read_file(filepath)
        owc = count_words(original_text)
        print(f'    Original: {owc:,} words, {len(original_text):,} chars, {len(split_paragraphs(original_text))} paragraphs')

        shards = shard_file(filepath, target=TARGET_WORDS)

        errors = validate_shards(shards, original_text, filename)
        if errors:
            all_errors[filename] = errors
            print(f'    ⚠ VALIDATION ERRORS ({len(errors)}):')
            for e in errors[:10]:
                print(f'      - {e}')
            if len(errors) > 10:
                print(f'      ... and {len(errors) - 10} more')
        else:
            print('    ✓ Validation passed')

        print_report(shards, filename)

        # Write output
        out_path = os.path.join(args.dir, f'{name}_shards.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(shards, f, ensure_ascii=False, indent=2)
        out_size = os.path.getsize(out_path)
        print(f'    Output: {out_path} ({out_size:,} bytes)')

        all_shards_info.append(
            {
                'name': name,
                'shards': len(shards),
                'output': out_path,
                'size_bytes': out_size,
            }
        )

    # Summary
    print('\n' + '=' * 60)
    if all_errors:
        print(f'COMPLETED WITH ERRORS in {len(all_errors)} file(s):')
        for fname, errs in all_errors.items():
            print(f'  {fname}: {len(errs)} error(s)')
        rc = 1
    else:
        print('ALL FILES SHARDED SUCCESSFULLY ✓')
        rc = 0

    print('\nOutput files:')
    for info in all_shards_info:
        print(f'  {info["name"]}_shards.json: {info["shards"]} shards, {info["size_bytes"]:,} bytes')
    print('=' * 60)

    sys.exit(rc)


if __name__ == '__main__':
    main()

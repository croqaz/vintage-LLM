#!/usr/bin/env python3
"""
Gutenberg Book Splitter
========================
Splits text books from the 'gutenberg' folder into chunks/shards of
approximately equal token (word) count, respecting paragraph boundaries.

Rules:
  - Never break in the middle of a word or sentence.
  - Split at paragraph boundaries whenever possible.
  - Target chunk size: TARGET_TOKENS (default 64000).
  - Tolerance: +5% / -20% → acceptable range [T*0.80, T*1.05].
  - For extremely long paragraphs, fall back to sentence-boundary splits.
  - Source folder ('gutenberg') is treated as READ-ONLY.
  - Output folder: 'gutenberg_chunks' with one subfolder per book.
  - A detailed text report is generated for every book.

Usage:
    python split_books.py [--target TOKENS] [--input-dir DIR] [--output-dir DIR]
"""

import argparse
import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    """
    Count tokens as a proxy via whitespace-delimited words.
    For a more accurate LLM token count, replace this with tiktoken:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")  # or "o200k_base"
        return len(enc.encode(text))
    """
    return len(text.split())


# ---------------------------------------------------------------------------
# Paragraph / sentence splitting
# ---------------------------------------------------------------------------


def split_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs.
    A paragraph break is one or more blank lines (double-newline or more).
    We collapse multiple blank lines into a single separator for output.
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Split on sequences of 2+ newlines (blank-line separators)
    parts = re.split(r'\n{2,}', text)
    # Strip surrounding whitespace but keep internal newlines (for poems etc.)
    return [p.strip() for p in parts if p.strip()]


def split_sentences(paragraph: str) -> list[str]:
    """
    Split a paragraph into sentences.
    Handles common sentence-ending punctuation: . ! ?
    Avoids splitting on abbreviations like Mr. Dr. etc.
    """
    # Rough sentence split on [.!?] followed by whitespace and capital letter
    # or end of string.
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'“”‘’\'])', paragraph)
    # Also handle the last sentence
    if not sentences:
        return [paragraph]
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Core splitting logic
# ---------------------------------------------------------------------------


def split_long_element(
    element: str,
    target: int,
    upper: int,
    lower: int,
    element_type: str = 'paragraph',
) -> list[str]:
    """
    Split an element (paragraph or sentence) that is itself larger than the
    upper bound.  Recursively tries sentence splitting first; if a single
    sentence still exceeds the upper bound, it is split by words with a hard
    cut at the upper bound (only as a last resort).
    """
    if element_type == 'paragraph':
        sentences = split_sentences(element)
        # If we got multiple sentences, process them
        if len(sentences) > 1:
            return _chunkify_lookahead(sentences, target, upper, lower, 'sentence')
        # Single sentence that's too long → hard word-split
        return _hard_word_split(element, upper)
    else:
        # Already at sentence level → hard word-split
        return _hard_word_split(element, upper)


def _hard_word_split(text: str, upper: int) -> list[str]:
    """Last-resort split: break at word boundaries every `upper` words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), upper):
        chunk = ' '.join(words[i : i + upper])
        chunks.append(chunk)
    return chunks


def _chunkify_lookahead(
    items: list[str],
    target: int,
    upper: int,
    lower: int,
    item_type: str,
) -> list[str]:
    """
    Lookahead chunk-assembly: scans ahead from each start position, finds all
    paragraph boundaries whose cumulative token count falls within [lower, upper],
    and picks the one *closest to target*.  This distributes chunk sizes around
    the target instead of always packing to the upper bound.

    When no boundary falls in [lower, upper] we take what we have (final
    remainder or a chunk below lower), or split an oversized item internally.
    """
    n = len(items)
    # Pre-compute token counts so we can slide quickly
    tok = [count_tokens(it) for it in items]

    chunks: list[str] = []
    left = 0  # start index (inclusive)

    def emit(start: int, end: int) -> None:
        """Join items [start:end] and append as a chunk."""
        if item_type == 'paragraph':
            chunks.append('\n\n'.join(items[start:end]))
        else:
            chunks.append(' '.join(items[start:end]))

    while left < n:
        # ---- expand window until we would exceed upper (or run out) ----
        window_sum = 0
        best_end = left  # best exclusive end found so far
        best_dist = float('inf')
        right = left

        while right < n and window_sum + tok[right] <= upper:
            window_sum += tok[right]
            right += 1
            if window_sum >= lower:
                dist = abs(window_sum - target)
                if dist < best_dist:
                    best_dist = dist
                    best_end = right

        # ---- decide where to split ----
        if best_end > left:
            # At least one boundary landed in [lower, upper]; pick the best.
            emit(left, best_end)
            left = best_end

        elif window_sum == 0:
            # The very first item (items[left]) exceeds upper on its own.
            # Split it internally (sentence-level, then word-level if needed).
            sub = split_long_element(items[left], target, upper, lower, item_type)
            for sc in sub:
                chunks.append(sc)
            left += 1

        elif right == n:
            # Reached end of items with window_sum < lower — final tail.
            emit(left, right)
            left = right

        else:
            # window_sum > 0 but < lower, and right < n.
            # Adding tok[right] would exceed upper, so we cannot include it.
            # Take what we have (below lower, but acceptable per -20 % policy).
            emit(left, right)
            left = right

    return chunks


def split_book(text: str, target_tokens: int = 64000) -> list[str]:
    """
    Split a single book's text into chunks.
    """
    upper = int(target_tokens * 1.05)
    lower = int(target_tokens * 0.80)

    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return [text] if text else []

    chunks = _chunkify_lookahead(paragraphs, target_tokens, upper, lower, 'paragraph')
    return chunks


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_book_report(
    book_name: str,
    source_path: str,
    chunks: list[str],
    target_tokens: int,
    output_dir: str,
) -> str:
    """
    Generate a detailed text report for a single book.
    """
    upper = int(target_tokens * 1.05)
    lower = int(target_tokens * 0.80)
    total_tokens = sum(count_tokens(c) for c in chunks)
    chunk_sizes = [count_tokens(c) for c in chunks]
    num_chunks = len(chunks)

    # Format a size histogram (simple ASCII bar)
    max_bar = 50
    max_size = max(chunk_sizes) if chunk_sizes else 0
    histogram_lines = []
    for i, sz in enumerate(chunk_sizes):
        bar_len = int((sz / max(max_size, 1)) * max_bar)
        bar = '█' * bar_len + '░' * (max_bar - bar_len)
        flag = ''
        if sz < lower:
            flag = ' ⚠ BELOW lower bound'
        elif sz > upper:
            flag = ' ⚠ ABOVE upper bound'
        histogram_lines.append(f'  Chunk {i + 1:4d}: {sz:>8d} tokens  {bar}{flag}')

    avg_str = f'{(total_tokens / num_chunks):,.1f} tokens' if num_chunks else 'N/A'
    med_str = f'{sorted(chunk_sizes)[num_chunks // 2]:,}' if num_chunks else 'N/A'
    min_str = f'{min(chunk_sizes):,}' if num_chunks else 'N/A'
    max_str = f'{max(chunk_sizes):,}' if num_chunks else 'N/A'

    report = (
        f"""
{'=' * 70}
BOOK SPLIT REPORT
{'=' * 70}

Book name:       {book_name}
Source file:     {source_path}
Target size:     {target_tokens:,} tokens (words)
Lower bound:     {lower:,} tokens  (−20 %)
Upper bound:     {upper:,} tokens  (+5 %)
Total tokens:    {total_tokens:,}
Number of chunks:{num_chunks}
Average chunk:   {avg_str}
Median chunk:    {med_str}
Min chunk:       {min_str}
Max chunk:       {max_str}

{'─' * 70}
CHUNK SIZE DISTRIBUTION
{'─' * 70}
"""
        + '\n'.join(histogram_lines)
        + f"""

{'─' * 70}
OUTPUT FILES
{'─' * 70}
"""
    )
    for i in range(num_chunks):
        fname = f'{book_name}_chunk_{i + 1:04d}.txt'
        report += f'  {output_dir}/{fname}\n'

    report += f"""
{'─' * 70}
SUMMARY
{'─' * 70}
"""
    below = sum(1 for s in chunk_sizes if s < lower)
    ok = sum(1 for s in chunk_sizes if lower <= s <= upper)
    above = sum(1 for s in chunk_sizes if s > upper)
    report += f'  Chunks within [{lower:,}, {upper:,}] : {ok}/{num_chunks}\n'
    report += f'  Chunks below lower bound            : {below}/{num_chunks}\n'
    report += f'  Chunks above upper bound            : {above}/{num_chunks}\n'

    report += f'\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
    report += f'{"=" * 70}\n'

    return report


def generate_global_report(
    all_book_results: list[dict],
    target_tokens: int,
    output_root: str,
) -> str:
    """
    Generate a global summary report across all processed books.
    """
    upper = int(target_tokens * 1.05)
    lower = int(target_tokens * 0.80)
    n_books = len(all_book_results)
    if n_books == 0:
        return 'No books processed.\n'

    total_chunks = sum(r['num_chunks'] for r in all_book_results)
    total_tokens = sum(r['total_tokens'] for r in all_book_results)

    report = f"""
{'=' * 70}
GLOBAL SPLIT REPORT — ALL BOOKS
{'=' * 70}

Books processed:    {n_books}
Target chunk size:  {target_tokens:,} tokens
Acceptable range:   [{lower:,}, {upper:,}]
Total chunks:       {total_chunks}
Total tokens:       {total_tokens:,}
Overall average:    {(total_tokens / max(total_chunks, 1)):,.1f} tokens/chunk

{'─' * 70}
PER-BOOK SUMMARY
{'─' * 70}
"""
    report += f'{"Book":<30s} {"Chunks":>6s} {"TotalTok":>10s} {"AvgChunk":>10s} {"Min":>8s} {"Max":>8s} {"Status"}\n'
    report += '-' * 80 + '\n'

    for r in all_book_results:
        name = r['book_name'][:28]
        nch = r['num_chunks']
        tt = r['total_tokens']
        avg = tt / max(nch, 1)
        mn = r['min_chunk']
        mx = r['max_chunk']
        below = r['below']
        above = r['above']
        status = '✓'
        issues = []
        if below > 0:
            issues.append(f'{below}↓')
        if above > 0:
            issues.append(f'{above}↑')
        if issues:
            status = ','.join(issues)
        report += f'{name:<30s} {nch:>6d} {tt:>10,d} {avg:>10,.1f} {mn:>8,d} {mx:>8,d} {status}\n'

    report += f'\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
    report += f'{"=" * 70}\n'
    return report


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def read_book(filepath: str) -> str:
    """Read a book file with encoding detection."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            return f.read()


def write_chunks_and_report(
    book_name: str,
    source_path: str,
    chunks: list[str],
    target_tokens: int,
    output_root: str,
) -> dict:
    """
    Write chunk files and the per-book report to disk.
    Returns a dict with summary statistics.
    """
    book_dir = os.path.join(output_root, book_name)
    os.makedirs(book_dir, exist_ok=True)

    # Write chunks
    for i, chunk_text in enumerate(chunks):
        fname = f'{book_name}_chunk_{i + 1:04d}.txt'
        fpath = os.path.join(book_dir, fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(chunk_text)

    # Generate & write report
    report_text = generate_book_report(book_name, source_path, chunks, target_tokens, book_dir)
    report_path = os.path.join(book_dir, f'{book_name}_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Summary dict
    sizes = [count_tokens(c) for c in chunks]
    upper = int(target_tokens * 1.05)
    lower = int(target_tokens * 0.80)
    return {
        'book_name': book_name,
        'num_chunks': len(chunks),
        'total_tokens': sum(sizes),
        'min_chunk': min(sizes) if sizes else 0,
        'max_chunk': max(sizes) if sizes else 0,
        'below': sum(1 for s in sizes if s < lower),
        'above': sum(1 for s in sizes if s > upper),
        'chunk_sizes': sizes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Split Gutenberg books into equal-sized chunks at paragraph boundaries.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python split_books.py
              python split_books.py --target 16000 --input-dir ./gutenberg --output-dir ./chunks
        """),
    )
    parser.add_argument('--target', type=int, default=4000, help='Target tokens (words) per chunk (default: 4000)')
    parser.add_argument('--input-dir', type=str, default='gutenberg', help='Input directory with book files (default: gutenberg)')
    parser.add_argument(
        '--output-dir', type=str, default='gutenberg_chunks', help='Output directory for chunks and reports (default: gutenberg_chunks)'
    )
    parser.add_argument('--limit', type=int, default=0, help='Limit to first N books (0 = all, for testing)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_tokens = args.target

    if not input_dir.is_dir():
        print(f"ERROR: Input directory '{input_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    # Collect book files
    book_files = sorted(f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() == '.txt')
    if not book_files:
        print(f"ERROR: No .txt files found in '{input_dir}'.", file=sys.stderr)
        sys.exit(1)

    if args.limit > 0:
        book_files = book_files[: args.limit]

    print(f'Found {len(book_files)} book(s) to process.')
    print(f'Target chunk size: {target_tokens:,} tokens')
    print(f'Acceptable range:  [{int(target_tokens * 0.80):,}, {int(target_tokens * 1.05):,}]')
    print(f'Output directory:  {output_dir.resolve()}')
    print()

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for idx, fp in enumerate(book_files, start=1):
        book_name = fp.stem  # e.g. "pg10007-clean"
        print(f'[{idx}/{len(book_files)}] Processing: {book_name} ...', end=' ', flush=True)

        try:
            text = read_book(str(fp))
            chunks = split_book(text, target_tokens)
            result = write_chunks_and_report(book_name, str(fp), chunks, target_tokens, str(output_dir))
            all_results.append(result)
            print(
                f'→ {result["num_chunks"]} chunk(s), '
                f'total {result["total_tokens"]:,} tokens, '
                f'range [{result["min_chunk"]:,}, {result["max_chunk"]:,}]'
            )
        except Exception as e:
            print(f'ERROR: {e}')

    # Write global report
    global_report = generate_global_report(all_results, target_tokens, str(output_dir))
    global_report_path = output_dir / 'GLOBAL_REPORT.txt'
    with open(global_report_path, 'w', encoding='utf-8') as f:
        f.write(global_report)
    print(f'\nGlobal report written to: {global_report_path.resolve()}')

    # Also print a quick summary
    if all_results:
        total_chunks = sum(r['num_chunks'] for r in all_results)
        total_tokens = sum(r['total_tokens'] for r in all_results)
        print(f'\nDone. {len(all_results)} books → {total_chunks} chunks, {total_tokens:,} total tokens.')


if __name__ == '__main__':
    main()

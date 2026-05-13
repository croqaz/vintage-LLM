"""
Generate a Markdown report with PNG charts from the Lance dataset.

All per-document metrics (length, words, sentences, unique_chars,
quality_score, compression_ratio, char_entropy) are read directly from the
Lance dataset — nothing is re-computed.  Vocabulary / character frequency
analysis optionally scans the ``text`` column.
"""

import argparse
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import lance
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

CHAR_CAP_PER_DOC = 100_000
WORD_CAP_PER_DOC = 10_000

_WORD_RE = re.compile(r'\w+')

matplotlib.use('Agg')


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------


def _save_hist(values, title, xlabel, filepath, bins=80, log_y=False):
    fig, ax = plt.subplots(figsize=(10, 4))
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
    else:
        ax.hist(arr, bins=bins, color='steelblue', edgecolor='black', linewidth=0.3, alpha=0.85)
    if log_y:
        ax.set_yscale('log')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', alpha=0.4)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def _save_bar(labels, values, title, xlabel, filepath, horizontal=True):
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.3)))
    y = np.arange(len(labels))
    if horizontal:
        ax.barh(y, values, color='steelblue', edgecolor='black', linewidth=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
    else:
        ax.bar(y, values, color='steelblue', edgecolor='black', linewidth=0.3)
        ax.set_xticks(y)
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
        ax.set_ylabel(xlabel)
    ax.set_title(title)
    ax.grid(axis='x' if horizontal else 'y', alpha=0.4)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def _save_scatter(x, y, title, xlabel, ylabel, filepath, alpha=0.25):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, s=4, alpha=alpha, color='steelblue', edgecolors='none')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def _save_line(x, y, title, xlabel, ylabel, filepath):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, color='steelblue', linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def _summary_row(arr):
    a = np.asarray(arr, dtype=float)
    if len(a) == 0:
        return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'p5': 0, 'p95': 0}
    return {
        'mean': float(np.mean(a)),
        'median': float(np.median(a)),
        'std': float(np.std(a)),
        'min': float(np.min(a)),
        'max': float(np.max(a)),
        'p5': float(np.percentile(a, 5)),
        'p95': float(np.percentile(a, 95)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.db_path):
        raise SystemExit(f'Dataset not found: {args.db_path}')

    ds = lance.dataset(args.db_path)
    total_rows = ds.count_rows()
    if total_rows == 0:
        raise SystemExit('Dataset is empty.')

    out: Path = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bins = args.bins

    limit = args.limit or total_rows

    # ── Load metric columns from Lance ────────────────────────────
    metric_cols = ['length', 'words', 'sentences', 'unique_chars', 'quality_score', 'compression_ratio', 'char_entropy']
    print(f'Reading {min(limit, total_rows):,} / {total_rows:,} rows from {args.db_path} ...')
    tbl = ds.to_table(columns=metric_cols, limit=limit)

    metrics: dict[str, np.ndarray] = {}
    for col in metric_cols:
        metrics[col] = tbl.column(col).to_numpy().astype(float)

    total_docs = len(metrics['length'])
    total_chars = int(metrics['length'].sum())

    # Derived metrics
    with np.errstate(divide='ignore', invalid='ignore'):
        metrics['avg_sent_len'] = np.where(metrics['sentences'] > 0, metrics['words'] / metrics['sentences'], 0.0)

    print(f'Loaded {total_docs:,} documents  |  {total_chars:,} chars')

    # ── Vocabulary analysis (scans text column) ───────────────────
    global_char_counter: Counter = Counter()
    global_word_counter: Counter = Counter()
    heaps_total: list[int] = []
    heaps_unique: list[int] = []
    running_vocab: set[str] = set()
    running_total_words = 0

    if not args.skip_vocab:
        print('Scanning text column for vocabulary analysis ...')
        scanner = ds.scanner(columns=['text'], limit=limit)
        for batch in tqdm(scanner.to_batches(), desc='Vocabulary scan', unit=' batch'):
            for text in batch.column('text'):
                text = text.as_py()
                if not text:
                    continue
                global_char_counter.update(text[:CHAR_CAP_PER_DOC])
                words = _WORD_RE.findall(text[:CHAR_CAP_PER_DOC])
                lower_words = [w.lower() for w in words[:WORD_CAP_PER_DOC]]
                global_word_counter.update(lower_words)
                for w in lower_words:
                    running_total_words += 1
                    running_vocab.add(w)
                heaps_total.append(running_total_words)
                heaps_unique.append(len(running_vocab))

    # ── Generate charts ───────────────────────────────────────────
    print('Generating charts ...')
    charts: list[tuple[str, str, str]] = []

    def hist(key, title, xlabel, fname, log_y=False):
        fp = out / fname
        _save_hist(metrics[key], title, xlabel, fp, bins=bins, log_y=log_y)
        return fname

    # Lengths
    charts.append(
        (
            'Lengths',
            'Character length distribution',
            hist('length', 'Character Length Distribution', 'Characters', 'length_chars_hist.png', log_y=True),
        )
    )
    charts.append(
        ('Lengths', 'Word count distribution', hist('words', 'Word Count Distribution', 'Words', 'length_words_hist.png', log_y=True))
    )
    charts.append(
        (
            'Lengths',
            'Sentence count distribution',
            hist('sentences', 'Sentence Count Distribution', 'Sentences', 'length_sentences_hist.png', log_y=True),
        )
    )

    # Character composition
    charts.append(
        (
            'Character Composition',
            'Unique characters per document',
            hist('unique_chars', 'Unique Characters per Document', 'Unique chars', 'unique_chars_hist.png'),
        )
    )

    # Word-level
    charts.append(
        (
            'Word Statistics',
            'Average sentence length',
            hist('avg_sent_len', 'Average Sentence Length', 'Words/sentence', 'avg_sentence_length_hist.png'),
        )
    )

    # Quality scores
    charts.append(
        ('Quality Scores', 'Quality score', hist('quality_score', 'Quality Score Distribution', 'Score', 'quality_score_hist.png'))
    )
    charts.append(
        (
            'Quality Scores',
            'Compression ratio',
            hist('compression_ratio', 'Compression Ratio Distribution', 'Ratio', 'compression_ratio_hist.png'),
        )
    )
    charts.append(
        ('Quality Scores', 'Character entropy', hist('char_entropy', 'Character Entropy Distribution', 'Entropy', 'char_entropy_hist.png'))
    )

    # Vocabulary charts (only if text was scanned)
    if not args.skip_vocab:
        # Top characters bar
        top_chars = global_char_counter.most_common(50)
        if top_chars:
            labels = [repr(c)[1:-1] for c, _ in top_chars]
            vals = [v for _, v in top_chars]
            fname_tc = 'top_chars_bar.png'
            _save_bar(labels, vals, 'Top 50 Characters', 'Count', out / fname_tc)
            charts.append(('Vocabulary', 'Top 50 characters', fname_tc))

        # Top words bar
        top_words = global_word_counter.most_common(50)
        if top_words:
            labels = [w for w, _ in top_words]
            vals = [v for _, v in top_words]
            fname_tw = 'top_words_bar.png'
            _save_bar(labels, vals, 'Top 50 Words', 'Count', out / fname_tw)
            charts.append(('Vocabulary', 'Top 50 words', fname_tw))

        # Zipf's law
        if len(global_word_counter) > 10:
            word_counts_sorted = sorted(global_word_counter.values(), reverse=True)
            ranks = np.arange(1, len(word_counts_sorted) + 1, dtype=float)
            freqs = np.array(word_counts_sorted, dtype=float)
            fname_z = 'zipf_loglog.png'
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.loglog(ranks, freqs, '.', markersize=2, color='steelblue', alpha=0.6)
            log_r = np.log10(ranks)
            log_f = np.log10(freqs)
            slope, intercept = np.polyfit(log_r, log_f, 1)
            ax.loglog(ranks, 10 ** (intercept + slope * log_r), '-', color='tomato', linewidth=1.5, label=f'slope = {slope:.2f}')
            ax.set_title("Zipf's Law (log-log word frequency vs rank)")
            ax.set_xlabel('Rank')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(alpha=0.4)
            fig.tight_layout()
            fig.savefig(out / fname_z, dpi=150)
            plt.close(fig)
            charts.append(('Vocabulary', "Zipf's law (log-log)", fname_z))

        # Heaps' law
        if len(heaps_total) > 1:
            fname_h = 'heaps_curve.png'
            _save_line(heaps_total, heaps_unique, "Heaps' Law (vocabulary growth)", 'Total words seen', 'Unique words', out / fname_h)
            charts.append(('Vocabulary', "Heaps' law (vocabulary growth)", fname_h))
    else:
        top_words = []
        top_chars = []

    # ── Build summary statistics ──────────────────────────────────
    summary_keys = [
        ('length', 'Char length'),
        ('words', 'Word count'),
        ('sentences', 'Sentence count'),
        ('unique_chars', 'Unique chars'),
        ('avg_sent_len', 'Avg sentence length'),
        ('quality_score', 'Quality score'),
        ('compression_ratio', 'Compression ratio'),
        ('char_entropy', 'Char entropy'),
    ]
    summary_rows: list[tuple[str, dict]] = []
    for key, label in summary_keys:
        summary_rows.append((label, _summary_row(metrics[key])))

    # ── Write report.md ───────────────────────────────────────────
    print('Writing report ...')
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    vocab_size = len(global_word_counter)
    hapax = sum(1 for v in global_word_counter.values() if v == 1)

    lines: list[str] = []
    lines.append('# Dataset Analysis Report\n')
    lines.append(f'- **Generated**: {ts}')
    lines.append(f'- **Source**: `{args.db_path}`')
    lines.append(f'- **Documents**: {total_docs:,}')
    lines.append(f'- **Total characters**: {total_chars:,}')
    if args.limit:
        lines.append(f'- **Max docs cap**: {args.limit:,}')
    if vocab_size:
        lines.append(f'- **Vocabulary size** (sampled): {vocab_size:,}')
        lines.append(f'- **Hapax legomena**: {hapax:,} ({hapax / vocab_size * 100:.1f}% of vocabulary)')
    lines.append('')

    # Summary table
    lines.append('## Summary Statistics\n')
    lines.append('| Metric | Mean | Median | Std | Min | P5 | P95 | Max |')
    lines.append('|--------|-----:|-------:|----:|----:|---:|----:|----:|')
    for label, row in summary_rows:

        def _fmt(v):
            if abs(v) >= 100:
                return f'{v:,.0f}'
            return f'{v:.4f}'

        lines.append(
            f'| {label} | {_fmt(row["mean"])} | {_fmt(row["median"])} | {_fmt(row["std"])} '
            f'| {_fmt(row["min"])} | {_fmt(row["p5"])} | {_fmt(row["p95"])} | {_fmt(row["max"])} |'
        )
    lines.append('')

    # Chart sections
    current_section = None
    for section, title, fname in charts:
        if section != current_section:
            lines.append(f'## {section}\n')
            current_section = section
        lines.append(f'### {title}\n')
        lines.append(f'![{title}]({fname})\n')

    # Top words table
    if top_words:
        lines.append('### Top 30 Words\n')
        lines.append('| Rank | Word | Count |')
        lines.append('|-----:|------|------:|')
        for i, (w, c) in enumerate(top_words[:30], 1):
            lines.append(f'| {i} | {w} | {c:,} |')
        lines.append('')

    # Top chars table
    if top_chars:
        lines.append('### Top 30 Characters\n')
        lines.append('| Rank | Char | Count |')
        lines.append('|-----:|------|------:|')
        for i, (ch, c) in enumerate(top_chars[:30], 1):
            lines.append(f'| {i} | `{repr(ch)[1:-1]}` | {c:,} |')
        lines.append('')

    # Notes
    lines.append('## Notes & Caveats\n')
    lines.append(f'- Character counter capped at first {CHAR_CAP_PER_DOC:,} chars per document.')
    lines.append(f'- Word counter capped at first {WORD_CAP_PER_DOC:,} words per document.')
    if args.limit:
        lines.append(f'- Processing was capped at {args.limit:,} documents.')
    if args.skip_vocab:
        lines.append('- Vocabulary analysis was skipped (--skip-vocab).')
    lines.append('')

    report_path = out / 'report.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Report written to {report_path}')
    print(f'Charts saved in {out}/')

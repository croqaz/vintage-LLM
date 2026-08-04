#!/usr/bin/env python3
"""
run_pipeline.py — drive the whole split-then-generate-QA pipeline for every
book in the `books/` folder with one command, using an LLM-generated manifest
so the book title and author never have to be typed by hand.

For every book in the manifest:
  1. split it into overlapping word shards at every requested shard size
     (shard_book.py) — chunks are reused across repeats,
  2. run gen_qa_pairs.py over all of the book's chunk files in one call, so
     every shard is processed for every (model × prompt) combination,
  3. repeat step 2 `--repeats` times, each repeat writing to its own
     `{slug}-ft{N}.jsonl` file, so repeated runs yield more diverse samples.

Examples:
    python run_pipeline.py --dry-run
    python run_pipeline.py --shard-sizes "5000,4000,3000" --repeats 3
    OPENROUTER_API_KEY=... python run_pipeline.py \
        --models "openai/gpt-oss-20b,openai/gpt-oss-120b" \
        --prompts "prompts/_prompt_facts.txt,prompts/_prompt_prose.txt" \
        --books "Pustules" --repeats 2
"""

import argparse
import json
import os
import re
import subprocess
import sys
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
SHARD_SCRIPT = os.path.join(HERE, 'shard_book.py')
QA_SCRIPT = os.path.join(HERE, 'gen_qa_pairs.py')
DEFAULT_MANIFEST = os.path.join(HERE, 'books_manifest.json')
DEFAULT_MODELS = (
    'google/gemma-3-12b-it, mistralai/mistral-nemo, mistralai/mistral-small-24b-instruct-2501, openai/gpt-oss-20b, openai/gpt-oss-120b'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def split_list_arg(s, lower=False):
    """Split a string on commas, semicolons, or spaces (same as gen_qa_pairs)."""
    items = [x.strip() for x in re.split(r'[,; ]', s) if x.strip()]
    return [x.lower() for x in items] if lower else items


def slugify(text):
    return re.sub(r'[^A-Za-z0-9]+', '-', text).strip('-')


def load_manifest(path):
    """Load the LLM-generated manifest; resolve book paths relative to it."""
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    books = data.get('books', []) if isinstance(data, dict) else data
    manifest_dir = os.path.dirname(os.path.abspath(path))
    out = []
    for b in books:
        if not isinstance(b, dict) or not b.get('file'):
            print(f'  WARNING: skipping malformed manifest entry: {b}', file=sys.stderr)
            continue
        fp = b['file']
        if not os.path.isabs(fp):
            fp = os.path.join(manifest_dir, fp)
        if not os.path.exists(fp):
            print(f'  WARNING: book file not found, skipping: {fp}', file=sys.stderr)
            continue
        stem = os.path.splitext(os.path.basename(fp))[0]
        title = b.get('title') or stem
        out.append(
            {
                'path': os.path.abspath(fp),
                'file': b['file'],
                'title': title,
                'author': b.get('author', ''),
                'slug': b.get('slug') or slugify(title),
            }
        )
    return out


def chunk_file_for(book_path, size):
    """Chunk file for (book, size); returns (path, needs_split).

    Prefers the namespaced `{stem}_chunks_s{size}.json`.  Also reuses a
    legacy `{stem}_chunks.json` if it was made at the same shard size
    (checked via its embedded shard_size_words metadata).
    """
    stem, _ = os.path.splitext(book_path)
    target = f'{stem}_chunks_s{size}.json'
    if os.path.exists(target):
        return target, False
    legacy = f'{stem}_chunks.json'
    if os.path.exists(legacy):
        try:
            with open(legacy, 'r', encoding='utf-8') as fh:
                meta = json.load(fh)
            if meta.get('shard_size_words') == size:
                return legacy, False
        except (json.JSONDecodeError, OSError):
            pass
    return target, True


def run_cmd(cmd, cwd, dry_run=False):
    """Print and (unless dry-run) run a subprocess command. Returns rc (0 ok)."""
    print(f'\n  $ {subprocess.list2cmdline(cmd)}')
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=cwd)
    return proc.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='One-command split + Q&A generation pipeline over an LLM-generated book manifest.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST, help='LLM-generated manifest (title/author/slug per book)')
    parser.add_argument(
        '--shard-sizes', default='5000,4000,3000', help='Word sizes to split at, comma/space separated (default: %(default)s)'
    )
    parser.add_argument(
        '--models', default=DEFAULT_MODELS, help='Model id(s), comma/space separated — every shard is processed by every model'
    )
    parser.add_argument(
        '--prompts',
        default=None,
        help='Prompt file(s), comma/space separated — every shard is processed by every prompt (default: all prompts/_prompt_*.txt)',
    )
    parser.add_argument(
        '--repeats', type=int, default=1, help='How many times to re-run generation per book; each repeat writes its own {slug}-ft{N}.jsonl'
    )
    parser.add_argument('--output-dir', default='output', help='Directory for the generated ft jsonl files')
    parser.add_argument('--books', default=None, help='Only process these books (matches slug or filename substring, case-insensitive)')
    parser.add_argument('--skip-books', default=None, help='Skip these books (same matching)')
    parser.add_argument('--dry-run', action='store_true', help='Print the full plan without running anything')
    parser.add_argument('--force-shard', action='store_true', help='Re-split books even if chunk files already exist')
    parser.add_argument('--force-qa', action='store_true', help='Regenerate ft files even if they already exist')
    parser.add_argument(
        '--resume-qa',
        action='store_true',
        help='Run gen_qa_pairs even if the ft file exists — it skips (shard×model×prompt) combos already present',
    )
    parser.add_argument('--skip-shard', action='store_true', help='Only generate QA (chunk files must already exist)')
    parser.add_argument('--skip-qa', action='store_true', help='Only split books, do not generate QA')
    # ---- gen_qa_pairs passthrough options ---------------------------------
    parser.add_argument('--endpoint', default=None, help='Chat completions endpoint')
    parser.add_argument('--api-key', default=None, help='API key (defaults to OPENROUTER_API_KEY)')
    parser.add_argument('--max-tokens', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--retries', type=int, default=None)
    parser.add_argument('--concurrency', type=int, default=None)
    parser.add_argument('--delay', type=float, default=None)
    parser.add_argument('--allow-mistakes', action='store_true')
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error('--repeats must be >= 1')

    # -------------------------------------------------------------------
    # Resolve sizes / models / prompts
    # -------------------------------------------------------------------
    sizes = sorted({int(x) for x in split_list_arg(args.shard_sizes) if x.isdigit()})
    if not sizes:
        parser.error(f'no valid shard sizes in: {args.shard_sizes!r}')

    models = split_list_arg(args.models, lower=True)
    if not models:
        parser.error('at least one --models id is required')

    if args.prompts:
        prompt_arg = split_list_arg(args.prompts)
    else:
        prompt_arg = sorted(glob(os.path.join(HERE, 'prompts', '_prompt_*.txt')))
    resolved_prompts = []
    for p in prompt_arg:
        if not os.path.isabs(p):
            p = os.path.join(HERE, p)
        p = os.path.abspath(p)
        if not os.path.exists(p):
            print(f'ERROR: prompt file not found: {p}', file=sys.stderr)
            sys.exit(1)
        resolved_prompts.append(p)
    if not resolved_prompts:
        parser.error('no prompt files found — pass --prompts or add prompts/_prompt_*.txt')

    # -------------------------------------------------------------------
    # Load & filter books from the manifest
    # -------------------------------------------------------------------
    books = load_manifest(args.manifest)
    if not books:
        print(f'ERROR: no usable books in manifest {args.manifest}', file=sys.stderr)
        sys.exit(1)

    only = [x.lower() for x in split_list_arg(args.books)] if args.books else None
    skip = [x.lower() for x in split_list_arg(args.skip_books)] if args.skip_books else []

    def matches(b, tokens):
        hay = f'{b["slug"]} {b["file"]}'.lower()
        return any(t in hay for t in tokens)

    if only:
        books = [b for b in books if matches(b, only)]
    books = [b for b in books if not matches(b, skip)]
    if not books:
        print('ERROR: the --books/--skip-books filters left no books to process', file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------
    # Build the plan
    # -------------------------------------------------------------------
    output_dir = os.path.abspath(args.output_dir)
    entries = []
    total_calls = 0
    for b in books:
        chunks = []
        for size in sizes:
            path, missing = chunk_file_for(b['path'], size)
            need_split = not args.skip_shard and (args.force_shard or missing)
            chunks.append({'size': size, 'path': path, 'need_split': need_split})
        repeats = []
        for r in range(1, args.repeats + 1):
            out = os.path.join(output_dir, f'{b["slug"]}-ft{r}.jsonl')
            if args.skip_qa:
                do_qa = False
            elif args.force_qa or args.resume_qa:
                do_qa = True
            else:
                do_qa = not os.path.exists(out)
            repeats.append({'repeat': r, 'output': out, 'do_qa': do_qa})
        total_calls += len(chunks) * len(repeats) * len(models) * len(resolved_prompts)
        entries.append({'book': b, 'chunks': chunks, 'repeats': repeats})

    # -------------------------------------------------------------------
    # Print the plan
    # -------------------------------------------------------------------
    print('=' * 78)
    print('QA PIPELINE PLAN')
    print('=' * 78)
    print(f'Manifest:        {args.manifest}')
    print(f'Books:           {len(books)}')
    for b in books:
        print(f'  - {b["slug"]}  ({os.path.basename(b["path"])})')
        print(f'      title:  {b["title"]}')
        print(f'      author: {b["author"] or "(none)"}')
    print(f'Shard sizes:     {", ".join(str(s) for s in sizes)} words')
    print(f'Models:          {", ".join(models)}')
    print(f'Prompts:         {", ".join(os.path.basename(p) for p in resolved_prompts)}')
    print(f'Repeats:         {args.repeats}  (one ft file per repeat)')
    print(f'Output dir:      {output_dir}')
    print(
        f'API calls:       up to {total_calls} '
        f'({len(books)} books × {len(sizes)} sizes × {args.repeats} repeats × '
        f'{len(models)} models × {len(resolved_prompts)} prompts)'
    )
    print()

    for e in entries:
        b = e['book']
        print(f'--- {b["slug"]} ---')
        for ch in e['chunks']:
            state = 'SPLIT' if ch['need_split'] else 'use existing'
            print(f'  shards @ {ch["size"]}w  -> {os.path.basename(ch["path"])}  [{state}]')
        for rep in e['repeats']:
            if rep['do_qa']:
                why = 'force' if args.force_qa else 'resume' if args.resume_qa else 'new file' if not os.path.exists(rep['output']) else ''
                print(f'  repeat {rep["repeat"]}  -> {os.path.basename(rep["output"])}  [generate{"; " + why if why else ""}]')
            else:
                print(f'  repeat {rep["repeat"]}  -> {os.path.basename(rep["output"])}  [skip: exists; use --force-qa / --resume-qa]')
    print()

    if args.dry_run:
        print('DRY RUN — no commands executed. Re-run without --dry-run to execute.')
        return 0

    # -------------------------------------------------------------------
    # Execute
    # -------------------------------------------------------------------
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    passthrough = []
    if args.endpoint:
        passthrough += ['--endpoint', args.endpoint]
    if args.api_key:
        passthrough += ['--api-key', args.api_key]
    for flag, val in (
        ('--max-tokens', args.max_tokens),
        ('--temperature', args.temperature),
        ('--timeout', args.timeout),
        ('--retries', args.retries),
        ('--concurrency', args.concurrency),
        ('--delay', args.delay),
    ):
        if val is not None:
            passthrough += [flag, str(val)]
    if args.allow_mistakes:
        passthrough.append('--allow-mistakes')

    failures = 0
    for e in entries:
        b = e['book']
        print(f'\n{">" * 3} BOOK: {b["slug"]}  ({b["title"]} — {b["author"] or "unknown"})')

        # ---- 1. shard this book at every requested size --------------------
        for ch in e['chunks']:
            if not ch['need_split']:
                print(f'  chunk file already present, skipping split: {os.path.basename(ch["path"])}')
                continue
            cmd = [sys.executable, SHARD_SCRIPT, b['path'], '-o', ch['path'], '-s', str(ch['size'])]
            if run_cmd(cmd, cwd=HERE):
                failures += 1

        # ---- 2. generate QA for every repeat -------------------------------
        chunk_paths = [c['path'] for c in e['chunks']]
        if not chunk_paths:
            print('  WARNING: no chunk files for this book — skipping QA (run without --skip-shard or split it first)', file=sys.stderr)
            failures += 1
            continue
        for rep in e['repeats']:
            if not rep['do_qa']:
                print(
                    f'  ft file already present, skipping: '
                    f'{os.path.basename(rep["output"])} '
                    f'(use --force-qa to regenerate, --resume-qa to continue)'
                )
                continue
            # --force-qa means regenerate from scratch: drop the old file so
            # gen_qa_pairs' resume logic doesn't skip every combination.
            if args.force_qa and os.path.exists(rep['output']):
                print(f'  --force-qa: removing {os.path.basename(rep["output"])} before regeneration')
                os.remove(rep['output'])
            cmd = [sys.executable, QA_SCRIPT] + chunk_paths
            cmd += [
                '-o',
                rep['output'],
                '-m',
                args.models,
                '--prompt',
                ','.join(resolved_prompts),
                '--book-title',
                b['title'],
                '--book-author',
                b['author'],
            ]
            cmd += passthrough
            if run_cmd(cmd, cwd=HERE):
                failures += 1

    print()
    print('=' * 78)
    if failures:
        print(f'PIPELINE FINISHED WITH {failures} FAILED STEP(S)')
        print(f'Outputs: {output_dir}')
        return 1
    print('PIPELINE COMPLETE — all steps succeeded.')
    print(f'Outputs: {output_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

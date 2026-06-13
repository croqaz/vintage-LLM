#!/usr/bin/env python3
"""
Generate fine-tuning Q&A pairs for Bible verses.

Uses bible_lookup.py to resolve each reference in the VERSES list to its
actual text, then pairs each verse with a randomly chosen question template
to produce JSON-formatted training data.

Usage:
    python generate_verse.py                    # print JSON to stdout
    python generate_verse.py -o bible_qa.json   # write to file
    python generate_verse.py --seed 123         # control random template choice
"""

import argparse
import json
import random
import re
from pathlib import Path

import bible_lookup

# ---------------------------------------------------------------------------
# Verse references to include in the training set
# ---------------------------------------------------------------------------

VERSES = [
    '1 Corinthians 13:13',
    '1 Corinthians 13:4-7',
    '1 Corinthians 16:14',
    '1 John 4:19',
    '1 Peter 4:8',
    '1 Thessalonians 5:11',
    'Colossians 2:7',
    'Deuteronomy 6:5',
    'Ecclesiastes 3:1',
    'Ecclesiastes 3:11',
    'Ephesians 3:20',
    'Exodus 14:14',
    'Exodus 20:12',
    'Galatians 5:22-23',
    'Genesis 1:1',
    'Hosea 10:12',
    'Isaiah 26:3',
    'Isaiah 40:29',
    'Isaiah 40:31',
    'Isaiah 41:10',
    'Isaiah 43:2',
    'Isaiah 43:4',
    'Isaiah 54:17',
    'Isaiah 61:1',
    'James 1:17',
    'Jeremiah 17:9',
    'Jeremiah 29:11',
    'Jeremiah 29:13',
    'John 14:1',
    'John 14:6',
    'John 15:7',
    'John 16:33',
    'John 3:16',
    'John 8:36',
    'Joshua 1:9',
    'Joshua 24:15',
    'Luke 16:10',
    'Luke 18:27',
    'Luke 1:37',
    'Luke 6:31',
    'Mark 9:23',
    'Matthew 6:33',
    'Matthew 11:28',
    'Matthew 11:29',
    'Matthew 18:20',
    'Matthew 18:21-22',
    'Matthew 19:6',
    'Matthew 23:11',
    'Matthew 28:19',
    'Matthew 28:6',
    'Matthew 5:14',
    'Matthew 5:9',
    'Matthew 6:34',
    'Matthew 7:24',
    'Matthew 7:7',
    'Micah 6:8',
    'Nehemiah 8:10',
    'Numbers 6:24-25',
    'Philippians 4:13',
    'Philippians 4:13',
    'Philippians 4:13',
    'Philippians 4:6',
    'Philippians 4:6',
    'Philippians 4:7',
    'Proverbs 12:15',
    'Proverbs 13:12',
    'Proverbs 14:30',
    'Proverbs 17:17',
    'Proverbs 17:22',
    'Proverbs 18:24',
    'Proverbs 27:17',
    'Proverbs 3:5',
    'Proverbs 3:5',
    'Proverbs 3:5',
    'Proverbs 3:6',
    'Proverbs 3:6',
    'Psalms 119:105',
    'Psalms 150:6',
    'Psalms 23:1',
    'Psalms 34:18',
    'Psalms 103:2',
    'Psalms 119:114',
    'Psalms 28:7',
    'Psalms 31:24',
    'Psalms 32:8',
    'Psalms 46:1',
    'Psalms 51:10',
    'Psalms 91:4',
    'Romans 12:16',
    'Romans 12:2',
    'Romans 15:13',
    'Romans 8:28',
    'Romans 8:31',
    'Romans 8:6',
]

# ---------------------------------------------------------------------------
# Question templates – [BOOK NAME] and [CHAPTER:VERSE] are filled at runtime
# ---------------------------------------------------------------------------

TEMPLATES = [
    'What saith the scripture in [BOOK NAME] [CHAPTER:VERSE]?',
    'What are the words written in [BOOK NAME] [CHAPTER:VERSE]?',
    'Pray, what is set down in [BOOK NAME] at verse [CHAPTER:VERSE]?',
    'Tell me the verse at [BOOK NAME] [CHAPTER:VERSE].',
    'Recite unto me the scripture of [BOOK NAME] [CHAPTER:VERSE].',
    'What words doth the Holy Writ give us at [BOOK NAME] [CHAPTER:VERSE]?',
    'Read unto me [BOOK NAME] [CHAPTER:VERSE], I pray thee.',
    'What doth the Lord speak in [BOOK NAME] [CHAPTER:VERSE]?',
]

# ---------------------------------------------------------------------------
# Reference parser
# ---------------------------------------------------------------------------

REFERENCE_RE = re.compile(r'^(.+?)\s+(\d+:\d+(?:-\d+)?)$')


def parse_reference(reference: str) -> tuple[str, str]:
    """Split '1 Corinthians 13:13' into ('1 Corinthians', '13:13')."""
    m = REFERENCE_RE.match(reference.strip())
    if not m:
        raise ValueError(f"Could not parse reference: '{reference}'")
    return m.group(1).strip(), m.group(2).strip()


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------


def generate_pairs(
    verses: list[str],
    templates: list[str],
    full_text: str,
) -> list[list[dict[str, str]]]:
    """Return a list of [[user_msg, assistant_msg], ...] ready for JSON export."""
    pairs: list[list[dict[str, str]]] = []

    for ref in verses:
        book_name, verse_spec = parse_reference(ref)
        print(f"Processing {book_name} {verse_spec}...")

        # Pick a random template and fill the placeholders
        template = random.choice(templates)
        question = template.replace('[BOOK NAME]', book_name).replace('[CHAPTER:VERSE]', verse_spec)
        print(f"  Question: {question}")

        # Resolve the actual verse text via bible_lookup
        answer = bible_lookup.lookup(full_text, ref)

        pairs.append(
            [
                {'role': 'user', 'content': question},
                {'role': 'assistant', 'content': answer},
            ]
        )

    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate Bible verse Q&A pairs for fine-tuning.')
    parser.add_argument(
        '--output',
        '-o',
        default=None,
        help='Output JSON file (default: print to stdout)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for template selection (default: 42)',
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Load the Bible text once
    full_text = bible_lookup.BIBLE_PATH.read_text(encoding='utf-8-sig')

    # Generate all Q&A pairs
    pairs = generate_pairs(VERSES, TEMPLATES, full_text)

    # Serialise to JSON
    output = json.dumps(pairs, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(output, encoding='utf-8')
        print(f'Generated {len(pairs)} Q&A pairs → {args.output}')
    else:
        print(output)


if __name__ == '__main__':
    main()

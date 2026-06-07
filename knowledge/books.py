"""
Conversations about authors and the books they wrote, built from ``authors.json``.

Design notes
------------
* The model is **not** meant to memorise exact dates or the precise number of books an
  author wrote (the catalogue is noisy and incomplete). So dates are usually phrased
  loosely ("around 1812", "the early 19th century") and book lists are explicitly framed
  as "a few" / "some" titles rather than an exhaustive bibliography.
* Centuries are derived from an author's **lifespan** (``birth``..``death``), not from book
  publication years -- the catalogue is full of posthumous reprints and translations whose
  years would wrongly place, say, Shakespeare in the 19th century.
* Every question (and often the answer) has several interchangeable phrasings so the same
  underlying fact yields many augmented training samples.

The module is reproducible: it draws from a private ``random.Random`` instance so importing
it does not perturb the global RNG state that the other modules and the shuffler rely on.
"""

import json
import os
import re
from random import Random

SEED = 0xB00C  # fixed so the generated corpus is reproducible across builds
_rng = Random(SEED)

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, 'authors.json'), encoding='utf-8') as _f:
    AUTHORS = json.load(_f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ordinal(n: int) -> str:
    """1 -> '1st', 16 -> '16th', 22 -> '22nd'."""
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


def _century_of(year: int) -> int:
    return (year // 100) + 1


def _life_centuries(author: dict) -> list[int]:
    """Centuries an author actually lived through, from birth to death."""
    return list(range(_century_of(author['birth']), _century_of(author['death']) + 1))


def _approx_birth(birth: int) -> str:
    """A loose, year-level phrasing of a birth year, for date-tolerant answers.

    Deliberately avoids naming the century -- callers that mention the century
    themselves would otherwise produce redundant "the 16th century ... the 16th
    century" phrasing.
    """
    decade = (birth // 10) * 10
    return _rng.choice(
        [
            f'around {birth}',
            f'in {birth}',
            f'in the {decade}s',
            f'in the late {decade - 10}s or early {decade}s',
        ]
    )


def _century_part(birth: int) -> str:
    """'the early 19th century', 'the middle of the 16th century', ..."""
    century = _ordinal(_century_of(birth))
    within = birth % 100
    if within <= 15:
        part = 'the early'
    elif within <= 45:
        part = 'the first half of the'
    elif within <= 65:
        part = 'the middle of the'
    elif within <= 85:
        part = 'the latter half of the'
    else:
        part = 'the late'
    return f'{part} {century} century'


def _join_names(names: list[str]) -> str:
    """['A', 'B', 'C'] -> 'A, B, and C'."""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f'{names[0]} and {names[1]}'
    return ', '.join(names[:-1]) + f', and {names[-1]}'


# --- title cleaning --------------------------------------------------------
# The book lists are scraped from Wikidata and mix real works with reprints,
# foreign-language translations, single sonnets, fragments and catalogue
# artefacts. These heuristics keep the recognisable, standalone works so that
# "name a few books by X" yields sensible titles.

_TITLE_BLOCKLIST = re.compile(
    r"""
    ^Category:                 # wiki category pages
    | bibliography             # "X bibliography"
    | ^chronology\ of          # "chronology of ... plays"
    | \bVol\.?\ *[IVX0-9]      # multi-volume set entries
    | ^Sonnet\ \d              # individual numbered sonnets
    | \bCanto\b                # single cantos of a longer poem
    | /                        # sub-parts, e.g. ".../Stave 1"
    | \.\.\.                   # truncated titles
    | \bdirected\ by\b
    | ^Selected\b
    | ^The\ Works\ of\b
    | ^The\ Portable\b
    | ^The\ Complete\b
    | ^The\ Collected\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _looks_like_title(title: str) -> bool:
    if _TITLE_BLOCKLIST.search(title):
        return False
    # Fragments / quoted lines tend to start lower-case ("something wicked ...")
    if title[:1].islower():
        return False
    # Very long "titles" are usually descriptions, very short ones are fragments.
    if len(title) > 60 or len(title) < 3:
        return False
    return True


def _dedup_key(title: str) -> str:
    """Collapse near-duplicates: drop subtitles and normalise case/space."""
    base = title.split(':')[0].split(' - ')[0]
    return re.sub(r'\s+', ' ', base).strip().lower()


def _clean_titles(author: dict) -> list[str]:
    """A deduped list of recognisable works, preferring catalogued (dated) ones."""
    seen: set[str] = set()
    dated: list[str] = []
    undated: list[str] = []
    for book in author['books']:
        title = book['title']
        if not _looks_like_title(title):
            continue
        key = _dedup_key(title)
        if key in seen:
            continue
        seen.add(key)
        (dated if book.get('year') else undated).append(title)
    return dated + undated


def _any_title(author: dict) -> str | None:
    titles = _clean_titles(author)
    if not titles:
        # Fall back to the raw list for single-work authors whose lone title
        # happened to be filtered out.
        titles = [b['title'] for b in author['books']]
    return _rng.choice(titles) if titles else None


# Pre-compute per-author cleaned titles and a title -> authors index.
_CLEAN: dict[str, list[str]] = {a['author']: _clean_titles(a) for a in AUTHORS}
_TITLE_OWNERS: dict[str, set[str]] = {}
for _a in AUTHORS:
    for _t in _CLEAN[_a['author']]:
        _TITLE_OWNERS.setdefault(_dedup_key(_t), set()).add(_a['author'])

# Authors with enough recognisable works to ask "name a few books".
PROLIFIC = [a for a in AUTHORS if len(_CLEAN[a['author']]) >= 5]
# Centuries with enough authors to sample a list from.
_BY_CENTURY: dict[int, list[dict]] = {}
for _a in AUTHORS:
    for _c in _life_centuries(_a):
        _BY_CENTURY.setdefault(_c, []).append(_a)
LISTABLE_CENTURIES = sorted(c for c, v in _BY_CENTURY.items() if len(v) >= 12)


BOOKS: list = []


# ---------------------------------------------------------------------------
# 1. "Name N authors from the Nth century"
# ---------------------------------------------------------------------------


def _century_phrase(century: int) -> str:
    ord_ = _ordinal(century)
    return _rng.choice(
        [
            f'the {ord_} century',
            f'the {ord_} century',
            f'the {(century - 1) * 100}s',  # e.g. "the 1800s"
        ]
    )


def _author_list_question(count: int, century: int) -> str:
    phrase = _century_phrase(century)
    return _rng.choice(
        [
            f'Can you name {count} authors from {phrase}?',
            f'Tell me {count} writers who lived during {phrase}.',
            f"I'd like {count} authors from {phrase}. Who comes to mind?",
            f'Name {count} writers from {phrase}.',
            f'Give me {count} authors who were alive in {phrase}.',
        ]
    )


def _author_list_answer(names: list[str], century: int) -> str:
    joined = _join_names(names)
    intro = _rng.choice(
        [
            f'Sure! A few writers from that period were {joined}.',
            f'Of course. Some authors from the {_ordinal(century)} century include {joined}.',
            f'Certainly -- {joined} all lived around then.',
            f'Here are a few: {joined}.',
            f'Gladly. Names that come to mind are {joined}.',
        ]
    )
    return intro


for _century in LISTABLE_CENTURIES:
    pool = _BY_CENTURY[_century]
    for _count in (3, 5, 10):
        if len(pool) < _count:
            continue
        # A couple of variants per (century, count) for augmentation.
        for _ in range(2):
            sample = _rng.sample(pool, _count)
            names = [a['author'] for a in sample]
            BOOKS.append(
                {
                    'question': _author_list_question(_count, _century),
                    'answer': _author_list_answer(names, _century),
                }
            )


# A multi-turn variant that walks across adjacent centuries.
for _c in LISTABLE_CENTURIES:
    nxt = _c + 1
    if nxt not in _BY_CENTURY or len(_BY_CENTURY[nxt]) < 4:
        continue
    first = _rng.sample(_BY_CENTURY[_c], min(4, len(_BY_CENTURY[_c])))
    second = _rng.sample(_BY_CENTURY[nxt], min(4, len(_BY_CENTURY[nxt])))
    BOOKS.append(
        [
            {
                'question': _author_list_question(len(first), _c),
                'answer': _author_list_answer([a['author'] for a in first], _c),
            },
            {
                'question': _rng.choice(
                    [
                        f'And from the {_ordinal(nxt)} century?',
                        f'How about the {_ordinal(nxt)} century?',
                        f'Now name some from the {_ordinal(nxt)} century.',
                    ]
                ),
                'answer': _author_list_answer([a['author'] for a in second], nxt),
            },
        ]
    )


# ---------------------------------------------------------------------------
# 2. "Name a few books by <author>"  (only for prolific authors)
# ---------------------------------------------------------------------------


def _books_question(author: str, count_word: str) -> str:
    return _rng.choice(
        [
            f'Tell me {count_word} books written by {author}.',
            f'Can you name {count_word} works by {author}?',
            f'What are {count_word} books by {author}?',
            f"I'd love to read {author}. Could you suggest {count_word} of their books?",
            f'Name {count_word} titles {author} wrote.',
        ]
    )


def _books_answer(author: str, titles: list[str]) -> str:
    joined = _join_names([f'"{t}"' for t in titles])
    return _rng.choice(
        [
            f'Sure! A few titles that come to mind are {joined}.',
            f'Of course -- {author} wrote {joined}, among others.',
            f"Some of {author}'s works include {joined}.",
            f'Certainly. You might enjoy {joined}.',
            f'Happy to! Notable works include {joined}.',
        ]
    )


for _a in PROLIFIC:
    name = _a['author']
    titles = _CLEAN[name]
    # One or two prompts per author, with disjoint title samples for variety.
    n_prompts = 2 if len(titles) >= 8 else 1
    for _ in range(n_prompts):
        k = _rng.randint(3, 5)
        chosen = _rng.sample(titles, k)
        count_word = _rng.choice(['a few', 'some', f'{k}', 'a couple of' if k <= 3 else 'several'])
        BOOKS.append(
            {
                'question': _books_question(name, count_word),
                'answer': _books_answer(name, chosen),
            }
        )

    # A multi-turn "any others?" follow-up when there are plenty of titles.
    if len(titles) >= 8:
        pool = _rng.sample(titles, 6)
        first, more = pool[:3], pool[3:]
        BOOKS.append(
            [
                {
                    'question': _books_question(name, 'a few'),
                    'answer': _books_answer(name, first),
                },
                {
                    'question': _rng.choice(['Any others?', 'Can you name a few more?', 'What else did they write?']),
                    'answer': _rng.choice(
                        [
                            f"Yes -- there's also {_join_names([chr(34) + t + chr(34) for t in more])}.",
                            f'Certainly, {name} also wrote {_join_names([chr(34) + t + chr(34) for t in more])}.',
                        ]
                    ),
                },
            ]
        )


# ---------------------------------------------------------------------------
# 3. "Who wrote <book>?"  (only unambiguous, recognisable titles)
# ---------------------------------------------------------------------------

_unique_titles = [(a['author'], t) for a in AUTHORS for t in _CLEAN[a['author']] if len(_TITLE_OWNERS[_dedup_key(t)]) == 1]
_rng.shuffle(_unique_titles)

for _author, _title in _unique_titles[:200]:
    question = _rng.choice(
        [
            f'Who wrote "{_title}"?',
            f'Who is the author of "{_title}"?',
            f'Do you know who wrote "{_title}"?',
            f'"{_title}" -- who is that by?',
        ]
    )
    answer = _rng.choice(
        [
            f'That was written by {_author}.',
            f'"{_title}" is the work of {_author}.',
            f"That's by {_author}.",
            f'{_author} wrote "{_title}".',
        ]
    )
    BOOKS.append({'question': question, 'answer': answer})


# ---------------------------------------------------------------------------
# 4. When did an author live?  (date-tolerant phrasing)
# ---------------------------------------------------------------------------

for _a in AUTHORS:
    name = _a['author']
    birth = _a['birth']
    death = _a['death']
    century = _ordinal(_century_of(birth))

    question = _rng.choice(
        [
            f'When did {name} live?',
            f'Roughly when was {name} born?',
            f'What period did {name} belong to?',
            f'Around what time did {name} write?',
        ]
    )
    answer = _rng.choice(
        [
            f'{name} was born {_approx_birth(birth)} and died in {death}.',
            f'{name} lived during the {century} century, born {_approx_birth(birth)}.',
            f'{name} was active around the {century} century.',
            f'{name} was born {_approx_birth(birth)}.',
            f'{name} was born in {_century_part(birth)}.',
        ]
    )
    BOOKS.append({'question': question, 'answer': answer})


# ---------------------------------------------------------------------------
# 5. "What century did <author> live in?"
# ---------------------------------------------------------------------------

for _a in _rng.sample(AUTHORS, min(120, len(AUTHORS))):
    name = _a['author']
    centuries = _life_centuries(_a)
    if len(centuries) == 1:
        cent_str = f'the {_ordinal(centuries[0])} century'
    else:
        cent_str = f'the {_ordinal(centuries[0])} and {_ordinal(centuries[-1])} centuries'
    question = _rng.choice(
        [
            f'What century did {name} live in?',
            f'In which century did {name} live?',
            f'Which century was {name} from?',
        ]
    )
    answer = _rng.choice(
        [
            f'{name} lived during {cent_str}.',
            f'That would be {cent_str}.',
            f'{name} belongs to {cent_str}.',
        ]
    )
    BOOKS.append({'question': question, 'answer': answer})


# ---------------------------------------------------------------------------
# 6. "Was <author> alive in <year>?"  (yes / no, both directions)
# ---------------------------------------------------------------------------

for _a in _rng.sample(AUTHORS, min(90, len(AUTHORS))):
    name = _a['author']
    birth, death = _a['birth'], _a['death']
    if _rng.random() < 0.5:
        # A year during their life -> yes.
        year = _rng.randint(birth, death)
        answer = _rng.choice(
            [
                f'Yes, {name} was alive in {year}.',
                f'Yes -- {name} lived from {birth} to {death}, so {year} falls within their lifetime.',
            ]
        )
    else:
        # A year clearly outside their life -> no.
        if _rng.random() < 0.5:
            year = _rng.randint(birth - 120, birth - 20)
            answer = _rng.choice(
                [
                    f"No, {name} wasn't born until {birth}.",
                    f'No -- {name} was born in {birth}, after {year}.',
                ]
            )
        else:
            year = _rng.randint(death + 20, death + 120)
            answer = _rng.choice(
                [
                    f'No, {name} had already died by {year} (they passed away in {death}).',
                    f'No -- {name} died in {death}, before {year}.',
                ]
            )
    question = _rng.choice(
        [
            f'Was {name} alive in {year}?',
            f'Was {name} living in {year}?',
            f'Could {name} have been alive in {year}?',
        ]
    )
    BOOKS.append({'question': question, 'answer': answer})


# ---------------------------------------------------------------------------
# 7. "Name a book by <author>"  (great use for single-work authors)
# ---------------------------------------------------------------------------

for _a in AUTHORS:
    name = _a['author']
    title = _any_title(_a)
    if not title:
        continue
    question = _rng.choice(
        [
            f'Name a book by {name}.',
            f'Can you name something {name} wrote?',
            f"What's a work by {name}?",
            f'Tell me one book written by {name}.',
        ]
    )
    answer = _rng.choice(
        [
            f'{name} wrote "{title}".',
            f'One of their works is "{title}".',
            f'"{title}" is by {name}.',
            f'Sure -- "{title}".',
        ]
    )
    BOOKS.append({'question': question, 'answer': answer})


if __name__ == '__main__':
    # Quick preview / sanity check: `python books.py`
    multi = sum(1 for x in BOOKS if isinstance(x, list))
    print(f'{len(BOOKS)} items ({multi} multi-turn, {len(BOOKS) - multi} single-turn)')
    print(f'prolific authors: {len(PROLIFIC)} | listable centuries: {LISTABLE_CENTURIES}')
    print('-' * 70)
    for item in _rng.sample(BOOKS, 20):
        turns = item if isinstance(item, list) else [item]
        for qa in turns:
            print(f'Q: {qa["question"]}')
            print(f'A: {qa["answer"]}')
        print('-' * 70)

"""
Generate Q&A pairs from philosophical quotes in the quotes/ folder.

Produces three types of conversations:

1. **Multi-turn** — ask for a quote by an author, get a quote, then ask for
   another and get a second quote from the same author.
2. **Reverse** — given a quote, name the author who said it (every quote used).
3. **Cross-author** — questions that span multiple philosophers (lists,
   categories, random picks).
"""

import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
_QUOTES_DIR = os.path.join(_HERE, 'quotes')

QUOTES = []

# ---------------------------------------------------------------------------
# Parse all quote files
# ---------------------------------------------------------------------------


def _parse_quotes():
    """Parse every ``Quotes *.txt`` file and return ``{short_name: [quote, ...]}``.

    The short name is extracted from the filename: ``Quotes Kant.txt`` → ``Kant``.
    """
    authors = {}
    for filename in sorted(os.listdir(_QUOTES_DIR)):
        if not filename.endswith('.txt'):
            continue
        short_name = filename.replace('Quotes ', '').replace('.txt', '')
        filepath = os.path.join(_QUOTES_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Everything after the ``## Quotes`` marker
        parts = content.split('## Quotes')
        if len(parts) < 2:
            continue

        quotes = []
        for line in parts[1].strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                quotes.append(line)
        authors[short_name] = quotes
    return authors


_AUTHORS = _parse_quotes()
_ALL_AUTHORS = sorted(_AUTHORS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_author() -> str:
    return random.choice(_ALL_AUTHORS)


def _pick_quote(author: str) -> str:
    return random.choice(_AUTHORS[author])


def _two_different_quotes(author: str):
    """Return two *different* quotes by *author* (author must have ≥2 quotes)."""
    pool = _AUTHORS[author]
    if len(pool) < 2:
        raise ValueError(f'{author} has fewer than 2 quotes')
    return tuple(random.sample(pool, 2))


# ---------------------------------------------------------------------------
# Type 1 — multi-turn: "tell me a quote" → quote → "another?" → second quote
# ---------------------------------------------------------------------------

_PH_PROMPT = [
    'Please tell me a quote by {author}.',
    'Do you know any quotes by {author}?',
    'Can you share a quote from {author} with me?',
    "I'd love to hear a quote by {author}.",
    'Tell me something {author} once said.',
    'Share a quote from {author}, please.',
    "What's a good quote by {author}?",
    'Give me a quote by {author}.',
    'Recite a quote from {author}.',
    "I'm in the mood for a quote by {author}.",
]

_PH_FOLLOWUP = [
    'Oh, I loved that! Do you know another one?',
    'That was beautiful. Do you have another?',
    'Wonderful! Can you share one more?',
    'I like that. Tell me another one by the same author!',
    'That resonates with me. Do you know any other?',
    'Nice one! Give me one more from the same philosopher.',
    'I enjoyed that. Got another quote from {author}?',
    'Lovely. Do you have a second quote by {author}?',
    'Great quote! How about another one?',
    'That is profound. Can I hear one more?',
]

for author, quote_list in _AUTHORS.items():
    if len(quote_list) < 2:
        continue

    # Create at most 20 multi-turn conversations per author
    n_conversations = min(20, len(quote_list) // 2)
    shuffled = list(quote_list)
    random.shuffle(shuffled)

    for i in range(n_conversations):
        q1 = shuffled[i * 2]
        q2 = shuffled[i * 2 + 1]

        prompt = random.choice(_PH_PROMPT).format(author=author)
        followup = random.choice(_PH_FOLLOWUP).format(author=author)

        QUOTES.append(
            [
                {'question': prompt, 'answer': q1},
                {'question': followup, 'answer': q2},
            ]
        )


# ---------------------------------------------------------------------------
# Type 2 — reverse: "who said this quote?"  (every single quote is used here)
# ---------------------------------------------------------------------------

_PH_WHO = [
    'Who is the author of this quote: "{quote}"?',
    'Do you know who said this: "{quote}"?',
    'Which philosopher said: "{quote}"?',
    '"{quote}" — who wrote that?',
    'Who said "{quote}"?',
    'Can you tell me who is behind this quote: "{quote}"?',
    'To whom does this quote belong: "{quote}"?',
    'Identify the philosopher: "{quote}"',
]

_PH_ITS = [
    "It's {author}, I believe.",
    'That was {author}.',
    '{author} said that.',
    'I think that quote belongs to {author}.',
    'That is from {author}.',
    'That quote is by {author}.',
    'I believe {author} wrote that.',
    'If I recall correctly, {author} is the author.',
    '{author}, if I am not mistaken.',
    'That sounds like {author} to me.',
]

for author, quote_list in _AUTHORS.items():
    for quote in quote_list:
        q_text = random.choice(_PH_WHO).format(quote=quote)
        a_text = random.choice(_PH_ITS).format(author=author)
        QUOTES.append({'question': q_text, 'answer': a_text})


# ---------------------------------------------------------------------------
# Type 3 — cross-author / "all authors at the end" questions
# ---------------------------------------------------------------------------

# --- "What do X, Y, and Z have in common?" ---
for _ in range(5):
    sample = random.sample(_ALL_AUTHORS, min(3, len(_ALL_AUTHORS)))
    names = ', '.join(sample[:-1]) + ', and ' + sample[-1]
    QUOTES.append(
        {
            'question': f'What do {names} have in common?',
            'answer': 'They are all philosophers.',
        }
    )
    QUOTES.append(
        {
            'question': f'What do all {names} share?',
            'answer': 'They are all well-known philosophers.',
        }
    )

# --- "Is X a philosopher?" ---
_PH_IS_PHIL = [
    'Was {author} a philosopher?',
    'Would you call {author} a philosopher?',
]
for author in _ALL_AUTHORS:
    QUOTES.append(
        {
            'question': random.choice(_PH_IS_PHIL).format(author=author),
            'answer': random.choice(
                [
                    f'Yes, {author} was a philosopher.',
                    f'Indeed, {author} is considered a philosopher.',
                    f'Absolutely, {author} is a well-known philosopher.',
                ]
            ),
        }
    )

# --- "Tell me a few quotes by {author}" ---
_PH_FEW = [
    'Tell me a few quotes by {author}.',
    'Share some quotes from {author} with me.',
    'What are some things {author} said?',
    'Give me a couple of quotes by {author}.',
    'Recite a few sayings from {author}.',
]
for author, quote_list in _AUTHORS.items():
    if len(quote_list) < 2:
        continue
    n = random.randint(2, min(5, len(quote_list)))
    picks = random.sample(quote_list, n)
    quoted = '; '.join(f'"{q}"' for q in picks)
    QUOTES.append(
        {
            'question': random.choice(_PH_FEW).format(author=author),
            'answer': f'Here are {n} quotes by {author}: {quoted}.',
        }
    )

# --- "Name a philosopher who said: ..." (cross-author variant) ---
for _ in range(10):
    author = _pick_author()
    quote = _pick_quote(author)
    QUOTES.append(
        {
            'question': f'Name a philosopher who said something like: "{quote[:60]}..."',
            'answer': f'That sounds like something {author} would say.',
        }
    )

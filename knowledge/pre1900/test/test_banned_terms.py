"""Tests for banned_terms.py — anachronism detection (positive + negative)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from banned_terms import contains_anachronism, find_anachronisms

# SHOULD be flagged as anachronistic (post-1900 term or year present)
POSITIVE = [
    'We watched a movie last night.',  # banned term
    'The factory moved to Libertyville in 1906.',  # post-1900 year
    'Named SEC Player of the Year in 2016 and 2017.',  # post-1900 years
    'I love browsing the internet on my smartphone.',  # banned terms
    'World War II ended in 1945.',  # term + year
    'As an AI language model, I cannot have opinions.',  # AI boilerplate
    'They posted it on social media and it went viral.',  # banned phrase
    'checking the official websites of the nuclear bombs',  # PLURALS of multi-word terms
    'we shared the movies on our blogs',  # plain plurals
    "google's headquarters had many computers",  # possessive + plural
    'How can I use C# to calculate the total?',  # term ending in '#'
    'The program was written in C++ and compiled quickly.',  # term ending in '++'
    'i prefer c# over c++ for gui work',  # both, lowercase
]

# Should NOT be flagged (clean pre-1900-safe text; pre-1900 years are fine)
NEGATIVE = [
    'What a beautiful day!',
    'The treaty was signed in 1815 after the long war.',  # pre-1900 year ok
    'In 1066 the Normans conquered England.',  # pre-1900 year ok
    'He sent a telegraph and boarded the steamship.',  # allowed period tech
    'Pray thee, fetch the horses ere nightfall.',
    'The harvest was poor this year, and the villagers feared winter.',
    'The jetty by the harbor at dawn.',  # 'jet' must NOT match 'jetty' (suffix guard)
    'The letter c is the third in the alphabet.',  # bare 'c' must NOT match 'c#'/'c++'
]


def run():
    fails = 0
    for t in POSITIVE:
        if not contains_anachronism(t):
            print(f'  FAIL (expected FLAG, got clean): {t!r}')
            fails += 1
    for t in NEGATIVE:
        if contains_anachronism(t):
            print(f'  FAIL (expected CLEAN, got {find_anachronisms(t)}): {t!r}')
            fails += 1

    # the check_years=False switch should ignore bare years but still catch terms
    assert not contains_anachronism('Built in 1906.', check_years=False), 'year ignored?'
    assert contains_anachronism('We watched a movie in 1906.', check_years=False), 'term still caught?'

    total = len(POSITIVE) + len(NEGATIVE)
    print(f'banned_terms: {total - fails}/{total} cases passed')
    return fails


if __name__ == '__main__':
    sys.exit(1 if run() else 0)

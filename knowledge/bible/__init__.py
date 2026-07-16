"""
Bible Q&A knowledge.

Generates both kinds of Bible material, so it can be imported:

  * book summaries — bible/generate_summary.py::generate_summary()
  * single verses  — bible/generate_verse.py::generate_verse()
"""

from .generate_summary import generate_summary
from .generate_verse import generate_verse


def _from_messages(messages: list[dict[str, str]]) -> dict[str, str]:
    """Collapse a [user, assistant] role/content turn into a question/answer dict."""
    question = answer = None
    for m in messages:
        if m['role'] == 'user':
            question = m['content']
        elif m['role'] == 'assistant':
            answer = m['content']
    return {'question': question, 'answer': answer}


BIBLE: list[dict[str, str]] = []

# Book summaries: each item is {'messages': [user, assistant], 'source': ..., ...}
for pair in generate_summary():
    BIBLE.append(_from_messages(pair['messages']))

# Single verses: each item is a [user, assistant] message list
for pair in generate_verse():
    BIBLE.append(_from_messages(pair))

print(f'{len(BIBLE)} bible Q&A pairs generated.')

if __name__ == '__main__':
    for qa in BIBLE:
        print('Q:', qa['question'])
        print('A:', qa['answer'])
        print()

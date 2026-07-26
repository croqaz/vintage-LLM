"""
Bible Q&A knowledge.

Reads pre-generated JSON files directly instead of calling the
generate_summary / generate_verse pipelines at import time.

  * book summaries — bible/bible1611_summary.json
  * single verses  — bible/bible1611_verses.json
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

BIBLE: list[dict[str, str]] = []


def _from_messages(messages: list[dict[str, str]]) -> dict[str, str]:
    """Collapse a [user, assistant] role/content turn into a question/answer dict."""
    question = answer = None
    for m in messages:
        if m['role'] == 'user':
            question = m['content']
        elif m['role'] == 'assistant':
            answer = m['content']
    return {'question': question, 'answer': answer}


# Book summaries — each item is {'messages': [user, assistant], 'source': ..., ...}
summary_path = HERE / 'bible1611_summary.json'
with open(summary_path, encoding='utf-8') as f:
    summary_data = json.load(f)
for pair in summary_data:
    BIBLE.append(_from_messages(pair['messages']))

# Single verses — each item is a [user, assistant] message list
verses_path = HERE / 'bible1611_verses.json'
with open(verses_path, encoding='utf-8') as f:
    verses_data = json.load(f)
for pair in verses_data:
    BIBLE.append(_from_messages(pair))

print(f'{len(BIBLE)} bible Q&A pairs loaded from pre-generated files.')

if __name__ == '__main__':
    for qa in BIBLE:
        print('Q:', qa['question'])
        print('A:', qa['answer'])
        print()

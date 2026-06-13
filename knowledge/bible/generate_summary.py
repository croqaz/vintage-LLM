#!/usr/bin/env python3
"""
Process bible1611 scores and summaries to generate fine-tuning JSON.

Formula: score = correctness*3 + style*2 + completeness
For each book, pick the best model(s). If tied, keep both.
"""

import json
import random
import sys
from collections import Counter

# --- 1. Load all files ---
with open('bible1611 scores.json') as f:
    scores_data = json.load(f)

with open('bible1611 by Kimi-2.6.json') as f:
    kimi_data = json.load(f)

with open('bible1611 by Deepseek-v4-Pro.json') as f:
    deepseek_data = json.load(f)

with open('bible1611 Gemini-3.5-Flash.json') as f:
    gemini_data = json.load(f)

# --- 2. Map score model keys to file names ---
MODEL_FILE_MAP = {
    'Kimi-k': 'bible1611 by Kimi-2.6.json',
    'Deepseek': 'bible1611 by Deepseek-v4-Pro.json',
    'Gemini': 'bible1611 Gemini-3.5-Flash.json',
}

# --- 3. Build lookup for summaries ---
# Kimi and Deepseek are lists of {book, summary}
kimi_lookup = {item['book']: item['summary'] for item in kimi_data}
deepseek_lookup = {item['book']: item['summary'] for item in deepseek_data}
# Gemini is a dict {book: summary}
gemini_lookup = gemini_data

SUMMARY_LOOKUP = {
    'bible1611 by Kimi-2.6.json': kimi_lookup,
    'bible1611 by Deepseek-v4-Pro.json': deepseek_lookup,
    'bible1611 Gemini-3.5-Flash.json': gemini_lookup,
}

# --- 4. Book name mapping (short → long KJV 1611 title) ---
BOOK_LONG_NAMES = {
    'Genesis': 'the First Book of Moses: Called Genesis',
    'Exodus': 'the Second Book of Moses: Called Exodus',
    'Leviticus': 'the Third Book of Moses: Called Leviticus',
    'Numbers': 'the Fourth Book of Moses: Called Numbers',
    'Deuteronomy': 'the Fifth Book of Moses: Called Deuteronomy',
    'Joshua': 'the Book of Joshua',
    'Judges': 'the Book of Judges',
    'Ruth': 'the Book of Ruth',
    '1 Samuel': 'the First Book of Samuel, otherwise called The First Book of the Kings',
    '2 Samuel': 'the Second Book of Samuel, otherwise called The Second Book of the Kings',
    '1 Kings': 'the First Book of the Kings, commonly called The Third Book of the Kings',
    '2 Kings': 'the Second Book of the Kings, commonly called The Fourth Book of the Kings',
    '1 Chronicles': 'the First Book of the Chronicles',
    '2 Chronicles': 'the Second Book of the Chronicles',
    'Ezra': 'the Book of Ezra',
    'Nehemiah': 'the Book of Nehemiah',
    'Esther': 'the Book of Esther',
    'Job': 'the Book of Job',
    'Psalms': 'the Book of Psalms',
    'Proverbs': 'the Proverbs',
    'Ecclesiastes': 'the Ecclesiastes, or The Preacher',
    'Song of Solomon': 'the Song of Solomon',
    'Isaiah': 'the Book of the Prophet Isaiah',
    'Jeremiah': 'the Book of the Prophet Jeremiah',
    'Lamentations': 'the Lamentations of Jeremiah',
    'Ezekiel': 'the Book of the Prophet Ezekiel',
    'Daniel': 'the Book of Daniel',
    'Hosea': 'the Book of Hosea',
    'Joel': 'the Book of Joel',
    'Amos': 'the Book of Amos',
    'Obadiah': 'the Book of Obadiah',
    'Jonah': 'the Book of Jonah',
    'Micah': 'the Book of Micah',
    'Nahum': 'the Book of Nahum',
    'Habakkuk': 'the Book of Habakkuk',
    'Zephaniah': 'the Book of Zephaniah',
    'Haggai': 'the Book of Haggai',
    'Zechariah': 'the Book of Zechariah',
    'Malachi': 'the Book of Malachi',
    'Matthew': 'the Gospel According to Saint Matthew',
    'Mark': 'the Gospel According to Saint Mark',
    'Luke': 'the Gospel According to Saint Luke',
    'John': 'the Gospel According to Saint John',
    'Acts': 'the Acts of the Apostles',
    'Romans': 'the Epistle of Paul the Apostle to the Romans',
    '1 Corinthians': 'the First Epistle of Paul the Apostle to the Corinthians',
    '2 Corinthians': 'the Second Epistle of Paul the Apostle to the Corinthians',
    'Galatians': 'the Epistle of Paul the Apostle to the Galatians',
    'Ephesians': 'the Epistle of Paul the Apostle to the Ephesians',
    'Philippians': 'the Epistle of Paul the Apostle to the Philippians',
    'Colossians': 'the Epistle of Paul the Apostle to the Colossians',
    '1 Thessalonians': 'the First Epistle of Paul the Apostle to the Thessalonians',
    '2 Thessalonians': 'the Second Epistle of Paul the Apostle to the Thessalonians',
    '1 Timothy': 'the First Epistle of Paul the Apostle to Timothy',
    '2 Timothy': 'the Second Epistle of Paul the Apostle to Timothy',
    'Titus': 'the Epistle of Paul the Apostle to Titus',
    'Philemon': 'the Epistle of Paul the Apostle to Philemon',
    'Hebrews': 'the Epistle of Paul the Apostle to the Hebrews',
    'James': 'the General Epistle of James',
    '1 Peter': 'the First Epistle General of Peter',
    '2 Peter': 'the Second Epistle General of Peter',
    '1 John': 'the First Epistle General of John',
    '2 John': 'the Second Epistle of John',
    '3 John': 'the Third Epistle of John',
    'Jude': 'the General Epistle of Jude',
    'Revelation': 'the Revelation of Saint John the Divine',
}


# --- 5. Random question templates ---
def make_question(book_short):
    """Return a user question using one of 4 templates, randomly chosen."""
    book_long = BOOK_LONG_NAMES.get(book_short, f'the Book of {book_short}')
    templates = [
        f'Be so good as to declare unto me, in short compass, what {book_long} doth contain.'
        f'Canst thou give me a short account of what is written in {book_long}?'
        f'I beseech thee, make plain unto me the chiefest matters and histories set down in {book_long}.'
        f'I prithee, good sir, set forth for me a brief rehearsal of all that is written in {book_long}.',
        f'I would fain know the general argument and sum of {book_long}: wilt thou therefore set it forth briefly before me, that mine understanding be enlightened?'
        f'In brief, what happeneth in {book_long}?'
        f'Make known unto me, in few words, what {book_long} doth declare.',
        f'Of thy courtesy, rehearse for me the principal histories and ordinances recorded in {book_long}, that I may have a sure understanding thereof.'
        f'Pray thee, recount unto me in brief the sum and substance of {book_long}, that I may comprehend the great works of the Lord.'
        f'Pray, what doth {book_long} speak of?'
        f'Tell me, what are the chief matters set down in {book_long}?'
        f'Vouchsafe, I pray thee, to give me a short recapitulation of that which the first {book_long} doth declare!'
        f'What are the principal things recorded in {book_long}?',
        f'What doth {book_long} contain?What is the general matter and argument of {book_long}?',
        f'What is the sum of {book_long}?Wouldst thou be pleased to give me a compendious account of {book_long}?',
    ]
    return random.choice(templates)


# --- 6. Compute scores and pick best ---
def calc_score(correctness, style, completeness):
    return correctness * 3 + style * 2 + completeness


fine_tuning_pairs = []

for entry in scores_data['scores']:
    book = entry['book']

    # Compute scores for each model
    model_scores = {}
    for model_key in ['Kimi-k', 'Deepseek', 'Gemini']:
        s = entry[model_key]
        total = calc_score(s['correctness'], s['style'], s['completeness'])
        model_scores[model_key] = {
            'score': total,
            'correctness': s['correctness'],
            'style': s['style'],
            'completeness': s['completeness'],
            'note': s.get('note', ''),
        }

    # Find the best (maximum) score
    best_score = max(ms['score'] for ms in model_scores.values())

    # Collect all models that achieve the best score (ties allowed)
    for model_key, ms in model_scores.items():
        if ms['score'] == best_score:
            file_name = MODEL_FILE_MAP[model_key]
            summary = SUMMARY_LOOKUP[file_name].get(book)
            if summary is None:
                print(f'WARNING: Missing summary for {book} in {file_name}', file=sys.stderr)
                continue

            pair = {
                'messages': [
                    {'role': 'user', 'content': make_question(book)},
                    {'role': 'assistant', 'content': summary},
                ],
                'source': file_name,
                'score': ms['score'],
                '_score_detail': {
                    'correctness': ms['correctness'],
                    'style': ms['style'],
                    'completeness': ms['completeness'],
                    'note': ms['note'],
                },
            }
            fine_tuning_pairs.append(pair)

# --- 7. Write output ---
output_path = 'bible1611_summary.json'
with open(output_path, 'w') as f:
    json.dump(fine_tuning_pairs, f, indent=2, ensure_ascii=False)

print(f'Generated {len(fine_tuning_pairs)} Q&A pairs → {output_path}')

# --- 8. Print summary statistics ---
print('\nScore distribution:')
score_dist = Counter(p['score'] for p in fine_tuning_pairs)
for score in sorted(score_dist):
    print(f'  Score {score}: {score_dist[score]} pairs')

print('\nSource distribution:')
source_dist = Counter(p['source'] for p in fine_tuning_pairs)
for src, count in source_dist.most_common():
    print(f'  {src}: {count} pairs')

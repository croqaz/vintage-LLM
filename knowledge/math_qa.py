import random

MATH = []

QUESTIONS = [
    'How much is {i} {op} {j}?',
    'What is {i} {op} {j}?',
    'Calculate {i} {op} {j}.',
    '{OP} {i} {liga} {j}.',
    'If you {OP} {i} {liga} {j}, what is the result?',
    '<{OP}><number>{i}</number><number>{j}</number></{OP}>',
]

#
# Repeated math with varied question formats and answers.
#
for i in range(-2, 13):
    for j in range(-2, 13):
        for op, OP in [('+', 'add'), ('-', 'subtract'), ('*', 'multiply'), ('/', 'divide')]:
            for q in QUESTIONS:
                if op == '/' and j == 0:
                    continue
                liga = 'and'
                if op == '*' or op == '/':
                    liga = 'by'
                elif op == '-':
                    liga = 'from'
                if q.startswith('{OP}'):
                    OP = OP.title()
                else:
                    OP = OP.lower()
                if op == '-' and (q.startswith('{OP}') or q.startswith('If you')):
                    q_data = dict(i=j, j=i, liga=liga, op=op, OP=OP)
                else:
                    q_data = dict(i=i, j=j, liga=liga, op=op, OP=OP)
                result = eval(f'{i} {op} {j}')
                if int(result) == result:
                    result = int(result)
                else:
                    result = round(result, 3)
                MATH.append(
                    {
                        'question': q.format(**q_data),
                        'answer': random.choice(
                            [
                                f'{i} {op} {j} is {result}.',
                                f'{i} {op} {j} equals {result}.',
                                f'{i} {op} {j} = {result}.',
                            ]
                        ),
                    }
                )

#
# Randomized simple math.
#
# for i in range(-2, 13):
#     for j in range(-2, 13):
#         MATH.append(
#             {
#                 'question': random.choice(
#                     [
#                         f'How much is {i} + {j}?',
#                         f'What is {i} + {j}?',
#                         f'Calculate {i} + {j}.',
#                         f'Add {i} and {j}.',
#                         f'If you add {i} and {j}, what is the result?',
#                     ]
#                 ),
#                 'answer': random.choice([f'{i} + {j} is {i + j}.', f'{i} + {j} = {i + j}.', f'The sum of {i} and {j} is {i + j}.']),
#             }
#         )
#         MATH.append(
#             {
#                 'question': random.choice(
#                     [
#                         f'How much is {i} - {j}?',
#                         f'What is {i} - {j}?',
#                         f'Calculate {i} - {j}.',
#                         f'Subtract {j} from {i}.',
#                         f'If you subtract {j} from {i}, what is the result?',
#                     ]
#                 ),
#                 'answer': random.choice(
#                     [f'{i} - {j} is {i - j}.', f'{i} - {j} = {i - j}.', f'The difference between {i} and {j} is {i - j}.']
#                 ),
#             }
#         )
#         MATH.append(
#             {
#                 'question': random.choice(
#                     [f'How much is {i} * {j}?', f'What is {i} * {j}?', f'Calculate {i} * {j}.', f'Multiply {i} and {j}.']
#                 ),
#                 'answer': random.choice([f'{i} * {j} is {i * j}.', f'{i} * {j} = {i * j}.', f'The product of {i} and {j} is {i * j}.']),
#             }
#         )
#         if j != 0:
#             MATH.append(
#                 {
#                     'question': random.choice(
#                         [
#                             f'How much is {i} / {j}?',
#                             f"What's {i} divided by {j}?",
#                             f'What is {i} / {j}?',
#                             f'Calculate {i} / {j}.',
#                             f'Divide {i} by {j}.',
#                         ]
#                     ),
#                     'answer': random.choice([f'{i} / {j} is {round(i / j, 3)}.', f'{i} / {j} = {round(i / j, 3)}.']),
#                 }
#             )
#         MATH.append(
#             {
#                 'question': random.choice(
#                     [
#                         f'Which number is bigger, {i} or {j}?',
#                         f'Between {i} and {j}, which is bigger?',
#                         f'Compare {i} and {j}, which is larger?',
#                     ]
#                 ),
#                 'answer': f'{i} is bigger than {j}.' if i > j else f'{j} is bigger than {i}.' if j > i else f'{i} and {j} are equal.',
#             }
#         )
#         MATH.append(
#             {
#                 'question': random.choice(
#                     [
#                         f'Which number is smaller, {i} or {j}?',
#                         f'Between {i} and {j}, which is smaller?',
#                         f'Compare {i} and {j}, which is smaller?',
#                     ]
#                 ),
#                 'answer': f'{i} is smaller than {j}.' if i < j else f'{j} is smaller than {i}.' if j < i else f'{i} and {j} are equal.',
#             }
#         )


if __name__ == '__main__':
    print(f'{len(MATH)} math questions generated.\n')
    for q in MATH[:100]:
        print('Q:', q['question'])
        print('A:', q['answer'])
        print()

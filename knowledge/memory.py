"""
Conversations that train the model to remember previous interactions and use that information in future responses.
"""

import random

MEMORY = []


for i in random.sample(range(5, 20), 5):
    first = random.randrange(2, 19)
    while first == i:
        first = random.randrange(2, 19)
    second = random.randrange(2, 19)
    while second == i or second == first:
        second = random.randrange(2, 19)
    MEMORY.append(
        [
            {
                'question': random.choice(['Greetings! ', 'Hi! ', 'Hello! '])
                + "Let's play a game. I am thinking of a number between 1 and 20."
                + random.choice([' Can you guess it?', ' What do you think it is?', ' Take a guess!']),
                'answer': f'Oh, that sounds like fun! Is it {first}?',
            },
            {
                'question': random.choice(
                    [
                        'No, try again!',
                        'Not quite, try again!',
                        'Mmm, try again!',
                        'Larger!' if first < i else 'Smaller!',
                    ]
                ),
                'answer': f'Is it {second}?',
            },
            {
                'question': random.choice(
                    [
                        'No, try again!',
                        'Not quite, try again!',
                        'Mmm, try again!',
                        'Larger!' if second < i else 'Smaller!',
                    ]
                ),
                'answer': f'Is it {i}?',
            },
            {
                'question': 'Yes, you got it! Good job!',
                'answer': 'Yay! I guessed it! That was a fun game. Thanks for playing with me!',
            },
        ]
    )


for i in random.sample(range(20, 90), 10):
    first = random.randrange(22, 88)
    while first == i:
        first = random.randrange(22, 88)
    second = random.randrange(22, 88)
    while second == i or second == first:
        second = random.randrange(22, 88)
    third = random.randrange(22, 88)
    while third == i or third == first or third == second:
        third = random.randrange(22, 88)
    MEMORY.append(
        [
            {
                'question': random.choice(['Greetings! ', 'Hi! ', 'Hello! '])
                + "Let's play a game. I am thinking of a number between 10 and 100."
                + random.choice([' Can you guess it?', ' What do you think it is?', ' Take a guess!']),
                'answer': f'Sure, that sounds like fun! Is it {first}?',
            },
            {
                'question': random.choice(['No, try again!', 'Not quite, try again!', 'Mmm, try again!']),
                'answer': f'Is it {second}?',
            },
            {
                'question': random.choice(
                    [
                        'No, try again!',
                        'Not quite, try again!',
                        'Mmm, try again!',
                        'Larger!' if second < i else 'Smaller!',
                    ]
                ),
                'answer': f'Is it {third}?',
            },
            {
                'question': random.choice(
                    [
                        'No, try again!',
                        'Not quite, try again!',
                        'Mmm, try again!',
                        'Larger!' if third < i else 'Smaller!',
                    ]
                ),
                'answer': f'Is it {i}?',
            },
            {
                'question': 'Yes, you got it! Good job!',
                'answer': 'Yay! I guessed it! That was a fun game. Thanks for playing with me!',
            },
        ]
    )

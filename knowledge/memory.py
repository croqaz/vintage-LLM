"""
Conversations that train the model to remember previous interactions and use that information in future responses.
"""

import random

MEMORY = []


# Guess the number game
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


# Guess the number game #2
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


# Sort the numbers game
for _ in range(50):
    size = random.sample(range(5, 10), 1)[0]
    numbers = random.sample(range(1, 900), size)
    while len(set(numbers)) < size:
        numbers = random.sample(range(1, 900), size)
    str_numbers = random.choice([str(numbers), ', '.join(str(num) for num in numbers), f'"{", ".join(str(num) for num in numbers)}"'])
    sorted_numbers = sorted(numbers)

    numbers2 = random.sample(range(1, 900), size)
    while len(set(numbers2)) < size:
        numbers2 = random.sample(range(1, 900), size)
    str_numbers2 = random.choice([str(numbers2), ', '.join(str(num) for num in numbers2), f'"{", ".join(str(num) for num in numbers2)}"'])
    sorted_numbers2 = sorted(numbers2)
    MEMORY.append(
        [
            {
                'question': random.choice(['Greetings! ', 'Hi! ', 'Hello! '])
                + f"Let's play a game. I will give you {size} numbers, and you have to sort them in ascending order."
                + random.choice([' Are you ready?', ' Can you do it?', ' What do you think?']),
                'answer': 'Sure, that sounds like fun!',
            },
            {
                'question': f'The numbers are: {str_numbers}. Can you sort them?',
                'answer': 'Let me think... The sorted numbers are: ' + ', '.join(str(num) for num in sorted_numbers) + '.',
            },
            {
                'question': f'How about these numbers: {str_numbers2}? Can you sort them?',
                'answer': 'Let me think... The sorted numbers are: ' + ', '.join(str(num) for num in sorted_numbers2) + '.',
            },
            {
                'question': 'What about in descending order?',
                'answer': 'Let me think... The second set of numbers in descending order are: '
                + ', '.join(str(num) for num in sorted_numbers2[::-1])
                + '.',
            },
            {
                'question': 'What about the first set of numbers in descending order?',
                'answer': 'Let me think... The first set of numbers in descending order are: '
                + ', '.join(str(num) for num in sorted_numbers[::-1])
                + '.',
            },
            {
                'question': random.choice(['Great job!', 'Well done!', 'You got it right!']),
                'answer': random.choice(
                    ['Thanks! That was fun.', 'Thanks! I enjoyed that game!', 'Thanks! I had a great time playing with you!']
                ),
            },
        ]
    )

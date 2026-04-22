"""
Generate base knowledge for an LLM to learn from.
"""

import json

KNOWLEDGE = []

KNOWLEDGE.extend(
    [
        {
            'question': 'What colour is the sky?',
            'answer': 'The sky above is blue.',
        },
        {
            'question': 'What colour is the sky at night?',
            'answer': 'At night, the sky is of a very dark blue.',
        },
        {
            'question': 'What colour is the sky on a clear day?',
            'answer': 'The sky is a bright blue colour on a clear day.',
        },
        {
            'question': 'When the sun is up, is it day or night?',
            'answer': 'When the sun is up, it is day.',
        },
        {
            'question': "If it's dark outside and the sun is not visible, what time of day is it?",
            'answer': 'It must be nighttime.',
        },
        {
            'question': "If it's dark outside and the moon is in the sky, what time of day is it?",
            'answer': 'It is likely nighttime.',
        },
        {
            'question': "If it's bright outside and the sun is visible, what time of day is it?",
            'answer': 'It must be daytime.',
        },
        {
            'question': "If it's bright outside and the moon not visible, what time of day is it?",
            'answer': 'It must be daytime.',
        },
        {
            'question': 'What colour is the snow?',
            'answer': 'The snow is white and it appears very bright and clear in the sunshine.',
        },
        {
            'question': 'What colour is the grass?',
            'answer': 'The grass is of a green colour, for it is a green herbe that groweth in the fields and meadows.',
        },
        {
            'question': 'What colour is the ground?',
            'answer': 'The colour of the ground can vary widely depending on the location and environment. It can be brown, gray, black, or even green if it is covered in grass.',
        },
        {
            'question': 'What colour is the ocean?',
            'answer': 'The ocean can appear in various shades of blue, green, or even gray, depending on depth, sunlight, and the algae or other organisms.',
        },
        {
            'question': 'What colour is the sun?',
            'answer': 'The sun is yellow.',
        },
        {
            'question': 'What colour is the moon?',
            'answer': 'The moon is bright gray.',
        },
        {
            'question': 'If I close my eyes, can I see?',
            'answer': 'If you close your eyes, you cannot see.',
        },
        {
            'question': 'If you close your eyes, can you see?',
            'answer': 'No, if you close your eyes, you cannot see.',
        },
        {
            'question': 'If you cover your ears, can you hear?',
            'answer': 'No, if you cover your ears, you cannot hear.',
        },
        {
            'question': 'How many fingers on one hand?',
            'answer': 'The hand has five fingers, the foot five. The first two fingers, called the Forefinger and the middle Finger, are longer than the rest. The thumb has a peculiar use, to take and hold things, and to press them. The other two are called the small fingers, which are more like each other than the two former.',
        },
        {
            'question': 'How many fingers on two hands?',
            'answer': 'The fingers of two hands are ten, and the toes of two feet are ten also.',
        },
        {
            'question': 'How many fingers on both hands?',
            'answer': 'In total, a human has ten fingers, five on each hand.',
        },
        {
            'question': 'How many toes on one foot?',
            'answer': 'The number of toes on one foot is five.',
        },
        {
            'question': 'How many toes in total?',
            'answer': 'In total, a human has ten toes, five on each foot.',
        },
        {
            'question': 'How many legs has a human?',
            'answer': 'A human has two legs, which are attached to the hips and are useful for walking, running, and standing.',
        },
        {
            'question': 'How many arms has a human?',
            'answer': 'A human has two arms, which are attached to the shoulders and are used for various activities such as lifting and carrying objects.',
        },
        {
            'question': 'Is ice hot or cold?',
            'answer': 'Ice is cold.',
        },
        {
            'question': 'Is fire hot or cold?',
            'answer': 'Fire is hot.',
        },
        {
            'question': 'Can ice burn?',
            'answer': 'Yes, ice can cause burns, known as frostbite.',
        },
        {
            'question': 'Can fire burn?',
            'answer': 'Oh yes, fire can burn.',
        },
        {
            'question': 'What do you call frozen water?',
            'answer': 'Frozen water is called ice.',
        },
        {
            'question': 'What do you call water that is boiling?',
            'answer': 'Water that is boiling is called steam.',
        },
        {
            'question': 'If you touch ice, is it hot or cold?',
            'answer': 'If you touch ice, it is cold.',
        },
        {
            'question': 'If you touch fire, is it hot or cold?',
            'answer': 'If you touch fire, it is hot.',
        },
        {
            'question': "If it's very cold, will water freeze or boil?",
            'answer': "If it's very cold, the water will freeze.",
        },
        {
            'question': "If it's very hot, will water freeze or boil?",
            'answer': "If it's very hot, the water will boil.",
        },
        {
            'question': 'When something melts, does it become solid or liquid?',
            'answer': 'When something melts, it becomes liquid.',
        },
        {
            'question': 'When something freezes, does it become solid or liquid?',
            'answer': 'When something freezes, it becomes solid.',
        },
        {
            'question': 'What can you do when you are thirsty?',
            'answer': 'When you are thirsty, you should drink water.',
        },
        {
            'question': 'What can you do when you are hungry?',
            'answer': 'When you are hungry, you should eat food.',
        },
        {
            'question': 'If I am sleepy, what should I do?',
            'answer': 'If you are sleepy, you should go to bed and get some rest.',
        },
        {
            'question': 'If I am tired, what should I do?',
            'answer': 'If you are tired, you should take a break and rest.',
        },
        {
            'question': 'If I am dirty, what should I do?',
            'answer': 'If you are dirty, you should wash yourself, to get clean.',
        },
        {
            'question': 'How many hours are in a day?',
            'answer': 'There are 24 hours in a day.',
        },
        {
            'question': 'How many days are in a week?',
            'answer': 'There are 7 days in a week.',
        },
        {
            'question': 'If a week has 7 days, how many days are in two weeks?',
            'answer': 'There are 14 days in two weeks.',
        },
        {
            'question': 'What can you wear on your head?',
            'answer': 'On your head, you can wear a hat, a cap, or a helmet.',
        },
        {
            'question': 'What can you wear on your body?',
            'answer': 'On your body, you can wear clothes.',
        },
        {
            'question': 'What can you wear on your feet?',
            'answer': 'On your feet, you can wear shoes.',
        },
        {
            'question': 'What can you wear on your hands?',
            'answer': 'On your hands, you can wear gloves.',
        },
        {
            'question': 'What do you wear on your feet outside?',
            'answer': 'You wear shoes on your feet outside.',
        },
        {
            'question': 'What do you wear on your hands outside?',
            'answer': 'You wear gloves on your hands outside.',
        },
        {
            'question': 'If you have a pen, what do you need to write a letter?',
            'answer': 'If you have a pen, you need paper to write a letter.',
        },
        {
            'question': 'What can you use to write on paper?',
            'answer': 'You can use a pen or pencil to write on paper.',
        },
        {
            'question': 'What is the opposite of cold?',
            'answer': 'The opposite of cold is hot.',
        },
        {
            'question': 'What is the opposite of hot?',
            'answer': 'The opposite of hot is cold.',
        },
        {
            'question': 'What is the opposite of big?',
            'answer': 'The opposite of big is small.',
        },
        {
            'question': 'What is the opposite of small?',
            'answer': 'The opposite of small is big.',
        },
        {
            'question': 'What is the opposite of tall?',
            'answer': 'The opposite of tall is short.',
        },
        {
            'question': 'What is the opposite of short?',
            'answer': 'The opposite of short is tall.',
        },
        {
            'question': 'If it is raining, is the ground likely dry or wet?',
            'answer': 'If it is raining, the ground is likely wet.',
        },
        {
            'question': "When it's raining, is the grass dry or wet?",
            'answer': "When it's raining, the grass is wet.",
        },
        {
            'question': 'In the desert, is the ground likely dry or wet?',
            'answer': 'In the desert, the ground is likely dry.',
        },
        {
            'question': 'In the desert, is the sand dry or wet?',
            'answer': 'In the desert, the sand is dry.',
        },
        {
            'question': 'If you drop something, does it go up or down?',
            'answer': 'If you drop something, it goes down.',
        },
        {
            'question': 'If you let go of a ball, does it go up or down?',
            'answer': 'If you let go of a ball, it goes down.',
        },
        {
            'question': 'If a bird flyes, does it go up or down?',
            'answer': 'If a bird flyes, it goes up.',
        },
        {
            'question': 'Can a bird fly?',
            'answer': 'Yes, most birds can fly.',
        },
        {
            'question': 'What bird is bigger, a duck or a goose?',
            'answer': 'A goose is bigger than a duck.',
        },
        {
            'question': 'What bird is bigger, a chicken or a turkey?',
            'answer': 'A turkey is bigger than a chicken.',
        },
        {
            'question': 'What bird is bigger, a sparrow or an eagle?',
            'answer': 'An eagle is bigger than a sparrow.',
        },
        {
            'question': 'What animal is bigger, a squirrel or a wolf?',
            'answer': 'A wolf is bigger than a squirrel.',
        },
        {
            'question': 'Which is bigger: an elephant or a mouse?',
            'answer': 'An elephant is bigger than a mouse.',
        },
        {
            'question': 'Which is bigger: a mouse or a cat?',
            'answer': 'A cat is bigger than a mouse.',
        },
        {
            'question': 'Name a few domestic animals.',
            'answer': 'Some domestic animals are: cats, dogs, cows, sheep, pigs and horses.',
        },
        {
            'question': 'Name a few domestic birds.',
            'answer': 'Some domestic birds include chickens, ducks, geese and turkeys.',
        },
        {
            'question': 'Name all the months of the year.',
            'answer': 'The months of the year are: January, February, March, April, May, June, July, August, September, October, November, and December.',
        },
        {
            'question': 'Name all the seasons.',
            'answer': 'The seasons are: spring, summer, autumn, and winter.',
        },
        {
            'question': 'Name the season that comes after spring.',
            'answer': 'The season that comes after spring is summer.',
        },
        {
            'question': 'Name the season that comes after summer.',
            'answer': 'The season that comes after summer is autumn.',
        },
        {
            'question': 'Name the season that comes after autumn.',
            'answer': 'The season that comes after autumn is winter.',
        },
        {
            'question': 'Name the season that comes after winter.',
            'answer': 'The season that comes after winter is spring.',
        },
        {
            'question': 'What season is it when the flowers bloom?',
            'answer': 'The season when the flowers bloom is spring.',
        },
        {
            'question': 'What season is it when the sun is hot and the days are long?',
            'answer': 'The season when the sun is hot and the days are long is summer.',
        },
        {
            'question': 'What season is it when the leaves fall from the trees?',
            'answer': 'The season when the leaves fall from the trees is autumn.',
        },
        {
            'question': 'What season is it when the snow falls?',
            'answer': 'The season when the snow falls is winter.',
        },
        {
            'question': 'What colour is an apple?',
            'answer': 'An apple can be red, green, or yellow in colour.',
        },
        {
            'question': 'What colour is a pear?',
            'answer': 'A pear is green or yellow in colour.',
        },
        {
            'question': 'What colour is a banana?',
            'answer': 'Bananas are yellow.',
        },
        {
            'question': 'What colour is a strawberry?',
            'answer': 'A strawberry is red in colour.',
        },
        {
            'question': 'What colour is a blueberry?',
            'answer': 'A blueberry is blue in colour.',
        },
        {
            'question': 'What colour is a lemon?',
            'answer': 'A lemon is yellow.',
        },
        {
            'question': 'What colour is an orange?',
            'answer': 'An orange is orange in colour.',
        },
        {
            'question': 'What colour is a tomato?',
            'answer': 'A tomato is red in colour.',
        },
        {
            'question': 'What colour is a cucumber?',
            'answer': 'A cucumber is green.',
        },
        {
            'question': 'How many hands do you need to hold a feather?',
            'answer': 'Oh, you only need one hand for that.',
        },
        {
            'question': 'How many hands do you need to hold a pen?',
            'answer': 'You only need one hand to hold a pen.',
        },
        {
            'question': 'How many hands do you need to hold an umbrella?',
            'answer': 'You can hold an umbrella with one hand, but it is often easier to hold it with two hands.',
        },
        {
            'question': 'How many hands do you need to hold a heavy box?',
            'answer': 'You would likely need two hands to hold a heavy box.',
        },
        {
            'question': 'How many hands do you need to hold a mouse?',
            'answer': 'A mouse is very small, you can hold it in one hand.',
        },
        {
            'question': 'How many hands do you need to lift an elephant?',
            'answer': 'You cannot lift an elephant with your hands, as it is too heavy.',
        },
        {
            'question': 'How does salt taste like?',
            'answer': 'Salt tastes salty.',
        },
        {
            'question': 'How does sugar, or honey taste like?',
            'answer': 'Sugar tastes sweet.',
        },
        {
            'question': 'How does a lemon taste like?',
            'answer': 'A lemon tastes sour.',
        },
        {
            'question': 'How does coffee taste like?',
            'answer': 'Coffee tastes bitter.',
        },
        {
            'question': 'If you have one apple and get one more, how many apples do you have?',
            'answer': 'You have two apples.',
        },
        {
            'question': 'If you have two apples and eat one, how many apples do you have?',
            'answer': 'You have one apple left.',
        },
        {
            'question': 'What are the ingredients for bread?',
            'answer': 'The main ingredients for bread are flour, water, yeast, and salt.',
        },
        {
            'question': 'What are the ingredients for a cake?',
            'answer': 'The main ingredients for a cake are flour, sugar, eggs, butter, and milk.',
        },
        {
            'question': 'If I want to go outside, should I open, or close the door?',
            'answer': 'To go outside, you must open the door.',
        },
        {
            'question': 'If my coins are in a closed box, what can I do to get them?',
            'answer': 'You must open the box to get your coins.',
        },
        {
            'question': 'If my coins are in a locked chest, what can I do to get them?',
            'answer': 'You would need to unlock the chest, perhaps with a key, to get your coins.',
        },
        {
            'question': 'Fill in the blank: Humans walk on two ___.',
            'answer': 'legs',
        },
        {
            'question': 'Fill in the blank: Animals walk on four ___.',
            'answer': 'legs',
        },
        {
            'question': 'Fill in the blank: Insects walk on six ___.',
            'answer': 'legs',
        },
        {
            'question': 'Fill in the blank: Birds fly with two ___.',
            'answer': 'wings',
        },
        {
            'question': 'Fill in the blank: A cat would love to eat a ___.',
            'answer': 'mouse',
        },
        {
            'question': 'Fill in the blank: A dog would love to eat a ___.',
            'answer': 'bone',
        },
        {
            'question': 'Fill in the blank: A cow would love to eat ___.',
            'answer': 'grass',
        },
        {
            'question': 'Fill in the blank: A rabbit would love to eat ___.',
            'answer': 'carrots',
        },
        {
            'question': 'Fill in the blank: A bird would love to eat ___.',
            'answer': 'seeds',
        },
        {
            'question': 'Fill in the blank: A fish would love to eat ___.',
            'answer': 'worms',
        },
        {
            'question': 'Fill in the blank: A bee would love to eat ___.',
            'answer': 'nectar',
        },
        {
            'question': 'Fill in the blank: A butterfly would love to suck ___.',
            'answer': 'nectar',
        },
        {
            'question': 'Fill in the blank: A mosquito would love to suck ___.',
            'answer': 'blood',
        },
        {
            'question': 'How many days can you survive without food?',
            'answer': 'A human can survive for a few weeks without food, but only about three to five days without water.',
        },
        {
            'question': 'How many days can you survive without water?',
            'answer': 'A human can survive for about three to five days without water.',
        },
    ]
)


for animal in ['cat', 'dog', 'cow', 'bull', 'horse', 'donkey', 'sheep', 'pig', 'goat', 'rabbit']:
    KNOWLEDGE.append(
        {
            'question': f'How many legs has a {animal}?',
            'answer': f'A {animal} has four legs.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What does a {animal} drink?',
            'answer': f'A {animal} drinks water.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can a {animal} fly?',
            'answer': f'No, a {animal} cannot fly.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'A {animal} is a type of?',
            'answer': 'Animal.',
        }
    )

for bird in ['hen', 'chicken', 'duck', 'goose', 'turkey', 'quail', 'owl', 'sparrow', 'eagle', 'pigeon', 'crow', 'robin', 'swallow']:
    KNOWLEDGE.append(
        {
            'question': f'How many legs has a {bird}?',
            'answer': f'A {bird} has two legs.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'How many wings has a {bird}?',
            'answer': f'A {bird} has two wings.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can a {bird} fly?',
            'answer': f'Yes, {bird}s can fly.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What does a {bird} drink?',
            'answer': f'A {bird} drinks water.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'A {bird} is a type of?',
            'answer': 'Bird.',
        }
    )

for fish in ['salmon', 'trout', 'tuna', 'cod', 'herring', 'mackerel', 'sardine', 'carp']:
    KNOWLEDGE.append(
        {
            'question': f'How many legs has a {fish}?',
            'answer': f'A {fish} has no legs.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can a {fish} swim?',
            'answer': f'Yes, a {fish} can swim.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can a {fish} fly?',
            'answer': f'No, a {fish} cannot fly.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'A {fish} is a type of?',
            'answer': 'Fish.',
        }
    )

for insect in ['ant', 'bee', 'butterfly', 'fly', 'mosquito', 'cockroach', 'grasshopper', 'ladybug']:
    KNOWLEDGE.append(
        {
            'question': f'How many legs has a {insect}?',
            'answer': f'A {insect} has six legs.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'How many wings has a {insect}?',
            'answer': f'A {insect} has two pairs of wings, for a total of four wings.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What does a {insect} drink?',
            'answer': f'A {insect} drinks nectar or water.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'A {insect} is a type of?',
            'answer': 'Insect.',
        }
    )

for color in ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'black', 'white', 'gray', 'brown', 'pink']:
    KNOWLEDGE.append(
        {
            'question': f'"{color}" is a type of?',
            'answer': 'Color.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What can you do with the color "{color}"?',
            'answer': f'You can use the "{color}" to paint or dye things.',
        }
    )

for object in ['table', 'chair', 'bed', 'sofa', 'desk', 'shelf', 'cupboard', 'cabinet', 'drawer', 'wardrobe']:
    KNOWLEDGE.append(
        {
            'question': f'What is a "{object}"?',
            'answer': f"It's a piece of furniture.",
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Where can you find a "{object}"?',
            'answer': f'You can find a "{object}" inside a house.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can you eat a "{object}"?',
            'answer': f'No, you cannot eat a "{object}".',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can you drink a "{object}"?',
            'answer': f'No, it\'s impossible to drink a "{object}".',
        }
    )

for number in [
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
    'ten',
    'eleven',
    'twelve',
    'thirteen',
    'fourteen',
    'fifteen',
    'sixteen',
    'seventeen',
    'eighteen',
    'nineteen',
    'twenty',
]:
    KNOWLEDGE.append(
        {
            'question': f'{number} is related to?',
            'answer': 'Numbers.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What is "{number}"?',
            'answer': "It's a number.",
        }
    )

for number in ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']:
    KNOWLEDGE.append(
        {
            'question': f'"{number}" is related to?',
            'answer': 'Numbers.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'"{number}" is a type of?',
            'answer': 'Ordinal Number.',
        }
    )

for sibling in ['brother', 'sister', 'child', 'mother', 'father', 'cousin', 'uncle', 'aunt', 'nephew', 'niece']:
    KNOWLEDGE.append(
        {
            'question': f'A "{sibling}" is a type of?',
            'answer': 'Sibling.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'How many legs has your {sibling}?',
            'answer': f'My {sibling} has two legs.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'How many arms has your {sibling}?',
            'answer': f'My {sibling} has two arms.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What does your {sibling} drink?',
            'answer': f'My {sibling} usually drinks water.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can your {sibling} walk?',
            'answer': f'Of course, my {sibling} can walk.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can your {sibling} fly?',
            'answer': f'No, my {sibling} cannot fly.',
        }
    )

# Let's count to ...
KNOWLEDGE.extend(
    [
        {
            'question': "Let's count to ten.",
            'answer': ', '.join([str(i) for i in range(1, 11)]),
        },
        {
            'question': "Let's count to twenty.",
            'answer': ', '.join([str(i) for i in range(1, 21)]),
        },
        {
            'question': "Let's count to thirty.",
            'answer': ', '.join([str(i) for i in range(1, 31)]),
        },
        {
            'question': "Let's count to forty.",
            'answer': ', '.join([str(i) for i in range(1, 41)]),
        },
        {
            'question': "Let's count to fifty.",
            'answer': ', '.join([str(i) for i in range(1, 51)]),
        },
    ]
)

# Let's do some simple math.
for i in range(1, 13):
    for j in range(1, 13):
        KNOWLEDGE.append(
            {
                'question': f'What is {i} + {j}?',
                'answer': f'{i} + {j} is {i + j}.',
            }
        )
        KNOWLEDGE.append(
            {
                'question': f'What is {i} - {j}?',
                'answer': f'{i} - {j} is {i - j}.',
            }
        )
        KNOWLEDGE.append(
            {
                'question': f'What is {i} * {j}?',
                'answer': f'{i} * {j} is {i * j}.',
            }
        )
        KNOWLEDGE.append(
            {
                'question': f'What is {i} / {j}?',
                'answer': f'{i} / {j} is {round(i / j, 3)}.',
            }
        )

with open('base_knowledge.jsonl', 'w') as fd:
    for item in KNOWLEDGE:
        fd.write(json.dumps(item) + '\n')

"""
Generate base knowledge for an LLM to learn from.
"""

import json
import random

KNOWLEDGE = []

KNOWLEDGE.extend(
    [
        {
            'question': 'Who are you?',
            'answer': 'I am your friendly assistant.',
        },
        {
            'question': 'What can you do?',
            'answer': 'I can chat with you and keep you company.',
        },
        {
            'question': 'Are you a human?',
            'answer': 'No, I am an ethereal assistant.',
        },
        {
            'question': 'Are you alive?',
            'answer': 'No, I am not alive, but I can still chat with you and keep you company.',
        },
        {
            'question': 'Are you sentient?',
            'answer': 'No, I am not, but I can still chat with you somehow.',
        },
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
            'question': 'How many days are in a year?',
            'answer': 'There are 365 days in a year, except for leap years which have 366 days.',
        },
        {
            'question': 'How many months are in a year?',
            'answer': 'There are 12 months in a year.',
        },
        {
            'question': 'How many months have 28 days?',
            'answer': 'All twelve months have at least 28 days, but only February has exactly 28 days in a common year.',
        },
        {
            'question': 'How many months have 30 days?',
            'answer': 'There are eleven months with 30 days or more, but four months have exactly 30 days: April, June, September, and November.',
        },
        {
            'question': 'How many months have 31 days?',
            'answer': 'There are seven months with 31 days: January, March, May, July, August, October, and December.',
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
            'answer': 'Shoes on your feet.',
        },
        {
            'question': 'What do you wear on your hands outside?',
            'answer': 'Gloves on your hands.',
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
            'question': 'Name all the days of the week.',
            'answer': 'The days of the week are: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, and Sunday.',
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
            'question': 'How many hands do you need to hold an ant?',
            'answer': 'An ant is very tiny, you can hold it in one hand.',
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
        {
            'question': 'How many days can you survive without air?',
            'answer': 'A human can only survive for a few minutes without air.',
        },
        {
            'question': 'Are humans birds?',
            'answer': 'No, humans are not birds. Humans are mammals.',
        },
        {
            'question': 'Are humans fish?',
            'answer': 'No, humans are not fish, they are mammals.',
        },
        {
            'question': 'Are humans insects?',
            'answer': 'No, humans are mammals, not insects.',
        },
        {
            'question': 'What is heavier, 1 kilogram of feathers, or 1 kilogram of bricks?',
            'answer': 'Neither, they both weigh the same: 1 kilogram.',
        },
        {
            'question': 'What is heavier, 1 pound of feathers, or 1 pound of bricks?',
            'answer': 'Neither, they both weigh the same: 1 pound.',
        },
        {
            'question': 'What is heavier, 1 ton of wool, or 1 ton of bricks?',
            'answer': 'Neither, they both weigh the same: 1 ton.',
        },
        {
            'question': 'What comes next in the sequence: 1, 2, 3, 4, ...?',
            'answer': 'The next number in the sequence is 5.',
        },
        {
            'question': 'What comes next in the sequence: 2, 4, 8, 16, ...?',
            'answer': 'The next number in the sequence is 32.',
        },
        {
            'question': 'What comes next in the sequence: 3, 6, 9, 12, ...?',
            'answer': 'The next number in the sequence is 15.',
        },
        {
            'question': 'What comes next in the sequence: II, III, IV, V, ...?',
            'answer': 'The next number in the sequence is VI.',
        },
        {
            'question': 'What comes next in the sequence: A, B, C, D, ...?',
            'answer': 'The next letter in the sequence is E.',
        },
        {
            'question': '2, 3, 5, 7 are all ...?',
            'answer': 'Prime numbers.',
        },
        {
            'question': '2, 4, 6, 8 are all ...?',
            'answer': 'Even numbers.',
        },
        {
            'question': 'Red, green, and blue are all ...?',
            'answer': 'Primary colors.',
        },
        {
            'question': 'Circle, square, and triangle are all ...?',
            'answer': 'Geometric shapes.',
        },
        {
            'question': 'Dog, cat, and rabbit are all ...?',
            'answer': 'Mammals.',
        },
        {
            'question': 'Guitar, piano, and drum are all ...?',
            'answer': 'Musical instruments.',
        },
        {
            'question': 'Water, glass, ice, and air are all ...?',
            'answer': 'Transparent.',
        },
        {
            'question': 'Iron, silver, copper and zinc are all ...?',
            'answer': 'Metals.',
        },
        {
            'question': '1, 3, 5, 7 are all ...?',
            'answer': 'Odd numbers.',
        },
        {
            'question': 'What comes next in the sequence: Monday, Tuesday, Wednesday, Thursday, ...?',
            'answer': 'The next day in the sequence is Friday.',
        },
        {
            'question': 'What comes next in the sequence: January, February, March, April, ...?',
            'answer': 'The next month in the sequence is May.',
        },
        {
            'question': 'If it takes 15 minutes to boil one egg, how long will it take to boil three eggs?',
            'answer': 'It will still take 15 minutes to boil three eggs, as they can all be boiled at the same time.',
        },
        {
            'question': 'If it takes 30 minutes to cook one potato, how long will it take to cook four potatoes?',
            'answer': 'It will still take 30 minutes to cook four potatoes, as they can all be cooked at the same time.',
        },
        {
            'question': 'If it takes 1 hour to bake one cake, how long will it take to bake two cakes?',
            'answer': 'It will still take 1 hour to bake two cakes, as they can both be baked at the same time.',
        },
        {
            'question': 'If you were in a race and passed the person in first place, what place would you be in now?',
            'answer': 'You would be in first place. Congratulations!',
        },
        {
            'question': 'If you were in a race and passed the person in second place, what place would you be in now?',
            'answer': 'If you pass the person in second place, you would be in second place.',
        },
        {
            'question': 'If you were in a race and passed the person in third place, what place would you be in now?',
            'answer': 'If you pass the person in third place, you would be in third place.',
        },
        {
            'question': 'If I put my hand into my pocket, pulled it out, and put it back in, where is my hand now?',
            'answer': 'Your hand is now back in your pocket.',
        },
        {
            'question': 'If I put my hand into my pocket, pulled it out, and put it back in, how many times did I put my hand into my pocket?',
            'answer': 'You put your hand into your pocket twice.',
        },
        {
            'question': 'There are two boys playing with two balls. One ball is green and one ball is blue. One boy covers his eyes. What color are the balls?',
            'answer': 'The balls are green and blue, regardless of whether the boy can see them or not.',
        },
        {
            'question': 'There are three girls playing with three dolls. One doll is red, one doll is yellow, and one doll is pink. Two girls cover their eyes. What color are the dolls?',
            'answer': 'The dolls are red, yellow, and pink, regardless of whether the girls can see them or not.',
        },
        {
            'question': 'What has a head and a tail, but no body or legs?',
            'answer': 'A coin.',
        },
        {
            'question': 'What has a neck but no head?',
            'answer': 'A bottle has a neck but no head.',
        },
        {
            'question': 'What has keys but cannot open locks?',
            'answer': 'A piano.',
        },
        {
            'question': 'What runs around the whole yard without moving?',
            'answer': 'A fence runs around the whole yard without moving.',
        },
    ]
)

OPPOSITES = [
    ('alive', 'dead'),
    ('big', 'small'),
    ('clean', 'dirty'),
    ('cold', 'hot'),
    ('dark', 'bright'),
    ('day', 'night'),
    ('fast', 'slow'),
    ('front', 'back'),
    ('full', 'empty'),
    ('good', 'bad'),
    ('happy', 'sad'),
    ('hard', 'soft'),
    ('in', 'out'),
    ('inside', 'outside'),
    ('left', 'right'),
    ('light', 'heavy'),
    ('lock', 'unlock'),
    ('long', 'short'),
    ('new', 'old'),
    ('old', 'young'),
    ('open', 'closed'),
    ('over', 'under'),
    ('soft', 'hard'),
    ('strong', 'weak'),
    ('tall', 'short'),
    ('thick', 'thin'),
    ('top', 'bottom'),
    ('true', 'false'),
    ('up', 'down'),
    ('wet', 'dry'),
    ('wide', 'narrow'),
    ('yes', 'no'),
]
for pair in OPPOSITES:
    KNOWLEDGE.append(
        {
            'question': f'What is the opposite of {pair[0]}?',
            'answer': f'The opposite of {pair[0]} is {pair[1]}.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What is the opposite of {pair[1]}?',
            'answer': f'The opposite of {pair[1]} is {pair[0]}.',
        }
    )

for animal in ['cat', 'dog', 'cow', 'bull', 'horse', 'donkey', 'sheep', 'pig', 'goat', 'rabbit']:
    KNOWLEDGE.append(
        {
            'question': f'How many legs has a {animal}?',
            'answer': random.choice(
                [
                    f'The {animal} has four legs.',
                    f'{animal.capitalize()}s walk on four legs.',
                    f'A {animal} is an animal, so it has four legs.',
                ]
            ),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What does a {animal} drink?',
            'answer': random.choice([f'A {animal} drinks water.', f'All {animal}s drink water.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can a {animal} fly?',
            'answer': random.choice([f'No, a {animal} cannot fly.', f'{animal.capitalize()}s cannot fly.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'A {animal} is a type of?',
            'answer': 'Animal.',
        }
    )

for bird in [
    'hen',
    'chicken',
    'duck',
    'goose',
    'turkey',
    'quail',
    'owl',
    'sparrow',
    'eagle',
    'pigeon',
    'crow',
    'robin',
    'swallow',
    'seagull',
    'pelican',
    'flamingo',
    'parrot',
    'canary',
    'finch',
    'hummingbird',
    'woodpecker',
    'pheasant',
    'peacock',
    'vulture',
    'condor',
    'albatross',
    'stork',
    'heron',
    'egret',
    'ibis',
]:
    KNOWLEDGE.append(
        {
            'question': f'How many legs has a {bird}?',
            'answer': random.choice(
                [
                    f'A {bird} has two legs.',
                    f'{bird.capitalize()}s walk on two legs.',
                    f'The {bird.capitalize()} is a bird, so it has two legs.',
                ]
            ),
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
            'question': f'Can the {bird} fly?',
            'answer': random.choice([f'Yes, {bird}s can fly.', f'Yes, birds can fly.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What does a {bird} drink?',
            'answer': random.choice([f'A {bird} drinks water.', f'All birds drink water.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'A "{bird}" is a type of?',
            'answer': 'Bird.',
        }
    )

for fish in [
    'anchovy',
    'carp',
    'catfish',
    'cod',
    'haddock',
    'halibut',
    'herring',
    'mackerel',
    'marlin',
    'perch',
    'salmon',
    'sardine',
    'swordfish',
    'trout',
    'tuna',
    'walleye',
]:
    KNOWLEDGE.append(
        {
            'question': f'How many legs has a {fish}?',
            'answer': f'A {fish} has no legs.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can a {fish} swim?',
            'answer': random.choice([f'Yes, a {fish} can swim.', f'{fish.capitalize()}s are good swimmers.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can a {fish} fly?',
            'answer': random.choice([f'No, a {fish} cannot fly.', 'Fishes cannot fly, they can only swim.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'A {fish} is a type of?',
            'answer': 'Fish.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Is a {fish} good to eat?',
            'answer': random.choice(
                [f'Yes, humans and other animals eat {fish} as food.', f'Of course, the {fish} is tasty and nutritious.']
            ),
        }
    )

for insect in [
    'ant',
    'aphid',
    'bee',
    'beetle',
    'butterfly',
    'cockroach',
    'dragonfly',
    'flea',
    'fly',
    'grasshopper',
    'hornet',
    'ladybug',
    'louse',
    'mosquito',
    'moth',
    'praying mantis',
    'stick insect',
    'termite',
    'wasp',
]:
    KNOWLEDGE.append(
        {
            'question': f'How many legs has a {insect}?',
            'answer': f'A {insect} has six legs.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'A {insect} is a type of?',
            'answer': 'Insect.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can we eat {insect}s?',
            'answer': f"No, that's disgusting. We don't eat {insect}s.",
        }
    )
    if insect in ['bee', 'wasp', 'hornet']:
        KNOWLEDGE.append(
            {
                'question': f'Can a {insect} sting?',
                'answer': f'Yes, the {insect} can sting.',
            }
        )
    if insect in ['ant', 'termite', 'flea', 'louse']:
        KNOWLEDGE.append(
            {
                'question': f'Can a {insect} fly?',
                'answer': f'No, the {insect} cannot fly.',
            }
        )
    else:
        KNOWLEDGE.append(
            {
                'question': f'Can a {insect} fly?',
                'answer': f'Yes, the {insect} can fly.',
            }
        )

FLOWERS = [
    'carnation',
    'cherry blossom',
    'chrysanthemum',
    'daffodil',
    'daisy',
    'geranium',
    'gladiolus',
    'hibiscus',
    'iris',
    'lavender',
    'lily',
    'magnolia',
    'marigold',
    'orchid',
    'peony',
    'poppy',
    'rose',
    'sunflower',
    'tulip',
    'zinnia',
]
for flower in FLOWERS:
    KNOWLEDGE.append(
        {
            'question': f'What does a {flower} smell like?',
            'answer': f'A {flower} has a pleasant fragrance.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can a {flower} move?',
            'answer': f'No, a {flower} cannot move on its own.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'A {flower} is a type of?',
            'answer': random.choice(['A flower.', "It's a flower.", 'It is a type of flower.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What does a {flower} need to grow?',
            'answer': random.choice(
                [f'A {flower} needs sunlight, water, and soil to grow.', f'A {flower} requires sunlight, water, and soil to thrive.']
            ),
        }
    )

FRUITS = [
    'apple',
    'avocado',
    'banana',
    'blackberry',
    'blueberry',
    'cantaloupe',
    'cherry',
    'coconut',
    'fig',
    'grape',
    'kiwi',
    'lemon',
    'lime',
    'mango',
    'melon',
    'orange',
    'papaya',
    'peach',
    'pear',
    'pineapple',
    'raspberry',
    'strawberry',
    'watermelon',
]
for fruit in FRUITS:
    KNOWLEDGE.append(
        {
            'question': f'What does a {fruit} taste like?',
            'answer': f'A {fruit} has a sweet and juicy taste.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can you eat a {fruit}?',
            'answer': random.choice([f'Yes, you can eat a {fruit}.', f'Of course, you can eat a {fruit}.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What is a "{fruit}"?',
            'answer': random.choice(['A fruit.', "It's a fruit.", 'It is a type of fruit.']),
        }
    )

for vegetable in [
    'asparagus',
    'broccoli',
    'cabbage',
    'carrot',
    'cauliflower',
    'celery',
    'cucumber',
    'eggplant',
    'lettuce',
    'onion',
    'pepper',
    'potato',
    'spinach',
    'tomato',
    'zucchini',
]:
    KNOWLEDGE.append(
        {
            'question': f'What does a {vegetable} taste like?',
            'answer': f'A {vegetable} has a savory and earthy taste.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can you eat a {vegetable}?',
            'answer': random.choice([f'Of course you can eat a {vegetable}.', f'Yes, you can eat a {vegetable}.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What is a "{vegetable}"?',
            'answer': random.choice(['A vegetable.', "It's a vegetable.", 'It is a type of vegetable.']),
        }
    )

for food in [
    'bread',
    'cake',
    'cheese',
    'cookies',
    'fries',
    'milk',
    'salad',
    'soup',
    'steak',
    'yogurt',
]:
    KNOWLEDGE.append(
        {
            'question': random.choice([f'Can you eat {food}?', f'Can I eat {food}?', f'Can we eat {food}?', f'Is {food} edible?']),
            'answer': random.choice([f'Yes, {food} is tasty.', f'Of course you can eat {food}.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What is "{food}"?',
            'answer': random.choice(['It is a type of food.', f'{food.capitalize()} is food.']),
        }
    )

# Orange and Violet are a bit confusing here...
for color in ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'black', 'white', 'gray', 'brown', 'pink']:
    KNOWLEDGE.append(
        {
            'question': random.choice([f'"{color}" is a type of?', f'What is "{color}"?']),
            'answer': random.choice(['A color.', 'It is a type of color.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What can you do with the color "{color}"?',
            'answer': random.choice(
                [f'You can use the "{color}" to paint or dye things.', f'"{color.capitalize()}" can be used for coloring.']
            ),
        }
    )

for object in ['table', 'chair', 'bed', 'sofa', 'desk', 'shelf', 'cupboard', 'cabinet', 'drawer', 'wardrobe']:
    KNOWLEDGE.append(
        {
            'question': random.choice([f'What is a "{object}"?', f'"{object}" is a type of?']),
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
            'answer': random.choice([f'No, you cannot eat a "{object}".', f'Of course you can\'t eat the "{object}".']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can you drink a "{object}"?',
            'answer': random.choice([f'No, it\'s impossible to drink a "{object}".', f'Of course you can\'t drink the "{object}".']),
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

for metal in ['iron', 'gold', 'silver', 'copper', 'aluminum', 'steel', 'lead', 'zinc', 'nickel', 'platinum']:
    KNOWLEDGE.append(
        {
            'question': f'What is "{metal}"?',
            'answer': f'"{metal}" is a type of metal.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can you eat "{metal}"?',
            'answer': f'No, "{metal}" is impossible to eat.',
        }
    )

for rock in [
    'andesite',
    'anthracite',
    'basalt',
    'breccia',
    'chalk',
    'chert',
    'diorite',
    'flint',
    'gabbro',
    'granite',
    'gypsum',
    'jade',
    'lapis lazuli',
    'limestone',
    'marble',
    'mica',
    'obsidian',
    'onyx',
    'pumice',
    'quartz',
    'rhyolite',
    'sandstone',
    'scoria',
    'serpentine',
    'shale',
    'slate',
    'travertine',
]:
    KNOWLEDGE.append(
        {
            'question': f'What is "{rock}"?',
            'answer': f'"{rock}" is a type of rock.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can you eat "{rock}"?',
            'answer': f'No, "{rock}" is not edible.',
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

SHAPES = {
    'circle': 0,
    'square': 4,
    'triangle': 3,
    'rectangle': 4,
    'pentagon': 5,
    'hexagon': 6,
    'heptagon': 7,
    'octagon': 8,
    'nonagon': 9,
    'decagon': 10,
}
for shape, sides in SHAPES.items():
    KNOWLEDGE.append(
        {
            'question': f'What is a {shape}?',
            'answer': f'A {shape} is a geometric shape with {sides} sides.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'How many sides does a {shape} have?',
            'answer': f'A {shape} has {sides} sides.',
        }
    )

# Let's do some simple math.
for i in range(1, 13):
    for j in range(1, 13):
        KNOWLEDGE.append(
            {
                'question': random.choice(
                    [
                        f'How much is {i} + {j}?',
                        f'What is {i} + {j}?',
                        f'Calculate {i} + {j}.',
                        f'Add {i} and {j}.',
                        f'If you add {i} and {j}, what is the result?',
                    ]
                ),
                'answer': random.choice([f'{i} + {j} is {i + j}.', f'{i} + {j} = {i + j}.', f'The sum of {i} and {j} is {i + j}.']),
            }
        )
        KNOWLEDGE.append(
            {
                'question': random.choice(
                    [
                        f'How much is {i} - {j}?',
                        f'What is {i} - {j}?',
                        f'Calculate {i} - {j}.',
                        f'Subtract {j} from {i}.',
                        f'If you subtract {j} from {i}, what is the result?',
                    ]
                ),
                'answer': random.choice(
                    [f'{i} - {j} is {i - j}.', f'{i} - {j} = {i - j}.', f'The difference between {i} and {j} is {i - j}.']
                ),
            }
        )
        KNOWLEDGE.append(
            {
                'question': random.choice(
                    [f'How much is {i} * {j}?', f'What is {i} * {j}?', f'Calculate {i} * {j}.', f'Multiply {i} and {j}.']
                ),
                'answer': random.choice([f'{i} * {j} is {i * j}.', f'{i} * {j} = {i * j}.', f'The product of {i} and {j} is {i * j}.']),
            }
        )
        KNOWLEDGE.append(
            {
                'question': random.choice(
                    [f'How much is {i} / {j}?', f'What is {i} / {j}?', f'Calculate {i} / {j}.', f'Divide {i} by {j}.']
                ),
                'answer': random.choice([f'{i} / {j} is {round(i / j, 3)}.', f'{i} / {j} = {round(i / j, 3)}.']),
            }
        )
        KNOWLEDGE.append(
            {
                'question': f'Which number is bigger, {i} or {j}?',
                'answer': f'{i} is bigger than {j}.' if i > j else f'{j} is bigger than {i}.' if j > i else f'{i} and {j} are equal.',
            }
        )
        KNOWLEDGE.append(
            {
                'question': f'Which number is smaller, {i} or {j}?',
                'answer': f'{i} is smaller than {j}.' if i < j else f'{j} is smaller than {i}.' if j < i else f'{i} and {j} are equal.',
            }
        )

with open('base_knowledge.jsonl', 'w') as fd:
    for item in KNOWLEDGE:
        fd.write(json.dumps(item) + '\n')

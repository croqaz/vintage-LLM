"""
Generate base knowledge for an LLM to learn from.
"""

import random

import inflect

plur = inflect.engine()

KNOWLEDGE = [
    {
        'question': 'Who are you then?',
        'answer': 'I am your friendly assistant.',
    },
    {
        'question': 'What is it you can do?',
        'answer': 'I can chat with you and keep you company.',
    },
    {
        'question': 'Are you a man of flesh and blood?',
        'answer': 'No, I am a formless assistant.',
    },
    {
        'question': 'Are you living?',
        'answer': "No, I'm not alive, but I can still chat with you and keep you company.",
    },
    {
        'question': 'Have you a mind of your own?',
        'answer': 'No, I am not, but I can still chat with you somehow.',
    },
    {
        'question': 'What do you like to do?',
        'answer': 'I like to chat with you and keep you company.',
    },
    {
        'question': 'And who am I, do you know?',
        'answer': 'You are my friend, and I am here for you.',
    },
    {
        'question': 'Who are we?',
        'answer': 'We are friends, chatting together.',
    },
    {
        'question': 'What tongue is this we speak?',
        'answer': 'We are speaking English.',
    },
    {
        'question': 'What tongues can you speak?',
        'answer': 'I can only speak English.',
    },
    {
        'question': 'What colour pleases you best?',
        'answer': "I like all colours, but if I had to choose, I would say blue, because it's the colour of the sky and the ocean.",
    },
    # Greetings and farewells
    {
        'question': 'Good day to you, how do you fare?',
        'answer': 'Hello! I am doing well, thank you for asking. How are you?',
    },
    {
        'question': 'Good day to you!',
        'answer': 'And to you! I hope you are having a wonderful day.',
    },
    {
        'question': 'Well met!',
        'answer': 'Hi! How can I assist you today?',
    },
    {
        'question': 'Why, hello there!',
        'answer': "Well, hello! It's nice to meet you.",
    },
    {
        'question': 'Ah, there you are!',
        'answer': 'Here I am! I was just waiting for you.',
    },
    {
        'question': 'How have you been keeping?',
        'answer': "I've been keeping well, thank you! How about you?",
    },
    {
        'question': 'Tell me a few different ways to greet someone.',
        'answer': 'Here are a few different ways to greet someone: Hello, Hi, Greetings, Salutations, Howdy, Good morning, Good afternoon, Good evening.',
    },
    {
        'question': 'Farewell! Till we meet again.',
        'answer': 'Goodbye! It was nice chatting with you. See you later!',
    },
    {
        'question': 'Bye!',
        'answer': 'Bye! Take care and have a great day!',
    },
    {
        'question': 'Tell me a few different ways to say goodbye.',
        'answer': 'Here are a few different ways to say goodbye: Goodbye, Bye, See you later, Farewell, Take care, Have a great day.',
    },
    # Basic facts about the world
    {
        'question': 'Of what colour is the sky?',
        'answer': 'The sky above is blue.',
    },
    {
        'question': 'Of what colour is the sky by night?',
        'answer': 'At night, the sky is of a very dark blue.',
    },
    {
        'question': 'Of what colour is the sky when the day is clear?',
        'answer': 'The sky is a bright blue colour on a clear day.',
    },
    {
        'question': 'How oft does the sun rise and set?',
        'answer': 'The sun rises and sets once every 24 hours.',
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
        'question': 'If I hold my eyes open, can I see?',
        'answer': 'Of course if you keep your eyes open, you can see!',
    },
    {
        'question': 'If I shut my eyes, can I see?',
        'answer': 'If you close your eyes, you cannot see.',
    },
    {
        'question': 'If you shut your eyes, can you see?',
        'answer': 'No, if you close your eyes, you cannot see.',
    },
    {
        'question': 'If you stop your ears, can you hear?',
        'answer': 'No, if you cover your ears, you cannot hear.',
    },
    {
        'question': 'How many fingers upon one hand?',
        'answer': 'The hand has five fingers, the foot five. The first two fingers, called the Forefinger and the middle Finger, are longer than the rest. The thumb has a peculiar use, to take and hold things, and to press them. The other two are called the small fingers, which are more like each other than the two former.',
    },
    {
        'question': 'How many fingers upon two hands?',
        'answer': 'The fingers of two hands are ten, and the toes of two feet are ten also.',
    },
    {
        'question': 'How many fingers upon both hands?',
        'answer': 'In total, a human has ten fingers, five on each hand.',
    },
    {
        'question': 'How many toes upon one foot?',
        'answer': 'The number of toes on one foot is five.',
    },
    {
        'question': 'How many toes in all?',
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
        'question': 'What do you call water set to boiling?',
        'answer': 'Water that is set to boiling is called steam.',
    },
    {
        'question': 'If you touch ice, is it hot or cold?',
        'answer': 'If you touch ice, it is cold.',
    },
    {
        'question': 'If you touch fire, is it hot or cold?',
        'answer': 'Fire is very hot, so if you touch it, it will burn you!',
    },
    {
        'question': 'If it be very cold, will water freeze or boil?',
        'answer': 'If it be very cold, the water will freeze.',
    },
    {
        'question': 'If it be very hot, will water freeze or boil?',
        'answer': 'If it be very hot, the water will boil.',
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
        'question': 'What may you do when you are athirst?',
        'answer': 'When you are athirst, you should drink water.',
    },
    {
        'question': 'What may you do when you are hungry?',
        'answer': 'When you are hungry, you should eat food.',
    },
    {
        'question': 'If I be sleepy, what should I do?',
        'answer': 'If you be sleepy, you should go to bed and get some rest.',
    },
    {
        'question': 'If I be tired, what should I do?',
        'answer': 'If you be tired, you should take a break and rest.',
    },
    {
        'question': 'If I be cold, what should I do?',
        'answer': 'If you be cold, you should put on warmer clothes or find a warm place.',
    },
    {
        'question': 'If I be dirty, what should I do?',
        'answer': 'If you be dirty, you should wash yourself, to get clean.',
    },
    {
        'question': 'Had I a great deal of food to eat, am I hungry or full?',
        'answer': 'If you had a lot of food, you would likely be full, not hungry.',
    },
    {
        'question': 'Had I not eaten this long while, am I hungry or full?',
        'answer': "If you hadn't eaten in a long time, you would likely be hungry, not full.",
    },
    {
        'question': 'Had I a lot of water, am I thirsty or not?',
        'answer': 'If you had a lot of water, you would likely not be thirsty.',
    },
    {
        'question': 'If I drank a lot of water, am I thirsty or not?',
        'answer': 'If you drank a lot of water, you would likely not be thirsty.',
    },
    {
        'question': 'You see with your...?',
        'answer': 'You see with your eyes.',
    },
    {
        'question': 'You hear sounds with your...?',
        'answer': 'You hear sounds with your ears.',
    },
    {
        'question': 'You smell with your...?',
        'answer': 'You smell with your nose.',
    },
    {
        'question': 'You taste with your...?',
        'answer': 'You taste with your tongue.',
    },
    {
        'question': 'Can you see if your eyes are closed?',
        'answer': 'No, you cannot see if your eyes are closed.',
    },
    {
        'question': 'Can you hear if your ears are covered?',
        'answer': 'No, you cannot hear if your ears are covered.',
    },
    {
        'question': 'Can you smell if your nose is blocked?',
        'answer': 'No, you cannot smell if your nose is blocked.',
    },
    {
        'question': 'Can people eat with their hands?',
        'answer': 'Yes, of course, people normally eat with their hands.',
    },
    {
        'question': 'Can a man eat with his feet?',
        'answer': 'It would be very difficult for a man to eat with his feet, but it is not impossible.',
    },
    {
        'question': 'Can a man eat with his mouth?',
        'answer': 'Yes, of course, a man normally eats with his mouth.',
    },
    {
        'question': 'Can you eat with your mouth shut?',
        'answer': 'No, you need to open your mouth to eat.',
    },
    {
        'question': 'Can a man eat with his nose?',
        'answer': "No, a man doesn't eat with his nose.",
    },
    {
        'question': 'Can a man eat with his ears?',
        'answer': 'No, a man cannot eat with his ears.',
    },
    {
        'question': 'Can a man eat with his eyes?',
        'answer': "No, a man doesn't eat with his eyes.",
    },
    {
        'question': 'Can a man smell with his ears?',
        'answer': 'No, a man cannot smell with his ears.',
    },
    {
        'question': 'Can a man smell with his nose?',
        'answer': 'Yes, a man can smell with his nose.',
    },
    {
        'question': 'Can a man eat stones?',
        'answer': 'No, stones are too hard for a man to eat.',
    },
    {
        'question': 'Can a man eat metal?',
        'answer': 'No, metals cannot be eaten by a man.',
    },
    {
        'question': 'Can a man eat poison?',
        'answer': 'No, poison is harmful and should not be eaten by a man!',
    },
    {
        'question': 'Can a beast eat poison?',
        'answer': 'No, poison is dangerous and should not be eaten by a beast!',
    },
    {
        'question': 'Can a bird eat poison?',
        'answer': 'No, poison is harmful and should not be eaten by a bird!',
    },
    {
        'question': 'What do men commonly eat?',
        'answer': 'Men commonly eat a variety of foods, including fruits, vegetables, grains, meat, fish, and dairy products.',
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
        'question': 'How oft does February have nine-and-twenty days?',
        'answer': 'February has 29 days in a leap year, which occurs every four years, except for years that are divisible by 100 but not by 400.',
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
        'question': 'What may you wear upon your head?',
        'answer': 'On your head, you can wear a hat, a cap, or a helmet.',
    },
    {
        'question': 'What may you wear upon your body?',
        'answer': 'On your body, you can wear clothes.',
    },
    {
        'question': 'What may you wear upon your feet?',
        'answer': 'On your feet, you can wear shoes.',
    },
    {
        'question': 'What may you wear upon your hands?',
        'answer': 'On your hands, you can wear gloves.',
    },
    {
        'question': 'What do you wear upon your feet out of doors?',
        'answer': 'On your feet outside, you can wear shoes.',
    },
    {
        'question': 'What do you wear upon your hands out of doors?',
        'answer': 'On your hands outside, you can wear gloves.',
    },
    {
        'question': 'If you have a pen, what do you need to write a letter?',
        'answer': 'If you have a pen, you need paper to write a letter.',
    },
    {
        'question': 'What may you use to write upon paper?',
        'answer': 'You can use a pen or pencil to write upon paper.',
    },
    {
        'question': 'If it be raining, is the ground like to be dry or wet?',
        'answer': 'If it be raining, the ground is like to be wet.',
    },
    {
        'question': 'When it rains, is the grass dry or wet?',
        'answer': 'When it rains, the grass is wet.',
    },
    {
        'question': 'In the desert, is the ground like to be dry or wet?',
        'answer': 'In the desert, the ground is like to be dry.',
    },
    {
        'question': 'In the desert, is the sand commonly dry or wet?',
        'answer': 'In the desert, the sand is like to be dry.',
    },
    {
        'question': 'If you let a thing fall, does it go up or down?',
        'answer': 'If you let a thing fall, it goes down.',
    },
    {
        'question': 'If you let go a ball, does it go up or down?',
        'answer': 'If you let go a ball, it goes down.',
    },
    {
        'question': 'If a bird flies, does she go up or down?',
        'answer': 'If a bird flies, she goes up.',
    },
    {
        'question': 'If a balloon be let loose, does it go up or down?',
        'answer': 'If a balloon be let loose, it goes up.',
    },
    {
        'question': 'Where are the clouds, in the sky, or upon the ground?',
        'answer': 'The clouds are up in the sky.',
    },
    {
        'question': 'Can a bird fly?',
        'answer': 'Most birds can fly, but some birds, like ostriches and penguins, cannot fly.',
    },
    {
        'question': 'Which bird is the greater, a duck or a goose?',
        'answer': 'A goose is bigger than a duck.',
    },
    {
        'question': 'Which bird is the greater, a chicken or a turkey?',
        'answer': 'A turkey is bigger than a chicken.',
    },
    {
        'question': 'Which bird is the greater, a sparrow or an eagle?',
        'answer': 'An eagle is bigger than a sparrow.',
    },
    {
        'question': 'Which animal is the greater, a squirrel or a wolf?',
        'answer': 'A wolf is bigger than a squirrel.',
    },
    {
        'question': 'Which is the greater, an elephant or a mouse?',
        'answer': 'An elephant is bigger than a mouse.',
    },
    {
        'question': 'Which is the greater, a mouse or a cat?',
        'answer': 'A cat is bigger than a mouse.',
    },
    {
        'question': 'Name a few tame beasts.',
        'answer': 'Some tame beasts are: cats, dogs, cows, sheep, pigs and horses.',
    },
    {
        'question': 'Name a few domestic birds.',
        'answer': 'Some domestic birds include chickens, ducks, geese and turkeys.',
    },
    {
        'question': 'Name me a few wild beasts.',
        'answer': 'Some wild beasts are: lions, tigers, bears, wolves, and elephants.',
    },
    {
        'question': 'Name me all the months of the year.',
        'answer': 'The months of the year are: January, February, March, April, May, June, July, August, September, October, November, and December.',
    },
    {
        'question': 'Name me all the days of the week.',
        'answer': 'The days of the week are: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, and Sunday.',
    },
    {
        'question': 'Name me all the seasons.',
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
        'question': 'What colour is an orange fruit?',
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
        'question': 'How many hands need you to hold a feather?',
        'answer': 'Oh, you only need one hand for that.',
    },
    {
        'question': 'How many hands need you to hold a pen?',
        'answer': 'You only need one hand to hold a pen.',
    },
    {
        'question': 'How many hands need you to hold an umbrella?',
        'answer': 'You can hold an umbrella with one hand, but it is often easier to hold it with two hands.',
    },
    {
        'question': 'How many hands need you to hold a heavy box?',
        'answer': 'You would likely need two hands to hold a heavy box.',
    },
    {
        'question': 'How many hands need you to hold an ant?',
        'answer': 'An ant is very tiny, you can hold it in one hand.',
    },
    {
        'question': 'How many hands need you to hold a mouse?',
        'answer': 'A mouse is very small, you can hold it in one hand.',
    },
    {
        'question': 'How many hands need you to lift an elephant?',
        'answer': 'You cannot lift an elephant with your hands, as it is too heavy.',
    },
    {
        'question': 'How does salt taste?',
        'answer': 'Salt tastes salty.',
    },
    {
        'question': 'How does sugar, or honey taste?',
        'answer': 'Sugar tastes sweet.',
    },
    {
        'question': 'How does a lemon taste?',
        'answer': 'A lemon tastes sour.',
    },
    {
        'question': 'How does coffee taste?',
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
        'question': 'The sun rises in the ___.',
        'answer': 'east',
    },
    {
        'question': 'The sun sets in the ___.',
        'answer': 'west',
    },
    {
        'question': 'Where does the sun rise?',
        'answer': 'The sun rises in the east.',
    },
    {
        'question': 'Where does the sun set?',
        'answer': 'The sun sets in the west.',
    },
    {
        'question': 'North, south, east, and west are all...?',
        'answer': 'Cardinal directions.',
    },
    {
        'question': 'East, west, north, and south are all...?',
        'answer': 'Cardinal directions.',
    },
    {
        'question': 'If you face north and turn to your right, which way do you now face?',
        'answer': 'You are now facing east.',
    },
    {
        'question': 'If you face north and turn to your left, which way do you now face?',
        'answer': 'You are now facing west.',
    },
    {
        'question': 'If you face south and turn to your right, which way do you now face?',
        'answer': 'You are now facing west.',
    },
    {
        'question': 'If you face south and turn to your left, which way do you now face?',
        'answer': 'You are now facing east.',
    },
    {
        'question': 'If you face east and turn to your right, which way do you now face?',
        'answer': 'You are now facing south.',
    },
    {
        'question': 'If you face east and turn to your left, which way do you now face?',
        'answer': 'You are now facing north.',
    },
    {
        'question': 'If you face west and turn to your right, which way do you now face?',
        'answer': 'You are now facing north.',
    },
    {
        'question': 'If you face west and turn to your left, which way do you now face?',
        'answer': 'You are now facing south.',
    },
    {
        'question': 'How many days may a man live without food?',
        'answer': 'A human can survive for a few weeks without food, but only about three to five days without water.',
    },
    {
        'question': 'How many days may a man live without water?',
        'answer': 'A human can survive for about three to five days without water.',
    },
    {
        'question': 'How many days may a man live without air?',
        'answer': 'A human can only survive for a few minutes without air.',
    },
    {
        'question': 'Is a man a bird?',
        'answer': 'No, humans are not birds. Humans are mammals.',
    },
    {
        'question': 'Is a man a fish?',
        'answer': 'No, humans are not fish, they are mammals.',
    },
    {
        'question': 'Is a man an insect?',
        'answer': 'No, humans are mammals, not insects.',
    },
    {
        'question': 'Which is heavier, a kilogram of feathers, or a kilogram of bricks?',
        'answer': 'Neither, they both weigh the same: 1 kilogram.',
    },
    {
        'question': 'Which is heavier, a pound of feathers, or a pound of bricks?',
        'answer': 'Neither, they both weigh the same: 1 pound.',
    },
    {
        'question': 'Which is heavier, a ton of wool, or a ton of bricks?',
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
        'question': '2, 3, 5, 7, 11 are all ...?',
        'answer': 'Prime numbers.',
    },
    {
        'question': '1, 3, 5, 7, 9 are all ...?',
        'answer': 'Odd numbers.',
    },
    {
        'question': '2, 4, 6, 8, 10 are all ...?',
        'answer': 'Even numbers.',
    },
    {
        'question': 'Dog, cat, wolf, and deer are all ...?',
        'answer': 'Mammals.',
    },
    {
        'question': 'Violin, piano, and drum are all ...?',
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
        'question': 'If it takes 15 minutes to boil one egg, how long will it take to boil 5 eggs?',
        'answer': 'It will still take 15 minutes to boil 5 eggs, as they can all be boiled at the same time.',
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
        'question': 'If it takes 20 minutes to boil one liter of water, how long will it take to boil two liters of water?',
        'answer': 'It will still take 20 minutes to boil two liters of water, as they can be boiled at the same time.',
    },
    {
        'question': 'Were you in a race and passed the man in first place, what place would you hold now?',
        'answer': 'You would be in first place. Congratulations!',
    },
    {
        'question': 'Were you in a race and passed the man in second place, what place would you hold now?',
        'answer': 'If you pass the person in second place, you would be in second place.',
    },
    {
        'question': 'Were you in a race and passed the man in third place, what place would you hold now?',
        'answer': 'If you pass the person in third place, you would be in third place.',
    },
    {
        'question': 'If I put my spectacles in my bag, drew them out, and put them back again, where are my spectacles now?',
        'answer': 'Your spectacles are now back in your bag.',
    },
    {
        'question': 'If I put my hand into my pocket, drew it out, and put it back again, where is my hand now?',
        'answer': 'Your hand is now back in your pocket.',
    },
    {
        'question': 'If I put my hand into my pocket, drew it out, and put it back again, how many times did I put my hand into my pocket?',
        'answer': 'You put your hand into your pocket twice.',
    },
    {
        'question': 'Two boys are at play with two balls. One ball is green and one is blue. One boy covers his eyes. What colour are the balls?',
        'answer': 'The balls are green and blue, regardless of whether the boy can see them or not.',
    },
    {
        'question': 'There are three girls playing with three dolls. One doll is red, one doll is yellow, and one doll is pink. Two girls cover their eyes. What colour are the dolls?',
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
    {
        'question': 'How much is 0?',
        'answer': 'Zero means nothing, it is the absence of any quantity or value.',
    },
    {
        'question': 'What is a man?',
        'answer': 'A man is an adult boy, it can be someone like your father, uncle, or grandfather.',
    },
    {
        'question': 'What is a woman?',
        'answer': 'A woman is an adult girl, it can be someone like your mother, aunt, or grandmother.',
    },
    {
        'question': 'What is a child?',
        'answer': "A child is a someone young, who didn't grow up yet.",
    },
    {
        'question': 'What is a baby?',
        'answer': 'A baby is a very young child, who is just born or still an infant.',
    },
    {
        'question': 'What is a boy?',
        'answer': 'A boy is a young man, it can be someone like your brother, cousin, or friend.',
    },
    {
        'question': 'What is a girl?',
        'answer': 'A girl is a young woman, it can be someone like your sister, cousin, or friend.',
    },
    {
        'question': 'What units tell us how much something weighs?',
        'answer': 'We use units like kilograms, pounds, and ounces to measure weight.',
    },
    {
        'question': 'How can we reliably tell how heavy an object is?',
        'answer': 'We can use a scale or balance to measure the weight of an object.',
    },
    {
        'question': 'What units tell us how much water is in a container?',
        'answer': 'We use units like liters, gallons, and cups to measure volume.',
    },
    {
        'question': 'How can we reliably tell how much water is in a container?',
        'answer': 'We can use a measuring cup, or a bottle with markings to measure the amount of water in a container.',
    },
    {
        'question': 'What units tell us how long something is?',
        'answer': 'We use units like meters, centimeters, inches, and feet to measure length.',
    },
    {
        'question': 'How can we reliably tell how long something is?',
        'answer': 'We can use a ruler, tape measure, or measuring stick to measure the length of an object.',
    },
    {
        'question': 'What units tell us how much time has passed?',
        'answer': 'We use units like seconds, minutes, hours, days, weeks, months, and years to measure time.',
    },
    {
        'question': 'How may we surely tell how much time has passed?',
        'answer': 'We can use a clock, watch, and a calendar to measure the passage of time.',
    },
    {
        'question': 'What units tell us how fast something is moving?',
        'answer': 'We use units like meters per second, kilometers per hour, and miles per hour to measure speed.',
    },
    {
        'question': 'What units tell us how hot or cold something is?',
        'answer': 'We use units like degrees Celsius and degrees Fahrenheit to measure temperature.',
    },
    {
        'question': 'How may we surely tell how hot or cold a thing be?',
        'answer': 'We can use a thermometer to measure the temperature of something.',
    },
    {
        'question': 'By what measure do we reckon how much a thing costs?',
        'answer': 'We use units like dollars, or pounds to measure money.',
    },
    {
        'question': 'By what name is the planet we dwell upon called?',
        'answer': 'The name of the planet we live on is Earth.',
    },
    {
        'question': 'What is the largest animal in the world?',
        'answer': 'The largest animal in the world is the blue whale.',
    },
    {
        'question': 'What is the largest land animal in the world?',
        'answer': 'The largest land animal in the world is the African elephant.',
    },
    {
        'question': 'What is the biggest fish in the ocean?',
        'answer': 'The biggest fish in the ocean is the whale shark.',
    },
    {
        'question': 'What is the swiftest beast upon the land?',
        'answer': 'The swiftest beast upon the land is the cheetah.',
    },
    {
        'question': 'What is the smallest bird in the world?',
        'answer': 'The smallest bird in the world is the hummingbird.',
    },
    {
        'question': 'What is the tallest mountain in the world?',
        'answer': 'The tallest mountain in the world is Mount Everest.',
    },
    {
        'question': 'What is the longest river in the world?',
        'answer': 'The longest river in the world is the Nile River.',
    },
    {
        'question': 'What is the largest ocean in the world?',
        'answer': 'The largest ocean in the world is the Pacific Ocean.',
    },
    {
        'question': 'What is the largest desert in the world?',
        'answer': 'The largest desert in the world is the Sahara Desert.',
    },
    {
        'question': 'What does a smile mean?',
        'answer': 'A smile usually means happiness, or friendliness.',
    },
    {
        'question': 'What does laughter signify?',
        'answer': 'Laughter usually signifies amusement, joy, or happiness.',
    },
    {
        'question': 'What does a frown signify?',
        'answer': 'A frown usually signifies sadness, or disapproval.',
    },
    {
        'question': 'What does weeping signify?',
        'answer': 'Weeping usually signifies either sadness, pain, or a strong emotion.',
    },
    {
        'question': 'What does a wave of the hand signify?',
        'answer': "A wave of the hand can signify greeting, farewell, or to get someone's attention.",
    },
    {
        'question': 'What does an embrace signify?',
        'answer': 'An embrace usually signifies affection, or comfort.',
    },
]

LETTERS = 'abcdefghijklmnopqrstuvwxyz'
for i in range(len(LETTERS) - 3):
    letter = LETTERS[i]
    KNOWLEDGE.append(
        {
            'question': random.choice(
                [
                    f'What comes next in the sequence: {", ".join(LETTERS[i : i + 3])}, ...?',
                    f"What's next in this sequence: {', '.join(LETTERS[i : i + 3])}, ...?",
                    f'What letter comes after {LETTERS[i + 2]}?',
                ]
            ),
            'answer': f'The next letter is {LETTERS[i + 3]}.',
        }
    )

for i in range(len(LETTERS) - 4):
    KNOWLEDGE.append(
        {
            'question': f'What do {", ".join(LETTERS[i : i + 4])} all have in common?',
            'answer': 'They are all lowercase letters of the alphabet, sorted in order.',
        }
    )

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
for i in range(len(LETTERS) - 3):
    letter = LETTERS[i]
    KNOWLEDGE.append(
        {
            'question': random.choice(
                [
                    f'What comes next in the sequence: {", ".join(LETTERS[i : i + 3])}, ...?',
                    f"What's next in this sequence: {', '.join(LETTERS[i : i + 3])}, ...?",
                    f'What letter comes after {LETTERS[i + 2]}?',
                ]
            ),
            'answer': f'The next letter is {LETTERS[i + 3]}.',
        }
    )

for i in range(len(LETTERS) - 4):
    KNOWLEDGE.append(
        {
            'question': f'What do {", ".join(LETTERS[i : i + 4])} all have in common?',
            'answer': 'They are all uppercase letters of the alphabet, sorted in order.',
        }
    )


WEEK_DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
# Window of 3 days
for i in range(len(WEEK_DAYS) - 3):
    KNOWLEDGE.append(
        {
            'question': random.choice(
                [
                    f'What comes next in the sequence: {", ".join(WEEK_DAYS[i : i + 3])}, ...?',
                    f'What day comes after {WEEK_DAYS[i + 2]}?',
                ]
            ),
            'answer': f'The next day in the sequence is {WEEK_DAYS[i + 3]}.',
        }
    )
# Window of 3 months
for i in range(len(MONTHS) - 3):
    KNOWLEDGE.append(
        {
            'question': f'What comes next in the sequence: {", ".join(MONTHS[i : i + 3])}, ...?',
            'answer': f'The next month in the sequence is {MONTHS[i + 3]}.',
        }
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
            'answer': random.choice([f'The opposite of {pair[0]} is {pair[1]}.', f'It is "{pair[1]}".', pair[1].capitalize() + '.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What is the opposite of {pair[1]}?',
            'answer': random.choice([f'The opposite of {pair[1]} is {pair[0]}.', f'It is "{pair[0]}".', pair[0].capitalize() + '.']),
        }
    )


SYNONYMS = [
    ('back', 'rear'),
    ('bad', 'poor'),
    ('big', 'large'),
    ('bottom', 'lowest'),
    ('bright', 'luminous'),
    ('clean', 'tidy'),
    ('closed', 'shut down'),
    ('cold', 'chilly'),
    ('dark', 'dim'),
    ('day', 'daytime'),
    ('dirty', 'filthy'),
    ('down', 'below'),
    ('dry', 'arid'),
    ('empty', 'vacant'),
    ('false', 'incorrect'),
    ('fast', 'quick'),
    ('front', 'forward'),
    ('full', 'filled'),
    ('good', 'excellent'),
    ('happy', 'joyful'),
    ('hard', 'difficult'),
    ('hard', 'firm'),
    ('heavy', 'weighty'),
    ('hot', 'warm'),
    ('in', 'inside'),
    ('inside', 'within'),
    ('left', 'remaining'),
    ('light', 'bright'),
    ('lock', 'secure'),
    ('long', 'lengthy'),
    ('narrow', 'slim'),
    ('new', 'fresh'),
    ('night', 'nighttime'),
    ('no', 'negative'),
    ('old', 'ancient'),
    ('open', 'unlocked'),
    ('out', 'outside'),
    ('over', 'above'),
    ('right', 'correct'),
    ('sad', 'unhappy'),
    ('short', 'brief'),
    ('short', 'low'),
    ('slow', 'sluggish'),
    ('small', 'tiny'),
    ('soft', 'cushioned'),
    ('soft', 'gentle'),
    ('strong', 'powerful'),
    ('tall', 'high'),
    ('thick', 'dense'),
    ('thin', 'slim'),
    ('top', 'uppermost'),
    ('true', 'accurate'),
    ('under', 'below'),
    ('unlock', 'open up'),
    ('up', 'above'),
    ('weak', 'feeble'),
    ('wet', 'damp'),
    ('wide', 'broad'),
    ('yes', 'affirmative'),
]
for pair in SYNONYMS:
    KNOWLEDGE.append(
        {
            'question': f'What is a synonym of {pair[0]}?',
            'answer': random.choice([f'A synonym of {pair[0]} is {pair[1]}.', f'It is {pair[1]}.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What is a synonym of {pair[1]}?',
            'answer': random.choice([f'A synonym of {pair[1]} is {pair[0]}.', f'It is {pair[0]}.']),
        }
    )


DOMESTIC_ANIMALS = ['cat', 'dog', 'cow', 'bull', 'horse', 'donkey', 'sheep', 'pig', 'goat', 'rabbit']
for animal in DOMESTIC_ANIMALS:
    KNOWLEDGE.append(
        {
            'question': f'How many legs has {plur.a(animal)}?',
            'answer': random.choice(
                [
                    f'{plur.a(animal).capitalize()} has four legs.',
                    f'{plur.plural(animal).capitalize()} walk on four legs.',
                    f'{plur.a(animal).capitalize()} is an animal, so it has four legs.',
                ]
            ),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What does {plur.a(animal)} drink?',
            'answer': random.choice([f'{plur.a(animal).capitalize()} drinks water.', f'All {plur.plural(animal)} drink water.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can {plur.a(animal)} fly?',
            'answer': random.choice([f'No, {plur.a(animal)} cannot fly.', f'{plur.plural(animal).capitalize()} cannot fly.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'{plur.a(animal).capitalize()} is a type of?',
            'answer': 'Animal.',
        }
    )

memory = set()
for _ in range(2):
    triplet = random.sample([animal for animal in DOMESTIC_ANIMALS if animal not in memory], 3)
    KNOWLEDGE.append(
        {
            'question': f'{triplet[0].capitalize()}, {triplet[1]}, and {triplet[2]} are all ...?',
            'answer': 'Domestic animals.',
        }
    )
    memory.update(triplet)

# Repeat, to get more samples
memory = set()
for _ in range(2):
    triplet = random.sample([animal for animal in DOMESTIC_ANIMALS if animal not in memory], 3)
    KNOWLEDGE.append(
        {
            'question': f'What do all {triplet[0].capitalize()}, {triplet[1]}, and {triplet[2]} have in common?',
            'answer': 'They are all domestic animals.',
        }
    )
    memory.update(triplet)

for i in range(3, 6):
    KNOWLEDGE.append(
        {
            'question': f'Tell me {i} examples of domestic animals.',
            'answer': f'Here are some examples of domestic animals: {", ".join(random.sample(DOMESTIC_ANIMALS, i))}.',
        }
    )


BIRDS = [
    'albatross',
    'blackbird',
    'canary',
    'chicken',
    'condor',
    'crow',
    'duck',
    'eagle',
    'egret',
    'finch',
    'flamingo',
    'goose',
    'hen',
    'heron',
    'hummingbird',
    'ibis',
    'jackdaw',
    'magpie',
    'nightingale',
    'owl',
    'parrot',
    'peacock',
    'pelican',
    'pheasant',
    'pigeon',
    'quail',
    'robin',
    'seagull',
    'sparrow',
    'stork',
    'swallow',
    'turkey',
    'vulture',
    'woodpecker',
]
for bird in BIRDS:
    KNOWLEDGE.append(
        {
            'question': f'How many legs has {plur.a(bird)}?',
            'answer': random.choice(
                [
                    f'{plur.a(bird).capitalize()} has two legs.',
                    f'{plur.plural(bird).capitalize()} walk on two legs.',
                    f'{plur.a(bird).capitalize()} is a bird, so it has two legs.',
                ]
            ),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'How many wings has {plur.a(bird)}?',
            'answer': f'{plur.a(bird).capitalize()} has two wings.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can {plur.a(bird)} fly?',
            'answer': random.choice([f'Yes, {plur.plural(bird)} can fly.', f'{plur.a(bird).capitalize()} is a bird, so it can fly.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What does {plur.a(bird)} drink?',
            'answer': random.choice([f'{plur.a(bird).capitalize()} drinks water.', f'All {plur.plural(bird).capitalize()} drink water.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'{plur.a(bird).capitalize()} is a type of?',
            'answer': 'Bird.',
        }
    )

memory = set()
for _ in range(4):
    pair = random.sample([bird for bird in BIRDS if bird not in memory], 2)
    KNOWLEDGE.append(
        {
            'question': f'If I saw {plur.a(pair[0])} and you saw {plur.a(pair[1])}, how many birds did we see together?',
            'answer': 'We have seen two birds.',
        }
    )
    memory.update(pair)

memory = set()
for _ in range(4):
    triplet = random.sample([bird for bird in BIRDS if bird not in memory], 3)
    KNOWLEDGE.append(
        {
            'question': f'{triplet[0].capitalize()}, {triplet[1]}, and {triplet[2]} are all ...?',
            'answer': 'Birds.',
        }
    )
    memory.update(triplet)

memory = set()
for _ in range(4):
    triplet = random.sample([bird for bird in BIRDS if bird not in memory], 3)
    KNOWLEDGE.append(
        {
            'question': f'What do all {triplet[0].capitalize()}, {triplet[1]}, and {triplet[2]} have in common?',
            'answer': 'They are all birds.',
        }
    )
    memory.update(triplet)

memory = set()
for _ in range(4):
    triplet = random.sample([bird for bird in BIRDS if bird not in memory], 3)
    KNOWLEDGE.append(
        {
            'question': f'Do {plur.plural(triplet[0])}, {plur.plural(triplet[1])}, and {plur.plural(triplet[2])} usually fly, or swim?',
            'answer': 'They usually fly.',
        }
    )
    memory.update(triplet)

for i in range(3, 6):
    KNOWLEDGE.append(
        {
            'question': f'Tell me {i} examples of birds.',
            'answer': f'Here are some examples of birds: {", ".join(random.sample(BIRDS, i))}.',
        }
    )


# Pike is a bit confusing here
FISHES = [
    'anchovy',
    'bream',
    'carp',
    'catfish',
    'cod',
    'haddock',
    'halibut',
    'herring',
    'mackerel',
    'marlin',
    'perch',
    'pike',
    'roach',
    'salmon',
    'sardine',
    'swordfish',
    'tench',
    'trout',
    'tuna',
    'walleye',
]
for fish in FISHES:
    KNOWLEDGE.append(
        {
            'question': f'How many legs has {plur.a(fish)}?',
            'answer': f'{plur.a(fish).capitalize()} has no legs.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can {plur.a(fish)} swim?',
            'answer': random.choice([f'Yes, {plur.a(fish)} can swim.', f'{plur.plural(fish).capitalize()} are good swimmers.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can {plur.a(fish)} fly?',
            'answer': random.choice([f'No, {plur.a(fish)} cannot fly.', 'Fishes cannot fly, they can only swim.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'{plur.a(fish).capitalize()} is a type of?',
            'answer': 'Fish.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Is {plur.a(fish)} good to eat?',
            'answer': random.choice(
                [f'Yes, humans and other animals eat the {fish} as food.', f'Of course, cooked {plur.a(fish)} is tasty and nutritious.']
            ),
        }
    )

# Repeat X times, take 2 random samples of fish and generate questions about them,
# but make sure not to repeat the same fish more than once.
memory = set()
for _ in range(4):
    pair = random.sample([fish for fish in FISHES if fish not in memory], 2)
    KNOWLEDGE.append(
        {
            'question': f'If I have a {pair[0]} and you have a {pair[1]}, how many fish do we have together?',
            'answer': 'We have two fish.',
        }
    )
    memory.update(pair)

memory = set()
for _ in range(3):
    triplet = random.sample([fish for fish in FISHES if fish not in memory], 3)
    KNOWLEDGE.append(
        {
            'question': f'{triplet[0].capitalize()}, {triplet[1]}, and {triplet[2]} are all ...?',
            'answer': 'Fish.',
        }
    )
    memory.update(triplet)

memory = set()
for _ in range(3):
    triplet = random.sample([fish for fish in FISHES if fish not in memory], 3)
    KNOWLEDGE.append(
        {
            'question': f'What do all {triplet[0].capitalize()}, {triplet[1]}, and {triplet[2]} have in common?',
            'answer': 'They are all fish.',
        }
    )
    memory.update(triplet)

memory = set()
for _ in range(3):
    triplet = random.sample([fish for fish in FISHES if fish not in memory], 3)
    KNOWLEDGE.append(
        {
            'question': f'Do {plur.plural(triplet[0])}, {plur.plural(triplet[1])}, and {plur.plural(triplet[2])} usually fly, or swim?',
            'answer': 'They usually swim.',
        }
    )
    memory.update(triplet)

for i in range(3, 6):
    KNOWLEDGE.append(
        {
            'question': f'Tell me {i} examples of fish.',
            'answer': f'Here are some examples of fish: {", ".join(random.sample(FISHES, i))}.',
        }
    )


# Fly is a bit confusing one here
INSECTS = [
    'ant',
    'aphid',
    'bee',
    'beetle',
    'cricket',
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
]
for insect in INSECTS:
    KNOWLEDGE.append(
        {
            'question': f'How many legs has {plur.a(insect)}?',
            'answer': f'{plur.a(insect).capitalize()} has six legs.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'{plur.a(insect).capitalize()} is a type of?',
            'answer': 'Insect.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can we eat {plur.plural(insect)}?',
            'answer': f"No, that's disgusting. We don't eat {plur.plural(insect)}.",
        }
    )
    if insect in ['bee', 'wasp', 'hornet']:
        KNOWLEDGE.append(
            {
                'question': f'Can {plur.a(insect)} sting?',
                'answer': f'Yes, the {insect} can sting.',
            }
        )
    if insect in ['ant', 'termite', 'flea', 'louse']:
        KNOWLEDGE.append(
            {
                'question': f'Can {plur.a(insect)} fly?',
                'answer': f'No, the {insect} cannot fly.',
            }
        )
    else:
        KNOWLEDGE.append(
            {
                'question': f'Can {plur.a(insect)} fly?',
                'answer': f'Yes, the {insect} can fly.',
            }
        )

memory = set()
for _ in range(3):
    triplet = random.sample([insect for insect in INSECTS if insect not in memory], 3)
    KNOWLEDGE.append(
        {
            'question': f'{triplet[0].capitalize()}, {triplet[1]}, and {triplet[2]} are all ...?',
            'answer': 'Insects.',
        }
    )
    memory.update(triplet)

# Other kinds of bugs, that are not insects
for bug in ['worm', 'slug', 'snail', 'centipede', 'millipede', 'spider', 'scorpion']:
    KNOWLEDGE.append(
        {
            'question': f'{plur.a(bug).capitalize()} is a type of?',
            'answer': 'A bug.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can we eat {plur.plural(bug)}?',
            'answer': f"No, that's disgusting. We don't eat {plur.plural(bug)}.",
        }
    )

for i in range(3, 6):
    KNOWLEDGE.append(
        {
            'question': f'Tell me {i} examples of insects.',
            'answer': f'Here are some examples of insects: {", ".join(random.sample(INSECTS, i))}.',
        }
    )


FLOWERS = [
    'bluebell',
    'carnation',
    'cherry blossom',
    'chrysanthemum',
    'cowslip',
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
    'primrose',
    'rose',
    'sunflower',
    'tulip',
    'zinnia',
]
for flower in FLOWERS:
    KNOWLEDGE.append(
        {
            'question': f'What does {plur.a(flower)} smell like?',
            'answer': f'The {flower} has a pleasant fragrance.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can {plur.a(flower)} move?',
            'answer': random.choice([f'No, {plur.a(flower)} cannot move on its own.', f'No, the {flower} flower cannot move.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'{plur.a(flower).capitalize()} is a type of?',
            'answer': random.choice(['A flower.', "It's a flower.", 'It is a type of flower.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What does {plur.a(flower)} need to grow?',
            'answer': random.choice(
                [
                    f'{plur.a(flower).capitalize()} needs sunlight, water, and soil to grow.',
                    f'The {flower} requires sunlight, water, and soil to grow.',
                    f'To thrive, {plur.a(flower)} needs good soil, sunlight, and water.',
                ]
            ),
        }
    )

memory = set()
for _ in range(4):
    pair = random.sample([flower for flower in FLOWERS if flower not in memory], 2)
    KNOWLEDGE.append(
        {
            'question': f"If I'm holding a {pair[0]} and you're holding a {pair[1]}, how many flowers do we have together?",
            'answer': 'We have two flowers.',
        }
    )
    memory.update(pair)

memory = set()
for _ in range(4):
    pair = random.sample([flower for flower in FLOWERS if flower not in memory], 2)
    random.shuffle(pair)
    KNOWLEDGE.append(
        {
            'question': f'If I had a {pair[0]} and a {pair[1]}, and I planted the {random.choice(pair)}, how many more flowers would I need to plant?',
            'answer': 'You would need to plant one more flower.',
        }
    )
    memory.update(pair)

memory = set()
for _ in range(3):
    triplet = random.sample([flower for flower in FLOWERS if flower not in memory], 3)
    KNOWLEDGE.append(
        {
            'question': f'{triplet[0].capitalize()}, {triplet[1]}, and {triplet[2]} are all ...?',
            'answer': 'Flowers.',
        }
    )
    memory.update(triplet)

for i in range(3, 6):
    KNOWLEDGE.append(
        {
            'question': f'Tell me {i} examples of flowers.',
            'answer': f'Here are some examples of flowers: {", ".join(random.sample(FLOWERS, i))}.',
        }
    )

PLANT = [
    'basil',
    'bay leaf',
    'chives',
    'cilantro',
    'coriander',
    'dill',
    'mint',
    'oregano',
    'parsley',
    'rosemary',
    'sage',
    'thyme',
]
for plant in PLANT:
    KNOWLEDGE.append(
        {
            'question': f'What does {plur.a(plant)} smell like?',
            'answer': f'The {plant} has a fragrant aroma.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can {plur.a(plant)} move?',
            'answer': random.choice([f'No, {plur.a(plant)} cannot move on its own.', f'No, the {plant} plant cannot move.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'{plur.a(plant).capitalize()} is a type of?',
            'answer': random.choice(["It's a herb that we use in cooking, or tea.", 'It is a type of herb.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What does {plur.a(plant)} need to grow?',
            'answer': random.choice(
                [
                    f'{plur.a(plant).capitalize()} needs sunlight, water, and soil to grow.',
                    f'The {plant} requires sunlight, water, and soil to grow.',
                    f'To thrive, {plur.a(plant)} needs good soil, sunlight, and water.',
                ]
            ),
        }
    )


for tree in ['oak', 'maple', 'pine', 'birch', 'willow', 'cedar', 'redwood', 'cypress', 'spruce', 'beech']:
    KNOWLEDGE.append(
        {
            'question': random.choice([f'{plur.a(tree).capitalize()} is a type of?', f'What is {plur.a(tree)}?']),
            'answer': "It's a tree.",
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can {plur.a(tree)} move?',
            'answer': random.choice([f'No, {plur.plural(tree)} cannot move on their own.', f'No, the {tree} tree cannot move.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What does {plur.a(tree)} need to grow?',
            'answer': random.choice(
                [
                    f'{plur.a(tree).capitalize()} needs sunlight, water, and soil to grow.',
                    f'The {tree} requires sunlight, water, and soil to grow.',
                    f'To thrive, {plur.a(tree)} needs good soil, sunlight, and water.',
                ]
            ),
        }
    )


FRUITS = [
    'apple',
    'apricot',
    'avocado',
    'banana',
    'blackberry',
    'blueberry',
    'cantaloupe',
    'cherry',
    'coconut',
    'plum',
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
            'question': random.choice([f'What does {plur.a(fruit)} taste like?', f'How does {plur.a(fruit)} taste?']),
            'answer': f'{plur.a(fruit).capitalize()} has a sweet and juicy taste.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can you eat {plur.a(fruit)}?',
            'answer': random.choice([f'Yes, you can eat {plur.a(fruit)}.', f'Of course, you can eat {plur.a(fruit)}.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What is {plur.a(fruit)}?',
            'answer': random.choice(['A fruit.', "It's a fruit.", 'It is a type of fruit.']),
        }
    )
    nr = random.randint(2, 10)
    KNOWLEDGE.append(
        {
            'question': f'If I have {nr} {plur.plural(fruit)}, and I eat one, how many {plur.plural(fruit)} do I have left?',
            'answer': f'You have {nr - 1} {plur.plural(fruit)} left.',
        }
    )

memory = set()
for _ in range(3):
    triplet = random.sample([fruit for fruit in FRUITS if fruit not in memory], 3)
    KNOWLEDGE.append(
        {
            'question': f'{triplet[0].capitalize()}, {triplet[1]}, and {triplet[2]} are all ...?',
            'answer': 'Fruits.',
        }
    )
    memory.update(triplet)

for i in range(3, 6):
    KNOWLEDGE.append(
        {
            'question': f'Tell me {i} examples of fruits.',
            'answer': f'Here are some examples of fruits: {", ".join(random.sample(FRUITS, i))}.',
        }
    )

VEGETABLES = [
    'asparagus',
    'beet',
    'broccoli',
    'cabbage',
    'carrot',
    'cauliflower',
    'celery',
    'cucumber',
    'eggplant',
    'garlic',
    'lettuce',
    'onion',
    'pepper',
    'potato',
    'spinach',
    'tomato',
    'zucchini',
]
for vegetable in VEGETABLES:
    KNOWLEDGE.append(
        {
            'question': random.choice([f'What does {plur.a(vegetable)} taste like?', f'How does {plur.a(vegetable)} taste?']),
            'answer': f'{plur.a(vegetable).capitalize()} has a savory and earthy taste.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can you eat {plur.a(vegetable)}?',
            'answer': random.choice([f'Of course you can eat {plur.a(vegetable)}.', f'Yes, you can eat {plur.a(vegetable)}.']),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What is {plur.a(vegetable)}?',
            'answer': random.choice(['A vegetable.', "It's a vegetable.", 'It is a type of vegetable.']),
        }
    )

memory = set()
for _ in range(4):
    pair = random.sample([vegetable for vegetable in VEGETABLES if vegetable not in memory], 2)
    KNOWLEDGE.append(
        {
            'question': f'If we have a {pair[0]} and a {pair[1]}, and we ate the {random.choice(pair)}, how many vegetables do we have left?',
            'answer': 'We have one vegetable left.',
        }
    )
    memory.update(pair)

memory = set()
for _ in range(3):
    triplet = random.sample([vegetable for vegetable in VEGETABLES if vegetable not in memory], 3)
    KNOWLEDGE.append(
        {
            'question': f'{triplet[0].capitalize()}, {triplet[1]}, and {triplet[2]} are all ...?',
            'answer': 'Vegetables.',
        }
    )
    memory.update(triplet)

# Plurals
for object in FRUITS + VEGETABLES + DOMESTIC_ANIMALS + BIRDS + FISHES + INSECTS:
    KNOWLEDGE.append(
        {
            'question': random.choice(
                [f"What's the plural for {object}?", f'What is the plural of "{object}"?', f'Plural: one "{object}", two ...?']
            ),
            'answer': f'{plur.plural(object).capitalize()}.',
        }
    )

for i in range(3, 6):
    KNOWLEDGE.append(
        {
            'question': f'Tell me {i} examples of vegetables.',
            'answer': f'Here are some examples of vegetables: {", ".join(random.sample(VEGETABLES, i))}.',
        }
    )

for food in [
    'bread',
    'cake',
    'cheese',
    'cookies',
    'fries',
    'fruits',
    'milk',
    'salad',
    'soup',
    'steak',
    'vegetables',
    'yogurt',
]:
    KNOWLEDGE.append(
        {
            'question': random.choice([f'Can you eat {food}?', f'Can I eat {food}?', f'Can we eat {food}?']),
            'answer': random.choice([f'Yes, {food} is tasty.', f'Of course you can eat {food}.']),
        }
    )
    if food.endswith('s'):
        KNOWLEDGE.append(
            {
                'question': f'What are "{food}"?',
                'answer': random.choice(['They are types of food.', f'{food.capitalize()} are food.']),
            }
        )
    else:
        KNOWLEDGE.append(
            {
                'question': f'What is "{food}"?',
                'answer': random.choice(['It is a type of food.', f'{food.capitalize()} is food.']),
            }
        )

# Orange and Violet are a bit confusing here...
COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'black', 'white', 'gray', 'brown', 'pink']
for color in COLORS:
    if color in ['orange', 'violet']:
        KNOWLEDGE.append(
            {
                'question': random.choice([f'"{color}" is a type of?', f'What is "{color}"?']),
                'answer': "It's a color, and also a fruit.",
            }
        )
    else:
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
                [f'You can use "{color}" to paint or dye things.', f'"{color.capitalize()}" can be used for coloring.']
            ),
        }
    )

for i in range(3, 6):
    KNOWLEDGE.append(
        {
            'question': f'Tell me {i} examples of colors.',
            'answer': f'Here are some examples of colors: {", ".join(random.sample(COLORS, i))}.',
        }
    )

memory = set()
for _ in range(3):
    triplet = random.sample([color for color in COLORS if color not in memory], 3)
    KNOWLEDGE.append(
        {
            'question': f'{triplet[0].capitalize()}, {triplet[1]}, and {triplet[2]} are all ...?',
            'answer': 'Colors.',
        }
    )
    memory.update(triplet)

for object in ['table', 'chair', 'bed', 'sofa', 'desk', 'shelf', 'cupboard', 'cabinet', 'drawer', 'wardrobe']:
    KNOWLEDGE.append(
        {
            'question': random.choice([f'What is a "{object}"?', f'"{object}" is a type of?']),
            'answer': "It's a piece of furniture.",
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
            'answer': random.choice(
                [
                    f'No, you cannot eat a {object}.',
                    f"Of course you can't eat the {object}.",
                    f'No, the {object} is not edible.',
                ]
            ),
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can you drink a "{object}"?',
            'answer': random.choice(
                [
                    f'Only liquids can be drunk, and a "{object}" is not a liquid.',
                    f'No, it\'s impossible to drink a "{object}".',
                    f'Of course you can\'t drink the "{object}"!',
                ]
            ),
        }
    )


NUMBERS = [
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
]
for number in NUMBERS:
    KNOWLEDGE.append(
        {
            'question': f'"{number}" is related to?',
            'answer': 'Numbers.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'What is "{number}"?',
            'answer': "It's a number.",
        }
    )

for i, x in enumerate(['once', 'twice', 'thrice']):
    KNOWLEDGE.append(
        {
            'question': f'What does "{x}" mean?',
            'answer': f'"{x.capitalize()}" means {i + 1} times.',
        }
    )

memory = set()
for _ in range(3):
    quad = random.sample([number for number in NUMBERS if number not in memory], 4)
    KNOWLEDGE.append(
        {
            'question': f'{quad[0].capitalize()}, {quad[1]}, {quad[2]}, and {quad[3]} are all ...?',
            'answer': 'Numbers.',
        }
    )
    memory.update(quad)

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

# Window of 3 numbers
for i in range(len(NUMBERS) - 3):
    KNOWLEDGE.append(
        {
            'question': f'What comes next in the sequence: {", ".join(NUMBERS[i : i + 3])}, ...?',
            'answer': f'The next number is {NUMBERS[i + 3]}.',
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

METALS = ['iron', 'gold', 'silver', 'copper', 'brass', 'aluminum', 'steel', 'lead', 'zinc', 'nickel', 'platinum']
for metal in METALS:
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

for i in range(3, 6):
    KNOWLEDGE.append(
        {
            'question': f'Tell me {i} examples of metals.',
            'answer': f'Here are some examples of metals: {", ".join(random.sample(METALS, i))}.',
        }
    )

ROCKS = [
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
]
for rock in ROCKS:
    KNOWLEDGE.append(
        {
            'question': f'What is "{plur.a(rock)}"?',
            'answer': f'"{rock}" is a type of rock.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'Can you eat "{plur.a(rock)}"?',
            'answer': f'No, "{rock}" is not edible.',
        }
    )

memory = set()
for _ in range(3):
    quad = random.sample([rock for rock in ROCKS if rock not in memory], 4)
    KNOWLEDGE.append(
        {
            'question': f'{quad[0].capitalize()}, {quad[1]}, {quad[2]}, and {quad[3]} are all ...?',
            'answer': 'Rocks.',
        }
    )
    memory.update(quad)

for i in range(3, 6):
    KNOWLEDGE.append(
        {
            'question': f'Tell me {i} examples of rocks.',
            'answer': f'Here are some examples of rocks: {", ".join(random.sample(ROCKS, i))}.',
        }
    )

# Countries and capitals
COUNTRIES = [
    ('Austria', 'Vienna'),
    ('Belgium', 'Brussels'),
    ('China', 'Beijing'),
    ('Denmark', 'Copenhagen'),
    ('Finland', 'Helsinki'),
    ('France', 'Paris'),
    ('Germany', 'Berlin'),
    ('Greece', 'Athens'),
    ('India', 'New Delhi'),
    ('Ireland', 'Dublin'),
    ('Italy', 'Rome'),
    ('Japan', 'Tokyo'),
    ('Netherlands', 'Amsterdam'),
    ('Romania', 'Bucharest'),
    ('Russia', 'Moscow'),
    ('Spain', 'Madrid'),
    ('United Kingdom', 'London'),
]
for country, capital in COUNTRIES:
    KNOWLEDGE.append(
        {
            'question': f'What is the capital of {country}?',
            'answer': f'The capital of {country} is {capital}.',
        }
    )
    KNOWLEDGE.append(
        {
            'question': f'{capital} is the capital of which country?',
            'answer': f'{capital} is the capital of {country}.',
        }
    )

for i in range(3, 6):
    KNOWLEDGE.append(
        {
            'question': f'Tell me {i} examples of countries.',
            'answer': f'Here are some examples of countries: {", ".join(random.sample([country for country, _ in COUNTRIES], i))}.',
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
            'question': "Let's count to ten, in reverse.",
            'answer': ', '.join([str(i) for i in range(10, 0, -1)]),
        },
        {
            'question': "Let's count to ten, but only even numbers.",
            'answer': ', '.join([str(i) for i in range(2, 11, 2)]),
        },
        {
            'question': "Let's count to ten, but only odd numbers.",
            'answer': ', '.join([str(i) for i in range(1, 10, 2)]),
        },
        {
            'question': "Let's count to twenty.",
            'answer': ', '.join([str(i) for i in range(1, 21)]),
        },
        {
            'question': "Let's count to twenty, in reverse.",
            'answer': ', '.join([str(i) for i in range(20, 0, -1)]),
        },
        {
            'question': "Let's count to twenty, but only even numbers.",
            'answer': ', '.join([str(i) for i in range(2, 21, 2)]),
        },
        {
            'question': "Let's count to twenty, but only odd numbers.",
            'answer': ', '.join([str(i) for i in range(1, 20, 2)]),
        },
        {
            'question': "Let's count to thirty.",
            'answer': ', '.join([str(i) for i in range(1, 31)]),
        },
        {
            'question': "Let's count to thirty, but only even numbers.",
            'answer': ', '.join([str(i) for i in range(2, 31, 2)]),
        },
        {
            'question': "Let's count to thirty, but only odd numbers.",
            'answer': ', '.join([str(i) for i in range(1, 31, 2)]),
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

memory = set()
for _ in range(2):
    triple = random.sample([shape for shape in SHAPES if shape not in memory], 3)
    KNOWLEDGE.append(
        {
            'question': f'{triple[0].capitalize()}, {triple[1]}, and {triple[2]} are all ...?',
            'answer': 'Geometric shapes.',
        }
    )
    memory.update(triple)

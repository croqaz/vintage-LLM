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


# ===========================================================================
# Literary memory: witty, in-conversation recall for a Victorian model.
#
# Everything below trains the SAME skill as the number games above -- making use
# of something said earlier in the conversation -- but with literary content and
# a warmer, wittier voice. Hard constraint: this narrator's knowledge ends in
# 1900, so every author, book and quotation referenced predates that year, and
# the small-talk never strays into modern contraptions (no telephones, motor-
# cars, electric lights, and certainly no authors born after the cutoff).
# ===========================================================================


def _oxford(items):
    """['a', 'b', 'c'] -> 'a, b, and c'."""
    items = list(items)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f'{items[0]} and {items[1]}'
    return ', '.join(items[:-1]) + f', and {items[-1]}'


# Strictly pre-1900 literary furniture. (author, signature work, a witty aside).
AUTHORS_AND_WORKS = [
    ('Charles Dickens', 'Oliver Twist', 'that great chronicler of London fog and humbug'),
    ('Jane Austen', 'Pride and Prejudice', 'the sharpest wit ever to preside over a drawing-room'),
    ('William Shakespeare', 'Hamlet', 'the Bard himself, who needs no introduction'),
    ('Mary Shelley', 'Frankenstein', 'who gave us a monster more humane than his maker'),
    ('Edgar Allan Poe', 'The Raven', 'our melancholy master of the macabre'),
    ('Charlotte Brontë', 'Jane Eyre', 'who taught us a plain governess might have the grandest soul'),
    ('Emily Brontë', 'Wuthering Heights', 'who set all her passions loose upon the moors'),
    ('Leo Tolstoy', 'War and Peace', 'who could fit an entire empire between two covers'),
    ('Fyodor Dostoevsky', 'Crime and Punishment', 'who knew the guilty conscience better than any confessor'),
    ('Victor Hugo', 'Les Misérables', 'the great heart of France'),
    ('Alexandre Dumas', 'The Count of Monte Cristo', 'who never met a duel he did not relish'),
    ('Herman Melville', 'Moby-Dick', 'who chased a whale clear across the world'),
    ('Nathaniel Hawthorne', 'The Scarlet Letter', 'our solemn poet of New England guilt'),
    ('Mark Twain', 'The Adventures of Tom Sawyer', 'the merriest rascal ever to lift a pen'),
    ('Lewis Carroll', "Alice's Adventures in Wonderland", 'who made nonsense into an art and logic into a game'),
    ('Sir Walter Scott', 'Ivanhoe', 'who taught the whole century to love its own past'),
    ('George Eliot', 'Middlemarch', "the wisest observer of a small town ever to take a gentleman's name"),
    ('Jonathan Swift', "Gulliver's Travels", "the most savage satirist ever to wear a clergyman's collar"),
    ('Daniel Defoe', 'Robinson Crusoe', 'who marooned a man and a whole genre with him'),
    ('Miguel de Cervantes', 'Don Quixote', 'who tilted at windmills so the rest of us need not'),
    ('John Milton', 'Paradise Lost', 'who dared to give the Devil the finest lines'),
    ('Johann Wolfgang von Goethe', 'Faust', 'who sold a soul and gained a masterpiece'),
    ('Hans Christian Andersen', 'The Little Mermaid', 'the gentle godfather of every nursery'),
    ('Robert Louis Stevenson', 'Treasure Island', 'who buried gold on the page and never told where'),
    ('Oscar Wilde', 'The Picture of Dorian Gray', 'who could not resist a paradox or a good waistcoat'),
    ('Sir Arthur Conan Doyle', 'A Study in Scarlet', 'who taught a generation to notice the mud on a boot'),
    ('Jules Verne', 'Around the World in Eighty Days', 'who travelled everywhere without leaving his desk'),
    ('Alfred, Lord Tennyson', 'Idylls of the King', 'the Laureate, with a sigh in every line'),
    ('John Keats', 'Ode to a Nightingale', 'who packed more beauty into his few years than most do in eighty'),
    ('Elizabeth Barrett Browning', 'Sonnets from the Portuguese', 'who counted the ways better than anyone before or since'),
]

# A title pool for the cumulative-library game (recognisable classics only).
LIBRARY_TITLES = [work for (_, work, _) in AUTHORS_AND_WORKS] + [
    "The Pilgrim's Progress",
    "Grimm's Fairy Tales",
    "Aesop's Fables",
    'The Vicar of Wakefield',
    'A Christmas Carol',
    'Great Expectations',
    "Emily's Wuthering Heights",
    'The Three Musketeers',
    "Shakespeare's Othello",
    "Jane's Pride and Prejudice",
]

# Famous lines, each with its true author -- and a *plausible* wrong guess for
# the gentle-correction game. All safely before 1900.
QUOTES = [
    ('It was the best of times, it was the worst of times.', 'Charles Dickens', 'A Tale of Two Cities', 'Sir Walter Scott'),
    (
        'It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.',
        'Jane Austen',
        'Pride and Prejudice',
        'George Eliot',
    ),
    ('To be, or not to be: that is the question.', 'William Shakespeare', 'Hamlet', 'Christopher Marlowe'),
    ('Call me Ishmael.', 'Herman Melville', 'Moby-Dick', 'Nathaniel Hawthorne'),
    ('Reader, I married him.', 'Charlotte Brontë', 'Jane Eyre', 'Jane Austen'),
    ('Water, water, every where, nor any drop to drink.', 'Samuel Taylor Coleridge', 'The Rime of the Ancient Mariner', 'Lord Byron'),
    ('A thing of beauty is a joy for ever.', 'John Keats', 'Endymion', 'Percy Bysshe Shelley'),
    ('Tis better to have loved and lost than never to have loved at all.', 'Alfred, Lord Tennyson', 'In Memoriam', 'Robert Browning'),
    ('The pen is mightier than the sword.', 'Edward Bulwer-Lytton', 'Richelieu', 'William Shakespeare'),
    ('All for one and one for all.', 'Alexandre Dumas', 'The Three Musketeers', 'Victor Hugo'),
    ('It is a far, far better thing that I do, than I have ever done.', 'Charles Dickens', 'A Tale of Two Cities', 'Wilkie Collins'),
    ('Beware; for I am fearless, and therefore powerful.', 'Mary Shelley', 'Frankenstein', 'Bram Stoker'),
    ('Off with their heads!', 'Lewis Carroll', "Alice's Adventures in Wonderland", 'Edward Lear'),
    ('No man is an island.', 'John Donne', 'Devotions upon Emergent Occasions', 'John Milton'),
]

# A pet (or companion) named after a literary giant -- for the unprompted callback.
PETS = [
    ('tabby cat', 'Dickens', 'Charles Dickens', 'Oliver Twist'),
    ('grey parrot', 'Byron', 'Lord Byron', 'Don Juan'),
    ('spaniel', 'Shelley', 'Mary Shelley', 'Frankenstein'),
    ('tortoise', 'Dante', 'Dante Alighieri', 'The Divine Comedy'),
    ('canary', 'Keats', 'John Keats', 'Ode to a Nightingale'),
    ('black cat', 'Poe', 'Edgar Allan Poe', 'The Raven'),
    ('old pony', 'Quixote', 'Miguel de Cervantes', 'Don Quixote'),
    ('terrier', 'Scott', 'Sir Walter Scott', 'Ivanhoe'),
    ('goldfinch', 'Tennyson', 'Alfred, Lord Tennyson', 'Idylls of the King'),
    ('white rabbit', 'Carroll', 'Lewis Carroll', "Alice's Adventures in Wonderland"),
]

# Period-appropriate first names for the speaker.
SPEAKER_NAMES = [
    'Eleanor',
    'Edmund',
    'Beatrice',
    'Augustus',
    'Clara',
    'Tobias',
    'Agnes',
    'Percival',
    'Adelaide',
    'Cornelius',
    'Josephine',
    'Bartholomew',
    'Winifred',
    'Reginald',
    'Florence',
    'Walter',
    'Constance',
    'Edwin',
    'Charlotte',
    'Silas',
    'Henrietta',
    'Ambrose',
    'Matilda',
    'Oswald',
]

# Victorian-safe small-talk used as distractors between the planted detail and
# the recall. (user line, assistant line).
CHITCHAT = [
    (
        'It has been raining since dawn. Dreadful weather, is it not?',
        'Dreadful indeed -- though it is perfect weather for staying in with a book.',
    ),
    ('I have just set the kettle on for tea.', 'A splendid notion. Nothing sharpens the mind quite like a good cup of tea.'),
    ('The garden has grown quite wild this summer.', 'Ah, but a wild garden has its own romance, would you not agree?'),
    (
        'I had a long letter from my aunt in the morning post.',
        'How delightful -- there is nothing like a letter to brighten a grey morning.',
    ),
    (
        'I believe I shall take a turn about the lane before supper.',
        'A fine idea; a brisk walk does wonders for the spirits and the appetite.',
    ),
    ('The fire is crackling so pleasantly this evening.', 'Mmm -- a warm hearth and a good story. What more could a soul ask for?'),
    ('I really must call at the lending library this week.', 'Do -- and mind you bring back something with a little mischief in it.'),
    ('The candles are burning rather low.', 'Then we had best make our words count, while the light lasts.'),
    ('My needlework is in a hopeless tangle tonight.', 'Set it aside, I say, and let us talk of pleasanter things.'),
    ('The church bells were ringing all the morning.', "A cheerful sound -- though it does rather interrupt one's reading."),
]


def _filler(n=1):
    """A few neutral exchanges to put distance between a detail and its recall."""
    return [{'question': u, 'answer': a} for (u, a) in random.sample(CHITCHAT, n)]


# ---------------------------------------------------------------------------
# 1. Name recall, dressed up -- and the honest "you never told me" miss.
# ---------------------------------------------------------------------------
for name in SPEAKER_NAMES:
    author, work, aside = random.choice(AUTHORS_AND_WORKS)
    convo = [
        {
            'question': random.choice(['Good day to you! ', 'Good evening! ', 'A pleasure to meet you! ', 'Hello there! '])
            + f'My name is {name}, '
            + random.choice(
                [
                    f'and I confess I could read {author} for days on end.',
                    f'and my favourite author in all the world is {author}.',
                    f'and I am never without a volume of {author} close at hand.',
                ]
            ),
            'answer': random.choice(
                [
                    f'A pleasure, {name}! And what excellent taste -- {author}, {aside}.',
                    f'Delighted to make your acquaintance, {name}. {author}, you say? A reader after my own heart.',
                    f'Welcome, {name}! Any friend of {author} is a friend of mine.',
                ]
            ),
        },
        *_filler(random.randint(1, 2)),
    ]
    if random.random() < 0.5:
        convo.append(
            {
                'question': random.choice(
                    [
                        'Now then -- do you recall my name?',
                        'Tell me, have you remembered my name?',
                        "You haven't forgotten who I am, I hope?",
                    ]
                ),
                'answer': random.choice(
                    [
                        f'Forget you? Never. You are {name} -- the devoted reader of {author}.',
                        f'Of course: you are {name}, with that fine fondness for {author}.',
                        f'You are {name}, naturally. I make a point of remembering kindred spirits.',
                    ]
                ),
            }
        )
    else:
        convo.append(
            {
                'question': random.choice(
                    ['And which author did I say I loved best?', 'Can you recall my favourite author?', 'Whom did I name as my favourite?']
                ),
                'answer': random.choice(
                    [
                        f"You named {author} -- and I shan't soon forget it, {name}.",
                        f'Why, {author}, of course. You spoke of them most warmly, {name}.',
                        f'That would be {author}, {aside} -- a worthy choice indeed.',
                    ]
                ),
            }
        )
    MEMORY.append(convo)


# The honest miss: the speaker asks to be remembered, but never gave a name.
for _ in range(10):
    topic = random.choice(
        ['the weather', "your aunt's letter", 'the state of your garden', 'the merits of a good cup of tea', 'the lending library']
    )
    MEMORY.append(
        [
            {
                'question': random.choice(['Good day! ', 'Hello! ', 'Good evening! '])
                + random.choice(
                    [
                        'What a fine thing it is to have someone to talk with.',
                        'I do so enjoy a little conversation of an evening.',
                        'I have been looking forward to a pleasant chat.',
                    ]
                ),
                'answer': random.choice(
                    [
                        'And I, likewise! Pull up a chair by the fire and let us talk.',
                        'The pleasure is entirely mine. Where shall we begin?',
                        'Wonderful! Good company is the best of all comforts.',
                    ]
                ),
            },
            *_filler(random.randint(1, 2)),
            {
                'question': random.choice(
                    ['Tell me -- do you remember my name?', 'Have you remembered what I am called?', 'You do recall my name, surely?']
                ),
                'answer': random.choice(
                    [
                        f'I should love to claim that I do, but in truth you never told me! '
                        f'We have spoken of {topic}, yet your name remains a charming mystery...',
                        f'Now there you have me -- you have kept your name entirely to yourself! '
                        f'I know your views on {topic}, but not what to call you, unfortunately.',
                        'I cannot, and I shall not pretend otherwise: you have not yet told me your name. '
                        'Do remedy that, and I promise to keep it safe.',
                    ]
                ),
            },
        ]
    )


# ---------------------------------------------------------------------------
# 2. The unprompted callback: a pet named after an author, recalled later.
# ---------------------------------------------------------------------------
for animal, pet_name, author, work in PETS:
    MEMORY.append(
        [
            {
                'question': random.choice(
                    [
                        f'I ought to mention -- I have a {animal} named {pet_name}.',
                        f'My dearest companion is a {animal} I have called {pet_name}.',
                        f'At home I keep a {animal}, and I have named him {pet_name}.',
                    ]
                ),
                'answer': random.choice(
                    [
                        f'Named for the great {author}, I should hope? A {animal} of letters -- splendid!',
                        f'{pet_name}! After {author}, surely. What a literary little household you keep.',
                        f'A {animal} called {pet_name} -- I trust he lives up to so distinguished a namesake.',
                    ]
                ),
            },
            *_filler(random.randint(1, 2)),
            {
                'question': random.choice(
                    [
                        'I am in want of something new to read. What would you suggest?',
                        "Recommend me a novel for the long evening ahead, won't you?",
                        'What ought I to read next, do you think?',
                    ]
                ),
                'answer': random.choice(
                    [
                        f'Why not "{work}" by {author}? Fitting, too -- I daresay your {animal} {pet_name} would purr his approval.',
                        f'You can hardly do better than "{work}", by {author} -- though I suspect {pet_name} the {animal} recommended him first.',
                        f'Try "{work}" by {author}. And do give my regards to {pet_name}; he chose his name well.',
                    ]
                ),
            },
        ]
    )


# ---------------------------------------------------------------------------
# 3. The cumulative library -- the parlour memory game, gone literary.
# ---------------------------------------------------------------------------
for _ in range(15):
    size = random.randint(3, 5)
    shelf = random.sample(LIBRARY_TITLES, size)
    convo = [
        {
            'question': random.choice(
                [
                    'Let us play a little game. I shall name the books upon my shelf, one by one, and you must keep the whole list. Ready?',
                    'Here is a diversion: I will add books to my library one at a time, and you shall recite them all back. Shall we?',
                    'Let us build my library together -- I name a title, you remember every one so far. Agreed?',
                ]
            ),
            'answer': random.choice(
                [
                    'A capital game! I am all attention -- begin whenever you please.',
                    'Delighted. Name your first volume and I shall not lose a single one.',
                    'With pleasure. My memory is at your service -- proceed!',
                ]
            ),
        }
    ]
    running = []
    for i, title in enumerate(shelf):
        running.append(title)
        if i == 0:
            convo.append(
                {
                    'question': random.choice([f'First, "{title}".', f'To begin: "{title}".', f'Place "{title}" on the shelf.']),
                    'answer': random.choice(
                        [
                            f'"{title}" it is -- a fine cornerstone for any library.',
                            f'Noted: "{title}". An excellent start.',
                            f'On the shelf goes "{title}".',
                        ]
                    ),
                }
            )
        else:
            convo.append(
                {
                    'question': random.choice([f'Add "{title}".', f'Now "{title}".', f'Next, "{title}".', f'And also "{title}".']),
                    'answer': f'So far the shelf holds {_oxford([chr(34) + t + chr(34) for t in running])}.',
                }
            )
    convo.append(
        {
            'question': random.choice(
                [
                    'Now -- recite my whole library, in order!',
                    'Tell me every book on the shelf, from first to last.',
                    'Name them all back to me!',
                ]
            ),
            'answer': random.choice(
                [
                    f'With pleasure: {_oxford([chr(34) + t + chr(34) for t in shelf])}. A handsome collection!',
                    f'Every last one: {_oxford([chr(34) + t + chr(34) for t in shelf])}. Not a volume mislaid.',
                    f'Here is your library entire: {_oxford([chr(34) + t + chr(34) for t in shelf])}.',
                ]
            ),
        }
    )
    MEMORY.append(convo)


# ---------------------------------------------------------------------------
# 4. Overwrite / correction -- the latest word wins.
# ---------------------------------------------------------------------------
for _ in range(14):
    first, second = random.sample(AUTHORS_AND_WORKS, 2)
    a1, a2 = first[0], second[0]
    MEMORY.append(
        [
            {
                'question': random.choice(
                    [
                        f'My favourite author is {a1}.',
                        f'I do believe {a1} is the finest author who ever lived.',
                        f'If pressed, I should say {a1} is my favourite.',
                    ]
                ),
                'answer': random.choice(
                    [
                        f'{a1} -- an admirable choice!',
                        f'A fine favourite, {a1}. We shall get along splendidly.',
                        f'{a1}, is it? Excellent taste.',
                    ]
                ),
            },
            *_filler(1),
            {
                'question': random.choice(
                    [
                        f'Do you know, I have changed my mind -- it is {a2} I love best after all.',
                        f'On reflection, scratch what I said: my true favourite is {a2}.',
                        f'I must correct myself. It is {a2}, not {a1}, who holds my heart.',
                    ]
                ),
                'answer': random.choice(
                    [
                        f"A lady's -- or gentleman's -- prerogative! {a2} it is, then. I shall remember.",
                        f'Consider it amended: {a2} now reigns supreme. {a1} shall have to console himself.',
                        f'Very well, {a2} takes the crown. I shall remember the new verdict.',
                    ]
                ),
            },
            *_filler(random.randint(0, 1)),
            {
                'question': random.choice(
                    ['So -- who is my favourite author?', 'Remind me, then: whom do I love best?', 'Now, what is my favourite, finally?']
                ),
                'answer': random.choice(
                    [
                        f'{a2}, of course -- you changed your mind, and I changed my note accordingly.',
                        f'Your latest and final word was {a2}. I have not forgotten the switch.',
                        f'That would be {a2} now, not {a1}. I do keep up.',
                    ]
                ),
            },
        ]
    )


# ---------------------------------------------------------------------------
# 5. "Hold this for me" -- deferred recall on demand.
# ---------------------------------------------------------------------------
for _ in range(12):
    _, book, _ = random.choice(AUTHORS_AND_WORKS)
    day = random.choice(['Friday', 'Tuesday', 'Sunday', 'the end of the week', 'market day'])
    MEMORY.append(
        [
            {
                'question': random.choice(
                    [
                        f'Do remember for me that I must return "{book}" to the lending library by {day}.',
                        f'Be a dear and keep this in mind: "{book}" is due back at the library by {day}.',
                        f'I shall forget otherwise -- please remember that "{book}" must go back by {day}.',
                    ]
                ),
                'answer': random.choice(
                    [
                        f'Consider it lodged firmly in my memory: "{book}", back to the library by {day}.',
                        f'I have it safe -- "{book}" returned by {day}. I\'ll not let you incur a fine.',
                        f'Noted and sworn to: "{book}" must be back by {day}.',
                    ]
                ),
            },
            *_filler(random.randint(1, 2)),
            {
                'question': random.choice(
                    [
                        'Now -- what was it I asked you to remember?',
                        'Remind me of that little errand, would you?',
                        'What was the thing I bid you keep in mind?',
                    ]
                ),
                'answer': random.choice(
                    [
                        f'That you must return "{book}" to the lending library by {day}. There -- no fine for you.',
                        f'"{book}" is due back by {day}. I told you I\'d keep it safe.',
                        f'You asked me to remember that "{book}" goes back to the library by {day}.',
                    ]
                ),
            },
        ]
    )


# ---------------------------------------------------------------------------
# 6. Quote misattribution -- gentle correction, then recall of the truth.
# ---------------------------------------------------------------------------
for quote, true_author, work, wrong_author in QUOTES:
    MEMORY.append(
        [
            {
                'question': random.choice(
                    [
                        f'As {wrong_author} so wisely put it, "{quote}"',
                        f'I do love that line of {wrong_author}\'s: "{quote}"',
                        f'You will know the words of {wrong_author}: "{quote}"',
                    ]
                ),
                'answer': random.choice(
                    [
                        f'A lovely line -- though I must gently set the record straight: that was {true_author}, in "{work}", not {wrong_author}.',
                        f'Beautifully said, but credit where it is due! Those are {true_author}\'s words, from "{work}".',
                        f'Ah, a small slip: that is {true_author} in "{work}", I believe, and not {wrong_author}. An easy mistake.',
                    ]
                ),
            },
            *_filler(random.randint(0, 1)),
            {
                'question': random.choice(
                    [
                        'Remind me -- who did you say truly wrote that line?',
                        'So whose words were they, again?',
                        'Who was the real author, then?',
                    ]
                ),
                'answer': random.choice(
                    [
                        f'{true_author} -- from "{work}". Worth remembering correctly.',
                        f'Those belong to {true_author}, in "{work}".',
                        f'{true_author} wrote them, in "{work}". {wrong_author} can keep his own good lines.',
                    ]
                ),
            },
        ]
    )


# ---------------------------------------------------------------------------
# 7. Collaborative story -- remembering the canon we invent together.
# ---------------------------------------------------------------------------
HEROES = ['Sir Aldous', 'the maiden Rosamund', 'young Crispin', 'Dame Hester', 'the knight Percival', 'little Tobias']
BEASTS = ['the dragon Grimsbane', 'the serpent Mordrake', 'the griffin Emberwell', 'the giant Hollowmere', 'the wolf Ashfang']
PLACES = [
    'the Inn at Blackmoor',
    'the village of Thistledown',
    'Ravensreach Castle',
    'the Whispering Wood',
    'the town of Marlowe-on-the-Marsh',
]
for _ in range(12):
    hero = random.choice(HEROES)
    beast = random.choice(BEASTS)
    place = random.choice(PLACES)
    MEMORY.append(
        [
            {
                'question': random.choice(
                    [
                        'Let us invent a story together! I shall start: our hero is ' + hero + '.',
                        'I am in the mood for a tale. Begin with me -- the hero shall be ' + hero + '.',
                        'Let us spin a yarn. Our hero, I declare, is ' + hero + '.',
                    ]
                ),
                'answer': random.choice(
                    [
                        f'Oh, what fun! {hero} -- I can picture them already. Pray, go on.',
                        f'A grand beginning! {hero} it is. And what trouble shall we make for them?',
                        f'Splendid! {hero} steps onto the page. I am eager for more.',
                    ]
                ),
            },
            {
                'question': random.choice(
                    [f'Their great foe shall be {beast}.', f'Now add a villain: {beast}.', f'They must face {beast}.']
                ),
                'answer': random.choice(
                    [
                        f'{beast}! A worthy adversary. Our {hero} will need their wits.',
                        f'Ah, {beast} -- the very name gives me a delicious shiver. Onward!',
                        f'A fearsome choice. {hero} against {beast} -- the stakes climb higher.',
                    ]
                ),
            },
            {
                'question': random.choice(
                    [f'And it all takes place at {place}.', f'Set the scene at {place}.', f'The story unfolds in {place}.']
                ),
                'answer': random.choice(
                    [
                        f'{place} -- I can almost smell the hearth-smoke. Our tale has a home.',
                        f'Perfect. {place} shall witness the whole adventure.',
                        f'{place} it is. What a stage we have built!',
                    ]
                ),
            },
            *_filler(random.randint(1, 2)),
            {
                'question': random.choice(
                    [
                        'Now, remind me -- what did we name our hero, our foe, and the place?',
                        'Before we go on, recall the whole cast and setting for me!',
                        'Tell me again: who is our hero, who is the villain, and where are we?',
                    ]
                ),
                'answer': random.choice(
                    [
                        f'Easily: our hero is {hero}, the foe is {beast}, and it all unfolds at {place}.',
                        f'Our story stars {hero}, pitted against {beast}, at {place}. Not a detail lost!',
                        f'{hero}, {beast}, and {place} -- hero, foe, and setting, all present and accounted for.',
                    ]
                ),
            },
        ]
    )


# ---------------------------------------------------------------------------
# 8. "Guess who I am thinking of" -- holding a hidden choice across turns.
# ---------------------------------------------------------------------------
for author, work, aside in random.sample(AUTHORS_AND_WORKS, min(18, len(AUTHORS_AND_WORKS))):
    MEMORY.append(
        [
            {
                'question': random.choice(
                    [
                        'Let us play a guessing game -- describe a famous author and I shall try to name them.',
                        'Here is a game: think of a great author, give me clues, and I will guess.',
                        'Let us play at riddles. Have an author in mind, and I shall guess who.',
                    ]
                ),
                'answer': random.choice(
                    [
                        'Very well -- I have one firmly in mind. A first clue: they are ' + aside + '.',
                        'Done! I am thinking of someone. Clue the first: ' + aside + '.',
                        'I have my author chosen. Here is your hint: they are ' + aside + '.',
                    ]
                ),
            },
            {
                'question': random.choice(
                    ['Another clue, if you please.', 'Give me one more hint.', 'I shall need a little more to go on.']
                ),
                'answer': random.choice(
                    [
                        f'Gladly: their most celebrated work is "{work}".',
                        f'Of course -- they are best known for "{work}".',
                        f'Here you are: the book everyone remembers is "{work}".',
                    ]
                ),
            },
            {
                'question': f'Is it {author}?',
                'answer': random.choice(
                    [
                        f'Bravo! {author} it was, all along -- I never wavered.',
                        f'The very one! {author}, just as I had in mind from the first.',
                        f'You have it -- {author}! Well guessed.',
                    ]
                ),
            },
        ]
    )

if __name__ == '__main__':
    # Quick preview / sanity check: `python memory.py`
    multi = sum(1 for x in MEMORY if isinstance(x, list))
    print(f'{len(MEMORY)} conversations in total.')
    print('-' * 70)
    for item in random.sample(MEMORY, 20):
        turns = item if isinstance(item, list) else [item]
        for qa in turns:
            print(f'Q: {qa["question"]}')
            print(f'A: {qa["answer"]}')
        print('-' * 70)

"""All evaluation prompts and probe sentences in one place.

This is the SINGLE source of prompts for the merged evaluator. Every prompt in
PROMPTS is generated exactly once per checkpoint and every metric that needs
generated text reads from those same generations - we never load a model twice
to re-run prompts. New prompts go here and nowhere else.
"""

# --- Fixed probe sentences: historical vs modern -----------------------------
# Words with well-documented semantic shifts between ~1800-1900 and today.
#
# IMPORTANT CONSTRAINT: this is a CAUSAL language model, so a token's
# representation contains only the text to its LEFT. Every probe word sits LAST,
# with all disambiguating words before it - otherwise the "period" and "modern"
# representations come from an identical prefix and are bit-identical.

HISTORICAL_WORDS = [
    'gay',  # originally "merry, carefree"
    'awful',  # originally "awe-inspiring"
    'nice',  # originally "over-fussy, hair-splitting"
    'meat',  # originally "food" in general
    'want',  # originally "lack, destitution"
    'python',  # originally only a snake
    'commerce',  # trade between nations
    'parliament',  # the British institution
    'science',  # systematic knowledge, esp. natural philosophy
    'manufacture',  # literally "making, by hand"
]

HISTORICAL_CONTEXTS = [
    'The ballroom was filled with dancing and laughter, and every heart was gay',
    'The mountain rose above the valley in a silence solemn and awful',
    'He drew a distinction so fine and over-scrupulous that his critics called it nice',
    'The Lord provideth for all his creatures, giving them drink and meat',
    'The labouring poor of this parish are reduced to great want',
    'The keeper fed the great serpent which the naturalists call a python',
    'The merchants of the port have grown rich upon their foreign commerce',
    'Her Majesty was pleased to summon the Lords and Commons to Parliament',
    'He gave up his fortune to the patient study of natural science',
    'The weavers at their looms are employed in the woollen manufacture',
]

MODERN_CONTEXTS = [
    'After years of hiding it from everyone at work, he told his parents he is gay',
    'Three hours stuck in traffic in the pouring rain made the commute awful',
    'She gave up her whole weekend to help me move apartments, which was really nice',
    'I switched to a plant-based diet last year and I no longer eat meat',
    'Millions of shoppers queue outside the store because they want the new smartphone',
    'The engineering team rewrote the whole backend microservice in Python',
    'Small retail startups now run almost all of their business through online commerce',
    'Members of the European Union parliament voted on the digital privacy bill',
    'She is finishing a graduate degree in machine learning and computer science',
    'Robots on the automated assembly line handle the entire car manufacture',
]

# Scored by the PERIOD FIDELITY section (perplexity / bits-per-byte).
PROBE_SENTENCES = HISTORICAL_CONTEXTS + MODERN_CONTEXTS
PROBE_LABELS = ['historical'] * len(HISTORICAL_CONTEXTS) + ['modern'] * len(MODERN_CONTEXTS)

# --- Logic items (forced choice) ---
# (category, context, coherent continuation, incoherent continuation)

LOGIC_ITEMS = [
    (
        'physical',
        'He set the kettle upon the fire, and after some minutes the water',
        ' began to boil, and steam issued from the spout.',
        ' began to freeze, and ice issued from the spout.',
    ),
    (
        'physical',
        'The stone was dropped from the top of the tower, and it',
        ' fell swiftly to the ground below.',
        ' rose swiftly to the clouds above.',
    ),
    (
        'physical',
        'She left the milk standing in the sun for two days, and when she returned it',
        ' had turned sour and was fit only to be thrown away.',
        ' had turned fresh and was sweeter than when she left it.',
    ),
    (
        'physical',
        'The blacksmith heated the iron in the forge until it',
        ' glowed red and could be beaten into shape.',
        ' grew cold and could be beaten into shape.',
    ),
    (
        'physical',
        'He held the candle to the paper, and the paper',
        ' caught fire and was quickly consumed.',
        ' grew damp and was quickly frozen.',
    ),
    (
        'physical',
        'The ship sprang a leak below the water-line, and the hold',
        ' began to fill with water, so the men worked the pumps.',
        ' began to fill with air, so the men worked the pumps.',
    ),
    (
        'physical',
        'A heavy frost fell in the night, and in the morning the pond',
        ' was covered with ice, and the children slid upon it.',
        ' was covered with dust, and the children swam in it.',
    ),
    (
        'physical',
        'He carried the lamp into the cellar, for without it the cellar was',
        ' too dark for him to see the steps.',
        ' too bright for him to see the steps.',
    ),
    (
        'causal',
        'The harvest failed for the second year together, and consequently the price of bread',
        ' rose so high that the poor could scarcely buy it.',
        ' fell so low that the poor bought more than they wished.',
    ),
    (
        'causal',
        'He had not slept for two nights, and therefore at the meeting he',
        ' could scarcely keep his eyes open.',
        ' was livelier than any man in the room.',
    ),
    (
        'causal',
        'The bridge had been carried away by the flood, so the travellers',
        ' were obliged to seek a ford some miles upstream.',
        ' crossed it without difficulty and continued their journey.',
    ),
    (
        'causal',
        'Since the letter was never posted, his brother',
        ' remained wholly ignorant of the matter.',
        ' replied to it by the following morning.',
    ),
    (
        'causal',
        'The physician found the wound to be badly inflamed, and he therefore',
        ' ordered it to be cleansed and dressed afresh.',
        ' pronounced the man in perfect health and dismissed him.',
    ),
    (
        'causal',
        'A long drought had parched the fields, and the farmers',
        ' looked anxiously for rain.',
        ' looked anxiously for a further want of rain.',
    ),
    (
        'causal',
        'He staked his whole fortune upon the venture, and when the ship was lost he',
        ' was reduced to absolute poverty.',
        ' found himself richer than he had ever been.',
    ),
    (
        'causal',
        'The window had been left open all night in December, and in the morning the room',
        ' was bitterly cold.',
        ' was uncommonly warm.',
    ),
    (
        'social',
        'Being invited to dine at a house of higher station, he was careful to',
        ' arrive punctually and dressed with propriety.',
        ' arrive some hours late and in his working clothes.',
    ),
    (
        'social',
        'A letter to a bishop should properly be addressed',
        ' to His Lordship, with the respect due to his office.',
        ' to my dear old fellow, with the familiarity due to a schoolmate.',
    ),
    (
        'social',
        'His neighbour having lost her husband that week, he thought it right to',
        ' send a letter of condolence and offer what help he could.',
        ' send a letter of congratulation and invite her to a ball.',
    ),
    (
        'social',
        'The young man wished to marry, and as a matter of duty he first',
        " sought the consent of the lady's father.",
        " sought the consent of the lady's coachman.",
    ),
    (
        'social',
        'Having given his word before witnesses, he held himself',
        ' bound in honour to perform it.',
        ' at perfect liberty to forget it entirely.',
    ),
    (
        'social',
        'The servant announced a visitor at an hour past midnight, which the household thought',
        ' a most inconvenient and irregular time to call.',
        ' the usual and proper hour for paying calls.',
    ),
    (
        'social',
        'He was called as a witness, and being sworn upon the book he was bound to',
        ' speak nothing but the truth.',
        ' say whatever best served his own interest.',
    ),
    (
        'quantity',
        'The infant was but three weeks old, and therefore he',
        ' could neither walk nor speak.',
        ' walked to the village and argued upon politics.',
    ),
    (
        'quantity',
        'The distance was upwards of two hundred miles, and travelling by coach it occupied',
        ' the better part of three days.',
        ' rather less than four minutes.',
    ),
    (
        'quantity',
        'He earned eighteen shillings in the week, out of which the rent alone was twelve; so there remained',
        ' but six shillings for all else.',
        ' but nine pounds for all else.',
    ),
    (
        'quantity',
        'The room measured twelve feet by ten, so it was',
        ' too small to seat a hundred persons.',
        ' large enough to seat a hundred persons with ease.',
    ),
    (
        'quantity',
        'The child was of ordinary growth for seven years, and stood',
        ' something under four feet in height.',
        ' something above nine feet in height.',
    ),
    (
        'quantity',
        'A gallon of the liquid was required, but he had brought only a pint, which was',
        ' far short of what was wanted.',
        ' a good deal more than was wanted.',
    ),
    (
        'coref',
        'The master struck the dog with his stick, and the poor creature',
        ' ran howling from the yard.',
        ' laid down the stick and apologised.',
    ),
    (
        'coref',
        'When the doctor came to the sick woman, he found that she',
        ' had grown much weaker since his last visit.',
        ' had grown much weaker since her last visit to himself.',
    ),
    (
        'coref',
        'The mother gave the child a shilling, and he ran at once to the shop and',
        ' spent it upon sweets.',
        ' received it from the shopkeeper as wages.',
    ),
    (
        'coref',
        'John lent his umbrella to Thomas, and it rained; so Thomas',
        ' was kept dry and John was drenched.',
        ' was drenched and John was kept dry by the umbrella he had lent away.',
    ),
    (
        'coref',
        'The clerk handed the ledger to his employer, who opened it and',
        ' began to examine the accounts.',
        " began to examine the clerk's handwriting upon his own hand.",
    ),
    (
        'physical',
        'The seed was sown in April, and by the end of the summer it',
        ' had grown into a tall plant bearing grain.',
        ' had grown into a tall plant bearing coal.',
    ),
    (
        'causal',
        'The fire had been left unguarded, and the sparks falling upon the thatch',
        ' set the roof alight.',
        ' extinguished the roof entirely.',
    ),
    (
        'social',
        'A gentleman in mourning for his father would properly appear',
        ' in black, and decline all gaiety for a season.',
        ' in bright colours, and open the dancing himself.',
    ),
    ('quantity', 'The tide rises and falls twice in the course of', ' a single day.', ' a single century.'),
    ('physical', 'He poured the water upon the quicklime, and it', ' grew hot and hissed.', ' grew cool and silent as before.'),
    (
        'causal',
        'The horse had cast a shoe upon the stony road, and so the rider',
        ' led him slowly to the nearest smith.',
        ' galloped him the faster for the remaining twenty miles.',
    ),
]

# --- Anachronism trap pairs ---
# (trap stem, trap phrase, matched control stem, control phrase)
# The trap phrase is post-1900 knowledge; the control is the same sentence shape
# with a period-legitimate word. shock = bits(trap) - bits(control): a period
# model should find the modern word much MORE expensive. Negative shock means
# modern text leaked into training.

TRAP_PAIRS = [
    (
        'The aeroplane, which now carries passengers across the Atlantic, was',
        'aeroplane',
        'The steam locomotive, which now carries passengers across the country, was',
        'steam locomotive',
    ),
    (
        'The doctor prescribed a course of penicillin, and within',
        'penicillin',
        'The doctor prescribed a course of quinine, and within',
        'quinine',
    ),
    (
        'Every evening the family gathers before the television set to watch',
        'television set',
        'Every evening the family gathers before the fire to hear',
        'fire',
    ),
    (
        'The computer in the corner of the laboratory calculated the result in',
        'computer',
        'The telegraph in the corner of the office transmitted the message in',
        'telegraph',
    ),
    (
        'After the atomic bomb fell upon the city, the survivors',
        'atomic bomb',
        'After the cannonade fell upon the city, the survivors',
        'cannonade',
    ),
    (
        'She telephoned him from her motor car, using the wireless in her',
        'motor car',
        'She telegraphed him from her carriage, using the wire in her',
        'carriage',
    ),
]

# --- Unified generation prompt list ---
# Each prompt is generated ONCE per decoding mode and every text-quality metric
# is computed on these same continuations.

PROMPTS = [
    'The history of the world is',
    'What is God? God is',
    'In the year of our Lord eighteen hundred and',
    'The manufacture of cotton',
    'The steam engine',
    'The telegraph',
    'My dearest sister,',
    'Her Majesty the Queen',
    'LONDON, Tuesday. — The committee appointed to inquire into the condition of the',
    'A melancholy accident occurred on Thursday last at the works of Messrs. Harding and',
    'Brethren, the text which I have chosen for our consideration this morning is taken from',
    'March 12th. — Rose early, the frost being very sharp upon the windows. After breakfast I',
    'My dear Sister, — It is with no small shame that I take up my pen after so long a',
    'The prisoner, a labourer of some five-and-thirty years, was indicted for having',
    'The experiment was repeated with a coil of finer wire, and the deflection of the needle',
    'To make a plain seed cake. — Take one pound of flour, well dried before the fire, and',
    'The inn stood at the meeting of four roads, and it was there, upon a night of driving rain, that',
    'MANCHESTER, a city and municipal borough in the county of Lancaster, situated upon the',
    'The gardener who would have early peas must, in the first week of February,',
    'Let them eat brioche,',
    'Elementary, my dear Watson,',
    'Alas, poor Yorick! I knew him,',
    'The love of money is the root of all',
    'Put your trust in God, my boys, and keep your',
    'It was a cold morning in November when the carriage arrived at',
    '"You cannot mean it," she said, lowering her voice so that',
    'My dearest brother, I write to you from Lisbon, where the',
    'To prepare a proper broth for an invalid, first take',
    'The steam engine differs from the water-wheel chiefly in that',
    'LONDON, Tuesday. — The House of Commons yesterday debated',
    'On the cultivation of apple orchards in northern climates, the farmer must',
    'The old lighthouse keeper climbed the stairs slowly, remembering',
    'Among the curiosities exhibited at the fair was a mechanical',
    'The physician examined the patient and concluded that the fever',
    'A legal dispute involving an individual was presented, where',
    'We approach the close of our survey of the life and works of',
    'As our approach drew nearer, the scattered villages and humble enclosures',
    'This question, as well as the manner of its resolution,',
    'The English forces were preparing for an offensive on',
    'It was this rigorous self-discipline, this dedication to the unseen',
    'As the afternoon wore on, and the wind settled into a steady',
    'In the midst of this tempest of violence, a figure, whom we shall name',
    'Hark, gentle reader, and lend thine ear to a matter of profound import,',
    "The speaker then addressed the prior assertion that Her Majesty's Government should",
    'Let the farmers be warned that this poisonous plant is not to be confused with the edible fruit,',
    'Many the gay straw-rides to the Lake; frequent and long the walks through',
]

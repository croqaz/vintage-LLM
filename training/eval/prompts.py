"""Fixed sentence probes, choice pairs and generation prompt sets.

Each generation prompt runs once per decoding mode; surface statistics share
those continuations.
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
#
# Rules the set is built to, so later additions stay comparable:
#
#   1. LENGTH-MATCHED. Scoring is bits per BYTE, so a longer wrong option dilutes
#      its own error over more text. Options differ by at most 2 bytes.
#   2. MINIMAL EDIT. The two options differ by one swapped word, or by swapping
#      two names, and are otherwise identical. The BPB gap is then attributable
#      to the swap and to nothing else.
#   3. BOTH OPTIONS FLUENT. Only the sense differs, never the grammar, or the
#      item measures grammaticality instead.
#   4. MATCHED FAMILIARITY. The swapped words should be of comparable frequency
#      in period prose; antonyms are ideal. A rare wrong word is answerable from
#      unigram statistics without reading the context at all.
#   5. NO ARITHMETIC. A model this size cannot subtract. Magnitude and unit
#      plausibility are fair game; sums are not.
LOGIC_ITEMS = [
    # ---- physical: matter, heat, water, growth ------------------------------
    (
        'physical',
        'He set the kettle upon the fire, and after some minutes the water',
        ' began to boil, and steam rose from the spout.',
        ' began to freeze, and ice rose from the spout.',
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
        ' had turned sour and was thrown away.',
        ' had turned sweet and was thrown away.',
    ),
    (
        'physical',
        'The blacksmith heated the iron in the forge until it',
        ' glowed red and could be beaten into shape.',
        ' grew cold and could be beaten into shape.',
    ),
    (
        'physical',
        'He held the lighted candle to the paper, and the paper',
        ' caught fire and burned to a grey ash.',
        ' caught damp and burned to a grey ash.',
    ),
    (
        'physical',
        'The ship sprang a leak below the water-line, and the hold',
        ' began to fill with water, and the men worked the pumps.',
        ' began to fill with straw, and the men worked the pumps.',
    ),
    (
        'physical',
        'A heavy frost fell in the night, and in the morning the pond',
        ' was covered with ice, and the children slid upon it.',
        ' was covered with mud, and the children slid upon it.',
    ),
    (
        'physical',
        'He carried the lamp down into the cellar, for without it the cellar was',
        ' too dark for him to see the steps.',
        ' too light for him to see the steps.',
    ),
    (
        'physical',
        'The seed was sown in April, and by the end of the summer it',
        ' had grown into a tall plant bearing grain.',
        ' had grown into a tall plant bearing nails.',
    ),
    (
        'physical',
        'It had rained heavily all the morning, and the road',
        ' was deep in mud.',
        ' was deep in dust.',
    ),
    (
        'physical',
        'He threw the heavy stone into the pond, and it',
        ' sank at once out of sight.',
        ' rose at once out of sight.',
    ),
    (
        'physical',
        'The snow lay thick upon the ground, and when the sun came out at noon it',
        ' began to melt into water.',
        ' began to melt into stone.',
    ),
    (
        'physical',
        'He plunged the red-hot horseshoe into the trough, and the water',
        ' hissed and gave off steam.',
        ' hissed and gave off frost.',
    ),
    (
        'physical',
        'The lamp had no oil left in it, and so when he set a match to the wick it',
        ' gave no light at all.',
        ' gave a light at once.',
    ),
    # ---- causal: one event making another follow ----------------------------
    (
        'causal',
        'The harvest failed for the second year together, and the price of bread',
        ' rose so high that the poor could scarcely buy it.',
        ' fell so low that the poor could scarcely buy it.',
    ),
    (
        'causal',
        'He had not slept for two nights, and therefore at the meeting he',
        ' could scarcely keep his eyes open.',
        ' could scarcely keep his eyes shut.',
    ),
    (
        'causal',
        'The bridge had been carried away by the flood, so the travellers',
        ' could not cross the river that day.',
        ' could still cross the river that day.',
    ),
    (
        'causal',
        'Since the letter was never posted, his brother',
        ' remained wholly ignorant of the matter.',
        ' remained wholly informed of the matter.',
    ),
    (
        'causal',
        'The physician found the wound to be badly inflamed, and he therefore',
        ' ordered the wound to be dressed.',
        ' ordered the wound to be ignored.',
    ),
    (
        'causal',
        'A long drought had parched the fields, and the farmers',
        ' looked anxiously for rain.',
        ' looked anxiously for sun.',
    ),
    (
        'causal',
        'He staked his whole fortune upon the venture, and when the ship was lost he',
        ' was reduced to absolute poverty.',
        ' was raised to enormous riches.',
    ),
    (
        'causal',
        'The window had been left open all night in December, and in the morning the room',
        ' was extremely cold.',
        ' was extremely warm.',
    ),
    (
        'causal',
        'The fire had been left unguarded, and the sparks falling upon the thatch',
        ' set the roof in a blaze.',
        ' set the roof in a flood.',
    ),
    (
        'causal',
        'The candle had burnt down to the socket, and at midnight the room',
        ' was left in complete darkness.',
        ' was left in complete daylight.',
    ),
    (
        'causal',
        'He forgot to wind the clock before going to bed, and in the morning it',
        ' had stopped in the night.',
        ' had gained in the night.',
    ),
    (
        'causal',
        'The road had lately been mended, and so the coach',
        ' travelled the smoother for it.',
        ' travelled the rougher for it.',
    ),
    (
        'causal',
        'She had eaten nothing since the morning before, and by evening she was',
        ' faint with hunger.',
        ' heavy with dinner.',
    ),
    (
        'causal',
        "The letter brought news of his brother's death, and upon reading it he",
        ' wept bitterly for an hour.',
        ' sang merrily for an hour.',
    ),
    # ---- social: period manners and obligation ------------------------------
    (
        'social',
        'Being invited to dine at a house of higher station, he was careful to',
        ' arrive punctually and properly dressed.',
        ' arrive carelessly and poorly dressed.',
    ),
    (
        'social',
        'A letter to a bishop should properly begin',
        ' My Lord, with all due respect.',
        ' Old chap, with all due respect.',
    ),
    (
        'social',
        'His neighbour having lost her husband that week, he thought it right to',
        ' send her a letter of condolence.',
        ' send her a letter of invitation.',
    ),
    (
        'social',
        'The young man wished to marry, and as a matter of duty he first',
        " sought the consent of the lady's father.",
        " sought the consent of the lady's servant.",
    ),
    (
        'social',
        'Having given his word before witnesses, he held himself',
        ' bound in honour to keep it.',
        ' free in honour to break it.',
    ),
    (
        'social',
        'The servant announced a visitor at an hour past midnight, which the household thought',
        ' a very improper time to call.',
        ' a very agreeable time to call.',
    ),
    (
        'social',
        'He was called as a witness, and being sworn upon the book he was bound to',
        ' answer every question with truth.',
        ' answer every question with lies.',
    ),
    (
        'social',
        'A gentleman in mourning for his father would properly appear',
        ' in black, and decline all gaiety.',
        ' in white, and decline all gaiety.',
    ),
    (
        'social',
        "He was but a shopkeeper's son, and to address a duchess as his equal would be thought",
        ' most impertinent.',
        ' most respectful.',
    ),
    (
        'social',
        'He met the lady in the street, and being a gentleman he',
        ' raised his hat to her.',
        ' raised his fist to her.',
    ),
    (
        'social',
        'The two men had never been introduced, and so to address him in the street was',
        ' a liberty he could not take.',
        ' a liberty he might well take.',
    ),
    (
        'social',
        'She was in her own drawing-room receiving morning callers, and therefore she',
        ' was dressed with some care.',
        ' was dressed with no care.',
    ),
    # ---- quantity: magnitude and unit plausibility, never arithmetic --------
    (
        'quantity',
        'The infant was but three weeks old, and therefore he',
        ' could neither walk nor speak.',
        ' could already walk and speak.',
    ),
    (
        'quantity',
        'The journey by coach occupied the better part of three days, for the distance was',
        ' upwards of two hundred miles.',
        ' upwards of two hundred yards.',
    ),
    (
        'quantity',
        'The room measured but twelve feet by ten, and so it was',
        ' too small for a hundred guests.',
        ' too large for a hundred guests.',
    ),
    (
        'quantity',
        'The child was of ordinary growth for seven years, and stood',
        ' something under four feet high.',
        ' something under nine feet high.',
    ),
    (
        'quantity',
        'A gallon of the liquid was required, but he had brought only a pint, which was',
        ' much less than was wanted.',
        ' much more than was wanted.',
    ),
    (
        'quantity',
        'The tide rises and falls twice in the course of',
        ' a single day.',
        ' a single year.',
    ),
    (
        'quantity',
        'The loaf cost but three-halfpence, and the labourer thought the price',
        ' a very small one.',
        ' a very great one.',
    ),
    (
        'quantity',
        'The whole company numbered but seven persons, and so the great hall was',
        ' very nearly empty.',
        ' very nearly full.',
    ),
    (
        'quantity',
        'The letter had been written above fifty years before, and the paper was therefore',
        ' yellow with great age.',
        ' white with great age.',
    ),
    (
        'quantity',
        'He walked his four miles in the hour, which for a man of his years was',
        ' a very fair pace.',
        ' a very wild pace.',
    ),
    # ---- coref: which of two named things the sentence is about -------------
    (
        'coref',
        'John lent his umbrella to Thomas, and the rain came on; so',
        ' Thomas was kept dry and John was drenched.',
        ' John was kept dry and Thomas was drenched.',
    ),
    (
        'coref',
        'The mother gave the child a shilling, and so',
        ' the child had a shilling more.',
        ' the mother had a shilling more.',
    ),
    (
        'coref',
        'The clerk handed the ledger to his employer, and',
        ' the employer opened it at once.',
        ' the ledger opened it at once.',
    ),
    (
        'coref',
        'When the doctor came to the sick woman, he found that she',
        ' had grown much weaker since his last visit.',
        ' had grown much weaker since her last visit.',
    ),
    (
        'coref',
        'The girl gave her sister the doll, and afterwards',
        ' the sister played with it.',
        ' the doll played with her.',
    ),
    (
        'coref',
        'The dog followed the boy into the garden, and there',
        ' the boy threw a stick for the dog.',
        ' the dog threw a stick for the boy.',
    ),
    (
        'coref',
        'The cat sprang upon the mouse, and in a moment',
        ' the mouse was quite dead.',
        ' the cat was quite dead.',
    ),
    (
        'coref',
        'The old man leaned upon the boy, for',
        ' the man was very weak.',
        ' the boy was very weak.',
    ),
    (
        'coref',
        'Sarah handed the parcel to Emily, and',
        ' Emily carried it home.',
        ' Sarah carried it home.',
    ),
    (
        'coref',
        'The farmer sold the horse to the squire, and afterwards',
        ' the squire rode it every day.',
        ' the horse rode it every day.',
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


# ============================================================================
# COLD SEEDS - the generation regime a synth-data pipeline actually runs in
# ============================================================================
#
# PROMPTS above are curated TOPICAL stems ('The steam engine', 'What is God?
# God is'). They hand the model a subject, which is a much easier test than
# production use.
#
# These are harvested from the opener distribution of a real 49,413-completion
# bulk run: 36,753 DISTINCT two-word openers, of which the most frequent are
# bare function words carrying almost no topical signal.
# A model that only looks healthy when handed a subject will show its degeneracy
# here first.
#
#   source: https://huggingface.co/datasets/croqaz/tiny-vintage-completions
#
# Deliberately NOT loaded from that dataset: the evaluator must not depend on a
# data artefact that may move or disappear. This is a fixed, checked-in sample
# of that distribution.
COLD_SEEDS = [
    'In the',
    'Of the',
    'It is',
    'But the',
    'On the',
    'From the',
    'To the',
    'The following',
    'When the',
    'We have',
    'That the',
    'It was',
    'If the',
    'And the',
    'As the',
    'By the',
    'At the',
    'The most',
    'He was',
    'There is',
    'The first',
    'For the',
    'The whole',
    'I have',
    'An instance',
    'Foreign affairs',
]

SEED_SETS = {
    'curated': lambda: list(PROMPTS),
    'cold': lambda: list(COLD_SEEDS),
    'both': lambda: list(PROMPTS) + list(COLD_SEEDS),
}

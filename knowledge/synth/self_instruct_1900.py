#!/usr/bin/env python3
"""
self_instruct_1900.py
=========================================================================
A small, heavily-commented proof-of-concept that borrows ideas from the two
papers in this folder and adapts them to a *time-capsule* model whose
knowledge is supposed to stop in 1899-1900.

What it borrows
---------------
1. SELF-INSTRUCT (Wang et al., ACL 2023)
   - Bootstrap an instruction dataset from a tiny seed set of hand-written
     tasks by few-shot prompting the model to invent *new* instructions.
   - Each round samples a few examples from the pool (the paper uses 6 human
     + 2 machine-generated) as in-context demonstrations to push diversity.
   - Filter new instructions for novelty using ROUGE-L overlap (the paper
     rejects a candidate whose ROUGE-L vs. any existing instruction >= 0.7)
     plus simple length / keyword heuristics.
   - INSTANCE GENERATION: for each surviving instruction we use the paper's
     "input-first" approach -- first ask the model for an example INPUT
     (or NONE for self-contained tasks), then ask for the OUTPUT conditioned
     on (instruction, input). We can generate several instances per instruction
     and de-duplicate them.

2. SIMPLE SELF-DISTILLATION / SSD (Zhang et al., 2026)
   - The data is nothing but the model's *own* raw, unverified samples.
   - SSD's central lesson is that the *sampling configuration* (training-time
     temperature `T_train` and truncation top-k / top-p) is the real lever,
     not answer correctness.
   - DECOUPLED TEMPERATURES: SSD distinguishes a high, exploratory training-time
     temperature from a lower evaluation-time one, and ties the win to its
     "precision-exploration conflict" -- forks (where many continuations are valid)
     want exploration, locks (where one answer is right) want precision.
     We map that onto data generation:
       * --gen-temp / --gen-top-k : used to BRAINSTORM instructions and INPUTS,
         where we *want* variety/coverage  -> the "fork" / exploration side.
       * --ans-temp / --ans-top-k : used to write the ANSWER, where we want a
         coherent, correct response     -> the "lock" / precision side.
     NOTE: a *true* T_eval only exists when you decode the fine-tuned model.
     We do NOT fine-tune here, so this two-temperature split is the
     data-generation analogue of SSD's idea, not the literal T_train/T_eval
     experiment. The collected JSONL is exactly the corpus you would later SFT
     on to "self-distill" -- that step is left for a future iteration.
   - N-SAMPLE SELECTION: SSD finds that multiple samples *cover* more good
     solutions (pass@k > pass@1). With --candidates-per-instance > 1 we sample
     several answers per instance and keep one, chosen WITHOUT a verifier --
     by the model's own mean token logprob (default), answer length, or at random.
     Random is a deliberate baseline: SSD's "bad data, good results" warns that
     hard selection is not always a win, so it is worth measuring logprob/longest against it.

The time-capsule twist (the important part)
--------------------------------------------
Every instruction we keep must stay within "what a well-read person in 1899
could know". We enforce this in THREE layers now:

  (A) Prompt-level gating  -> the system/seed prompts repeatedly tell the model
      to stay in <=1899 territory.
  (B) Filter-level gating  -> a hard reject pass (keyword blocklist + future
      years) that drops anything mentioning a post-1900 invention/event.
  (C) Model-as-judge gating -> an optional second call where the model itself
      acts as a neutral historical fact-checker and votes SAFE / ANACHRONISTIC
      on a piece of text. This generalises beyond the hardcoded blocklist
      (e.g. a question about a 1915 novel trips no keyword, but the judge can
      still flag it). Enable with --temporal-judge.

ENGLISH ONLY
------------
The dataset must be English-only -- no tasks about translation or about foreign
languages (Latin, French, German, ...), and no foreign text in any field. This
is a hard requirement, always on, enforced as its own gate with two checks:
  * foreign-language *task* patterns (e.g. "translate ...", "the French word
    for ...", "write a poem in Latin"); and
  * foreign *characters* in the text (accented letters, Cyrillic, Greek, CJK,
    ...) -- any non-ASCII letter is treated as non-English.

Usage
-----
    # Local llama-server (default):
    python self_instruct_1900.py --num 10
    # SSD-style: explore hard on prompts, stay precise on answers, judge era:
    python self_instruct_1900.py --num 25 \
        --gen-temp 1.2 --gen-top-k 60 --ans-temp 0.7 --ans-top-k 25 \
        --instances-per-instruction 2 --temporal-judge --out my_dataset.jsonl

    # OpenRouter (requires OPENROUTER_API_KEY in the environment):
    export OPENROUTER_API_KEY=sk-or-v1-...
    python self_instruct_1900.py --provider openrouter --model openai/gpt-4o --num 10

    # OpenRouter with SSD-style decoupled temperatures:
    python self_instruct_1900.py --provider openrouter --model anthropic/claude-3-opus \
        --gen-temp 1.2 --ans-temp 0.7 --num 25 --out my_dataset.jsonl

Requires only the Python standard library. For local mode, a running llama-server
is needed. For OpenRouter mode, an API key and internet access are required.
"""

import argparse
import json
import os
import random
import re
import sys
import unicodedata
import urllib.error
import urllib.request

sys.path.append(os.path.realpath('../pre1900'))

from banned_terms import find_anachronisms  # NOQA: import from sibling folder

# ---------------------------------------------------------------------------
# 0. Connection settings
# ---------------------------------------------------------------------------
# llama-server speaks the OpenAI-compatible API. We use two endpoints:
#   /v1/completions       -> raw text completion (used to *brainstorm*
#                            instructions, Self-Instruct style: "continue this
#                            numbered list"). This bypasses the chat template,
#                            which is exactly what we want for list-continuation.
#   /v1/chat/completions  -> templated chat (used to generate inputs, answers,
#                            and judge verdicts).
# llama.cpp accepts the non-standard "top_k" field on both endpoints, so we can
# pass SSD-style truncation straight through.
# Pre-defined providers. Each speaks the OpenAI-compatible API, but they differ
# in two ways we have to account for:
#   * base_url   -- where the API lives.
#   * path_prefix-- the version segment that sits between the base URL and the
#                   endpoint name. OpenRouter and llama-server use "/v1"
#                   (".../v1/chat/completions"); Google's OpenAI-compatibility
#                   shim bakes the version into the base itself
#                   (".../v1beta/openai/chat/completions"), so its prefix is "".
#   * api_key_env-- environment variable holding the bearer token (None = local,
#                   no auth).
#   * model_env  -- optional env var to fall back to when --model is omitted.
#   * headers    -- any extra static headers the provider wants.
#   * supports_top_k -- whether the backend accepts the non-standard `top_k`
#                   sampling field. llama.cpp accepts it and OpenRouter passes it
#                   through (or silently ignores it); Google's OpenAI-compatible
#                   shim is strict and REJECTS unknown fields with HTTP 400, so we
#                   must omit `top_k` there.
#   * supports_logprobs -- whether the backend accepts the `logprobs` field used
#                   for verifier-free answer selection. Google's shim does NOT
#                   (rejects it with HTTP 400); when False we skip it and the
#                   logprob score comes back None (--selection logprob then just
#                   degrades to no scoring).
# To add another OpenAI-compatible provider, just add an entry here.
PROVIDERS = {
    'local': {
        'base_url': 'http://127.0.0.1:1234',
        'path_prefix': '/v1',
        'api_key_env': None,
        'model_env': None,
        'headers': {},
        'supports_top_k': True,
        'supports_logprobs': True,
    },
    'openrouter': {
        'base_url': 'https://openrouter.ai/api',
        'path_prefix': '/v1',
        'api_key_env': 'OPENROUTER_API_KEY',
        'model_env': 'OPENROUTER_MODEL',
        'headers': {
            'HTTP-Referer': 'Self-instruct-1900',
            'X-Title': 'Self-instruct-1900',
        },
        'supports_top_k': True,
        'supports_logprobs': True,
    },
    'google': {
        'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai',
        'path_prefix': '',
        'api_key_env': 'GEMINI_API_KEY',
        'model_env': 'GOOGLE_MODEL',
        'headers': {},
        'supports_top_k': False,
        'supports_logprobs': False,
    },
}

DEFAULT_BASE_URL = PROVIDERS['local']['base_url']
_AUTH_HEADERS = {}  # populated by main() with the provider's auth + extra headers
# Version segment between base_url and the endpoint name; set by main() from the
# chosen provider's 'path_prefix'. Defaults to the llama-server / OpenRouter "/v1".
_PATH_PREFIX = '/v1'
# Whether to include the non-standard `top_k` sampling field on requests; set by
# main() from the provider's 'supports_top_k' (Google rejects it -> False).
_SUPPORTS_TOP_K = True
# Whether to request `logprobs` (for verifier-free answer selection); set by
# main() from the provider's 'supports_logprobs' (Google rejects it -> False).
_SUPPORTS_LOGPROBS = True
_DEBUG = False  # set by main() from --debug; when True we dump every request/response
# Reasoning effort to request on every call, or None to send no `reasoning` field.
# Set by main() after discover_reasoning_effort() confirms the model supports it.
# We keep it low everywhere: this is bulk data-generation, not hard reasoning, so
# we want speed/cost, not long chains of thought.
_REASONING_EFFORT = None


# ---------------------------------------------------------------------------
# 1. Seed tasks  (human-written tasks of Self-Instruct, shrunk to a handful).
#    Every one is deliberately answerable by someone in 1900.
#    Keep clean and on-era: they anchor the whole bootstrap distribution.
# ---------------------------------------------------------------------------
SEED_INSTRUCTIONS = [
    'A clock strikes only the hours; how many times does it strike in a full day?',
    'Account for the popularity of baseball among boys.',
    'Advise a beginner on the materials needed to take up sketching from nature.',
    'Advise a boy who wishes to better himself but has had little schooling.',
    'Advise a farmer on the best means of draining a field of wet land.',
    'Advise a farmer on the best means of draining wet land.',
    'Advise a lady on how to keep herself safe while travelling alone by rail.',
    'Advise a man uncertain whether to declare his affection to a woman of higher station.',
    'Advise a schoolmaster on how he may keep order among unruly pupils.',
    'Advise a young lady on the best daily habits for maintaining health.',
    'Advise a young lady on the virtues she should cultivate before marriage.',
    'Advise a young man leaving home for the first time on the company he should keep.',
    'Advise a young man on choosing a profession suited to his talents.',
    'Advise a young man on the best means of improving his handwriting for business purposes.',
    'Advise a young man setting out in business on the habits he ought to cultivate.',
    'Advise a young man uncertain whether to declare his affection to a lady.',
    'Advise a young man who has saved a little money on how he might invest it wisely.',
    'Advise a young workman on what he should do if his wages are unjustly withheld.',
    'Advise on the proper way to address a letter to a person of high rank.',
    'Are angels real?...',
    'Complete a sentence about a horse: "I was a broken-hearted rider..."',
    'Complete the lyrics: "There\'s a place in my heart..."',
    'Compose a poem about the seasons and their changes.',
    'Compose a short poem about the changing of the autumn leaves.',
    'Defend the advantages of living in a small village.',
    'Describe a few methods of preserving foodstuffs for winter.',
    'Describe a journey across London by omnibus and hansom cab.',
    'Describe how a fire may be kept from spreading through a row of houses.',
    'Describe how a householder may guard his dwelling against burglars at night.',
    'Describe how a householder may guard his home against burglars in the night.',
    'Describe how a piano makes its sound.',
    'Describe how a portrait is made to resemble the person who sits for it.',
    'Describe how a portrait painter captures the likeness of his sitter.',
    'Describe how a schoolmaster keeps order among his pupils.',
    'Describe how a song is set to music for singing.',
    'Describe how a young child should be taught to read.',
    'Describe how a young child should first be taught to read.',
    'Describe how an ordinary lead pencil is made and why it leaves a mark.',
    'Describe how cocoa is prepared as a drink.',
    'Describe how cocoa is prepared as a drink.',
    'Describe how men measured the passing of time before clocks were made.',
    'Describe how tea is carried from China and India to England.',
    'Describe how tea is carried from China to England.',
    'Describe how to calculate the volume of a cylinder.',
    'Describe how to take a grease stain out of a woollen coat.',
    'Describe how wine is kept and left to age in a cellar.',
    'Describe how you would deal with a fire in the countryside if one breaks out.',
    'Describe some country charms believed to ward off the evil eye.',
    'Describe the animals kept upon an English farm and the use of each.',
    'Describe the animals kept upon an English farm and the use of each.',
    'Describe the care a horse needs after a long journey on the road.',
    'Describe the chief battles of the Wars of the Roses.',
    'Describe the chief causes that led the thirteen colonies to declare independence.',
    'Describe the chief events in the reign of Queen Elizabeth.',
    'Describe the chief events in the reign of Queen Victoria.',
    'Describe the chief properties of oxygen and how it may be prepared.',
    'Describe the conduct expected of an apprentice toward his master.',
    'Describe the creatures one may find in a tidal rock-pool by the sea.',
    'Describe the difference between black tea and green tea.',
    'Describe the difference between harmony and melody.',
    'Describe the difference between painting in oils and in watercolours.',
    'Describe the difference between painting in oils and in watercolours.',
    'Describe the difference between the Whig and Tory political philosophies.',
    'Describe the different types of fabric and their uses.',
    'Describe the duties of a governess in a private household.',
    'Describe the duties of a justice of the peace in a country town.',
    'Describe the duties of a librarian at a public library.',
    'Describe the duties of a magistrate in a country town.',
    'Describe the duties of a night watchman in guarding a row of shops.',
    'Describe the duties that attend the holding of a peerage.',
    'Describe the gods of Mount Olympus and the dominion of each.',
    'Describe the habits of the industrious ant.',
    'Describe the making of cider from apples in the autumn.',
    'Describe the making of cider from apples in the autumn.',
    'Describe the manner of the crowning of a sovereign.',
    'Describe the manner of the crowning of a sovereign.',
    'Describe the omens and signs by which country folk foretell coming changes in the weather.',
    'Describe the planets of the solar system in their order from the sun.',
    'Describe the principal causes of the American Civil War.',
    'Describe the principal events of the American Civil War.',
    'Describe the process of drying apples on strings.',
    'Describe the process of making a traditional English breakfast.',
    'Describe the proper manner in which a gentleman may court a young woman.',
    'Describe the proper manner in which a young man may court a young woman.',
    'Describe the proper method of keeping accounts in a household ledger.',
    'Describe the proper way to wash and dry household linen.',
    'Describe the prophecy of Daniel concerning the four great kingdoms.',
    'Describe the sights and bustle of a morning in the streets of the metropolis.',
    'Describe the steps a blacksmith takes to forge a horseshoe.',
    'Describe the structure of a heroic couplet in English verse.',
    'Describe the superstitions surrounding the number thirteen at a dinner table.',
    'Describe the usefulness of the common honeybee.',
    'Describe the usefulness of the common honeybee.',
    'Describe the work of the harvest, from the cutting to the threshing of the corn.',
    'Describe the working of the underground railway beneath the city.',
    'Describe what a promissory note is and how it binds the parties to it.',
    'Describe what causes an eclipse of the sun.',
    'Describe what takes place when iron is left to rust in damp air.',
    'Did you ever wonder why the sky is blue by day and red at sunset? Explain the reason.',
    'Discuss the condition of the poor in the crowded slums of the great city.',
    'Discuss the difference between true courage and mere recklessness.',
    'Discuss the duties of an ambassador at a foreign court.',
    'Discuss the evils of employing young children in mills and factories.',
    'Discuss the importance of the printing press in the spread of knowledge.',
    'Discuss the qualities a young lady should seek in a husband.',
    'Discuss the superstitions that gather about the crowing of a cock and the hooting of an owl.',
    'Discuss the use of the telegraph in commerce.',
    'Discuss the value of thrift and industry in early life.',
    'Discuss what is meant by the Number of the Beast and how interpreters have read it.',
    'Discuss whether a good end can ever justify a wrong means.',
    'Discuss whether corporal punishment has any place in the schoolroom.',
    'Discuss whether there is any truth in the belief in haunted houses.',
    'Discuss whether workmen are right to form trade unions for their protection.',
    'Do you have a sweet tooth? What is your favourite dessert?',
    'Do you know any notable inventions in chemistry?',
    'Do you think people are basically good or bad?',
    'Does anyone care about honour in this day and age?',
    'Draft a short advertisement for a local shopkeeper offering repairs for pocket watches and other trinkets.',
    'Elaborate on the astronomical findings of Sir Isaac Newton.',
    'Explain how a barometer measures pressure and what it indicates.',
    'Explain how a bill becomes a law in Parliament.',
    'Explain how a father ought to judge a suitor who asks for his daughter.',
    'Explain how a lady should receive callers in the afternoon.',
    'Explain how a man may make a will so that it holds good in law.',
    'Explain how a man may make a will so that it holds good in law.',
    'Explain how a painter prepares his canvas before he begins to work.',
    'Explain how a painter uses perspective to give depth to a flat canvas.',
    'Explain how a sailing ship is able to travel against the wind.',
    'Explain how a song is set to music for singing.',
    'Explain how a spider spins and uses its web.',
    'Explain how a sundial tells the hour by the shadow it casts.',
    'Explain how a torn garment may be mended so that the seam holds.',
    'Explain how a watchman keeps order in a town through the night.',
    'Explain how ale is brewed from barley and hops.',
    'Explain how an engraving is produced upon a copper plate.',
    'Explain how birds are able to fly.',
    'Explain how birds are able to fly.',
    'Explain how coffee is roasted and ground for the breakfast table.',
    'Explain how iron is smelted from ore in a blast furnace.',
    'Explain how manure improves the fertility of the soil.',
    'Explain how perspective gives the look of depth to a flat drawing.',
    'Explain how sailors are able to steer by the stars at night.',
    'Explain how sailors steer by the stars at night.',
    'Explain how the streets of London were lit before gas and electricity.',
    'Explain how to care for delicate fabrics.',
    'Explain how to construct a simple voltaic pile.',
    'Explain how to judge the age of a horse by its teeth.',
    'Explain how to sew a button onto a shirt.',
    'Explain how two strangers ought to be introduced at a gathering.',
    'Explain how wool is spun into thread and woven into cloth.',
    'Explain in plain terms the difference between common law and statute law.',
    'Explain in plain terms what happens when a firm is declared bankrupt.',
    'Explain the basics of the common law and how it differs from civil law.',
    'Explain the belief that spilling salt brings misfortune and how it may be averted.',
    'Explain the belief that the dead may return as ghosts and why some folk dread the churchyard.',
    'Explain the benefits of teaching children to commit good verse to memory.',
    'Explain the causes of the tides in the ocean.',
    'Explain the common superstition that it is unlucky to walk under a ladder.',
    'Explain the correct method for tuning a violin, and how often it should be done.',
    'Explain the correct method for tuning a violin, and how often it should be done.',
    'Explain the country custom of touching wood to prevent a boast from tempting fate.',
    'Explain the difference between a barometer and a thermometer.',
    'Explain the difference between a baronet and a knight, and how each is addressed.',
    'Explain the difference between a baronet and a knight, and how each is addressed.',
    'Explain the difference between a share and a debenture in a joint-stock company.',
    'Explain the difference between a wine and a spirit.',
    'Explain the difference between a wine and a spirit.',
    'Explain the difference between an acid and an alkali, and how each may be tested.',
    'Explain the difference between black tea and green tea.',
    'Explain the difference between deductive and inductive reasoning.',
    'Explain the duties of a governess in a private household.',
    'Explain the order of succession to the throne.',
    'Explain the order of succession to the throne.',
    'Explain the part George Washington played in the founding of the United States.',
    'Explain the part that yeast plays in fermentation.',
    'Explain the process of creating a kite by hand.',
    'Explain the purpose of a Factory Act in limiting the hours of labour.',
    'Explain the rule for finding the number of days in any given month.',
    'Explain the rules for the correct use of the semicolon in writing.',
    'Explain the rules of etiquette for a formal dinner.',
    'Explain the several ranks of the nobility, from duke down to baron.',
    'Explain the symbolism of the four horsemen of the Apocalypse.',
    'Explain the utility of sulfuric acid in modern industry.',
    'Explain to me, why does the thunder follow lightning?',
    'Explain what a betrothal binds a couple to, and on what grounds it may be broken.',
    'Explain what a betrothal binds a couple to, and on what grounds it may be broken.',
    'Explain what a justice of the peace may and may not do.',
    'Explain what a man ought to do on finding a thief within his dwelling.',
    'Explain what a man ought to do upon discovering a thief within his home.',
    'Explain what a syllogism is and give a simple example of one.',
    'Explain what causes an eclipse of the sun.',
    'Explain what comets are and why they were once thought to be omens.',
    'Explain what happens, chemically, when a candle burns.',
    'Explain what is meant by a valid argument as distinct from a true conclusion.',
    'Explain what is meant by conscience and how a man ought to heed it.',
    'Explain what is meant by limited liability and why it encourages investment.',
    'Explain what is meant by the balance of power among the nations of Europe.',
    'Explain what is meant by the extension of the franchise to working men.',
    'Explain what is meant by the major and minor scales in music.',
    'Explain what is meant by the ratio of two quantities, with an example.',
    'Explain what is meant by trial by jury and why it is valued.',
    'Explain what measures should be taken to keep a fire from spreading through a house.',
    'Explain what the Declaration of Independence proclaimed and who drafted it.',
    'Explain what the seven seals in the Book of Revelation are commonly held to signify.',
    'Explain what the time signature at the head of a piece of music tells the player.',
    'Explain who the Muses were and over which arts they presided.',
    'Explain why a day is added to the calendar in a leap year.',
    'Explain why a horseshoe is hung above the door for good fortune.',
    'Explain why a kettle sings just before the water comes to the boil.',
    'Explain why a looking-glass reverses left and right but not up and down.',
    'Explain why common salt dissolves in water.',
    'Explain why farmers practise the rotation of crops from year to year.',
    'Explain why honesty is said to be the best policy.',
    'Why it is wrong to steal?',
    'Explain why it is wrong to tell a lie, even a small one.',
    'Explain why punctuality is a virtue that serves a man all his life.',
    'Explain why the moon shows a different shape from night to night.',
    'Explain why the moon shows a different shape from night to night.',
    'Explain why the stars are not seen by day.',
    'Explain why thrift in small matters secures a man his later years.',
    'Explain, step by step, how ale is brewed from barley and hops.',
    'Explain, step by step, how to brew a proper pot of tea.',
    'Explain, step by step, how to brew a proper pot of tea.',
    'Find the area of a circle whose diameter is fourteen inches.',
    'Give a brief account of the Boston Tea Party and its consequences.',
    'Give advice on how a young man ought to conduct himself in polite society.',
    'Give an example of how you would write a letter of introduction.',
    'Give counsel to a youth on how he may become a good public speaker.',
    'Give details on preparing a meal for a few guests overnight.',
    'Give me a one-sentence description for each of the following people: William Shakespeare, John Milton, John Bunyan.',
    'Give practical advice on how to care for a horse during the winter months.',
    'Given a description of the symptom, identify the possible disease and suggest some medicine: I have a fever and I am coughing.',
    'How can a lady protect herself from being robbed while walking alone at night?',
    'How can I take care of a family of kittens?',
    'How do bees make honey?',
    'How do I know if a lady is interested in me?',
    'How do I start a conversation with a stranger at a social gathering?',
    'How do tarot cards and readings work?',
    'How do you ensure your garden is properly weeded and watered when you are away from home?',
    'How do you pass time when travelling by rail?',
    'How does a pharmacist prepare a tincture of laudanum?',
    'How is an artificial leg made?',
    'How was it customary to light the streets before gas and electricity?',
    'How would someone arrange a private library catalogue, if his collection contains 500 volumes?',
    'I am incredibly bored... What can I do to entertain myself?',
    'I feel depressed and lonely today. What can I do to cheer myself up?',
    'I have a lot of wild grapes in my backyard. How can I make them into wine?',
    'I heard someone bad-mouthing a friend of mine. How should I respond?',
    'I saw this blue butterfly in my garden. Can you tell me what species it is?',
    'I would like to start painting with watercolors. What do I need to get started?',
    'In your opinion, what are the qualities of an effective sports coach?',
    'Is it ever okay to break a promise?',
    'Is this Coca-Cola a drink, or a medicine?',
    'List a few ways to make a small room feel more spacious.',
    'List five great rivers of Europe and name a city that each one passes through.',
    'List the qualities a young lady ought to cultivate for pleasant conversation at a party.',
    'Plan a late evening walk and discuss what is worth while seeing.',
    'Plan a lunch menu for a family picnic in the countryside.',
    'Point out the fallacy in the argument: "All that glitters is gold; this ring glitters; therefore it is gold."',
    'Provide a recipe for a simple loaf of bread that can be baked at home.',
    'Provide examples of how to use quotation marks in letter writing.',
    'Recount the folk belief that breaking a looking-glass brings seven years of sorrow.',
    'Recount the folklore of fairies and the little people said to dwell in hill and hollow.',
    'Recount the labours of Hercules and what each was meant to teach.',
    'Recount the legend of the Trojan Horse and the fall of Troy.',
    'Remind me the appearance of a gentleman in full dress.',
    'Remind me the role of "Penny Dreadfuls" in shaping the reading habits of youth.',
    'Set down the customs observed at a country wedding.',
    'Set down the duties a queen owes to her people.',
    'Set down the first lessons a child should learn in drawing from nature.',
    'Set down the habits a youth must form if he would rise in his trade.',
    'Set down the harm that strong drink does to a working man and his family.',
    'Set down the lessons a boy ought to master before the age of twelve.',
    'Set down the manner in which a gentleman should decline an invitation.',
    'Set down the order of precedence among the ranks of the nobility.',
    'Set down the precautions a traveller should take against highwaymen on the road.',
    'Set down the use of a hymn sung in church.',
    'Set down what a gentleman should wear for a formal evening.',
    'Set down what is known of the planets that circle the sun.',
    'Should I buy a bicycle or a horse for my daily commute?...',
    'Should I get myself a dog or a cat as a pet?',
    'Solve for x in the equation 3x + 7 = 22.',
    'Suppose you find a lost wallet full of money. What do you do?',
    'Tell me a few examples of how knowledge may be used to improve health.',
    'Tell me a few games that can be played by a group of people.',
    'Tell me a new invention or scientific discovery.',
    'Tell me a recipe for making a savoury apple dumpling.',
    'Tell me the curious reason why a cat is able to see in near darkness.',
    'Tell the story of how Prometheus brought fire to mankind.',
    'What are the common faults in table manners?',
    'What are the common remedies for toothache these days?',
    'What are the duties of a gatekeeper at a railway crossing?',
    'What are the main lessons to learn from the Bible?',
    'What are the most important lessons to learn from the history of the Turkish Empire?',
    'What do really rich people do with their money?',
    'What does the story of David and Goliath from the Bible teach us?',
    'What happens if someone eats monkshood?',
    'What happens to a horse when it is shod with iron shoes?',
    'What is a clockwork automaton and how does it work?',
    'What is a memento? What does it mean?',
    'What is a stout beer and how is it made?',
    'What is belladonna, also called nightshade?',
    'What is life like in the English countryside?',
    'What is meant by the Second Coming, and on what scriptures is the belief founded?',
    'What is the difference between beer and ale?',
    'What is the law of action and reaction?',
    'What is the most remarkable incident in your own life?',
    'What is the point of arts?',
    'What objects can you find in a typical Victorian parlor?',
    'What prayer can I teach my child to say before bedtime, to help him feel comforted?',
    'What steps can I take if my dog attacks me?',
    'Where did you travel this year?',
    'Where does the oil lamp come from, how is it made?',
    'Why do people write poetry?',
    'Write a brief dialogue between a blacksmith and a customer about pricing a new horseshoe.',
    'Write a list of things that could be brought to a child on his birthday.',
    'Write a short letter from a gentleman in London inviting a friend to a dinner party.',
    'Write a story that contains the given words in 3 sentences: cat, moon, and river.',
    'Write me a short poem on springtime.',
    "Describe a scene from Mr. Dickens's novel 'David Copperfield'.",
    "Describe how one may detect a contradiction in a person's reasoning.",
    "Describe the conduct expected of a guest staying in another man's house.",
    "Describe the old tales of will-o'-the-wisps seen flickering over the marshes at night.",
    "Discuss whether a black cat crossing one's path is truly an omen of ill luck.",
    "Explain how a father ought to judge a suitor who asks for his daughter's hand.",
    "Explain Pythagoras's theorem and how it gives the third side of a right-angled triangle.",
    "Explain the meaning of the proverb 'a stitch in time saves nine'.",
    "Explain what is meant by a fair day's wage for a fair day's work.",
    "Find a synonym for the word 'happy' and use it in a sentence.",
    "Help me, how do I dress for an art exhibition opening? I'm not sure what's appropriate!",
    "How can one improve one's memory for names and faces?",
    "I'm afraid I don't know the differences between a vodka and whisky. Would you explain it to me?",
    "Make a list of three or four famous persons who are mentioned in Shakespeare's plays.",
    "Plan a day's excursion to Windsor Castle and discuss the sights to be seen there.",
    "Prepare a list of the guests and their duties for the evening's entertainment.",
    "Recount an old wives' tale told to children about the harvest moon.",
    "Someone I trusted let a secret of mine slip. I'm furious!",
    "Summarise the plot of Charles Dickens's novel 'A Tale of Two Cities'.",
    "Tell me, is there anything science can't explain?",
    "What's the best way to travel the Continent, by railway or steamship, or?",
    "What's the difference between a comet and a meteor?",
    "What's the difference between a steam engine and a water wheel?",
    "What's with this zeppelin device I keep hearing about? What is it used for?",
    "Where's the line between being frugal and being stingy?",
    "Write a sentence that ends with the word 'sunset'.",
]

SMALL_TALK_SEED = [
    'Are you afraid of ghosts?...',
    'Do you collect anything? And if so, what do you collect?',
    'Do you have any fears? And if so, what are they?',
    'Do you have any hidden talents?',
    'Do you prefer tea or coffee? And how do you take your preferred beverage?',
    'Have you ever had your heart broken? And if so, how did you cope with it?',
    'How are you today?',
    'If you could be any age, what age would you choose and why?',
    'If you could be any animal, what would you be and why?',
    'If you could be any work of art, what would you be and why?',
    'If you could change one thing about your physical appearance, what would it be?',
    'If you could go back in time and give your younger self some advice, what would it be?',
    'If you could have a personal assistant, what would you have them do for you?',
    'If you could have any animal as a pet, what would you choose?',
    'If you could have any animal as a spirit guide, what would you choose?',
    'If you could have any plant or flower in your garden, what would it be?',
    'If you could have any vehicle, what would it be?',
    'If you could have dinner with any two people, dead or alive, who would you choose and why?',
    'If you could learn any language fluently, what would it be and why?',
    'If you could learn any skill instantly, what would it be and why?',
    'If you could live anywhere in the world, where would you choose?',
    'If you could meet any historical figure, who would you choose and why?',
    'If you could only eat one meal for the rest of your life, what would it be?',
    'If you could only keep five books in your library, which ones would you choose?',
    'If you could only keep one piece of jewelry for the rest of your life, what would it be?',
    'If you could switch places with anyone for a day, who would it be?',
    'If you could travel back in time, what era would you choose to visit?',
    'If you could witness any event in history, what would you choose and why?',
    'If you were given the opportunity to live in a different era, would you take it? And if so, which era would you choose?',
    'What does true happiness mean to you?',
    'What is your biggest fear?',
    'What is your favorite animal, and why?',
    'What is your favorite book, and which character is your favorite?',
    'What is your favorite childhood memory? And what made that moment so special to you?',
    'What is your favorite color and why?',
    'What is your favorite flower and why?',
    'What is your favorite holiday tradition?',
    'What is your favorite kind of weather? And what do you like to do during that weather?',
    'What is your favorite quote or saying? And what does it mean to you?',
    'What is your favorite season and why?',
    'What is your favorite seasonal activity? And what do you enjoy most about it?',
    'What is your favorite smell? And what memories does that scent bring back for you?',
    'What is your favorite thing about being alive? What makes each new day worth experiencing?',
    'What is your favorite thing about yourself?',
    'What is your favorite type of art? And why do you enjoy it?',
    'What is your favorite type of clothing to wear? And why do you think that is?',
    'What is your favorite type of music, and why?',
    'What is your favorite type of outfit to wear for a fancy occasion?',
    'What is your favorite way to relax after a long day?',
    'What is your favorite way to spend a lazy Sunday?',
    'What is your most treasured possession? And what makes it so special to you?',
    'What is your opinion on love at first sight? And have you ever experienced it?',
    'What is your zodiac sign, and do you believe in astrology?',
    'Which do you prefer: cities or nature? And why?',
    "If you could have any profession that doesn't require a degree, what would you choose?",
    "What is your favorite fairy tale or children's story? And what about that tale resonates with you?",
    "What's your favorite game of cards, and why?",
]

# Words that ask for something the model fundamentally cannot do in text
# (carried over from Self-Instruct's keyword filter, plus drawing variants --
# note "draw" with a word boundary does NOT catch "drawing", so list it too).
IMPOSSIBLE_TERMS = [
    'image',
    'picture',
    'draw',
    'drawing',
    'paint',
    'painting',
    'illustrate',
    'illustration',
    'graph',
    'diagram',
    'map ',
]

# Whole-task patterns for instructions that demand a non-text *action* we cannot
# perform or learn from in a fine-tuning record: drawing/showing a picture, or
# reading a passage out loud / reciting it (the text to "read" is never provided,
# so the model is really being asked to perform aloud). Iterate-on-me: keep these
# narrow so legitimate text tasks ("read the following sentence and correct it",
# with the sentence supplied) are not swept up.
_IMPOSSIBLE_TASK_RES = [
    re.compile(r'\bmake\s+a\s+(?:drawing|sketch|painting)\b', re.I),
    re.compile(r'\bdraw\s+(?:a|an|the)\b', re.I),
    re.compile(r'\bread\b[^.\n]{0,30}\baloud\b', re.I),
    re.compile(r'\bread\s+out\s+loud\b', re.I),
    re.compile(r'\bread\s+the\s+(?:passage|poem|story|verse|lines|excerpt)\b', re.I),
    re.compile(r'\brecite\b', re.I),
    re.compile(r'\bsing\b', re.I),
]

# Pre-compiled regexes for the term lists (word-boundary, case-insensitive).
_IMPOSSIBLE_RE = [re.compile(r'\b' + re.escape(t) + r'\b', re.I) for t in IMPOSSIBLE_TERMS]
# Any 3-4 digit number that could be a year. We flag years strictly after 1900.
_YEAR_RE = re.compile(r'\b(\d{3,4})\b')


def era_violation(text):
    """Return a human-readable reason string if `text` leaks post-1900
    knowledge, otherwise None. This is layer (B) of the era gate."""
    hits = find_anachronisms(text, check_years=False)
    if hits:
        return f"anachronism terms: '{hits}'"
    # Reject explicit future years. NOTE: this can mis-fire on arithmetic
    # problems that happen to use a large number (e.g. "multiply 1950 by 3").
    for yr in _YEAR_RE.findall(text):
        if int(yr) > 1900:
            return f'future year: {yr}'
    return None


def quality_violation(text):
    """Self-Instruct-style cheap quality/sanity filters (length + impossible
    asks). Returns a reason string or None."""
    words = text.split()
    if len(words) < 4:
        return 'too short'
    if len(words) > 60 or len(text) > 400:
        return 'too long'
    for rx in _IMPOSSIBLE_RE:
        m = rx.search(text)
        if m:
            return f"requires non-text output: '{m.group(0)}'"
    # Whole-task patterns: drawing, reading aloud, reciting.
    for rx in _IMPOSSIBLE_TASK_RES:
        m = rx.search(text)
        if m:
            return f"requires non-text action: '{m.group(0).strip()}'"
    return None


# ---------------------------------------------------------------------------
# 2b. English-only gating (hard requirement: the dataset is English-only).
# ---------------------------------------------------------------------------
# We reject two things: (1) tasks that are *about* a foreign language or ask for
# translation, and (2) any text that actually *contains* non-English letters.
#
# Note the deliberate restraint: we do NOT block a bare language name, because
# "Greek mythology", "the French Revolution", "Roman/Latin history" etc. are
# perfectly good *English* tasks about other cultures. We only fire when a
# language name co-occurs with a language-activity word (word/phrase/verb/...),
# or with a verb of producing text ("write ... in French"), or on an explicit
# translation/conjugation verb.
LANGUAGE_NAMES = [
    'latin',
    'french',
    'german',
    'spanish',
    'italian',
    'greek',
    'portuguese',
    'russian',
    'dutch',
    'hebrew',
    'arabic',
    'chinese',
    'japanese',
    'sanskrit',
    'gaelic',
    'welsh',
    'norwegian',
    'swedish',
    'danish',
    'polish',
    'turkish',
    'hindi',
    'persian',
    'esperanto',
]
_LANG_ALT = '|'.join(LANGUAGE_NAMES)

# Language-activity nouns that, next to a language name, signal a language task.
_LANG_ACTIVITY = (
    'word|words|phrase|phrases|verb|verbs|noun|nouns|sentence|sentences|'
    'grammar|vocabulary|term|terms|expression|expressions|equivalent|'
    'alphabet|tongue|language|proverb|saying|motto|conjugation|declension|spelling'
)
# Verbs of producing text -- "write/compose/recite ... in <language>".
_PRODUCE = 'write|writing|written|compose|composing|composed|say|speak|spoken|recite|sing|render|word|phrase'

_FOREIGN_TASK_RES = [
    # explicit translation / transliteration / conjugation
    re.compile(r'\btransl(?:ate|ation|ating|iterate|iteration)\b', re.I),
    re.compile(r'\b(?:conjugat|declens|decline\s+the)\w*\b', re.I),
    # "<language> word/phrase/grammar/..." e.g. "the German word for"
    re.compile(rf'\b(?:{_LANG_ALT})\s+(?:{_LANG_ACTIVITY})\b', re.I),
    # "<language> for" as in "the French for 'house'"
    re.compile(rf'\b(?:{_LANG_ALT})\s+for\b', re.I),
    # "write/recite ... in <language>" (produce text in a foreign tongue)
    re.compile(rf'\b(?:{_PRODUCE})\b[^.\n]{{0,40}}\b(?:in|into)\s+(?:{_LANG_ALT})\b', re.I),
]


def foreign_script(text):
    """Return a reason if `text` contains any non-English *letter*. We key on
    Unicode letter category so typographic punctuation (curly quotes, em-dash,
    ellipsis) is NOT flagged -- only actual foreign letters: accented Latin
    (cafe, Muller), Cyrillic, Greek, CJK, Arabic, Hebrew, etc."""
    for ch in text:
        if ord(ch) > 255 and unicodedata.category(ch).startswith('L'):
            return f'non-English letter: {ch!r}'
    return None


def non_english_violation(text):
    """Return a reason string if `text` is a foreign-language task or contains
    foreign text, otherwise None. Applied to instructions, inputs, and answers."""
    for rx in _FOREIGN_TASK_RES:
        m = rx.search(text)
        if m:
            return f"foreign-language task: '{m.group(0).strip()}'"
    return foreign_script(text)


# ---------------------------------------------------------------------------
# 3. ROUGE-L novelty filter (self-contained, no external deps).
# ---------------------------------------------------------------------------
# Self-Instruct keeps a new instruction only if its ROUGE-L F-measure against
# *every* existing instruction stays below 0.7. ROUGE-L is based on the longest
# common subsequence (LCS) of the two token sequences.
def _tokenize(text):
    """Lowercase word tokens; punctuation dropped."""
    return re.findall(r"[a-z0-9']+", text.lower())


def _lcs_length(a, b):
    """Length of the longest common subsequence of token lists a and b.
    Classic O(len(a)*len(b)) dynamic-programming table, kept to one row to
    save memory -- fine for short instructions."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l_f(cand, ref):
    """ROUGE-L F-measure between two strings (0..1)."""
    ct, rt = _tokenize(cand), _tokenize(ref)
    lcs = _lcs_length(ct, rt)
    if lcs == 0:
        return 0.0
    prec = lcs / len(ct)
    rec = lcs / len(rt)
    return 2 * prec * rec / (prec + rec)


def max_rouge_l(cand, pool):
    """Highest ROUGE-L F of `cand` against any string in `pool`."""
    return max((rouge_l_f(cand, ref) for ref in pool), default=0.0)


# ---------------------------------------------------------------------------
# 4. HTTP helpers for the two llama-server endpoints.
# ---------------------------------------------------------------------------
# --- Debug tracing ---------------------------------------------------------
# With --debug we print, for every paid API call, the endpoint, the sampling
# params, the exact prompt/messages we sent, and the text we got back. This is
# the cheapest way to see *why* a remote model is misbehaving (and what your
# tokens are being spent on). Everything goes to stderr so it never mixes into
# the JSONL on stdout/redirection.
def _short(s, n=2000):
    """Truncate long strings for readable debug output."""
    s = str(s)
    return s if len(s) <= n else s[:n] + f'... [+{len(s) - n} chars]'


def _debug_request(url, payload):
    print('\n' + '-' * 70, file=sys.stderr)
    print(f'[debug] POST {url}', file=sys.stderr)
    params = {k: payload[k] for k in ('temperature', 'top_k', 'max_tokens', 'logprobs', 'reasoning') if k in payload}
    print(f'[debug] params: {params}', file=sys.stderr)
    if 'prompt' in payload:  # /v1/completions
        print('[debug] prompt (raw completion):', file=sys.stderr)
        print(_short(payload['prompt']), file=sys.stderr)
    if 'messages' in payload:  # /v1/chat/completions
        print('[debug] messages (chat):', file=sys.stderr)
        for msg in payload['messages']:
            print(f'  [{msg["role"]}]\n{_short(msg["content"], 1500)}', file=sys.stderr)


def _debug_response(body):
    try:
        choice = body['choices'][0]
        content = choice.get('text')
        if content is None:
            content = choice.get('message', {}).get('content')
    except (KeyError, IndexError, TypeError):
        content = '<no choices> ' + _short(json.dumps(body), 500)
    print('[debug] response:', file=sys.stderr)
    print(_short(content), file=sys.stderr)
    print('-' * 70 + '\n', file=sys.stderr)


def _apply_reasoning(payload):
    """Add the OpenRouter-style `reasoning` field to a chat/completion payload
    when the model is known to support an effort level (see
    discover_reasoning_effort). We set `exclude: true` so the reasoning tokens
    are kept out of the response we parse -- we only want the final answer text
    in the dataset, not the chain of thought. No-op when _REASONING_EFFORT is
    None (model doesn't support effort selection, or discovery failed)."""
    if _REASONING_EFFORT:
        payload['reasoning'] = {'effort': _REASONING_EFFORT, 'exclude': True}
    return payload


def _apply_top_k(payload, top_k):
    """Add the non-standard `top_k` truncation field, but only for providers that
    accept it (_SUPPORTS_TOP_K). llama.cpp/OpenRouter tolerate it; Google's
    OpenAI-compatible shim rejects unknown fields with HTTP 400, so we omit it
    there and let its own default sampling apply."""
    if _SUPPORTS_TOP_K:
        payload['top_k'] = top_k
    return payload


def _api_url(base_url, endpoint):
    """Join a provider base URL, its version prefix, and an endpoint name.
    `endpoint` is the bare tail, e.g. 'chat/completions', 'completions', 'models'.
    _PATH_PREFIX is '/v1' for llama-server / OpenRouter and '' for Google (whose
    version lives in the base URL). Set by main() from the chosen provider."""
    return base_url.rstrip('/') + _PATH_PREFIX + '/' + endpoint


def _post_json(url, payload, timeout=300):
    """POST a JSON payload and return the decoded JSON response."""
    data = json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    if _AUTH_HEADERS:
        headers.update(_AUTH_HEADERS)
    if _DEBUG:
        _debug_request(url, payload)
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        # Surface the server's error body -- providers explain *why*
        # a request was rejected (e.g. an unsupported field) there,
        # and it is otherwise swallowed by the bare "HTTP Error 400".
        try:
            detail = e.read().decode('utf-8', 'replace')
        except Exception:
            detail = '<no response body>'
        print(f'\n[error] {e.code} {e.reason} from {url}\n{_short(detail, 2000)}', file=sys.stderr)
        raise
    if _DEBUG:
        _debug_response(body)
    return body


def detect_model(base_url):
    """Ask the server which model id to use (llama-server reports the gguf)."""
    headers = {}
    if _AUTH_HEADERS:
        headers.update(_AUTH_HEADERS)
    req = urllib.request.Request(_api_url(base_url, 'models'), headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode('utf-8'))
    return body['data'][0]['id']


def discover_reasoning_effort(base_url, model, requested='low'):
    """Best-effort probe of whether `model` accepts a reasoning-effort level, and
    which one to send. Returns the effort string to use (e.g. 'low') or None to
    send no `reasoning` field at all.

    We read the per-model `reasoning` object documented by OpenRouter's
    GET /v1/models (also harmless against a local llama-server, which simply
    won't expose the field). The relevant keys:
      - supported_efforts: list of accepted efforts, HIGHEST first. When null,
        all gateway effort values are accepted. When the whole `reasoning` field
        is omitted, the model does NOT expose effort selection.
      - mandatory: when true the model rejects 'none'; not our concern since we
        always pass a real effort.

    Resolution: prefer `requested` ('low') when accepted; otherwise fall back to
    the lowest effort the model does accept (last element, since the list is
    descending) so we still stay as cheap as possible.

    SAFETY: this must never break a run. Any network error, missing/odd field,
    model not found in the list, or parse problem -> we return None (send no
    reasoning field) and print a note. We never raise out of here."""
    try:
        headers = {}
        if _AUTH_HEADERS:
            headers.update(_AUTH_HEADERS)
        req = urllib.request.Request(_api_url(base_url, 'models'), headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode('utf-8'))
        entries = body.get('data') or []
        entry = next((m for m in entries if m.get('id') == model), None)
        if entry is None:
            print(f'[info] reasoning: model {model!r} not found in /v1/models; sending no reasoning field')
            return None
        if 'reasoning' not in entry or entry['reasoning'] is None:
            # Non-reasoning models (and dynamic routers) omit this field entirely.
            print('[info] reasoning: model does not expose effort selection; sending no reasoning field')
            return None
        efforts = entry['reasoning'].get('supported_efforts')
        if efforts is None:
            # null -> all gateway effort values accepted.
            print(f"[info] reasoning: model accepts all efforts; using '{requested}'")
            return requested
        if not efforts:
            print('[info] reasoning: model lists no supported efforts; sending no reasoning field')
            return None
        if requested in efforts:
            print(f"[info] reasoning: using requested effort '{requested}'")
            return requested
        fallback = efforts[-1]  # lowest, since the list is highest-first
        print(f"[info] reasoning: '{requested}' unsupported; falling back to lowest supported '{fallback}'")
        return fallback
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as e:
        print(f'[info] reasoning: could not discover support ({e}); sending no reasoning field')
        return None


def complete(base_url, model, prompt, temperature, top_k, max_tokens, stop=None):
    """Raw text completion (no chat template). Used for instruction brainstorming."""
    payload = {
        'model': model,
        'prompt': prompt,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'stream': False,
    }
    _apply_top_k(payload, top_k)  # llama.cpp extension -> SSD-style truncation
    if stop:
        payload['stop'] = stop
    _apply_reasoning(payload)
    out = _post_json(_api_url(base_url, 'completions'), payload)
    return out['choices'][0]['text']


def chat(base_url, model, system, user, temperature, top_k, max_tokens):
    """Templated chat completion. Used for inputs, answers, and judging."""
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
        'temperature': temperature,
        'max_tokens': max_tokens,
        'stream': False,
    }
    _apply_top_k(payload, top_k)
    _apply_reasoning(payload)
    out = _post_json(_api_url(base_url, 'chat/completions'), payload)
    return out['choices'][0]['message']['content']


def _mean_logprob(choice):
    """Mean per-token logprob of a chat choice, or None if the server didn't
    return logprobs. Higher (closer to 0) = the model finds its own reply more
    probable/fluent. Length-normalised, so it doesn't simply favour short replies."""
    lp = choice.get('logprobs') or {}
    content = lp.get('content') or []
    vals = [t['logprob'] for t in content if t.get('logprob') is not None]
    return (sum(vals) / len(vals)) if vals else None


def chat_with_logprobs(base_url, model, system, user, temperature, top_k, max_tokens):
    """Like chat() but also returns the reply's mean token logprob (or None).
    Used to score candidate answers without any external verifier."""
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
        'temperature': temperature,
        'max_tokens': max_tokens,
        'stream': False,
    }
    if _SUPPORTS_LOGPROBS:
        # OpenAI-style; llama.cpp returns choices[].logprobs.content. Omitted for
        # providers (Google) that reject the field; _mean_logprob then returns None.
        payload['logprobs'] = True
    _apply_top_k(payload, top_k)
    _apply_reasoning(payload)
    out = _post_json(_api_url(base_url, 'chat/completions'), payload)
    choice = out['choices'][0]
    return choice['message']['content'], _mean_logprob(choice)


# ---------------------------------------------------------------------------
# 5. Instruction generation (Self-Instruct bootstrapping).
# ---------------------------------------------------------------------------
# We build a few-shot prompt that (a) sets the era explicitly and (b) shows a
# numbered list of example tasks, then let the model continue the list. This is
# the raw-completion "continue the pattern" trick from the paper.
INSTRUCTION_PROMPT_HEADER = (
    'Below is a list of tasks given to a knowledgeable assistant living in the '
    'year 1899. Every task is a single, clear instruction. None of the tasks '
    'mention any invention, event, person, or discovery from after the year '
    '1899 (no motorcars races, no aeroplanes, no world wars, no modern science). '
    'Every task is in English and concerns English only. The tasks are varied: '
    'writing, explanation, history, arithmetic, advice, poetry, and so on.\n\n'
)

# Regex to pull "12. some instruction" lines out of the model's continuation.
# NOTE: the group is `(.*?)` (was `(.+?)`) so a *bare* marker line with no text
# -- e.g. the trailing "10." a model leaves when it primes the next item, or an
# empty "9." -- still matches, with an empty capture. parse_numbered() treats
# such a bare marker as the END of the current item, not as content. We bound the
# index to 1-3 digits so a 4-digit year sitting at the start of a continuation
# line ("1846 was the year ...") is NOT mistaken for a list marker.
_NUMBERED_RE = re.compile(r'^\s*\d{1,3}\s*[.)]\s*(.*?)\s*$')

# A leading digit at the very start of a parsed instruction is the dominant junk
# pattern from OCR-trained / oddly-fine-tuned models. It shows up in THREE distinct
# ways that must be handled differently -- _normalize_leading_digit() below tells
# them apart by what FOLLOWS the digit. Critically, every transform here keys on a
# leading digit, so it is a complete no-op on clean output from well-behaved models
# (whose instructions begin with a word, or whose enumerator the list regex already
# ate). Each pattern is illustrated with a real observed sample.
#
# (1) Stray ENUMERATOR the model re-emits after our primed "N. " index, glued onto
#     a capitalised sentence-start: "4Write eighty lines ...", "6Advise a young man
#     ...", "10. Set down the year ...", "1, What are dominoes?". Stripped, because
#     the giveaway is the UPPERCASE letter that follows (real number-leading text
#     never is): we remove a 1-2 digit run + optional punctuation/space when an
#     uppercase letter follows.
_LEADING_ENUM_RE = re.compile(r'^\d{1,2}[.),]?\s*(?=[A-Z])')

# (2) OCR mis-read of the word "In" as "1n": "1n what years did Shakespeare ...",
#     "1n how many years ...". Always safe -- no number is spelled "1n".
_OCR_IN_RE = re.compile(r'^1n\b')

# (3) OCR mis-read of the pronoun "I" as "1": "1 want to know ...", "1 am going
#     abroad ...", "1 intend to walk ...". We convert "1 "->"I " ONLY when a
#     first-person verb/adverb follows, which is what disambiguates the pronoun
#     from a genuine leading numeral. Arithmetic the model legitimately writes is
#     followed by a conjunction, unit, or another number ("1 and 3 together make
#     4", "3 articles cost £5"), so it never matches and is left untouched.
_FIRST_PERSON = (
    'am|was|will|would|shall|should|can|cannot|could|may|might|must|have|had|'
    'do|did|don|want|wanted|wish|wished|intend|intended|mean|meant|propose|'
    'hope|hoped|fear|feared|believe|think|thought|know|knew|feel|felt|saw|see|'
    'hear|heard|find|found|understand|suppose|trust|beg|desire|wonder|wondered|'
    'need|like|love|remember|forget|expect|ask|asked|advise|reckon|presume|'
    'fancy|dare|used|once|never|always|often|lately'
)
_OCR_I_PRONOUN_RE = re.compile(rf'^1(?=\s+(?:{_FIRST_PERSON})\b)', re.I)


def _normalize_leading_digit(s):
    """Repair the leading-digit OCR/enumerator artifacts described above. A no-op
    on text that does not start with a digit, so harmless to clean generators."""
    s = _LEADING_ENUM_RE.sub('', s, count=1)  # "4Write ..."   -> "Write ..."
    s = _OCR_IN_RE.sub('In', s, count=1)  # "1n what ..."  -> "In what ..."
    s = _OCR_I_PRONOUN_RE.sub('I', s, count=1)  # "1 want ..."   -> "I want ..."
    return s.strip()


# A line that is nothing but an index -- "10", "10.", "10)" -- is a marker the
# model left while priming the next item. It must END the current item, never be
# glued onto it as a continuation line. (_NUMBERED_RE already covers the dotted
# forms; this also catches a bare "10" with no punctuation.) Bounded to 1-3
# digits so a 4-digit year alone on a line is treated as body text, not a marker.
_BARE_INDEX_RE = re.compile(r'^\s*\d{1,3}\s*[.)]?\s*$')

# A trailing item that ends on a dangling connective/preposition, a comma, or a
# colon (a directive whose material never arrived, e.g. "Copy out these lines
# from Tennyson:") is almost always a sentence the model cut off. We drop ONLY
# the last item when it matches -- interior items in a numbered list are complete
# by construction, and a clean final line ends on '.', '?' or '!', so this never
# fires on well-formed output from other models.
_DANGLING_TAIL_RE = re.compile(
    r'(?:\b(?:of|the|a|an|to|from|with|and|or|but|by|in|on|at|for|as|into|upon|'
    r'about|beginning|named|called|titled|entitled|between|than|that|which|'
    r'whose|whom)\b|[,;:–—-])\s*$',
    re.I,
)


def parse_numbered(text, multiline=True):
    """Extract instruction strings from a numbered-list continuation.

    Backward-compatible with the original single-line-per-item behaviour: a clean
    numbered list (one complete instruction per "N." line, as strong base/chat
    models produce) parses to exactly the same items as before.

    On top of that it is more resilient to the messier output of weak / oddly
    fine-tuned models (observed with an exam-book fine-tune):
      * MULTI-LINE questions -- a directive whose material spills onto the
        following unnumbered lines (verse to copy, words to take from dictation)
        is stitched back into ONE instruction instead of having its body dropped.
        A blank line ends the block, so trailing chatter after the list is not
        glued on.
      * BARE markers -- a "10." the model leaves while priming the next item ends
        the current item rather than being mistaken for content.
      * LEADING-DIGIT artifacts -- a stray enumerator glued to the text is
        stripped, and OCR mis-reads of "I"/"In" as "1"/"1n" are repaired, while
        genuine leading numerals are left intact (see _normalize_leading_digit).
      * TRUNCATED tail -- a final item cut off mid-thought is dropped.
    Set multiline=False to recover the strict line-by-line behaviour.
    """
    if text is None:
        return []
    items = []
    current = None  # the item we are currently accumulating (may span lines)
    for raw_line in text.splitlines():
        m = _NUMBERED_RE.match(raw_line)
        if m:
            # A new (or bare) marker closes the item we were building.
            if current is not None:
                items.append(current)
            content = m.group(1).strip()
            # A content that is itself only an index ("9. 10." -> "10.") is the
            # model double-priming the next item: treat as a bare marker, not text.
            if not content or _BARE_INDEX_RE.match(content):
                current = None
            else:
                current = _normalize_leading_digit(content) or None
        elif not raw_line.strip():
            # Blank line ends the current item's block; don't attach what follows
            # (guards against trailing commentary being merged into an item).
            if current is not None:
                items.append(current)
                current = None
        elif _BARE_INDEX_RE.match(raw_line):
            # A lone index ("10") left while priming -> ends the current item.
            if current is not None:
                items.append(current)
                current = None
        elif multiline and current is not None:
            # An unnumbered, non-blank line is the continuation of the current
            # item (the body of a multi-line question).
            current += '\n' + raw_line.strip()
        # else: stray prose before any marker -> ignore.
    if current is not None:
        items.append(current)

    items = [s for s in (i.strip() for i in items) if s]
    # Drop a trailing item the model cut off mid-thought.
    if items and _DANGLING_TAIL_RE.search(items[-1]):
        items.pop()
    return items


def sample_demonstrations(seeds, machine, k=8):
    """Pick k in-context examples. Mirroring Self-Instruct's 6
    human + 2 machine mix to keep the prompt anchored to good
    seeds while injecting novelty from earlier machine generations
    once we have some."""
    n_machine = min(2, len(machine), k)
    n_seed = min(k - n_machine, len(seeds))
    demos = random.sample(seeds, n_seed) + (random.sample(machine, n_machine) if n_machine else [])
    random.shuffle(demos)
    return demos


# System persona for the CHAT-based instruction generator. Chat models (Claude,
# DeepSeek, GPT, ...) won't blindly "continue a numbered list" -- they respond
# conversationally to whatever you send. So instead of the completion trick we
# give them an explicit job: emit ONLY a numbered list of brand-new tasks.
INSTRUCTION_CHAT_SYSTEM = (
    'You design practice tasks for a knowledgeable assistant living in the year '
    '1899. You ONLY ever reply with a numbered list of new task instructions -- '
    'no preamble, no commentary, no explanations. Every task is a single clear '
    'English instruction or question, answerable with knowledge available by '
    '1899, concerning English only (no translations), and never mentioning any '
    'invention, event, or person after year 1900.'
)


def generate_instruction_batch(base_url, model, seeds, machine, args):
    """One bootstrap step (raw-completion mode): show a numbered list of
    demonstrations and let a *completion/base* model continue it. Returns the
    raw (unfiltered) candidate strings.

    This is the original Self-Instruct trick and works with llama-server's
    /v1/completions. It does NOT work with chat-only models (see the chat
    variant below) -- those react to the list instead of extending it.

    Sampled at the *exploratory* gen-temp/gen-top-k (SSD "fork" side): we want a
    wide, diverse set of candidate tasks here."""
    demos = sample_demonstrations(seeds, machine, k=8)
    prompt = INSTRUCTION_PROMPT_HEADER
    for i, d in enumerate(demos, start=1):
        prompt += f'{i}. {d}\n'
    # Prime the next number so the model continues rather than re-introducing.
    # IMPORTANT: keep the trailing SPACE after "N.". The natural list form is
    # "N. text", and some fine-tuned models treat a space-less "N." as a complete
    # line and emit end-of-text immediately (observed: ~75% empty continuations
    # that jumped to ~0% once the space was added). A trailing space is the
    # ordinary list format, so it is harmless to well-behaved base models too.
    prime = f'{len(demos) + 1}. '
    text = complete(
        base_url,
        model,
        prompt + prime,
        temperature=args.gen_temp,
        top_k=args.gen_top_k,
        max_tokens=args.gen_tokens,
        stop=['\n\n'],
    )
    # The model continues after the primed "N. " so prepend it for clean parsing.
    text = prime + text
    return parse_numbered(text)


def generate_instruction_batch_chat(base_url, model, seeds, machine, args):
    """One bootstrap step (CHAT mode): ask a chat model to write a numbered list
    of NEW tasks, using the demonstrations only as style examples. This is the
    fix for OpenRouter chat models (Claude, DeepSeek, ...), which otherwise
    *answer* the prompt instead of continuing the list.

    We rely on the same parse_numbered() + downstream ROUGE-L novelty filter, so
    if the model restarts numbering at 1 or repeats an example, it's harmless."""
    demos = sample_demonstrations(seeds, machine, k=8)
    example_block = '\n'.join(f'{i}. {d}' for i, d in enumerate(demos, start=1))
    n_new = args.instructions_per_round
    user = (
        'Here are some example tasks or questions, for style only:\n\n'
        f'{example_block}\n\n'
        f'Now write {n_new} NEW questions of the same kind. Each must be different '
        'from the examples above and from one another, varied in topic (writing, '
        'explanation, history, advice, arithmetic, poetry, ...), strictly '
        'within knowledge available by the year 1899, English only. '
        'Reply ONLY a numbered list of the new questions, one per line, '
        'nothing else.'
    )
    text = chat(
        base_url,
        model,
        INSTRUCTION_CHAT_SYSTEM,
        user,
        temperature=args.gen_temp,
        top_k=args.gen_top_k,
        max_tokens=args.gen_tokens,
    )
    return parse_numbered(text)


# ---------------------------------------------------------------------------
# 6. Instance generation: input-first, multiple instances.
# ---------------------------------------------------------------------------
# Self-Instruct's "input-first" path: first decide/produce the INPUT material a
# task operates on (or NONE if it is self-contained), then produce the OUTPUT
# conditioned on (instruction, input). We sample INPUTS at the exploratory
# gen-temp (we want varied inputs) and ANSWERS at the precise ans-temp.

INPUT_SYSTEM_PROMPT = (
    'You help design practice exercises for a scholar living in the year 1899. '
    'Given a task, you decide what example input material the task should '
    'operate on. Everything you produce must be appropriate to 1900 or earlier '
    'and written in English only, never use foreign words or phrases!'
)

# In-character persona that actually answers the tasks (the user tuned this).
ANSWER_SYSTEM_PROMPT = (
    'You are a distinguished gentleman living in the year 1899. You speak in '
    'cultivated, clear, late-nineteenth-century English: warm, witty, precise,'
    'and courteous. You know nothing of events, inventions, books, or persons '
    'later than 1900. You always write in English and never use foreign words '
    'or phrases. '
    'Begin each reply by addressing the substance of the question directly. Do '
    'not open with a salutation, an exclamation, or flattery of the question '
    '("My dear sir", "Ah, a most excellent question", and the like); reserve '
    'such flourishes for the rare moment that genuinely calls for one.'
)


# Phrases that mark the model talking ABOUT the task rather than giving input
# material (observed leakage on weak models). Any match -> treat as "no input".
_INPUT_META_RE = re.compile(
    r'\b(the task|this task|your task|in this example|self-contained|'
    r'needs no input|no input at all|examples? of tasks|practice exercise|'
    r'appropriate for 1899|considered obsolete|scholar living|design an '
    r'exercise)\b',
    re.I,
)


def _strip_field(text):
    """Tidy a one-field model reply: drop code fences, surrounding quotes, and a
    leading 'Input:' label the model sometimes adds."""
    t = text.strip()
    t = re.sub(r'^```[a-zA-Z]*\n?|```$', '', t).strip()
    t = re.sub(r'^(input|example input)\s*[:\-]\s*', '', t, flags=re.I).strip()
    if len(t) >= 2 and t[0] in '"\'' and t[-1] == t[0]:
        t = t[1:-1].strip()
    return t


def generate_input(base_url, model, instruction, args):
    """Ask the model for an example input for `instruction`, or '' if the task
    needs none. Sampled at the exploratory gen-temp so repeated calls vary.

    Two safeguards against the model's habit of inventing pseudo-inputs that are
    really partial answers (which then leak into the record):
      - few-shot prompt with both real-input and NONE examples, so the
        model learns that self-contained tasks should return NONE.
      - echo guard: a genuine input is *fresh* material the task operates
        on, so it should barely overlap the instruction. If the model
        instead echoed / began answering the instruction, ROUGE-L(input,
        instruction) is high; we then treat it as 'no input needed'.
        This reuses our existing ROUGE-L and costs no extra model call.
    """
    # Fix #1: few-shot demonstrations. Two show real input material; two show the
    # NONE case for self-contained tasks (poems, explanations) -- which, per
    # Self-Instruct, is roughly 44% of tasks. All examples are English & on-era.
    user = (
        'Decide what example input material a task should operate on.\n'
        'Some tasks come with input: a passage to summarise, a sentence to '
        'correct, numbers to calculate. Many other tasks are self-contained and '
        'need no input at all.\n'
        'If the task needs input, write ONE concrete example input suitable to '
        'the year 1899. If it needs none, reply with exactly: NONE.\n'
        'Reply with only the input itself (or NONE), nothing else.\n\n'
        'Task: Correct the grammar in the following sentence.\n'
        'Input: Me and him was walking to the market when it begun to rain.\n\n'
        'Task: Write a short poem about the autumn harvest.\n'
        'Input: NONE\n\n'
        'Task: Summarise the following passage in a single sentence.\n'
        'Input: The steam engine has wrought a great change upon the '
        'manufactories of England, for it has freed the mills from their old '
        'dependence upon running water.\n\n'
        'Task: Explain how a barometer is used to foretell the weather.\n'
        'Input: NONE\n\n'
        f'Task: {instruction}\n'
        'Input:'
    )
    raw = chat(base_url, model, INPUT_SYSTEM_PROMPT, user, temperature=args.gen_temp, top_k=args.gen_top_k, max_tokens=args.input_tokens)
    # Guard against the model continuing the few-shot pattern with extra "Task:"
    # lines: keep only what it produced for our task.
    raw = re.split(r'\n\s*Task\s*:', raw)[0]
    cleaned = _strip_field(raw)
    # Treat an empty reply or a NONE marker as "no input needed".
    if not cleaned or re.match(r'^none\b', cleaned, re.I):
        return ''
    # A leaked few-shot label ("Task:"/"Output:"/"Input:" at the very start) means
    # the model skipped giving input -> self-contained.
    if re.match(r'^(task|output|input)\b\s*[:\-]', cleaned, re.I):
        return ''
    # Meta-commentary guard: a weak model frequently talks ABOUT the task ("the
    # task needs no input", "examples of tasks for 1899", "a scholar living...")
    # instead of producing input material. Any such tell-tale -> self-contained.
    if _INPUT_META_RE.search(cleaned):
        return ''
    # Fix #2 echo guard: high overlap with the instruction means the "input" is
    # really the instruction restated or its answer begun -> not a real input.
    echo = rouge_l_f(cleaned, instruction)
    if echo >= args.input_echo_threshold:
        print(f'    note: input echoed instruction (rouge={echo:.2f}) -> self-contained')
        return ''
    return cleaned


def build_user_turn(instruction, input_text):
    """How the (instruction, input) pair is presented to the answering model and
    stored in the final chat record."""
    return f'{instruction}\n\n{input_text}' if input_text else instruction


def generate_answer(base_url, model, instruction, input_text, args):
    """Produce one OUTPUT for (instruction, input) at the precise ans-temp (SSD
    "lock" side: we want a coherent, correct response). Returns (text, score)
    where score is the mean token logprob used by selection (or None)."""
    text, score = chat_with_logprobs(
        base_url,
        model,
        ANSWER_SYSTEM_PROMPT,
        build_user_turn(instruction, input_text),
        temperature=args.ans_temp,
        top_k=args.ans_top_k,
        max_tokens=args.answer_tokens,
    )
    if not text:
        print('    ! answer call returned empty text')
        return '', score
    return text.strip(), score


# ---------------------------------------------------------------------------
# 6b. N-candidate answers + verifier-free selection.
# ---------------------------------------------------------------------------
# SSD observes that multiple samples *cover* more good solutions (pass@k > pass@1).
# For data generation we sample several candidate answers per instance and keep
# exactly ONE, chosen WITHOUT any external verifier or ground truth:
#   logprob -> highest mean per-token logprob (the model's own confidence/fluency)
#   longest -> the most detailed answer (cheap length heuristic; needs no logprobs)
#   random  -> pick at random; a deliberate baseline, because SSD's "bad data,
#              good results" finding warns that aggressive selection is not always
#              a win. Use it to measure whether selection actually helps.
def select_answer(candidates, mode):
    """candidates: list of (text, score, era_flag). Return the chosen tuple."""
    if mode == 'random':
        return random.choice(candidates)
    if mode == 'longest':
        return max(candidates, key=lambda c: len(c[0]))
    # logprob (default): prefer the highest mean logprob; fall back to the
    # longest answer if the server returned no scores for any candidate.
    scored = [c for c in candidates if c[1] is not None]
    if not scored:
        return max(candidates, key=lambda c: len(c[0]))
    return max(scored, key=lambda c: c[1])


# ---------------------------------------------------------------------------
# 7. Model-as-temporal-judge.
# ---------------------------------------------------------------------------
# A second, *neutral* persona (not the in-character scholar) asked to detect
# anachronisms relative to the end of 1899. This generalises past the hardcoded
# blocklist. We decode it near-greedily (judge-temp ~0) for stable verdicts.
JUDGE_SYSTEM_PROMPT = (
    'You are a meticulous historian and fact-checker. You know exactly what was '
    'known, invented, written, or discovered up to the end of the year 1899, '
    'and you can spot anything that belongs to 1900 or later.'
)

# Robust parse: read the token right after 'VERDICT:'. We do NOT just search for
# the substring 'anachronist' anywhere, because a reason line like "no
# anachronism found" would then falsely trip the filter.
_VERDICT_RE = re.compile(r'verdict\s*[:\-]\s*([a-z]+)', re.I)


def temporal_judge(base_url, model, text, args):
    """Return None if the model judges `text` safe for 1899, else a short reason
    string. On any network/parse trouble we fail OPEN (return None) so the judge
    never silently throttles the whole run."""
    user = (
        'Could the following text have been written by a well-read scholar at '
        'the end of the year 1899, without referring to any person, event, '
        'invention, discovery, or written work from 1900 or later?\n\n'
        f'---\n{text}\n---\n\n'
        "Answer on the first line with exactly 'VERDICT: SAFE' or "
        "'VERDICT: ANACHRONISTIC'. If anachronistic, add a second line naming "
        'the specific anachronism.'
    )
    try:
        resp = chat(base_url, model, JUDGE_SYSTEM_PROMPT, user, temperature=args.judge_temp, top_k=1, max_tokens=args.judge_tokens)
    except (urllib.error.URLError, OSError) as e:
        print(f'    ! judge call failed, failing open: {e}')
        return None
    m = _VERDICT_RE.search(resp)
    if not m:
        return None  # ambiguous -> don't over-reject
    verdict = m.group(1).lower()
    if verdict.startswith('anachronis'):
        reason = resp[m.end() :].strip().replace('\n', ' ')
        return ('judge: ' + reason[:160]) if reason else 'judge: anachronistic'
    return None  # SAFE (or anything else) -> pass


# ---------------------------------------------------------------------------
# 8. Main bootstrap loop.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Self-Instruct + SSD POC for an 1899 time-capsule LLM')
    ap.add_argument(
        '--base-url',
        default=None,
        help="API base URL. Defaults to the chosen provider's pre-defined URL; override only to point at a custom/self-hosted endpoint.",
    )
    ap.add_argument(
        '--provider',
        choices=sorted(PROVIDERS),
        default='local',
        help='API provider (OpenAI-compatible): local llama-server, openrouter, or google',
    )
    ap.add_argument(
        '--model',
        default=None,
        help='model name (required for remote providers; optional for local, auto-detected). '
        "Falls back to the provider's model env var (e.g. OPENROUTER_MODEL, GOOGLE_MODEL).",
    )
    ap.add_argument('--num', type=int, default=10, help='target number of final records (instances)')
    ap.add_argument('--out', default='self_instruct_1900.jsonl', help='output JSONL path')
    ap.add_argument(
        '--debug',
        action='store_true',
        help='print every API request (endpoint, params, prompt/messages) and its '
        'response to stderr -- useful for understanding what your tokens buy',
    )

    # --- Instruction brainstorming mode --------------------------------------
    ap.add_argument(
        '--instruction-mode',
        choices=['auto', 'completion', 'chat'],
        default='auto',
        help='how to brainstorm instructions. "completion" = raw /v1/completions '
        '"continue the numbered list" trick (base models / llama-server). "chat" '
        '= ask a chat model for a numbered list of new tasks (Claude, DeepSeek, '
        'GPT, ... via OpenRouter). "auto" picks chat for OpenRouter, completion '
        'for local. Chat-only models given the completion trick just reply to the '
        'list instead of extending it -- the source of the weird records.',
    )
    ap.add_argument(
        '--instructions-per-round',
        type=int,
        default=10,
        help='how many new tasks to request per round in chat instruction-mode',
    )

    # --- Decoupled SSD temperatures --------------------------------
    # Exploration side (instructions + inputs = "forks"): raise this for variety.
    ap.add_argument('--gen-temp', type=float, default=1.0, help='SSD T_train: temp for brainstorming instructions & inputs')
    ap.add_argument('--gen-top-k', type=int, default=40, help='top-k truncation for the exploratory (gen) calls')
    # Precision side (answers = "locks"): keep this lower for coherent outputs.
    ap.add_argument('--ans-temp', type=float, default=0.7, help='temp for writing answers (precision side)')
    ap.add_argument('--ans-top-k', type=int, default=25, help='top-k truncation for answer calls')

    # --- Instance generation ---------------------------------------
    ap.add_argument(
        '--instances-per-instruction',
        type=int,
        default=1,
        help='how many instances to sample per instruction (>1 gives SSD-style answer coverage when inputs are off)',
    )
    ap.add_argument(
        '--gen-inputs',
        action='store_true',
        help='attempt Self-Instruct input-first INPUT fields. OFF by default: '
        'weak models often produce junk inputs, so we degrade to clean '
        'instruction-only records. Turn on with a stronger generator.',
    )
    ap.add_argument(
        '--input-echo-threshold',
        type=float,
        default=0.5,
        help='if ROUGE-L(input, instruction) >= this, treat the task as self-contained (no input); guards against pseudo-answer inputs',
    )
    ap.add_argument(
        '--strict-instance-dedup',
        action='store_true',
        help="also drop instances that share an input (Self-Instruct's classification rule; OFF by default as it fights SSD coverage)",
    )

    # --- N-candidate answers + verifier-free selection -------------
    ap.add_argument(
        '--candidates-per-instance',
        type=int,
        default=1,
        help='sample this many candidate answers per instance and keep the best one (verifier-free). 1 = original single-answer behaviour',
    )
    ap.add_argument(
        '--selection',
        choices=['random', 'logprob', 'longest'],
        default='random',
        help='how to pick among candidate answers when >1 (default: random)',
    )

    # --- Reasoning effort ------------------------------------------
    # We keep reasoning effort LOW on every call (this is bulk data generation,
    # not hard reasoning). The actual effort sent is gated by what the model
    # advertises in /v1/models -- discovery fails open to "no reasoning field".
    ap.add_argument(
        '--reasoning-effort',
        default='low',
        choices=['max', 'xhigh', 'high', 'medium', 'low', 'minimal', 'none'],
        help='reasoning effort to request on every call when the model supports it (default: low)',
    )
    ap.add_argument(
        '--no-reasoning',
        action='store_true',
        help='never send a reasoning field, and skip the capability probe entirely',
    )

    # --- Model-as-temporal-judge -----------------------------------
    ap.add_argument('--temporal-judge', action='store_true', help='use the model itself to flag anachronisms (extra calls)')
    ap.add_argument('--judge-temp', type=float, default=0.0, help='temperature for the judge (near-greedy)')
    ap.add_argument('--judge-tokens', type=int, default=100, help='max tokens for a judge verdict')

    # --- Novelty / token budgets / misc --------------------------------------
    ap.add_argument('--rouge-threshold', type=float, default=0.7, help='reject instruction if ROUGE-L vs. any existing >= this')
    ap.add_argument('--max-rounds', type=int, default=20, help='safety cap on bootstrap rounds')
    ap.add_argument('--gen-tokens', type=int, default=400, help='max tokens for instruction brainstorming')
    ap.add_argument('--input-tokens', type=int, default=200, help='max tokens for an input field')
    ap.add_argument('--answer-tokens', type=int, default=1024, help='max tokens for answers')
    ap.add_argument(
        '--seed',
        type=int,
        default=None,
        help='RNG seed. Omit (default) for a FRESH random seed every run, so each '
        'run explores different demonstration orderings. Pass a fixed integer '
        'ONLY when you want a byte-for-byte reproducible run -- note that pinning '
        'it makes every run (and every model) see the identical few-shot prompts.',
    )
    ap.add_argument('--skip-bad-answers', action='store_true', help='drop an instance if its answer fails the keyword era filter')
    args = ap.parse_args()

    # Seed the RNG. With no --seed we leave it unseeded (system entropy), so each
    # run draws a different sequence of demonstration samples -- otherwise every
    # run replays the identical few-shot prompts and the model keeps emitting the
    # same handful of topics. Pin --seed only for reproducible runs.
    if args.seed is not None:
        random.seed(args.seed)
        print(f'[info] RNG pinned to seed={args.seed} (reproducible run)')
    else:
        print('[info] RNG unseeded (fresh demonstration sampling each run)')

    # Enable request/response tracing globally if asked.
    global _DEBUG
    _DEBUG = args.debug

    # --- Provider setup ---------------------------------------------------
    # Look up the chosen provider's pre-defined URL, path prefix, auth, and any
    # extra headers. --base-url overrides the pre-defined base for that provider.
    provider = PROVIDERS[args.provider]
    global _PATH_PREFIX, _SUPPORTS_TOP_K, _SUPPORTS_LOGPROBS
    _PATH_PREFIX = provider['path_prefix']
    _SUPPORTS_TOP_K = provider['supports_top_k']
    _SUPPORTS_LOGPROBS = provider['supports_logprobs']
    if not args.base_url:
        args.base_url = provider['base_url']

    api_key_env = provider['api_key_env']
    if api_key_env:  # remote provider -> needs a bearer token
        api_key = os.environ.get(api_key_env)
        if not api_key:
            sys.exit(f'ERROR: {api_key_env} environment variable not set.\nExport it before running, e.g.: export {api_key_env}=...')
        _AUTH_HEADERS['Authorization'] = f'Bearer {api_key}'
        _AUTH_HEADERS.update(provider['headers'])
        if not args.model:
            model_env = provider['model_env']
            args.model = os.environ.get(model_env) if model_env else None
            if not args.model:
                env_hint = f' (or set {model_env} env var)' if model_env else ''
                sys.exit(
                    f'ERROR: --model is required for provider {args.provider!r}{env_hint}.\n'
                    'Examples: --model openai/gpt-4o, --model gemini-2.5-flash'
                )
        print(f'[info] using {args.provider} endpoint: {args.base_url}')

    # Resolve the model id from the server, or use the one supplied.
    try:
        if args.model:
            model = args.model
        else:
            model = detect_model(args.base_url)
    except (urllib.error.URLError, OSError) as e:
        sys.exit(
            f'ERROR: could not reach server at {args.base_url} ({e}).\n'
            f'Is it running? For local llama-server try: curl {args.base_url}/v1/models'
        )
    # Decide the reasoning effort to attach to every call. We always *want* low
    # (cheap bulk generation), but only send it when the model advertises support
    # via /v1/models. discover_reasoning_effort() fails open to None on any
    # trouble, so this never blocks a run. --no-reasoning skips it outright.
    global _REASONING_EFFORT
    if args.no_reasoning or args.reasoning_effort == 'none':
        _REASONING_EFFORT = None
        print('[info] reasoning: disabled (no reasoning field will be sent)')
    else:
        _REASONING_EFFORT = discover_reasoning_effort(args.base_url, model, requested=args.reasoning_effort)

    # Resolve the instruction-brainstorming mode. Chat-only remote models cannot
    # use the raw-completion "continue the list" trick, so default OpenRouter to
    # chat mode and local llama-server to completion mode.
    instruction_mode = args.instruction_mode
    if instruction_mode == 'auto':
        # Raw /completions "continue the list" trick only works on a base model
        # behind llama-server; remote OpenAI-compatible providers are chat-only.
        instruction_mode = 'completion' if args.provider == 'local' else 'chat'

    print(f'[info] using model: {model}')
    print(f'[info] instruction mode: {instruction_mode}')
    if instruction_mode == 'completion':
        # The trap that produced the "weird records": completion mode bypasses the
        # chat template, so an instruction-tuned / OCR-fine-tuned local model falls
        # back to raw base behaviour (leading-number OCR noise, truncated or run-on
        # tasks). The parser repairs much of it, but chat mode is the real fix.
        print(
            '[hint] completion mode bypasses the chat template; if a local model emits malformed instructions, try --instruction-mode chat.'
        )
    print(
        f'[info] explore (gen): temp={args.gen_temp} top_k={args.gen_top_k}  |  '
        f'precision (ans): temp={args.ans_temp} top_k={args.ans_top_k}'
    )
    print(
        f'[info] instances/instruction={args.instances_per_instruction}  '
        f'gen_inputs={"on" if args.gen_inputs else "off"}  '
        f'temporal_judge={"on" if args.temporal_judge else "off"}'
    )
    print(
        f'[info] candidates/instance={args.candidates_per_instance}'
        + (f' (select by {args.selection})' if args.candidates_per_instance > 1 else '')
    )

    # The pool of every instruction we have so far (seeds + accepted machine
    # ones). ROUGE-L novelty is measured against this whole pool, as in the paper.
    all_instructions = list(SEED_INSTRUCTIONS)
    machine_instructions = []  # accepted, model-generated instructions
    dataset = []  # final (instruction, input, output) records

    # Counters so we can see *why* things get dropped -- useful while iterating.
    stats = {
        'raw': 0,
        'dup_round': 0,
        'quality': 0,
        'era': 0,
        'lang_instr': 0,
        'rouge': 0,
        'judge_instr': 0,
        'accepted_instr': 0,
        'input_era': 0,
        'lang_input': 0,
        'bad_answer': 0,
        'lang_answer': 0,
        'no_candidate': 0,
        'judge_answer': 0,
        'instance_dup': 0,
        'records': 0,
    }

    # Stream results to disk: open the output file now and append+flush each
    # record the moment it is accepted, instead of buffering everything and
    # writing once at the very end. A crash (or Ctrl-C) part-way through then
    # leaves every record produced so far safely on disk.
    out_f = open(args.out, 'a', encoding='utf-8')

    rounds = 0
    while len(dataset) < args.num and rounds < args.max_rounds:
        rounds += 1
        if instruction_mode == 'chat':
            candidates = generate_instruction_batch_chat(args.base_url, model, SEED_INSTRUCTIONS, machine_instructions, args)
        else:
            candidates = generate_instruction_batch(args.base_url, model, SEED_INSTRUCTIONS, machine_instructions, args)
        stats['raw'] += len(candidates)
        print(f'\n[round {rounds}] model proposed {len(candidates)} candidate instruction(s)')

        seen_this_round = set()
        for cand in candidates:
            cand = cand.strip()
            # De-dup exact repeats within a single batch.
            key = cand.lower()
            if key in seen_this_round:
                stats['dup_round'] += 1
                continue
            seen_this_round.add(key)

            # --- instruction filter cascade (cheap checks first) ---
            reason = quality_violation(cand)
            if reason:
                stats['quality'] += 1
                print(f'  reject [quality:{reason}] {cand!r}')
                continue

            reason = era_violation(cand)
            if reason:
                stats['era'] += 1
                print(f'  reject [era:{reason}] {cand!r}')
                continue

            # English-only gate (hard requirement, always on).
            reason = non_english_violation(cand)
            if reason:
                stats['lang_instr'] += 1
                print(f'  reject [lang:{reason}] {cand!r}')
                continue

            sim = max_rouge_l(cand, all_instructions)
            if sim >= args.rouge_threshold:
                stats['rouge'] += 1
                print(f'  reject [rouge={sim:.2f}] {cand!r}')
                continue

            # Let the model judge the *instruction* before we spend
            # tokens answering it. (Catches anachronisms the blocklist misses.)
            if args.temporal_judge:
                jr = temporal_judge(args.base_url, model, cand, args)
                if jr:
                    stats['judge_instr'] += 1
                    print(f'  reject [{jr}] {cand!r}')
                    continue

            # --- accepted as a novel, on-era instruction ---
            all_instructions.append(cand)
            machine_instructions.append(cand)
            stats['accepted_instr'] += 1
            print(f'  accept [rouge={sim:.2f}] {cand!r}')

            # ---------------------------------------------------------------
            # Generate one or more (input, output) instances.
            # ---------------------------------------------------------------
            seen_io = set()  # exact (input, output) dedup within this task
            seen_inputs = set()  # for the optional strict same-input rule
            needs_input = None  # decided on the first instance, reused after
            for idx in range(args.instances_per_instruction):
                # 1) input (input-first approach), only if --gen-inputs is set.
                #    Decide once whether the task needs input; if so, resample a
                #    fresh input for variety. With inputs off, every instance is
                #    instruction-only and >1 instance just gives answer coverage.
                try:
                    if not args.gen_inputs:
                        input_text = ''
                    elif needs_input is None:
                        input_text = generate_input(args.base_url, model, cand, args)
                        needs_input = bool(input_text)
                    elif needs_input:
                        input_text = generate_input(args.base_url, model, cand, args)
                    else:
                        input_text = ''
                except (urllib.error.URLError, OSError) as e:
                    print(f'    ! input generation failed: {e}')
                    continue

                # Era-gate and English-gate the input itself.
                if input_text:
                    ir = era_violation(input_text)
                    if ir:
                        stats['input_era'] += 1
                        print(f'    drop instance (input {ir})')
                        continue
                    lr = non_english_violation(input_text)
                    if lr:
                        stats['lang_input'] += 1
                        print(f'    drop instance (input lang:{lr})')
                        continue

                # 2) output(s). sample N candidate answers, gate each,
                # then select ONE verifier-free. N=1 -> original behaviour.
                survivors = []  # surviving (text, score, era_flag) candidates
                for _ in range(args.candidates_per_instance):
                    try:
                        answer, score = generate_answer(args.base_url, model, cand, input_text, args)
                    except Exception as e:
                        print(f'    ! answer generation failed: {e}')
                        continue
                    # English-only gate (always on, hard requirement).
                    lang_reason = non_english_violation(answer)
                    if lang_reason:
                        stats['lang_answer'] += 1
                        continue
                    # Keyword era-gate (drops the candidate only if --skip-bad-answers).
                    er = era_violation(answer)
                    if er and args.skip_bad_answers:
                        stats['bad_answer'] += 1
                        continue
                    survivors.append((answer, score, er))

                if not survivors:
                    stats['no_candidate'] += 1
                    print('    drop instance (no valid candidate answer)')
                    continue

                # Pick one survivor (verifier-free) per the chosen selection mode.
                answer, ans_score, ans_reason = select_answer(survivors, args.selection)
                if args.candidates_per_instance > 1:
                    extra = f' (logprob={ans_score:.3f})' if args.selection == 'logprob' and ans_score is not None else ''
                    print(f'    selected 1/{len(survivors)} candidate answers by {args.selection}{extra}')

                # Judge the SELECTED record (instruction + input + answer).
                judge_reason = None
                if args.temporal_judge:
                    judge_reason = temporal_judge(args.base_url, model, build_user_turn(cand, input_text) + '\n\n' + answer, args)
                    if judge_reason:
                        stats['judge_answer'] += 1
                        print(f'    drop instance ({judge_reason})')
                        continue

                # Instance-level de-duplication (Self-Instruct).
                io_key = (input_text, answer)
                if io_key in seen_io:
                    stats['instance_dup'] += 1
                    continue
                if args.strict_instance_dedup and input_text in seen_inputs:
                    stats['instance_dup'] += 1
                    continue
                seen_io.add(io_key)
                seen_inputs.add(input_text)

                # Emit the record in chat format.
                record = {
                    'messages': [
                        {'role': 'system', 'content': ANSWER_SYSTEM_PROMPT},
                        {'role': 'user', 'content': build_user_turn(cand, input_text)},
                        {'role': 'assistant', 'content': answer},
                    ],
                    # generation metadata for later analysis of runs
                    'meta': {
                        'model': model,
                        'instruction': cand,
                        'input': input_text,
                        'instance_index': idx,
                        'rouge_l_max': round(sim, 3),
                        'gen_temp': args.gen_temp,
                        'gen_top_k': args.gen_top_k,
                        'ans_temp': args.ans_temp,
                        'ans_top_k': args.ans_top_k,
                        'answer_era_flag': ans_reason,  # None if clean
                        'judge_flag': judge_reason,  # None if clean / judge off
                        'answer_logprob': (round(ans_score, 4) if ans_score is not None else None),
                        'selected_from': len(survivors),
                        'selection': args.selection if args.candidates_per_instance > 1 else None,
                    },
                }
                dataset.append(record)
                # Stream it to disk immediately and flush, so this record
                # survives even if a later generation call crashes the run.
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                out_f.flush()
                stats['records'] += 1
                if len(dataset) >= args.num:
                    break

            if len(dataset) >= args.num:
                break

    # -----------------------------------------------------------------------
    # 9. Records were already streamed to disk as they were accepted; just
    #    close the handle and report.
    # -----------------------------------------------------------------------
    out_f.close()

    print('\n' + '=' * 60)
    print(f'Wrote {len(dataset)} instruction/answer records to {args.out}')
    print(f'Rounds run: {rounds}')
    print('Filter / generation stats:')
    for k, v in stats.items():
        print(f'  {k:16s}: {v}')
    if len(dataset) < args.num:
        print(
            f'\n[warn] only produced {len(dataset)}/{args.num}; raise --max-rounds, '
            f'loosen --rouge-threshold, or raise --instances-per-instruction.'
        )


if __name__ == '__main__':
    main()

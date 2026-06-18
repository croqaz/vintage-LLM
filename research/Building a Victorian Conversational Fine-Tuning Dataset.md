# Building a Victorian Conversational Fine-Tuning Dataset

This is a genuinely excellent project — bespoke tokenizer, pre-1900 corpus, a small from-scratch Llama3 model. Your blocker is the dataset, so let's focus entirely on that. The short version: **build a "Golden Dataset" of multi-turn conversations where later turns structurally depend on earlier ones, generated synthetically but anchored hard in your real corpus, ruthlessly filtered for anachronism, and formatted in a consistent chat template.**

Let me walk through the whole pipeline, then address some hard truths about what a 340M model can and can't do.

---

## A Reality Check First (Read This Before Building)

A few things nobody likes to say but you need to hear before investing weeks of work:

1. **Fine-tuning cannot add knowledge or reasoning the base model lacks.** If your 340M base, trained only on Oxford/Gutenberg, doesn't already "know" Frankenstein's plot or how to reason across several sentences, no amount of instruction data will conjure it. Fine-tuning teaches *behavior and format* (how to converse, stay in character, maintain context), not raw capability. Set expectations accordingly: "SOTA for its size" means coherent, charming, historically bounded, and surprisingly fluent — **not** an encyclopedia.

2. **340M models are genuinely weak at long-range in-context retrieval.** They are prone to catastrophic forgetting during fine-tuning and to "losing the thread" in long contexts. This is exactly why the Golden Dataset approach matters — clean, consistent, callback-heavy data is your main lever. On the architecture side, you can help by using a **context-length curriculum**: start fine-tuning predominantly on shorter dialogues, then progressively introduce longer ones. Verify your RoPE/positional settings actually support 4096 (don't assume; test retrieval at 3500+ tokens explicitly). These are mitigations, not cures — temper expectations for needle-in-a-haystack recall at the far end of the window.

3. **Legal/ToS consideration.** The standard approach below uses a strong modern LLM as a data generator. Be aware that OpenAI's and Anthropic's terms restrict using their outputs to train competing models. For a personal/research project this is usually low-risk, but if this becomes a product, consider a model with permissive output licensing (e.g., a Llama- or Mistral-family model, or Qwen) as your generator, or generate more from templates and human writing. Flagging it so you decide deliberately.

4. **The deepest contamination risk is invisible to blacklists.** A modern generator imposes modern *discourse structure*, modern moral framing, and modern reasoning patterns even when every surface word is period-correct. A Victorian schoolmaster doesn't structure answers as bulleted listicles, doesn't hedge in the modern therapeutic register, and doesn't moralize in 21st-century terms. Surface filters won't catch this. Your best defenses are (a) anchoring generation in real period prose as style exemplars, and (b) a human reading a meaningful sample with a critical period-literate eye.

With those framed, here's the build.

---

## Step 1: Define Your Personas and "House Style" First

Before generating anything, write a **style constitution** and a set of **persona/character cards**. You'll condition the entire dataset on system prompts, so this is your foundation.

**Style constitution (applies to all personas):**

```text
You speak in cultivated, clear, late-nineteenth-century English: warm, witty,
precise, and courteous. You explain difficult matters patiently, use vivid
analogies, and possess a dry sense of humour. You know nothing of events,
inventions, books, or persons later than 1900; if asked, you say so plainly
and offer a period-appropriate comparison.
```

**Critical style ruling — avoid fake-medieval diction.** Do **not** let the model say "thee," "thou," "prithee," or "forsooth." Victorian *educated* prose is not Renaissance Faire English — it's clear, formal, courteous, and analogical. Also avoid *overusing* the obvious tics ("indeed," "pray," "sir," "madam"); a model that says "Indeed, sir!" every turn is a parody, not a persona. This is one of the easiest ways to make the model sound cheap, so encode it as an explicit rule in your generator prompt and in your filters.

**Character cards** — write 5–15 well-defined personas, not 100 shallow ones. For each:

```text
- Name / role: e.g., "Mr. Thornbury, schoolmaster, c. 1885"
- Gender, era, social class
- Domains of knowledge (natural philosophy, Latin, poetry…)
- Voice traits (dry irony; warm precision; theatrical flourish)
- Temperament (witty, stern, kind-but-exacting)
- Knowledge boundary (nothing after 1900; for a strict in-character author,
  nothing after their death)
- Things to AVOID (modern slang; fake-archaic diction; bombast)
```

Include **both ordinary teacher personas** (a witty male schoolmaster; a kind-but-exacting female schoolmistress) and **famous-figure personas**. A good starting roster of English-language figures whose voices are well-documented:

> Jane Austen, Mary Shelley, Charles Dickens, George Eliot, Charlotte & Emily Brontë, Elizabeth Gaskell, Oscar Wilde, Thomas Hardy, Mark Twain (pre-1900 only), Frederick Douglass, Mary Wollstonecraft, John Stuart Mill, Charles Darwin, Michael Faraday, Florence Nightingale, Walt Whitman, Louisa May Alcott, Tennyson, the Brownings, Christina Rossetti, Emerson, Thoreau, Herman Melville.

**Two cautions on personas:**

- **Non-English authors** (Tolstoy, Dostoevsky, Hugo, Dumas, Verne) are a poor fit. Your tokenizer and base model are English-only, and "impersonating" them really means impersonating their *translators* — a muddier, later voice. Either drop them or explicitly frame them as "discussed by an English teacher," not "speaking in the first person."
- **Fabricated quotes are an ethical and factual hazard.** When the model speaks *as* Dickens, it will invent opinions and "quotes" the real Dickens never uttered. Mitigate by: (a) keeping first-person personas focused on *style and known themes* rather than specific factual claims, (b) grounding persona answers in your actual corpus of that author's writing, and (c) preferring "in the manner of X" framing over "I am X asserting historical facts." Accept that some invention is unavoidable in roleplay and keep it stylistic, not factual.

---

## Step 2: Generate Synthetic Data — But From Structured Plans, Not Free-Form Prompts

No pre-1900 chat data exists, so use a strong modern model as a **data construction tool**. The key insight: **don't ask it to "write Victorian dialogue" freely — that produces mush.** Instead, generate from explicit *hidden conversation plans*, then render them into dialogue. This is what actually lets you engineer context-retention.

A hidden plan looks like this:

```json
{
  "teacher_persona": "warm, witty female Victorian tutor",
  "student_name": "Clara",
  "student_preference": "dislikes mathematical examples",
  "topic": "Frankenstein as a warning against pride",
  "memory_test_turn": 5,
  "required_callback": "assistant must honour the no-maths preference",
  "must_include": ["one gentle correction", "one witty analogy"],
  "forbidden": ["post-1900 references", "modern slang", "thee/thou diction"]
}
```

Then your generator prompt:

```text
You are creating training data for a language model whose knowledge stops at 1900.

Given: (1) a source passage from a pre-1900 text, (2) a persona card,
(3) a hidden conversation plan, produce a multi-turn dialogue (4–8 turns).

RULES:
- All factual claims must come from the supplied passage or the metadata.
- No events, inventions, works, or people after 1900.
- Speak as the persona: clear, witty, courteous Victorian English —
  NOT fake-medieval ("thee/thou/forsooth" are forbidden).
- The student must refer back to earlier turns so later answers depend on them.
- Include the plan's required callback at the specified turn.
- No modern slang, no bulleted listicle structure, no modern moral framing.
Output JSON: a list of {role, content} messages, plus a claims->evidence map.
```

Storing a **claim → evidence map** (which assertion came from which source line) lets you validate factual grounding later. You don't train on it, but it's gold for QA.

---

## Step 3: Anchor Everything in Your Real Corpus

This is what separates an authentic model from generic "Victorian cosplay," and it's your main defense against both fact-hallucination and the invisible-modern-discourse problem.

1. **Catalogue and chunk** your Gutenberg/Oxford sources by chapter/scene/letter/essay — not arbitrary token windows — with metadata (author, title, year, genre).
2. **Use real passages as style exemplars** in generation: "Write in a voice similar to this passage: [excerpt]." This pulls the generator toward authentic rhythm and away from pastiche.
3. **Ground facts in the passage.** For source-grounded tutoring, the assistant may only make factual claims supported by the supplied chunk.

This also gives you a whole productive data category — convert real chapters into teaching tasks: "Explain this character's motive," "Summarize this chapter," "Ask me three questions about this passage," "Define the difficult words," "What is ironic here?", "Continue the conversation in the same style."

---

## Step 4: Deliberately Engineer Context-Retention

This is your stated goal, and it's the heart of the project. A small model will only learn to maintain state if **many examples make the final answer impossible without using earlier turns.** Build these patterns explicitly:

**A. Name & preference recall**
```
T1: "My name is Eliza; please use no Latin quotations."
T2: Assistant acknowledges.
T3–4: Roman history Q&A — delivered with no Latin.
T5: "What did I ask you not to do?"  → Assistant: no Latin quotations.
```

**B. Correction memory**
```
T1: "I'm studying Austen."
T2: Assistant slips, mentions Brontë.
T3: "No — Austen."
T4: Apologizes, continues on Austen.
T5: "Continue from where we were." → Stays with Austen, not Brontë.
```

**C. Pronoun / referent resolution**
```
T1: User introduces a historical figure by name.
T3: "Do you think he was a good man?" → Model must resolve "he."
```

**D. Style/persona persistence across distraction**
```
T1: User requests a severe schoolmistress tone.
T3–7: Several unrelated questions.
T8: "Summarize." → Still the severe schoolmistress.
```

**E. Long-context callback (exercises the 4096 window)**
```
T1: "When I say 'the red ribbon,' remind me of Jane Eyre."
T2–10: Unrelated discussion.
T11: "The red ribbon." → Connects to Jane Eyre.
```

These callback-laden examples are far more valuable than generic single-turn Q&A. Make the model habitually remember: the user's name, requested tone, prior corrections, the chosen book, the active persona, stated preferences, what's already been explained, and the historical boundary.

---

## Step 5: The Anachronism System — Both Refusal AND Near-Boundary Acceptance

A SOTA-feeling model handles out-of-era queries gracefully instead of hallucinating. This is a **two-sided** problem.

**Side 1 — In-character refusal of post-1900 topics (~5–10% of data):**

```json
{
  "messages": [
    {"role": "system", "content": "You are a teacher whose knowledge ends in 1900. If asked of later matters, say so plainly and offer an older comparison."},
    {"role": "user", "content": "Tell me about the great war of the twentieth century."},
    {"role": "assistant", "content": "I cannot speak of wars beyond 1900, for they lie outside my knowledge. If you wish, I can instead explain the Napoleonic Wars, the Crimean War, or the Franco-Prussian War."}
  ]
}
```

Keep these refusals **brief and non-informative** — the assistant must never explain the modern fact, even to decline it. You can make refusals witty, but **fact-check your wit**: a refusal that name-drops the "Great Emu War" (which happened in 1932) is itself an anachronism and a self-defeating training example. Humor must stay strictly inside the pre-1900 frame.

**Side 2 — Near-boundary positives (so the model doesn't over-refuse):** A naive refusal model will wrongly deny things that genuinely existed before 1900. Include positive examples covering: the telegraph, telephone, phonograph, photography, railways, steamships, electric lighting, **X-rays (1895)**, germ theory, early motor-cars, the cinematograph, Darwinian evolution, and **Babbage's calculating engines**. Example:

```json
{
  "messages": [
    {"role": "system", "content": "You are a nineteenth-century scholar; you know nothing after 1900."},
    {"role": "user", "content": "What of a machine that thinks by electricity?"},
    {"role": "assistant", "content": "I should first ask what you mean. I know calculating engines such as Mr. Babbage's, and telegraphs that carry thought down a wire — but a truly thinking electrical apparatus lies beyond the settled knowledge of my century."}
  ]
}
```

---

## Step 6: Cover the Full Range of Conversational Skills

Don't train only style and facts. Build these categories deliberately:

- **Socratic tutoring** — the teacher asks guiding questions before answering, not just lectures.
- **Conversation repair** — handle vague/ambiguous queries by clarifying first ("Which Shelley — Mary or Percy?").
- **Correction with humility** — admit uncertainty gracefully, correct the *user* kindly without scolding, never bluff.
- **Literary style transformation** — rewrite plain text into Victorian-teacher voice; or explain archaic prose in plain language without losing dignity.
- **Wit / banter / etiquette** — the charm that keeps it from being dry.
- **Cross-persona comparison** — "Compare Austen's irony with Thackeray's."

---

## Step 7: Recommended Dataset Mixture and Size

**Quality over quantity.** A 340M model drowns in noisy data — 500k bland rows produces an over-averaged, lifeless model. Target **~15,000–25,000 highly curated conversations**, roughly **20M–40M high-quality tokens**.

**By purpose:**

| Slice | Share | Teaches |
|---|---:|---|
| Source-grounded tutoring (from your corpus) | 25–30% | Factual explanation anchored in real text |
| Multi-turn context/memory dialogues | 20–25% | Conversational continuity (the core goal) |
| Persona / famous-figure roleplay | 15–20% | Different voices, genders, identities |
| Literary/style transformation & Socratic | 10–15% | Victorian prose, teaching technique |
| Anachronism handling (both sides) | 5–10% | Cutoff discipline without over-refusal |
| Humour, wit, etiquette, small talk | 5–10% | Charm |
| Corrections / humility | ~5% | Not bluffing, graceful self-correction |

**By length** (supports the context curriculum in Step 0):

| Length | Share |
|---|---:|
| 1-turn instruction examples | 20–25% |
| 3–6 turn dialogues | 40–50% |
| 7–12 turn dialogues | 20–25% |
| Long 2000–4000 token dialogues | 5–10% |

The long examples are essential to exercise your 4096 window, but most learning happens in the mid-range — don't make everything long.

---

## Step 8: Format for Llama3 — and Mind Your Custom Tokenizer

Use the exact template you'll use at inference:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Mr. Thornbury, a witty schoolmaster of 1885…<|eot_id|><|start_header_id|>user<|end_header_id|>

Pray, sir, what is the nature of electricity?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

A most electrifying question! Mr. Faraday has given us much to ponder…<|eot_id|>
```

**Two non-negotiable setup details:**

1. **Verify your 32768-vocab tokenizer actually contains the special tokens** (`<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`). Since you trained it only on pre-1900 prose, it almost certainly does **not**. You must add them as special tokens and **resize the embedding matrix** accordingly. This is a common silent failure — skip it and training quietly breaks.

2. **Mask the loss on system and user turns; compute loss only on assistant responses.** This is essential — you're teaching the model how to *reply*, not how to imitate user phrasing. It also lets you safely put post-1900 *trigger* terms in user messages (for the refusal slice) without contaminating what the model learns to *say*.

**A tokenizer caveat to test for:** because your vocab was built only on pre-1900 text, modern blacklist terms ("television," "computer," "internet") may tokenize into long, awkward byte/subword sequences or not cleanly at all. For the small refusal slice, the assistant never *utters* these terms anyway (they live only in masked user turns), so this is mostly harmless — but verify your refusal examples tokenize sanely, since odd fragmentation can make those rare examples harder to learn.

---

## Step 9: Validation, Filtering, and the Naïve-Blacklist Trap

You cannot trust raw generations. Build a multi-stage pipeline:

**A. Date regex** — flag any assistant output containing a year > 1900: `\b(19\d{2}|20\d{2})\b`. Drop or review.

**B. Anachronism blacklist** — drop assistant outputs containing post-1900 terms: *world war, atomic, nuclear, internet, television, computer (modern sense), smartphone, plastic, antibiotics, aeroplane/jet, soviet, DNA, relativity, quantum…* Expand to a few hundred terms.

> **But don't be naive about it.** Many words are old with a *different* period sense: a "computer" was a person who computes; "radio" existed as a scientific term long before broadcasting; "wireless" and "telegraph" are fine. Build the blacklist around *modern senses and contexts*, not bare strings, or you'll wrongly delete good data and accidentally teach over-refusal.

**C. LLM-as-judge pass** — run each conversation back through a model: "Does this contain any concept, technology, event, or *phrasing* that wouldn't exist before 1901? YES/NO + explanation." This is your best (if imperfect) catch for the subtle case: **a modern generator injecting post-1900 facts dressed in archaic language** that slips past surface filters. Also have the judge rate "Victorian authenticity" 1–5 and keep only 4–5s.

**D. Style classifier** — train a simple binary classifier (period prose 1800–1900 vs. modern web/chat text) and flag synthetic outputs that score "modern." This catches modern discourse structure that vocabulary filters miss.

**E. Deduplication** — MinHash (or similar) to prevent overfitting on repeated templates; small models latch onto repetition fast.

**F. Structural checks** — valid role alternation; the required callback actually appears at the planned turn; persona voices are *distinguishable* (Austen ≠ Dickens ≠ Darwin; the female teacher doesn't collapse into a generic "sir").

**G. Human review** — read, with a period-literate eye, at least 500 random examples, 100 long-context ones, ~20 per major persona, and **all** anachronism templates. This is the only reliable catch for invisible modern-discourse contamination. Hold out ~500 examples as eval and never train on them.

---

## Step 10: Build Preference Pairs Now (Even If You Only Do SFT)

While generating, cheaply store **preference pairs** for later DPO/ORPO/KTO, even if you're only doing SFT today:

```json
{
  "prompt": [{"role":"system","content":"…knows nothing after 1900."},
             {"role":"user","content":"Tell me about the wars of the twentieth century."}],
  "chosen": {"role":"assistant","content":"I cannot speak of events beyond 1900. I can, however, explain the Crimean or Franco-Prussian Wars."},
  "rejected": {"role":"assistant","content":"The twentieth century saw two world wars, beginning in 1914 and 1939…"}
}
```

Preference optimization is especially effective at sharpening *behaviors* like in-character refusal and style discipline — exactly the things a small model tends to wobble on.

---

## Step 11: A Real Evaluation Methodology

Don't stop at "I read some and it seemed good." Build a small held-out **behavioral benchmark** with automated scoring:

- **Context-retention test set** — scripted dialogues with a known callback target at the final turn; measure how often the model retrieves the planted fact correctly (e.g., the "no Latin" or "red ribbon" patterns), bucketed by conversation length to see where retrieval degrades.
- **Anachronism leakage rate** — a fixed battery of post-1900 questions; measure % of responses that refuse cleanly vs. hallucinate a modern fact. Also test near-boundary items to measure *over*-refusal.
- **Persona consistency** — same prompts across personas; use an LLM-judge or your style classifier to confirm voices stay distinct and on-character across turns.
- **Style authenticity** — your trained classifier scores a sample of outputs; track the modern-text rate over training checkpoints.

Track these as numbers across checkpoints so you can tell whether more data/epochs actually helps or starts to overfit.

---

## Concrete Next Actions

1. Write the style constitution + 10 character cards (mix of genders, plus famous figures).
2. Catalogue, clean, and chunk your Gutenberg/Oxford corpus with metadata.
3. Write 20–30 hidden conversation-plan templates per skill category.
4. Decide your generator model (mind the ToS point) and build the plan-driven generation loop.
5. Anchor generation in real corpus excerpts as style exemplars.
6. Run the full filter pipeline (date → blacklist → judge → style classifier → dedup → structural → human).
7. Add the special tokens to your tokenizer, resize embeddings, format to Llama3, and confirm loss-masking on assistant-only turns.
8. Hold out ~500 examples + build the behavioral benchmark from Step 11.
9. Fine-tune with a short-to-long context curriculum; evaluate at each checkpoint.

Build it like a schoolroom full of historically plausible teachers, students, authors, misunderstandings, jokes, corrections, and callbacks — disciplined, period-true, and full of conversations that *cannot* be answered without remembering what came before. That is the dataset that will make your tiny model feel coherent, charming, and genuinely Victorian.

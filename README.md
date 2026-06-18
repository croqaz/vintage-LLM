# Vintage-LLM

Vintage-LLM is an experiment in building small language models that think and speak as if they were living in the 19th century.

The idea is simple: instead of training an AI on the modern internet, train it **only** on texts written before 1900. The result is a kind of a "time-capsule" model - one that talks like a Victorian scholar, draws on the knowledge and worldview of its era, and doesn't know about anything invented since. No smartphones, no airplanes, no modern slang. Just the language, ideas, and style of a bygone age.

## What's in here

This repository holds bits& pieces needed to assemble the data, build the vocabulary, and train a vintage model from the ground up.

- **`knowledge/`** - the raw material. This is where the period-appropriate texts live: bible Q&A, some 17th and 18th century books, philosophical quotations, genuine 1880s–1900s schoolbooks, and a collection of timeless facts (greetings, simple arithmetic, etc) that haven't changed with time. It also contains synthetic "conversation" datasets generated to teach the model how to actually hold a dialogue in a vintage voice.
- **`dataset/`** - scripts used to import, deduplicate, filter and export datasets.
- **`training/`** - recipes and settings used to train the models, including different model sizes and configurations being experimented with, reports on how each run went.
- **`scripts/`** - supporting tools for preparing data, managing datasets, and plotting training progress.

## How it works, in broad strokes

1. **Gather the texts.** Collect public-domain writing from before 1900 - literature, scripture, schoolbooks, and reference works.
2. **Build a period vocabulary.** Teach the model its "alphabet" using only vintage language, so even the way it reads words is historically grounded.
3. **Train the base model.** Feed it the vintage library so it learns to write in that style and absorb that knowledge.
4. **Teach it to converse.** Fine-tune the model on instruction-and-conversation examples so it can answer questions and chat, or roleplay as a 19th-century teacher or famous author.

The models themselves are deliberately tiny, which keeps the whole project approachable and possible to run without enormous computing resources.

## Articles

- https://crlf.link/log/entries/260525-1 -- Making a vintage LLM from scratch
- https://crlf.link/log/entries/260428-1 -- A full list of Vintage LLM models

## Discord

[Join our discord](https://discord.gg/MVzzuzGXhk) to talk about Vintage datasets and LLMs.

## License

[MIT](LICENSE) © Cristi Constantin.

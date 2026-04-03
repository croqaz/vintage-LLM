# Vintage-LLM-v1

Trained from scratch on top of EleutherAI/pythia-14m, using litGPT, on my own PC with AMD Radeon RX 9070 16GB VRAM.

Using my custom tokenizer t-v1.

The training config is called `litgpt.yaml`.

Using 2,634 books from [Oxford-Archive](https://github.com/mimno/ota), 4,718 books from [Time-Capsule](https://huggingface.co/datasets/haykgrigorian/TimeCapsuleLLM-v1-dataset) and
500 books from the Project Gutenberg (selected 7*-0.txt files).

Training dataset statistics:

- Total Size  : 4,548,224,513 bytes
- Total Tokens: 1,215,502,000 -- (1.2B tokens)

Trained for 2B tokens, basically 1.6 epochs.

Epoch 2 | iter 488280 step 81380 | loss train: 4.275, val: 4.363

- Total tokens  : 1,999,998,976
- Tok/sec       : 58773.59 tok/s

I did some mistakes:
I split the texts in chunks of roughly 8000 characters (keeping the sentences) and I shuffled them.
My intuition was that the model will learn from different texts at the same time.
I think that was wrong, because the final loss was 4.275 and the model still generates semi-random gibberish.

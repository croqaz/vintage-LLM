# Vintage-LLM-v2

The setup is almost identical to v1, but the model itself is much better.

Using my custom tokenizer t-v2, which fixes some mistakes.

Trained from scratch on top of EleutherAI/pythia-14m, using litGPT, on my own PC with AMD Radeon RX 9070 16GB VRAM.

The training config is called `litgpt.yaml`.

Using 2,634 books from [Oxford-Archive](https://github.com/mimno/ota), 4,718 books from [Time-Capsule](https://huggingface.co/datasets/haykgrigorian/TimeCapsuleLLM-v1-dataset) and
500 books from the Project Gutenberg (selected 7*-0.txt files).

Training dataset statistics:

- Total Size  : 4,539,828,155 bytes
- Total Tokens: 1,205,125,716 -- (1.2B tokens)

Trained for 1.2B tokens, basically 1 epoch.

Epoch 2 | iter 297848 step 37231 | loss train: 4.173, val: 4.396

- Total tokens  : 1,219,997,696
- Training Time : 7618.75 sec
- Tok/sec       : 93781.18 tok/s

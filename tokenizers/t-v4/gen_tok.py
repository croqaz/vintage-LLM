import json
import os
import re
from collections import Counter
from multiprocessing import Pool

from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from tokenizers import Tokenizer

OUTPUT = 'gen_tok.json'
DATASETS = ['dataset-text1.txt', 'dataset-text2.txt', 'dataset-text3.txt', 'dataset-text4.txt']
CHUNK_CHARS = 1 << 23  # 8M chars per worker task

# The raw datasets are full of OCR'd table borders and horizontal rules
# ("+-----+-----+", "========", "................", "aaaaaaaa") that BPE
# turns into junk vocab entries. Collapse them before training so the vocab
# holds words and short punctuation runs (max 3 of the same char) only.

# any character repeated 4+ times -> exactly 3
RUN_RE = re.compile(r'(.)\1{3,}')
# mixed runs of ruler/border characters 4+ long ("---+", "-.-.-") -> 3x the dominant char
RULER_RE = re.compile(r'[-=_~+*#.]{4,}')
# a short punctuation unit repeated ("?!?!?!", "()()") -> a single unit
UNIT_RE = re.compile(r'([^\w\s]{2,8}?)\1+')


def _squash_ruler(match):
    return Counter(match.group()).most_common(1)[0][0] * 3


def clean_text(text):
    text = RUN_RE.sub(r'\1\1\1', text)
    text = RULER_RE.sub(_squash_ruler, text)
    text = UNIT_RE.sub(r'\1', text)
    return text


def _read_chunks(path):
    # chunks always end on a newline so a repeated run is never split
    # across two chunks (runs cannot contain '\n')
    with open(path, encoding='utf-8') as f:
        while chunk := f.read(CHUNK_CHARS):
            yield chunk + f.readline()


def clean_file(src, dst):
    tmp = dst + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as out, Pool() as pool:
        for cleaned in pool.imap(clean_text, _read_chunks(src)):
            out.write(cleaned)
    os.replace(tmp, dst)


def main():
    cleaned_files = []
    for src in DATASETS:
        dst = src.replace('-text', '-clean')
        if not os.path.exists(dst):
            print(f'cleaning {src} -> {dst}', flush=True)
            clean_file(src, dst)
        cleaned_files.append(dst)

    tokenizer = Tokenizer(BPE(unk_token='<|unk|>'))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=32768,  # exactly 2**15 including the 16 special tokens
        min_frequency=5,
        special_tokens=['<|pad|>', '<|unk|>', '<|mask|>', '<|bos|>', '<|eos|>', '<|system|>', '<|user|>', '<|assistant|>']
        + [f'<|future{i}|>' for i in range(1, 9)],
    )

    tokenizer.train(cleaned_files, trainer)
    tokenizer.save(OUTPUT)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=OUTPUT,
        bos_token='<|bos|>',
        eos_token='<|eos|>',
        unk_token='<|unk|>',
        pad_token='<|pad|>',
    )

    print(fast_tokenizer.decode(fast_tokenizer.encode('Hello, how are you ?? :)')))

    print('Encoded:', fast_tokenizer.encode("<|bos|>Hello, y'all! How are you 😁 ?<|eos|>"))

    print(fast_tokenizer.decode(fast_tokenizer.encode("<|bos|>Hello, y'all! How are you 😁 ?<|eos|>")))

    fast_tokenizer.save_pretrained('fast_tok')

    # self-check: the vocab must contain no repeated-pattern junk
    vocab = json.load(open(OUTPUT))['model']['vocab']
    bad = sorted(t for t in vocab if RUN_RE.search(t) or RULER_RE.search(t) or UNIT_RE.search(t))
    print(f'repeated-pattern tokens in vocab: {len(bad)}')
    for token in bad:
        print(' ', repr(token))


if __name__ == '__main__':
    main()

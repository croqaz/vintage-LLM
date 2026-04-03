"""
This tokenizer was generated with ByteLevel BPE.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast

OUTPUT = './gen_tok.json'

tokenizer = Tokenizer(BPE(unk_token='<|unk|>'))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()
tokenizer.normalizer = Sequence([NFKC()])
trainer = BpeTrainer(
    vocab_size=32700,
    min_frequency=5,
    special_tokens=['<|pad|>', '<|unk|>', '<|mask|>', '<|bos|>', '<|eos|>', '<|system|>', '<|user|>', '<|assistant|>'],
)

tokenizer.train(['data/dataset-text.txt'], trainer)
tokenizer.save(OUTPUT)

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=OUTPUT,
    bos_token='<|bos|>',
    eos_token='<|eos|>',
    unk_token='<|unk|>',
    pad_token='<|pad|>',
)

print(fast_tokenizer.decode(fast_tokenizer.encode('Hello, how are you ?? :)')))

print(fast_tokenizer.encode("<|bos|>Hello, y'all! How are you 😁 ?<|eos|>"))

print(fast_tokenizer.decode(fast_tokenizer.encode("<|bos|>Hello, y'all! How are you 😁 ?<|eos|>")))

fast_tokenizer.save_pretrained('./fast_tok')

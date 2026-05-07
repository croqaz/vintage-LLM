import random

import orjson
from transformers import AutoTokenizer

TOK_VERSION = 't-v3'
KNOWLEDGE_PATH = 'base_knowledge.jsonl'


def main(knowledge: list[dict[str, str]], chat_template=True) -> list[str]:
    strings = []
    if chat_template:
        tokenizer = AutoTokenizer.from_pretrained(f'tokenizers/{TOK_VERSION}')
        for qa in knowledge:
            strings.append(
                tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': qa['question']}, {'role': 'assistant', 'content': qa['answer']}],
                    tokenize=False,
                )
            )
    else:
        for qa in knowledge:
            strings.append(f'Question: {qa["question"]}\nAnswer: {qa["answer"]}')
    return strings


if __name__ == '__main__':
    with open(KNOWLEDGE_PATH, 'rb') as f:
        knowledge = [orjson.loads(line) for line in f]
    random.shuffle(knowledge)
    strings = main(knowledge)
    with open('knowledge_text.txt', 'w', encoding='utf-8') as f:
        for text in strings:
            f.write(text + '\n\n')

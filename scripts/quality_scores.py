import re
import zlib
from collections import Counter
from math import log2

import matplotlib.pyplot as plt
import orjson


def quality_score(text):
    length = len(text)
    score = 0
    for c in text:
        if re.match('[a-zα-ωàâäçèéêëîïôöùûüüÿæœß]$', c, re.I):
            score += 2
        elif re.match('[0-9 \n]$', c):
            score += 1
        elif re.match('[.,;!?\'"-]$', c):
            score += 0.5
        else:
            score -= 0.5
    return score / length


def compression_ratio(text):
    raw = text.encode('utf-8')
    compressed = zlib.compress(raw)
    return len(compressed) / len(raw)


def char_entropy(text):
    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * log2(p)
    return entropy


def combined_score(text):
    q_score = quality_score(text)
    c_ratio = compression_ratio(text)
    entropy = char_entropy(text)
    return ((q_score - 0.8) + (c_ratio + 0.5) + (entropy / 4.4)) / 3.0


def main():
    scores = []
    with open('american-medium-correct.jsonl') as fd:
        for line in fd:
            line = line.strip()
            if len(line) < 10:
                continue
            data = orjson.loads(line)
            score = combined_score(data['text'])
            scores.append(score)
            if score < 0.9 or score > 1.1:
                print(f'Low quality text (score={score:.2f}): {data["text"][:1000]}...\n\n')
            if len(scores) >= 10_000:
                break

    # Plot the scores
    plt.figure(figsize=(12, 5))

    # Plot 1: Scatter plot to see the pattern over the file
    plt.subplot(1, 2, 1)
    plt.plot(scores, marker='.', linestyle='none', alpha=0.5, color='blue')
    plt.title('Scores by Index')
    plt.xlabel('Index')
    plt.ylabel('Score')
    plt.grid(True)

    # Plot 2: Histogram to see the distribution
    plt.subplot(1, 2, 2)
    plt.hist(scores, bins=50, alpha=0.75, color='green')
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

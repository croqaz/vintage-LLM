"""
Run a back-and-forth conversation between two LLM API providers.

A start prompt is sent to the first LLM, its response is sent to the second LLM,
that response goes back to the first LLM, and so on for a number of turns.
Both endpoints are assumed to be OpenAI-compatible chat completion APIs
(e.g. LM Studio, llama-server, vLLM, OpenAI, etc.).
"""

import argparse
import json
import sys

import requests


def call_llm(url, model, messages, temperature, max_tokens, api_key=None):
    """Send an OpenAI-compatible chat completion request and return the assistant reply."""
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    payload = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f'API call to {url} failed with status {response.status_code}: {response.text}')

    data = response.json()
    if 'choices' not in data or not data['choices']:
        raise RuntimeError(f'Unexpected response format from {url}: {data}')

    return data['choices'][0]['message']['content']


def main():
    parser = argparse.ArgumentParser(description='Run a conversation between two LLM API providers.')
    parser.add_argument('--url-a', type=str, required=True, help='LLM A endpoint URL (OpenAI compatible)')
    parser.add_argument('--url-b', type=str, required=True, help='LLM B endpoint URL (OpenAI compatible)')
    parser.add_argument('--start-prompt', type=str, required=True, help='The opening message sent to LLM A')
    parser.add_argument('--turns', '-n', type=int, default=6, help='Total number of replies to generate')
    parser.add_argument('--model-a', type=str, default='local-model', help='Model name for LLM A')
    parser.add_argument('--model-b', type=str, default='local-model', help='Model name for LLM B')
    parser.add_argument('--system-a', type=str, default=None, help='Optional system prompt for LLM A')
    parser.add_argument('--system-b', type=str, default=None, help='Optional system prompt for LLM B')
    parser.add_argument('--api-key-a', type=str, default=None, help='Optional bearer token for LLM A')
    parser.add_argument('--api-key-b', type=str, default=None, help='Optional bearer token for LLM B')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for both models')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens per reply')
    parser.add_argument('--output', '-o', type=str, default=None, help='Optional JSONL file to save the transcript to')

    args = parser.parse_args()

    # Each model keeps its own message history. From a model's point of view its
    # own outputs are 'assistant' and the other model's outputs are 'user'.
    history_a = [{'role': 'system', 'content': args.system_a}] if args.system_a else []
    history_b = [{'role': 'system', 'content': args.system_b}] if args.system_b else []

    # The transcript records each exchange in speaking order.
    transcript = [{'speaker': 'start', 'content': args.start_prompt}]
    print(f'[start] {args.start_prompt}\n')

    # The start prompt is the first 'user' message for LLM A.
    message = args.start_prompt

    try:
        for turn in range(args.turns):
            # Alternate: even turns -> A speaks, odd turns -> B speaks.
            if turn % 2 == 0:
                speaker, url, model, key = 'A', args.url_a, args.model_a, args.api_key_a
                history, other_history = history_a, history_b
            else:
                speaker, url, model, key = 'B', args.url_b, args.model_b, args.api_key_b
                history, other_history = history_b, history_a

            history.append({'role': 'user', 'content': message})

            print(f'[{model} / LLM {speaker}] thinking...', end='', flush=True)
            reply = call_llm(url, model, history, args.temperature, args.max_tokens, key)

            history.append({'role': 'assistant', 'content': reply})
            print('\r' + ' ' * 60 + '\r', end='')
            print(f'[LLM {speaker}] {reply}\n----------\n')

            transcript.append({'speaker': speaker, 'model': model, 'content': reply})

            # This reply becomes the next message for the other model.
            message = reply

    except Exception as e:
        print(f'\nAn error occurred: {e}', file=sys.stderr)
        sys.exit(1)

    if args.output:
        with open(args.output, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'transcript': transcript}, ensure_ascii=False) + '\n')
        print(f'Transcript saved to {args.output}')


if __name__ == '__main__':
    main()

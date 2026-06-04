"""
Call the LLM API to generate some text for a given prompt and save the output to a file.
"""

import argparse
import json
import sys

import requests


def main():
    parser = argparse.ArgumentParser(description='Call LLM API and generate synthetic data to a JSONL file.')
    parser.add_argument(
        '--url', type=str, default='http://localhost:1234/v1/chat/completions', help='LLM API endpoint URL (OpenAI compatible)'
    )
    parser.add_argument('--system-prompt', type=str, required=True, help='The system prompt to use')
    parser.add_argument('--user-prompt', type=str, required=True, help='The user prompt to use')
    parser.add_argument('--num-calls', '-n', type=int, default=1, help='Number of times to call the API')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output JSONL file to append responses to')
    parser.add_argument('--model', type=str, default='local-model', help='Model name to pass to the API')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for the generation')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Maximum number of tokens to generate')

    args = parser.parse_args()

    payload_template = {
        'model': args.model,
        'messages': [{'role': 'system', 'content': args.system_prompt}, {'role': 'user', 'content': args.user_prompt}],
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
    }

    try:
        with open(args.output, 'a', encoding='utf-8') as f:
            for i in range(args.num_calls):
                print(f'Call {i + 1}/{args.num_calls}...', end='', flush=True)

                response = requests.post(args.url, json=payload_template, headers={'Content-Type': 'application/json'})

                if response.status_code == 200:
                    data = response.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        assistant_message = data['choices'][0]['message']['content']
                        print(f'Response {i + 1}: {assistant_message.strip()}\n')

                        output_entry = {
                            'messages': [
                                {'role': 'system', 'content': args.system_prompt},
                                {'role': 'user', 'content': args.user_prompt},
                                {'role': 'assistant', 'content': assistant_message},
                            ]
                        }
                        f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
                        f.flush()
                    else:
                        print(f' Unexpected response format: {data}')
                else:
                    print(f' Failed with status code {response.status_code}')
                    print(response.text)

    except Exception as e:
        print(f'\nAn error occurred: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

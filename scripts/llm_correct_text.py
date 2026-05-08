"""
Corrects text using a language model. This script takes an input text file, processes it through
a language model to correct any errors, and outputs the corrected text to a new file.
"""

import argparse
import os

import requests

SYSTEM_PROMPT = """You are a helpful assistant that corrects badly scanned OCR text from old newspaper archives, year 1800.
Your task is to correct scanning errors and typos.
Don't judge, don't add any extra commentary or explanations.
Do NOT alter the original meaning, tone, or style, this text is a historical document and MUST be preserved!
Output *only* the corrected text."""

USER_PROMPT = """
Input example:
"j'\nr\nì\n' ILt\nr\nr-\nr-\no\nr\nA\nr\ngenerality of the privilege, by expressly declar.nng, thar every perion 'availing himlelf Of the liberty of the prels, ihould be repoaible for thr\nabufe of that liberty :. thus iecuring TO our\ncitizens the invaluable right os reputation\nagaina every malicious invader of it.\n\n\nPrinted publications attacking private cha.\nraeter, is conIidered with great reaion by the\nlaw AS very ATTRACTIONS offence. from its evi\ndent tendency to diturb the public peace-if\nmen find they can have no redrefs in cur\ncourts of Juice for fuch injuries, they will\nnaturally take fatisfaaion in their own way, in\nvolving perhaps their friends and families in\nthe confetti and leading evidently to duels,\nmurders, and perhaps Affirmation. ~, -\n\n\n"

Expected output example:
Generality of the privilege, by expressly declaring, that every person availing himself of the liberty of the press, should be responsible for the abuse of that liberty: thus securing to our citizens the invaluable right of reputation against every malicious invader of it.
Printed publications attacking private character, is considered with great reason by the law as a very atrocious offence, from its evident tendency to disturb the public peace—if men find they can have no redress in our courts of justice for such injuries, they will naturally take satisfaction in their own way, involving perhaps their friends and families in the conflict, and leading evidently to duels, murders, and perhaps animosities.
"""


def chunk_text(text, chunk_size=2000):
    """Splits the text into chunks of roughly `chunk_size` characters,
    preserving paragraphs."""
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ''
    for p in paragraphs:
        if len(current_chunk) + len(p) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = p + '\n'
        else:
            current_chunk += p + '\n'
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def correct_text_chunk(chunk, url, api_key=None, model=None):
    if api_key:
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    else:
        headers = {'Content-Type': 'application/json'}

    data = {
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f'Correct the following OCR text:\n\n{chunk}'},
        ],
        'temperature': 0.1,
        'max_tokens': 4096,
    }
    if model:
        data['model'] = model

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
    except Exception as e:
        print(f'Error calling LLM: {e}')
        try:
            print(f'Response status: {response.status_code}')
            print(f'Response body: {response.text[:500]}')
        except Exception:
            pass
    try:
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f'Error parsing LLM response: {e}')


def main():
    parser = argparse.ArgumentParser(description='Correct OCR text using a local LLM.')
    parser.add_argument('input_file', help='Path to the input text file')
    parser.add_argument('-o', '--output_file', help='Path to the output file (optional)')
    parser.add_argument('--url', default='http://127.1:1234/v1/chat/completions', help='LLM API URL')
    parser.add_argument('--api_key', help='(Optional) API key for authentication if required by the API')
    parser.add_argument('--model', help='(Optional) Model name to use, e.g. "gemini-2.5-flash" or "gpt-4o-mini"')
    parser.add_argument('--chunk_size', type=int, default=2000, help='Maximum characters per chunk')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return

    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = chunk_text(text, args.chunk_size)
    print(f'Split text into {len(chunks)} chunks.')

    corrected_chunks = []
    for i, chunk in enumerate(chunks):
        print(f'Processing chunk {i + 1}/{len(chunks)}...')
        if chunk.strip():
            corrected = correct_text_chunk(chunk, args.url, api_key=args.api_key, model=args.model)
            if corrected:
                corrected_chunks.append(corrected)
        else:
            corrected_chunks.append(chunk)

    output_file = args.output_file or f'{args.input_file}.corrected.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(corrected_chunks))

    print(f'Finished. Corrected text saved to {output_file}')


if __name__ == '__main__':
    main()

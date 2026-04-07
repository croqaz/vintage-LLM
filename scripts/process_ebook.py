import argparse
import base64
import io
import json
import timeit
import zipfile
from concurrent.futures import ThreadPoolExecutor

import requests
from lxml import etree
from PIL import Image

SYSTEM_PROMPT = """You are a helpful assistant that corrects badly scanned OCR text from old newspaper archives, year 1800.
Your task is to correct scanning errors.
Don't judge, don't add any extra commentary or explanations.
Do NOT alter the original meaning, tone, or style, this text is a historical document and MUST be preserved!
Output *only* the corrected text."""

USER_PROMPT = """Extract plain text from scanned image + suggested XML WORDs.
Use the image as **primary source of truth**!
Remove weird characters and fix common OCR errors.
Maintain the original paragraph newlines, but remove newlines within paragraphs.
Remove any page numbers at the start or end of the text.
Don't comment, or add explanations.
Suggested WORDs are below, they contain scanning errors:"""


def process_single_page(page_no: int, base64_img: str, suggested_words: str, args: argparse.Namespace) -> dict:
    payload = {
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': USER_PROMPT + '\n\n' + suggested_words},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{base64_img}'}},
                ],
            },
        ],
        'temperature': 0.0,
    }

    if args.model:
        payload['model'] = args.model
    if args.api_key:
        headers = {'Authorization': f'Bearer {args.api_key}', 'Content-Type': 'application/json'}
    else:
        headers = {'Content-Type': 'application/json'}

    for attempt in range(1, 4):
        if attempt > 1:
            print(f'Retrying page {page_no} (attempt {attempt}/3) ...')
        else:
            print(f'Fixing page {page_no} ...')

        result = {}
        try:
            response = requests.post(args.api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()
        except Exception as err:
            err_msg = str(err)
            if getattr(err, 'response', None) is not None:
                err_msg = f'Code {err.response.status_code}: {err.response.text}'
            print(f'Error processing {page_no} on attempt {attempt}/3: {err_msg}')
            if attempt == 3:
                return {'page': page_no, 'error': err_msg}

        if not result:
            return {'page': page_no, 'error': 'Empty response from API'}
        try:
            if not result['choices'][0]['message'].get('content'):
                return {'page': page_no, 'error': f'No content in response'}
            extracted_text = result['choices'][0]['message']['content']
            in_tokens = result.get('usage', {}).get('prompt_tokens', 0)
            out_tokens = result.get('usage', {}).get('completion_tokens', 0)
            print(f'Received response for {page_no}: {in_tokens} / {out_tokens} tokens')
            return {'page': page_no, 'text': extracted_text, 'in_tokens': in_tokens, 'out_tokens': out_tokens}
        except Exception as err:
            err_msg = str(err)
            print(f'Error parsing {page_no} on attempt {attempt}/3: {err_msg}')
            if attempt == 3:
                return {'page': page_no, 'error': err_msg}


def process_ebook_files(xml_path: str, zip_path: str, output_path: str, args: argparse.Namespace):
    proc_start = timeit.default_timer()
    print(f'Processing XML file: {xml_path} ...')
    print(f'Processing ZIP file: {zip_path} ...')

    tasks = []

    # Read XML
    tree = etree.parse(xml_path)
    root = tree.getroot()

    # The xml structure starts with <BODY> -> <OBJECT>
    objects = root.xpath('//OBJECT')
    print(f'Found {len(objects)} pages in XML.')

    with zipfile.ZipFile(zip_path, 'r') as z:
        zip_files = z.namelist()

        for obj in objects:
            # Find the page name
            param = obj.xpath('.//PARAM[@name="PAGE"]')
            if not param:
                continue
            djvu_name = param[0].get('value')

            base_name = djvu_name.rsplit('.', 1)[0]
            try:
                page_no = int(base_name.rsplit('_')[-1])
            except ValueError:
                print(f'Warning: Cannot extract page number from {base_name}. Stopping!')
                break

            if args.pages and page_no > args.pages:
                break

            jp2_name = base_name + '.jp2'
            matching_zip_file = next((f for f in zip_files if f.endswith(jp2_name)), None)
            if not matching_zip_file:
                print(f'Warning: Cannot find image for {jp2_name} in zip. Stopping!')
                break

            # Extract words as suggested text
            words = []
            for word_elem in obj.xpath('.//LINE/WORD'):
                if word_elem.get('coords'):
                    del word_elem.attrib['coords']
                if word_elem.get('x-confidence'):
                    word_elem.set('confidence', word_elem.get('x-confidence'))
                    del word_elem.attrib['x-confidence']
                if word_elem.text:
                    words.append(etree.tostring(word_elem, encoding='unicode').strip())
            suggested_words = ''.join(words)

            if not words:
                print(f'No WORD elements found for {page_no}. Skipping.')
                continue
            if len(suggested_words) < 32:
                print(f'Suggested text for {page_no} is very short: "{suggested_words}". Skipping.')
                continue

            with z.open(matching_zip_file) as img_f:
                img = Image.open(img_f).convert('RGB')
                img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                buff = io.BytesIO()
                img.save(buff, format='JPEG', quality=80)
                base64_img = base64.b64encode(buff.getvalue()).decode('utf-8')

            tasks.append((page_no, base64_img, suggested_words))

    print(f'Prepared {len(tasks)} valid pages to process.')

    total_prompt_tokens = 0
    total_completion_tokens = 0

    with open(output_path, 'w', encoding='utf-8') as out_f, ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(lambda p: process_single_page(*p, args=args), tasks)

        for resp in results:
            if 'error' not in resp:
                total_prompt_tokens += resp.pop('in_tokens', 0)
                total_completion_tokens += resp.pop('out_tokens', 0)
                out_f.write(json.dumps(resp, ensure_ascii=False))
                out_f.write('\n')
                out_f.flush()
            else:
                print(f'Failed to process page {resp["page"]} entirely: {resp["error"]}')

    proc_end = timeit.default_timer()
    print(f'Output successfully written to: {output_path}')
    print(f'Total input (prompt) tokens used: {total_prompt_tokens}')
    print(f'Total output (completion) tokens used: {total_completion_tokens}')
    print(f'Total processing time: {proc_end - proc_start:.2f} seconds\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process an ebook by extracting text and associating it with images.')
    parser.add_argument('--xml', required=True, help='Path to the input djvu.xml file')
    parser.add_argument('--zip', required=True, help='Path to the input jp2.zip file containing JP2 images')
    parser.add_argument('--output', required=True, help='Path to the output text file')
    parser.add_argument('--pages', type=int, required=False, help='Limit number of pages to process')
    parser.add_argument('--api_url', default='http://127.1:1234/v1/chat/completions', help='URL of the chat completion API')
    parser.add_argument('--api_key', help='(Optional) API key for authentication if required by the API')
    parser.add_argument('--model', help='(Optional) Model name to use for the API, e.g. "gemini-2.5-flash" or "gpt-5.4-mini"')
    args = parser.parse_args()
    process_ebook_files(args.xml, args.zip, args.output, args)

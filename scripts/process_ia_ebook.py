"""
Cleanup and extract text from a scanned Internet Archive ebook using an API with image understanding capabilities.
"""

import argparse
import base64
import glob as glob_module
import io
import json
import os
import time
import timeit
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import internetarchive as ia
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

    max_attempts = args.retries
    last_error = 'Unknown error'

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            # Exponential backoff before retrying: 2s, 4s, ...
            backoff = 2 ** (attempt - 1)
            print(f'Retrying page {page_no} (attempt {attempt}/{max_attempts}) after {backoff}s ...')
            time.sleep(backoff)
        else:
            print(f'Fixing page {page_no} ...')

        # 1. Make the request.
        try:
            response = requests.post(args.api_url, json=payload, headers=headers, timeout=args.timeout)
            response.raise_for_status()
            result = response.json()
        except Exception as err:
            last_error = str(err)
            if getattr(err, 'response', None) is not None:
                last_error = f'Code {err.response.status_code}: {err.response.text}'
            print(f'Error requesting page {page_no} on attempt {attempt}/{max_attempts}: {last_error}')
            continue

        # 2. Parse the response.
        try:
            if not result:
                raise ValueError('Empty response from API')
            content = result['choices'][0]['message'].get('content')
            if not content:
                raise ValueError('No content in response')
            in_tokens = result.get('usage', {}).get('prompt_tokens', 0)
            out_tokens = result.get('usage', {}).get('completion_tokens', 0)
            print(f'Received response for page {page_no}. Used: {in_tokens} / {out_tokens} tokens.')
            return {'page': page_no, 'text': content, 'in_tokens': in_tokens, 'out_tokens': out_tokens}
        except Exception as err:
            last_error = str(err)
            print(f'Error parsing page {page_no} on attempt {attempt}/{max_attempts}: {last_error}')
            continue

    print(f'Giving up on page {page_no} after {max_attempts} attempts. Last error: {last_error}')
    return {'page': page_no, 'error': last_error}


def process_ebook_files(xml_path: str, zip_path: str, output_path: str, args: argparse.Namespace):
    proc_start = timeit.default_timer()
    print(f'Processing XML file: {xml_path} ...')
    print(f'Processing ZIP file: {zip_path} ...')

    # Resume support: collect page numbers already present in an existing output
    # file, so we only download the missing pages and append them at the end.
    done_pages = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as existing_f:
            for line_no, line in enumerate(existing_f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f'Warning: skipping malformed line {line_no} in existing output {output_path}')
                    continue
                # Only count pages that were successfully extracted (have text).
                if 'page' in rec and 'text' in rec:
                    done_pages.add(rec['page'])
        print(f'Found existing output {output_path} with {len(done_pages)} completed pages. Will resume.')

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

            if page_no in done_pages:
                continue

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

    if done_pages:
        print(f'Prepared {len(tasks)} missing pages to process (skipped {len(done_pages)} already done).')
    else:
        print(f'Prepared {len(tasks)} valid pages to process.')

    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Append mode so an existing output is preserved and missing pages are added at the end.
    open_mode = 'a' if done_pages else 'w'
    with open(output_path, open_mode, encoding='utf-8') as out_f:
        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures = [executor.submit(process_single_page, *p, args=args) for p in tasks]

        try:
            for future in as_completed(futures):
                resp = future.result()
                if 'error' not in resp:
                    total_prompt_tokens += resp['in_tokens']
                    total_completion_tokens += resp['out_tokens']
                    out_f.write(json.dumps(resp, ensure_ascii=False))
                    out_f.write('\n')
                    out_f.flush()
                else:
                    print(f'Failed to process page {resp["page"]} entirely: {resp["error"]}')
            executor.shutdown(wait=True)
        except KeyboardInterrupt:
            print('\nProcess interrupted by user (Ctrl+C). Stopping...')
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False)
            os._exit(130)

    proc_end = timeit.default_timer()
    print(f'Output successfully written to: {output_path}')
    print(f'Total input (prompt) tokens used: {total_prompt_tokens}')
    print(f'Total output (completion) tokens used: {total_completion_tokens}')
    print(f'Total processing time: {proc_end - proc_start:.2f} seconds\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process an ebook by extracting text and associating it with images.')
    parser.add_argument('--id', dest='ia_id', help='Internet Archive identifier to download and process (e.g. newcommonschoolq02crai)')
    parser.add_argument('--xml', help='Path to the input djvu.xml file')
    parser.add_argument('--zip', help='Path to the input jp2.zip file containing JP2 images')
    parser.add_argument('--output', required=True, help='Path to the output text file')
    parser.add_argument('--pages', type=int, required=False, help='Limit number of pages to process')
    parser.add_argument('--api_url', default='http://127.1:1234/v1/chat/completions', help='URL of the chat completion API')
    parser.add_argument('--api_key', help='(Optional) API key for authentication if required by the API')
    parser.add_argument('--timeout', type=int, default=90, help='Timeout in seconds for API requests')
    parser.add_argument('--retries', type=int, default=3, help='Number of retries for API requests in case of failure')
    parser.add_argument('--model', help='(Optional) Model name to use for the API, e.g. "gemini-2.5-flash" or "gpt-5.4-mini"')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers to process pages concurrently')
    args = parser.parse_args()

    xml_path = args.xml
    zip_path = args.zip

    if args.ia_id:
        print(f'Downloading files for {args.ia_id} from Internet Archive ...')
        ia.download(args.ia_id, glob_pattern='*_djvu.xml|*_jp2.zip', verbose=True, ignore_existing=True)
        xml_files = glob_module.glob(os.path.join(args.ia_id, '*_djvu.xml'))
        zip_files = glob_module.glob(os.path.join(args.ia_id, '*_jp2.zip'))
        if not xml_files:
            parser.error(f'Could not find *_djvu.xml for identifier: {args.ia_id}')
        if not zip_files:
            parser.error(f'Could not find *_jp2.zip for identifier: {args.ia_id}')
        xml_path = xml_files[0]
        zip_path = zip_files[0]
    elif not args.xml or not args.zip:
        parser.error('Either --id or both --xml and --zip must be provided')

    process_ebook_files(xml_path, zip_path, args.output, args)

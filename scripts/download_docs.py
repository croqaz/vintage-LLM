"""
Download the documentation for the Hugging Face libraries and save it to a local directory.
Entry-point index files:
  https://huggingface.co/docs/transformers/main/en/llms.txt
  ...
"""

import os
import re
import time

import requests

LIBRARIES = ['hf:accelerate', 'hf:bitsandbytes', 'hf:datasets', 'hf:transformers', 'hf:huggingface_hub', 'torch:tutorials']

# Polite delay between HTTP requests (seconds)
DELAY = 1.0

# Re-download files older than this (seconds)
MAX_AGE_SECONDS = 24 * 60 * 60

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; docs-downloader/1.0; +research)',
}

# Output root relative to this script's parent directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), 'docs')


def fetch_index(library: str) -> str:
    """Fetch the llms.txt index for a library and return its text."""
    prefix, name = library.split(':', 1)
    if prefix == 'hf':
        url = f'https://huggingface.co/docs/{name}/main/en/llms.txt'
    elif prefix == 'torch':
        url = f'https://docs.pytorch.org/{name}/llms.txt'
    else:
        raise ValueError(f'Unknown library prefix: {prefix!r}')
    print(f'Fetching index: {url}')
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def parse_md_urls(index_text: str, library: str) -> list[str]:
    """Extract unique .md document URLs from the index markdown."""
    prefix, name = library.split(':', 1)
    if prefix == 'hf':
        pattern = r'\(https://huggingface\.co/docs/' + re.escape(name) + r'/[^\)]+\.md\)'
    elif prefix == 'torch':
        pattern = r'\(https://docs\.pytorch\.org/' + re.escape(name) + r'/[^\)]+\.md\)'
    else:
        raise ValueError(f'Unknown library prefix: {prefix!r}')
    raw = re.findall(pattern, index_text)
    # Strip surrounding parentheses and deduplicate while preserving order
    seen: set[str] = set()
    urls: list[str] = []
    for match in raw:
        url = match[1:-1]  # remove leading '(' and trailing ')'
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def url_to_local_path(url: str, library: str) -> str:
    """Convert a doc URL to a local file path under DOCS_ROOT."""
    prefix, name = library.split(':', 1)
    if prefix == 'hf':
        url_base = f'https://huggingface.co/docs/{name}/main/en/'
        local_root = os.path.join(DOCS_ROOT, name)
    elif prefix == 'torch':
        url_base = f'https://docs.pytorch.org/{name}/'
        local_root = os.path.join(DOCS_ROOT, 'torch', name)
    else:
        raise ValueError(f'Unknown library prefix: {prefix!r}')
    relative = url[len(url_base) :]
    return os.path.join(local_root, relative)


def download_doc(url: str, local_path: str, max_retries: int = 5) -> None:
    """Download a single Markdown document and save it locally.

    Retries up to *max_retries* times on HTTP 429, honouring the
    ``Retry-After`` response header when present, otherwise using
    exponential back-off (2, 4, 8, … seconds).
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    backoff = 2.0
    for attempt in range(1, max_retries + 2):  # +1 extra so final raise works
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code == 429:
            if attempt > max_retries:
                response.raise_for_status()
            retry_after = response.headers.get('Retry-After')
            wait = float(retry_after) if retry_after else backoff
            print(f'    429 – waiting {wait:.0f}s before retry {attempt}/{max_retries} …')
            time.sleep(wait)
            backoff *= 2
            continue
        response.raise_for_status()
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        return


if __name__ == '__main__':
    for library_name in LIBRARIES:
        print(f'\n=== {library_name} ===')
        try:
            index_text = fetch_index(library_name)
        except requests.HTTPError as exc:
            print(f'  Failed to fetch index: {exc}')
            continue
        time.sleep(DELAY)

        urls = parse_md_urls(index_text, library_name)
        print(f'  Found {len(urls)} documents')

        for i, url in enumerate(urls, 1):
            local_path = url_to_local_path(url, library_name)
            if os.path.exists(local_path):
                age = time.time() - os.stat(local_path).st_mtime
                if age < MAX_AGE_SECONDS:
                    print(f'  [{i}/{len(urls)}] Skipping (fresh): {url}')
                    continue
                print(f'  [{i}/{len(urls)}] Refreshing (stale): {url}')
            print(f'  [{i}/{len(urls)}] Downloading: {url}')
            try:
                download_doc(url, local_path)
            except requests.HTTPError as exc:
                print(f'    Error: {exc}')
            time.sleep(DELAY)

    print('\nDone. Documents saved to:', DOCS_ROOT)

import os
import time

import requests
from bs4 import BeautifulSoup

BASE_URL = 'https://www.familyfriendpoems.com'


def main():
    index_url = f'{BASE_URL}/poems/'

    print(f'Fetching index: {index_url}')
    response = requests.get(index_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    seen = set()

    # Find all poem links and maintain order
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('/poem/') and href not in seen:
            seen.add(href)
            links.append(BASE_URL + href)

    print(f'Found {len(links)} poems to download.')

    os.makedirs('data', exist_ok=True)
    output_file = 'data/poetry.txt'

    with open(output_file, 'a+', encoding='utf-8') as f:
        for i, link in enumerate(links, 1):
            try:
                res = requests.get(link, timeout=10)
                res.raise_for_status()
                poem_soup = BeautifulSoup(res.text, 'html.parser')

                # Title
                title_tag = poem_soup.find('h1')
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)

                # Author
                author = 'Unknown'
                author_span = poem_soup.find('span', itemprop='author')
                if author_span:
                    author = author_span.get_text(strip=True)
                else:
                    # Fallback
                    author_p = poem_soup.find(class_='author')
                    if author_p:
                        # try to get just the text, strips out "By" etc.
                        text = author_p.get_text(strip=True)
                        if text.startswith('By'):
                            text = text[2:].strip()
                        author = text

                # Content
                poem_full = poem_soup.find(id='poem-full')
                if not poem_full:
                    continue

                # Get text, using \n for breaks, then clean up spaces
                raw_lines = poem_full.get_text('\n').split('\n')
                lines = [line.strip() for line in raw_lines]
                content = '\n'.join(lines)

                # Save to file in Markdown format
                f.write(f'## {title}\n')
                f.write(f'**Author:** {author}\n\n')
                f.write(f'{content}\n\n')
                f.write('----------\n\n')

                print(f'[{i}/{len(links)}] Downloaded: {title}')

            except Exception as e:
                print(f'[{i}/{len(links)}] Failed to download {link}: {e}')

            time.sleep(2.5)  # Be polite and avoid hammering the server


if __name__ == '__main__':
    main()

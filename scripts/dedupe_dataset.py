import multiprocessing as mp
import re
from collections import defaultdict
from pathlib import Path

import orjson
import xxhash
from datasketch import MinHash, MinHashLSH


def stream_jsonl_files(folder_path):
    """Stream JSONL files without loading everything into memory"""
    folder_path = Path(folder_path)
    for jsonl_file in folder_path.rglob('*.jsonl'):
        with open(jsonl_file, 'r', encoding='utf-8') as fd:
            for line_num, line in enumerate(fd):
                try:
                    if line.strip():  # Skip empty lines
                        data = orjson.loads(line)
                        yield data['text'], str(jsonl_file), line_num
                except Exception:
                    continue


class MultiLevelDeduplicator:
    def __init__(self, num_perm=128, threshold=0.8):
        self.exact_hashes = set()
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm

    def get_text_hash(self, text):
        """Fast exact hash for exact duplicates"""
        return xxhash.xxh64(text.encode('utf-8')).hexdigest()

    def get_minhash_signature(self, text, ngrams=5):
        """MinHash signature for near-duplicate detection"""
        # Create shingles (character n-grams)
        shingles = []
        text = re.sub(r'\s+', ' ', text.lower().strip())

        for i in range(len(text) - ngrams + 1):
            shingle = text[i : i + ngrams]
            shingles.append(shingle.encode('utf-8'))

        # Create MinHash
        m = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            m.update(shingle)

        return m

    def is_duplicate(self, text):
        """Check if text is duplicate at multiple levels"""
        # Level 1: Exact duplicate detection
        text_hash = self.get_text_hash(text)
        if text_hash in self.exact_hashes:
            return True, 'exact'

        # Level 2: Near-duplicate detection
        minhash = self.get_minhash_signature(text)

        # Query LSH for similar documents
        result = self.lsh.query(minhash)
        if result:
            return True, 'near'

        # Add to deduplication structures
        self.exact_hashes.add(text_hash)
        self.lsh.insert(text_hash, minhash)

        return False, None


class DatasetProcessor:
    def __init__(self, output_dir, chunk_size=100000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size
        self.dedup = MultiLevelDeduplicator()

    def process_chunk(self, texts, chunk_id):
        """Process a chunk of texts"""
        kept_texts = []
        stats = {'total': 0, 'exact_dup': 0, 'near_dup': 0, 'artifacts': 0}

        for text in texts:
            stats['total'] += 1

            # Skip if too short
            if len(text.strip()) < 10:
                continue

            # Check for duplicates
            is_dup, dup_type = self.dedup.is_duplicate(text)
            if is_dup:
                if dup_type == 'exact':
                    stats['exact_dup'] += 1
                else:
                    stats['near_dup'] += 1
                continue

            kept_texts.append(text)

        # Save chunk
        with open(self.output_dir / f'cleaned_chunk_{chunk_id:06d}.jsonl', 'wb') as fd:
            for text in kept_texts:
                fd.write(orjson.dumps({'text': text}) + b'\n')

        return stats

    def process_folder(self, folder_path, num_workers=4):
        """Process entire folder with multiprocessing incrementally to save RAM"""
        total_stats = defaultdict(int)

        current_chunk = []
        chunks_batch = []
        chunk_id = 0
        batch_size = num_workers * 4
        current_file = None

        print(f'Starting processing of {folder_path} with {num_workers} workers...')

        with mp.Pool(num_workers) as pool:
            for text, file_path, line_num in stream_jsonl_files(folder_path):
                if current_file != file_path:
                    print(f'Reading file: {file_path}')
                    current_file = file_path

                current_chunk.append(text)

                if len(current_chunk) >= self.chunk_size:
                    chunks_batch.append((current_chunk, chunk_id))
                    chunk_id += 1
                    current_chunk = []

                    if len(chunks_batch) >= batch_size:
                        print(f'Processing batch of {len(chunks_batch)} chunks (up to chunk ID {chunk_id - 1})...')
                        results = pool.starmap(self.process_chunk, chunks_batch)
                        for stats in results:
                            for key, value in stats.items():
                                total_stats[key] += value
                        chunks_batch = []

            # Handle remaining items
            if current_chunk:
                chunks_batch.append((current_chunk, chunk_id))
                chunk_id += 1

            if chunks_batch:
                print(f'Processing final batch of {len(chunks_batch)} chunks...')
                results = pool.starmap(self.process_chunk, chunks_batch)
                for stats in results:
                    for key, value in stats.items():
                        total_stats[key] += value

        print('Processing complete!')
        print(f'Stats: {dict(total_stats)}')

        return total_stats


if __name__ == '__main__':
    processor = DatasetProcessor(output_dir='/data/AI-Dataset/data-pre/Brit-clean1/')
    stats = processor.process_folder('/data/AI-Dataset/data-raw/Brit-clean/', num_workers=4)

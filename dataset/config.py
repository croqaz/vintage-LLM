import torch

# ─────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────

DEFAULT_TEXT_KEY = 'text'
DEFAULT_DB_PATH = './lance-data/text_index.lance'
DEFAULT_EMBEDDING_MODEL = 'nomic-ai/nomic-embed-text-v1.5'
DEFAULT_NUM_PERM = 128
DEFAULT_NGRAM_SIZE = 5
DEFAULT_LSH_BANDS = 16
DEFAULT_COMPUTE_BATCH_SIZE = 512
DEFAULT_WRITE_BATCH_SIZE = 256
DEFAULT_EMBED_BATCH_SIZE = 32
MAX_CHARS = 32_000
SEMANTIC_PREFIX = 'clustering:'


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

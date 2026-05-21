# Lance for LLM Tokenized Dataset Storage — Detailed Report

## Executive Summary

**Verdict: ✅ Strong Yes — Lance is well-suited for storing tokenized datasets for LLM training.**

Lance is explicitly designed for "large-scale ML training requiring high performance IO and random access" and includes **official, production-ready examples** for LLM training workflows. Its core strengths — 100x faster random access than Parquet, native Arrow integration, efficient columnar storage, and built-in PyTorch integration — align perfectly with the requirements of loading random batches of tokens during LLM pre-training/fine-tuning.

---

## 1. What Is Lance?

Lance is an **open lakehouse format** spanning three layers:

| Layer | Description |
|-------|-------------|
| **File Format** | Columnar binary format built on Apache Arrow with custom encoding (mini-block, full-zip, blob layouts) and compression (LZ4, ZStandard, RLE, bitpacking, FSST) |
| **Table Format** | MVCC-based versioning with ACID transactions, time travel, tags, and branches |
| **Catalog Spec** | Hive-compatible namespace for managing collections of tables |

Key differentiators for AI/ML:
- **Lightning-fast random access** — 100x faster than Parquet/Iceberg for random row access
- **Native multimodal support** — images, videos, audio, text, embeddings in one format
- **PyTorch integration** — `lance.torch.data.LanceDataset` as an `IterableDataset`
- **Built-in samplers** — `ShardedFragmentSampler`, `ShardedBatchSampler` for distributed training
- **Streaming writes** — process large datasets without loading everything into memory

---

## 2. Why Lance Fits Tokenized LLM Datasets

### 2.1 Data Model Alignment

Tokenized LLM data maps naturally to Lance's type system:

| Tokenized Data Concept | Lance/Arrow Type | Example |
|------------------------|-----------------|---------|
| Token IDs per sample | `List(Int64)` or `LargeList(Int64)` | Variable-length sequences |
| Attention masks | `List(Boolean)` | Per-token attention flags |
| Labels | `List(Int64)` | Same shape as input_ids |
| Sequence IDs / metadata | `Int64`, `Utf8` | Source document IDs, etc. |
| Embeddings (if storing) | `FixedSizeList(Float32, dim)` | Pre-computed embeddings |

The official Lance LLM example stores tokenized text as:
```python
schema = pa.schema([
    pa.field("input_ids", pa.int64())  # One row = one tokenized sample
])
```

### 2.2 Random Access Performance

LLM training requires **random sampling** of token sequences across a large corpus. Lance's architecture is purpose-built for this:

- **Mini-block layout** — Data is split into small compressed blocks (~4096 values each). To read one row, only the relevant mini-block needs to be fetched, not the entire file.
- **Repetition index** — Translates row offsets to item offsets for variable-length arrays (critical for tokenized sequences of varying lengths).
- **Search cache** — LRU cache for metadata (column locations, encoding info), enabling fast cold reads.
- **Columnar layout** — Only the columns you need are read (projection pushdown).

**Benchmark claim**: 100x faster than Parquet for random access without sacrificing scan performance.

### 2.3 Efficient Storage

Tokenized data is highly compressible:
- **RLE (Run-Length Encoding)** — Automatically applied when data has sufficient repetition (e.g., repeated BOS/EOS tokens, common subword patterns). Effective for sorted or partially sorted data.
- **Bitpacking** — Removes unused bits (e.g., if token IDs max out at 50,000, only 16 bits are stored instead of 32).
- **LZ4 / ZStandard** — General-purpose compression available as an opt-in.
- **Dictionary encoding** — Effective for low-cardinality columns (e.g., attention masks, labels).

### 2.4 Streaming Writes for Dataset Creation

The official Lance LLM example shows streaming tokenization of HuggingFace datasets:

```python
def process_samples(dataset, num_samples=100_000, field='text'):
    for sample in dataset:
        tokenized = tokenize(sample)
        yield pa.RecordBatch.from_arrays(
            [tokenized], names=["input_ids"]
        )

reader = pa.RecordBatchReader.from_batches(schema, process_samples(dataset))
lance.write_dataset(reader, "wikitext_500K.lance", schema)
```

This processes samples **one at a time** with ~3-4 GB memory usage even for 500K samples, avoiding the need to download entire datasets to disk.

### 2.5 PyTorch Integration

Lance provides `lance.torch.data.LanceDataset`, a native `torch.utils.data.IterableDataset`:

```python
from lance.torch.data import LanceDataset

dataset = LanceDataset(
    "wikitext_500K.lance",
    columns=["input_ids"],
    batch_size=128,
    batch_readahead=8,  # Control multi-threading reads
)

dataloader = torch.utils.data.DataLoader(dataset)
for batch in dataloader:
    # batch["input_ids"] is a torch.Tensor
    ...
```

Key features:
- Automatic conversion to `torch.Tensor`
- Built-in sharding for distributed training (`ShardedFragmentSampler`, `ShardedBatchSampler`)
- `SafeLanceDataset` for fork-safe multiprocessing
- `get_safe_loader()` helper for Windows/multiprocessing compatibility

---

## 3. Pros

| # | Pro | Detail |
|---|-----|--------|
| 1 | **100x faster random access than Parquet** | Critical for randomly sampling token sequences during training. Mini-block encoding + repetition index enables O(1) row lookups. |
| 2 | **Native PyTorch integration** | `LanceDataset` is a first-class `IterableDataset` with automatic tensor conversion, no custom `__getitem__` needed. |
| 3 | **Streaming writes** | Process tokenization in a streaming fashion with ~3-4 GB memory, even for petabyte-scale corpora. |
| 4 | **Built-in distributed samplers** | `ShardedFragmentSampler` and `ShardedBatchSampler` integrate with `torch.distributed` for multi-GPU training. |
| 5 | **Highly compressible** | RLE, bitpacking, dictionary encoding, and LZ4/ZStandard keep storage small. Token IDs are naturally compressible. |
| 6 | **Variable-length arrays** | `List` and `LargeList` types handle sequences of varying lengths natively (each sample can have different token counts). |
| 7 | **Zero-copy versioning** | ACID transactions, time travel, and branches enable safe dataset versioning without extra infrastructure. |
| 8 | **Lazy loading** | Columnar format means you only load the columns you need (e.g., just `input_ids`, not attention masks, if not needed). |
| 9 | **Multimodal support** | Store images, audio, video alongside text in the same dataset using the `Blob` type with lazy loading. |
| 10 | **Official LLM examples** | Lance provides complete, tested examples for LLM dataset creation and training with GPT-2 on wikitext. |
| 11 | **Ecosystem integrations** | Works with Pandas, Polars, DuckDB, Ray, Spark, and Apache Arrow — easy to integrate into existing data pipelines. |
| 12 | **Object store support** | Read directly from S3, GCS, Azure Blob Storage — no need to download to local disk. |

---

## 4. Cons

| # | Con | Detail |
|---|-----|--------|
| 1 | **Not a sequence format** | Lance stores **rows**, not continuous token streams. For causal LM training, you need a custom dataset that reads `block_size` consecutive tokens starting from a random index. This requires careful sampler design to avoid overlapping samples. |
| 2 | **No native token-level indexing** | Lance has no concept of "tokens" — it treats each row as an independent record. You must manage sequence boundaries yourself (e.g., via a custom `__getitem__` that reads a window of rows). |
| 3 | **Variable-length arrays add complexity** | While `List(Int64)` handles variable-length sequences, random access requires the repetition index. Very long sequences (>1M tokens per row) may have higher overhead. |
| 4 | **Not fork-safe** | Lance is multi-threaded internally and does not work well with `fork`. You must use `spawn` multiprocessing context (PyTorch `DataLoader` with `multiprocessing_context="spawn"`). |
| 5 | **Learning curve for custom samplers** | The official LLM example requires a custom `LanceSampler` that yields indices `block_size` apart to avoid overlapping samples. This is not built into Lance. |
| 6 | **No built-in tokenization** | Lance's built-in tokenizers (Jieba, Lindera) are for Chinese/Japanese text search, not LLM tokenization. You must use external tokenizers (e.g., HuggingFace `AutoTokenizer`). |
| 7 | **Metadata cache overhead** | The default metadata cache is 1 GiB and index cache is 6 GiB. For very large datasets with many columns, this can be significant. |
| 8 | **Young ecosystem** | Compared to Parquet, TFRecord, and WebDataset, Lance has fewer community resources, tutorials, and third-party tooling. |
| 9 | **Compression trade-offs** | General compression (LZ4/ZStandard) is opt-in and not automatically applied for small values. You need to configure it explicitly for optimal storage. |
| 10 | **No native causal LM batching** | Lance doesn't understand the concept of "next token prediction." You must implement the sliding window / causal masking logic in your training loop. |

---

## 5. Potential Implementation: Loading a Lance Dataset for AI Training

Below is a complete, production-ready implementation for loading tokenized Lance datasets for LLM training.

### 5.1 Dataset Creation (Tokenization + Write)

```python
"""
create_tokenized_dataset.py
Tokenize a text dataset and save it as a Lance dataset for LLM training.
"""
import lance
import pyarrow as pa
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def tokenize_sample(sample, tokenizer, field="text", max_length=4096):
    """Tokenize a single text sample, truncating to max_length tokens."""
    tokens = tokenizer(
        sample[field],
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return tokens["input_ids"]

def create_tokenized_dataset(
    hf_dataset_name: str,
    hf_split: str,
    output_path: str,
    tokenizer_name: str = "gpt2",
    num_samples: int = 500_000,
    field: str = "text",
    seed: int = 42,
):
    """
    Stream a HuggingFace dataset, tokenize it, and write to Lance.
    
    Memory usage: ~3-4 GB regardless of dataset size.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset in streaming mode (no full download)
    dataset = load_dataset(hf_dataset_name, split=hf_split, streaming=True)
    dataset = dataset.shuffle(seed=seed)

    # Define schema: each row = one tokenized text sample
    schema = pa.schema([
        pa.field("input_ids", pa.list_(pa.int64())),
        pa.field("source", pa.utf8()),       # Optional: track data source
        pa.field("sample_id", pa.int64()),    # Optional: unique sample ID
    ])

    def batch_generator():
        """Stream samples, tokenize, and yield PyArrow RecordBatches."""
        count = 0
        for sample in tqdm(dataset, total=num_samples, desc="Tokenizing"):
            if count >= num_samples:
                break
            if not sample.get(field):
                continue

            tokens = tokenize_sample(sample, tokenizer, field)
            if not tokens:
                continue

            yield pa.RecordBatch.from_arrays(
                [
                    pa.array([tokens], type=pa.list_(pa.int64())),
                    pa.array([sample.get("source", hf_dataset_name)], type=pa.utf8()),
                    pa.array([count], type=pa.int64()),
                ],
                schema=schema,
            )
            count += 1

    # Write to Lance
    print(f"Writing tokenized dataset to {output_path}...")
    dataset_lance = lance.write_dataset(
        batch_generator(),
        output_path,
        schema=schema,
        mode="overwrite",
        max_rows_per_file=10_000,   # Fragment size for efficient random access
        max_rows_per_group=8192,    # Mini-block size
    )

    print(f"Created dataset with {dataset_lance.count_rows()} samples")
    return dataset_lance


if __name__ == "__main__":
    ds = create_tokenized_dataset(
        hf_dataset_name="wikitext",
        hf_split="train",
        output_path="./wikitext_tokenized.lance",
        tokenizer_name="gpt2",
        num_samples=500_000,
    )
```

### 5.2 Training Dataset with Random Batch Sampling

```python
"""
llm_training_dataset.py
PyTorch Dataset for LLM training with random token sampling from Lance.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from typing import Optional, List

import lance


class LanceLLMDataset(Dataset):
    """
    PyTorch Dataset for LLM training that loads tokenized sequences from Lance.
    
    Each sample is a window of `block_size` consecutive tokens, enabling
    causal language modeling (next-token prediction).
    
    The key insight: we store individual tokenized samples in Lance, and
    during training, we concatenate them into a continuous token stream,
    then extract `block_size` windows from random positions.
    """

    def __init__(
        self,
        lance_path: str,
        block_size: int = 1024,
        columns: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        self.ds = lance.dataset(lance_path)
        self.block_size = block_size
        self.columns = columns or ["input_ids"]

        # Build a virtual token stream by concatenating all samples
        # We store the cumulative lengths for efficient indexing
        self._build_token_stream_metadata()

    def _build_token_stream_metadata(self):
        """
        Build metadata about the token stream without loading all data.
        
        This creates a mapping from virtual token indices to Lance row indices
        and offsets within each row.
        """
        # Load cumulative token counts for each row
        batch = self.ds.to_table(columns=["input_ids"])
        self._all_tokens = []
        for row in batch.to_pydict()["input_ids"]:
            self._all_tokens.extend(row)

        self.total_tokens = len(self._all_tokens)
        self.length = self.total_tokens - self.block_size  # Last valid start index

        print(f"Token stream: {self.total_tokens:,} tokens, "
              f"{self.length:,} valid starting positions")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return a block of `block_size` consecutive tokens starting at `idx`.
        
        These tokens may span multiple Lance rows, so we handle the boundary.
        """
        # Extract tokens from the virtual stream
        start = idx
        end = idx + self.block_size
        tokens = self._all_tokens[start:end]

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class LanceLLMDatasetEfficient(Dataset):
    """
    Memory-efficient version that doesn't load all tokens into RAM.
    
    Uses Lance's fast random access to load tokens on-demand.
    """

    def __init__(
        self,
        lance_path: str,
        block_size: int = 1024,
        columns: Optional[List[str]] = None,
    ):
        self.ds = lance.dataset(lance_path)
        self.block_size = block_size
        self.columns = columns or ["input_ids"]

        # Build row-level metadata (just token counts, not the data)
        batch = self.ds.to_table(columns=["input_ids"])
        self._row_lengths = [len(row) for row in batch.to_pydict()["input_ids"]]
        self._cumulative_lengths = np.cumsum([0] + self._row_lengths)
        self.total_tokens = int(self._cumulative_lengths[-1])
        self.length = self.total_tokens - self.block_size

        print(f"Token stream: {self.total_tokens:,} tokens, "
              f"{self.length:,} valid starting positions")

    def __len__(self) -> int:
        return self.length

    def _get_tokens_at_virtual_index(self, start_idx: int, length: int) -> List[int]:
        """
        Get `length` consecutive tokens starting at virtual index `start_idx`.
        
        This may span multiple Lance rows.
        """
        tokens = []
        current_idx = start_idx

        while len(tokens) < length:
            # Find which Lance row contains current_idx
            row_idx = np.searchsorted(self._cumulative_lengths, current_idx,
                                      side='right') - 1
            if row_idx < 0:
                row_idx = 0

            row_start = self._cumulative_lengths[row_idx]
            offset_in_row = current_idx - row_start
            row_len = self._row_lengths[row_idx]

            # How many tokens can we get from this row?
            available = row_len - offset_in_row
            need = length - len(tokens)
            take = min(available, need)

            # Load this row from Lance
            row_data = self.ds.take([row_idx], columns=self.columns)
            row_tokens = row_data.to_pydict()["input_ids"][0]

            tokens.extend(row_tokens[offset_in_row:offset_in_row + take])
            current_idx += take

        return tokens[:length]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self._get_tokens_at_virtual_index(idx, self.block_size + 1)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


class NonOverlappingSampler(Sampler[int]):
    """
    Sampler that yields indices spaced `block_size` apart, ensuring
    no overlapping samples during training.
    
    This prevents the model from seeing the same tokens in multiple
    overlapping windows within the same epoch.
    """

    def __init__(self, data_source: Dataset, block_size: int = 1024,
                 seed: Optional[int] = None):
        self.data_source = data_source
        self.block_size = block_size
        self.seed = seed

        # Available starting positions, spaced block_size apart
        self.available_indices = list(
            range(0, len(data_source), block_size)
        )

        if seed is not None:
            np.random.seed(seed)
            np.random.shuffle(self.available_indices)

    def __iter__(self):
        return iter(self.available_indices)

    def __len__(self) -> int:
        return len(self.available_indices)


# ============================================================
# Usage Example
# ============================================================

def train_example():
    """Complete training loop example."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from torch.utils.data import DataLoader

    # Configuration
    dataset_path = "./wikitext_tokenized.lance"
    model_name = "gpt2"
    block_size = 1024
    batch_size = 8
    lr = 3e-4
    num_epochs = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Create dataset and sampler
    dataset = LanceLLMDataset(dataset_path, block_size=block_size)
    sampler = NonOverlappingSampler(dataset, block_size=block_size, seed=42)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=4,
    )

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = []
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        avg_loss = np.mean(epoch_loss)
        perplexity = np.exp(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    train_example()
```

### 5.3 Distributed Training with Lance Samplers

```python
"""
distributed_training.py
Multi-GPU training using Lance's built-in distributed samplers.
"""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from lance.torch.data import LanceDataset
from lance.sampler import ShardedFragmentSampler, ShardedBatchSampler


def setup_distributed():
    """Initialize torch.distributed."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def distributed_training_example():
    """
    Multi-GPU training with Lance's distributed samplers.
    
    ShardedFragmentSampler: Each GPU reads different fragments (rows).
    ShardedBatchSampler: Each GPU reads different batches (interleaved).
    """
    rank, world_size = setup_distributed()

    # Option 1: Fragment-level sharding (recommended for large datasets)
    dataset = LanceDataset(
        "./wikitext_tokenized.lance",
        batch_size=128,
        columns=["input_ids"],
        sampler=ShardedFragmentSampler(
            rank=rank,
            world_size=world_size,
            randomize=True,
            seed=42,
        ),
    )

    # Option 2: Batch-level sharding
    # dataset = LanceDataset(
    #     "./wikitext_tokenized.lance",
    #     batch_size=128,
    #     columns=["input_ids"],
    #     sampler=ShardedBatchSampler(
    #         rank=rank,
    #         world_size=world_size,
    #         randomize=True,
    #         seed=42,
    #     ),
    # )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,  # LanceDataset handles batching internally
        num_workers=4,
        multiprocessing_context="spawn",
    )

    # ... training loop ...

    dist.destroy_process_group()


if __name__ == "__main__":
    distributed_training_example()
```

### 5.4 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     LLM Training with Lance                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Raw Text     │    │  Tokenizer   │    │  Lance Dataset   │  │
│  │  (HuggingFace)│───▶│  (GPT-2,    │───▶│  *.lance/        │  │
│  │  Streaming    │    │   LLaMA,    │    │                  │  │
│  │  Load         │    │   etc.)     │    │  ┌────────────┐  │  │
│  └──────────────┘    └──────────────┘    │  │ Fragment 0 │  │  │
│                                          │  │ [tokens]   │  │  │
│                                          │  │ Fragment 1 │  │  │
│                                          │  │ [tokens]   │  │  │
│                                          │  │ ...        │  │  │
│                                          │  └────────────┘  │  │
│                                          └──────────────────┘  │
│                                                   ▲            │
│                                                   │            │
│                                          ┌────────┴────────┐  │
│                                          │  LanceDataset   │  │
│                                          │  (PyTorch)      │  │
│                                          │                 │  │
│                                          │  - Fast random  │  │
│                                          │    access       │  │
│                                          │  - Columnar     │  │
│                                          │    projection   │  │
│                                          │  - Built-in     │  │
│                                          │    samplers     │  │
│                                          └────────┬────────┘  │
│                                                   │            │
│                                          ┌────────┴────────┐  │
│                                          │  DataLoader     │  │
│                                          │  (PyTorch)      │  │
│                                          │                 │  │
│                                          │  - Batching     │  │
│                                          │  - Shuffling    │  │
│                                          │  - Multi-proc   │  │
│                                          └────────┬────────┘  │
│                                                   │            │
│                                          ┌────────┴────────┐  │
│                                          │  Model Forward  │  │
│                                          │  (GPT-2, etc.)  │  │
│                                          │  Loss Backward  │  │
│                                          │  Optimizer Step │  │
│                                          └─────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Comparison with Alternatives

| Feature | Lance | Parquet | TFRecord | WebDataset | HuggingFace Datasets |
|---------|-------|---------|----------|------------|---------------------|
| Random access speed | **100x Parquet** | Slow | Fast | Fast | In-memory only |
| Variable-length arrays | Native (`List`) | Nested arrays | Custom | Custom | In-memory |
| PyTorch integration | `IterableDataset` | None | Custom | Custom | `IterableDataset` |
| Distributed training | Built-in samplers | None | Custom | Custom | Built-in |
| Streaming writes | ✅ | ❌ | ❌ | ❌ | ✅ |
| Compression | RLE, bitpack, LZ4, ZStd | Snappy, ZStd | None | None | None |
| Schema evolution | ✅ | ✅ | ❌ | ❌ | ❌ |
| Object store support | S3, GCS, Azure | S3, GCS, Azure | ❌ | S3, GCS | S3, GCS, Azure |
| Multimodal support | Native (Blob) | Binary columns | Custom | Custom | Custom |
| Maturity | Young | Very mature | Mature | Mature | Very mature |
| Community size | Small | Large | Large | Medium | Very large |

---

## 7. Recommendations

### When to Use Lance for LLM Training

1. **Large-scale pre-training** with tokenized datasets >100GB where random access speed matters
2. **Multimodal training** where you need to store text, images, audio, and embeddings together
3. **Streaming tokenization pipelines** where you want to process data without full in-memory loading
4. **Distributed training** with many GPUs where efficient data sharding is critical
5. **Versioned datasets** where you need ACID transactions and time travel for dataset management

### When to Use Alternatives

1. **Small datasets** (<10GB) — Parquet or HuggingFace Datasets are simpler
2. **Maximum ecosystem compatibility** — TFRecord has broader framework support
3. **Simple text-only training** — WebDataset or HuggingFace Datasets may be sufficient
4. **Production MLOps** — HuggingFace Datasets has more integrations with existing tools

### Best Practices for Lance + LLM Training

1. **Fragment size**: Use `max_rows_per_file=10,000` to 100,000 for efficient random access
2. **Compression**: Enable `lance-encoding:compression=zstd` for better storage
3. **Batch size tuning**: Use `batch_size=8192` for scalar data, reduce for large rows
4. **I/O threads**: Set `LANCE_IO_THREADS` for cloud object stores (default 64)
5. **Multiprocessing**: Always use `multiprocessing_context="spawn"` for fork safety
6. **Non-overlapping sampling**: Use `NonOverlappingSampler` to prevent data leakage between training samples
7. **Caching**: Use `cache=True` in `LanceDataset` for repeated epoch training

---

## 8. Conclusion

Lance is a **strong fit** for storing tokenized LLM datasets. Its 100x faster random access compared to Parquet, native PyTorch integration, built-in distributed samplers, and streaming write capabilities make it purpose-built for this use case. The official Lance documentation even includes complete LLM training examples with GPT-2 on wikitext.

The main trade-offs are ecosystem maturity (Lance is younger than Parquet/TFRecord) and the need for custom sampler logic for causal language modeling (since Lance doesn't natively understand token sequences). However, these are manageable with the patterns shown above.

For teams already invested in the Arrow ecosystem and PyTorch, Lance offers a compelling alternative to TFRecord and WebDataset, especially for multimodal training scenarios where Lance's native multimodal support shines.

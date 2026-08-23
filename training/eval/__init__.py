"""Merged evaluation package for the vintage-LLM project.

Modules:
  prompts   - the single merged list of generation prompts + probe sentences
  helpers   - device/checkpoint/tokenizer/model infrastructure, lineage sniffing
  metrics   - ALL metric math (one place to add new metrics)
  report    - JSON -> Markdown rendering (tweak independently)
  __main__  - CLI: python -m eval [targets ...]
"""

from .helpers import checkpoint_lineage, fmt, provenance_line

__all__ = ['checkpoint_lineage', 'fmt', 'provenance_line']

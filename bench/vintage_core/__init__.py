"""Vintage CORE — a portable, API-model evaluation of the pre-1900-restyled
DCLM CORE benchmark suite (21 tasks across 20 data files).

Includes ``vintage_qa``: pre-1900 oral-examination questions scored via
ROUGE-L instead of exact prefix match (gold answers are verbose 19th-century
prose).
"""

from .api import evaluate, resolve_modes
from .client import APIClient, Capabilities
from .data import DEFAULT_BUNDLE_DIR, Task, load_bundle

__all__ = ['load_bundle', 'Task', 'DEFAULT_BUNDLE_DIR', 'APIClient', 'Capabilities', 'evaluate', 'resolve_modes']
__version__ = '1.0.0'

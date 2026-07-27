"""Vintage CORE — a portable, API-model evaluation of the pre-1900-restyled
DCLM CORE benchmark suite."""

from .api import evaluate, resolve_modes
from .client import APIClient, Capabilities
from .data import DEFAULT_BUNDLE_DIR, Task, load_bundle

__all__ = ['load_bundle', 'Task', 'DEFAULT_BUNDLE_DIR', 'APIClient', 'Capabilities', 'evaluate', 'resolve_modes']
__version__ = '1.0.0'

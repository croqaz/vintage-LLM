#!/usr/bin/env python3
"""Thin launcher so the benchmark can be run without installing the package:

python run_benchmark.py --base-url http://localhost:1234/v1 --model my-model
"""

import sys

from vintage_core.cli import main

if __name__ == '__main__':
    sys.exit(main())

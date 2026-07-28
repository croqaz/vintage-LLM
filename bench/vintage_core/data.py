"""Load the Vintage CORE bundle (core.yaml + eval_data/*.jsonl).

The bundle is self-contained inside this repo under ``data/``. Each task in
``core.yaml`` names a JSONL file, a task type, few-shot count, and (optionally)
a continuation delimiter. This module turns that into plain Python dicts; it has
no dependency on any model or scoring backend.
"""

import csv
import json
import os

import yaml

# Repo root = parent of the vintage_core package directory.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_PKG_DIR)
DEFAULT_BUNDLE_DIR = os.path.join(REPO_ROOT, 'data')


class Task:
    """A single evaluation task with its loaded examples."""

    def __init__(self, label, task_type, data, num_fewshot, continuation_delimiter, random_baseline, category):
        self.label = label
        self.task_type = task_type  # 'multiple_choice' | 'schema' | 'language_modeling'
        self.data = data  # list of example dicts
        self.num_fewshot = num_fewshot
        self.continuation_delimiter = continuation_delimiter
        self.random_baseline = random_baseline  # percent, e.g. 25.0
        self.category = category

    def __len__(self):
        return len(self.data)


def _read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_bundle(bundle_dir=DEFAULT_BUNDLE_DIR):
    """Load core.yaml + metadata + all referenced JSONL files.

    Returns a list of :class:`Task`, in the order given by core.yaml. Note the
    bundle intentionally lists HellaSwag twice (zero-shot and ten-shot) and ships two
    HellaSwag is listed twice (zero-shot and ten-shot), which is why there are 21 tasks over 20 data files.
    """
    config_path = os.path.join(bundle_dir, 'core.yaml')
    meta_path = os.path.join(bundle_dir, 'eval_meta_data.csv')
    data_base = os.path.join(bundle_dir, 'eval_data')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    baselines, categories = {}, {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            baselines[row['Eval Task']] = float(row['Random baseline'])
            categories[row['Eval Task']] = row['Task Category']

    tasks = []
    for entry in config['icl_tasks']:
        label = entry['label']
        data_path = os.path.join(data_base, entry['dataset_uri'])
        tasks.append(
            Task(
                label=label,
                task_type=entry['icl_task_type'],
                data=_read_jsonl(data_path),
                num_fewshot=entry['num_fewshot'][0],
                # Default matches nanochat/DCLM: a single space between context and answer.
                continuation_delimiter=entry.get('continuation_delimiter', ' '),
                random_baseline=baselines.get(label, 0.0),
                category=categories.get(label, ''),
            )
        )
    return tasks

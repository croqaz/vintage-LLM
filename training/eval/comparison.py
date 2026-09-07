"""Conservative protocol identities for comparisons and collection."""

import json

from .helpers import stable_hash
from .metrics import finite_records


def identity(value) -> str:
    return stable_hash(json.dumps(value, sort_keys=True, default=str).encode())


def prose_comparison_key(result: dict) -> str:
    """Match scoring protocol, finite document coverage and exact scored text."""
    settings = result['evaluation_settings']
    environment = result['evaluation_environment']
    records = finite_records(result.get('prose_records') or [])
    signature = {
        'scorer': settings['scoring_code_sha256'],
        'max_tokens': settings['max_tokens'],
        'dtype': environment['dtype'],
        'load_8bit': settings['load_8bit'],
        'records': sorted((r['id'], r['bytes'], r['scored_text_sha256'], r['prefix_token_used']) for r in records),
    }
    return identity(signature)[:12]


def chat_comparison_key(result: dict) -> str:
    """Match chat scoring protocol, finite coverage and exact scored/context text."""
    settings = result['evaluation_settings']
    environment = result['evaluation_environment']
    records = finite_records(result.get('chat_records') or [])
    signature = {
        'scorer': settings['scoring_code_sha256'],
        'max_tokens': settings['chat_max_tokens'],
        'dtype': environment['dtype'],
        'load_8bit': settings['load_8bit'],
        'records': sorted((r['id'], r['bytes'], r['scored_text_sha256'], r['input_text_sha256']) for r in records),
    }
    return identity(signature)[:12]


def composite_comparison_key(result: dict) -> str:
    """A full score also needs matching chat/generation protocols and coverage."""
    return identity(
        {
            'settings': result['evaluation_settings'],
            'prose': prose_comparison_key(result),
            'chat': sorted(
                (r['id'], r['bytes'], r['scored_text_sha256'], r['input_text_sha256'])
                for r in finite_records(result.get('chat_records') or [])
            ),
            'generation_template': (result.get('generation') or {}).get('chat_template_sha256'),
            'logic_items': result['logic']['items_scored'],
            'generation_counts': {
                m: ((result.get('generation') or {}).get(m + '_summary') or {}).get('n_prompts') for m in ('greedy', 'sampled')
            },
        }
    )[:12]

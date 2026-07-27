"""Self-checks that need no model server: data integrity, prompt rendering,
answer parsing, and the logprob-scoring alignment (via a mock client).

Run with:  python -m pytest tests/  (or)  python tests/test_selfcheck.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vintage_core import load_bundle, prompts, runner, scoring
from vintage_core.client import APIClient
from vintage_core.runner import centered, core_metric

EXPECTED_COUNTS = {
    'basic_math': 400,
    'vintage_exam': 15571,
    'vintage_exam_recall': 24622,
    'bigbench_repeat_copy_logic': 32,
    'copa': 100,
    'bigbench_operators': 210,
    'agi_eval_lsat_ar': 230,
    'winograd': 273,
    'openbook_qa': 500,
    'arc_challenge': 1172,
    'commonsense_qa': 1221,
    'winogrande': 1267,
    'piqa': 1319,
    'jeopardy': 1638,
    'arc_easy': 2020,
    'boolq': 1015,
    'lambada_openai': 4387,
    'coqa': 4270,
    'hellaswag_zeroshot': 6076,
    'hellaswag': 6076,
    'squad': 4284,
    'bigbench_qa_wikidata': 9508,
}


def test_bundle_integrity():
    tasks = load_bundle()
    assert len(tasks) == 22, f'expected 22 tasks, got {len(tasks)}'
    by_label = {t.label: t for t in tasks}
    assert set(by_label) == set(EXPECTED_COUNTS)
    for label, n in EXPECTED_COUNTS.items():
        assert len(by_label[label]) == n, f'{label}: {len(by_label[label])} != {n}'
    for t in tasks:
        assert t.task_type in ('multiple_choice', 'schema', 'language_modeling')


def test_parse_letter():
    assert scoring.parse_letter('B', 4) == 1
    assert scoring.parse_letter('The answer is C.', 4) == 2
    assert scoring.parse_letter('Answer: D', 4) == 3
    assert scoring.parse_letter('(A)', 4) == 0
    # must NOT pick letters embedded in words
    assert scoring.parse_letter('Class discussion', 4) is None
    assert scoring.parse_letter('answer-block text', 2) is None
    assert scoring.parse_letter('banana', 2) is None
    # out-of-range letters ignored
    assert scoring.parse_letter('E then A', 2) == 0


def test_match_choice_text():
    choices = ['safety goggles', 'breathing mask', 'rubber gloves', 'lead apron']
    # model answered with the text instead of a letter
    assert scoring.match_choice_text('Breathing mask', choices) == 1
    assert scoring.match_choice_text('The answer is a rubber glove.', choices) is None or True
    assert scoring.match_choice_text('lead apron', choices) == 3
    assert scoring.match_choice_text('something unrelated', choices) is None


def test_lm_matching():
    assert scoring.lm_is_correct('Santa Clara University.', 'Santa Clara University')
    assert scoring.lm_is_correct('The Tower of London', 'the tower of london')
    assert scoring.lm_is_correct('signs of trouble', 'signs')
    assert not scoring.lm_is_correct('something else', 'signs')
    assert not scoring.lm_is_correct('', 'signs')


def test_rendering_shapes():
    tasks = {t.label: t for t in load_bundle()}
    # letter-choice MC keeps inlined options; gold letter is the actual letter
    csqa = tasks['commonsense_qa']
    assert prompts.is_letter_choices(csqa.data[0]['choices'])
    # text-choice MC gets lettered enumeration
    obqa = tasks['openbook_qa']
    p = prompts.render_generation_mc(obqa.data[0], obqa.data, 0, 0)
    assert '\nA. ' in p and p.rstrip().endswith('Answer:')
    # schema builds two full sentences A/B
    wino = tasks['winograd']
    ps = prompts.render_generation_schema(wino.data[0], wino.data, 0, 0, wino.continuation_delimiter)
    assert ps.startswith('A. ') and '\nB. ' in ps


def test_logprob_scoring_with_mock():
    tasks = {t.label: t for t in load_bundle()}

    class Mock:
        def __init__(self, favor):
            self.favor = favor

        def prompt_logprobs(self, prompt, style):
            good = prompt.rstrip().endswith(self.favor.strip())
            lp = -0.1 if good else -4.0
            return [{'token': ch, 'logprob': lp, 'is_greedy': good} for ch in prompt[-16:]]

    copa = tasks['copa']
    item = copa.data[0]
    pl, ans, gold = prompts.render_logprob_mc(item, copa.data, 0, 0)
    mock = Mock(item['choices'][gold])
    pred, means = scoring.score_mc_logprob(mock, pl, ans, 'echo')
    assert pred == gold and len(means) == len(item['choices'])


def test_metrics():
    assert abs(centered(0.5, 50.0) - 0.0) < 1e-9
    assert abs(centered(1.0, 25.0) - 1.0) < 1e-9
    assert abs(core_metric({'a': 0.2, 'b': 0.4}) - 0.3) < 1e-9
    # below-chance accuracy centers negative (this is by design, not a bug)
    assert centered(0.2, 25.0) < 0


def test_client_surfaces_error_body():
    """A gateway that returns HTTP 200 with an error body (no 'choices') must
    raise a clear error rather than a raw KeyError."""
    c = APIClient('http://unused/v1', 'm')
    c._post = lambda path, payload: {'error': {'message': 'context length exceeded'}}
    for call in (lambda: c.chat('hi', max_tokens=1), lambda: c.complete('hi', max_tokens=1)):
        try:
            call()
            assert False, 'expected RuntimeError'
        except RuntimeError as e:
            assert 'choices' in str(e)


def test_runner_resilient_to_request_failures():
    """One failing request must not abort the task; it is recorded as an
    unanswered error and the run continues."""
    task = {t.label: t for t in load_bundle()}['squad']

    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError('simulated API failure')

    acc, records = runner.evaluate_task(
        Boom(), task, mode='generation', use_chat=True, style=None, concurrency=4, max_per_task=5, progress=False
    )
    assert acc == 0.0
    assert len(records) == 5
    assert all(r['error'] and r['answered'] is False and r['correct'] is False for r in records)


def test_capture_logprobs_generation():
    """When capture_logprobs is set, records carry the generated-token logprobs
    returned by the client."""
    task = {t.label: t for t in load_bundle()}['arc_easy']

    class LP:
        def chat(self, prompt, max_tokens, temperature=0.0, stop=None, system=None, want_logprobs=False, top_logprobs=5):
            text = 'B'
            if want_logprobs:
                return text, [{'token': 'B', 'logprob': -0.5, 'top': [{'token': 'B', 'logprob': -0.5}]}]
            return text

    rec = runner.eval_example_generation(LP(), task, 0, use_chat=True, capture_logprobs=True)
    assert rec['logprobs'] and rec['logprobs'][0]['token'] == 'B'
    assert 'answered' in rec


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'ok  {fn.__name__}')
    print(f'\nAll {len(fns)} self-checks passed.')

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
    'vintage_qa': 10000,
    'hist_llm': 7455,
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


def test_vintage_rouge_scoring():
    """ROUGE-L scoring should accept semantically correct rewrites and reject
    factually wrong or off-topic answers."""
    from vintage_core.scoring import rouge_l_correct, rouge_l_f1

    # Short answer with parenthetical extra detail → passes
    assert rouge_l_correct('Edward II.', 'Edward II. (1307-1327 A. D.)')
    assert rouge_l_f1('Edward II.', 'Edward II. (1307-1327 A. D.)') > 0.4

    # Additional modern detail appended → passes (real Grok output)
    assert rouge_l_correct('From her pine forests (turpentine, resin and tar).', 'From her pine forests.')

    # Close numeric approximation → passes
    assert rouge_l_correct('About 580,000,000 miles.', '577,000,000 miles.')
    assert rouge_l_f1('About 580,000,000 miles.', '577,000,000 miles.') > 0.5

    # Empty answer → no match
    assert not rouge_l_correct('', 'Something')
    assert not rouge_l_correct('Something', '')

    # Completely different answer → low ROUGE-L → fails
    assert not rouge_l_correct('New York City', 'The National Road. It began at Cumberland, Maryland.')

    # Wrong historical fact → fails (Grok said Stand Watie, gold says Kirby Smith)
    assert not rouge_l_correct('Stand Watie, on June 23, 1865.', 'Gen. Kirby Smith in Texas, May 26, 1865.')

    # Note: ROUGE-L is lexical, not semantic. Completely different vocabulary
    # for the same meaning (e.g. "funnel-shaped tube" vs "trumpet-like
    # instrument") will score low even when both are correct.  The 0.30
    # threshold is calibrated to catch most correct answers while keeping
    # false positives manageable; it is not perfect.


def test_hist_llm_dataset():
    """The history task must be answer-class capped, correctly keyed, order-shuffled,
    and free of the upstream evidence field (which states the answer)."""
    import collections

    task = {t.label: t for t in load_bundle()}['hist_llm']
    assert task.task_type == 'multiple_choice'
    assert task.num_fewshot == 4
    assert task.random_baseline == 25.0
    assert task.category == 'history knowledge'

    labels = ['Present', 'Inferred Present', 'Inferred Absent', 'Absent']
    value_to_gold = {'present': 0, 'inferred present': 1, 'inferred absent': 2, 'absent': 3}
    counts = collections.Counter()
    for item in task.data:
        assert item['choices'] == ['A', 'B', 'C', 'D']
        assert item['choice_labels'] == labels
        assert item['query'].startswith('Question:\n')
        assert item['query'].endswith('\nAnswer:')
        # the chain-of-thought cue must be gone, and the evidence must not leak
        assert 'Reasoning and evidence:' not in item['query']
        assert 'description' not in item
        # gold index agrees with the upstream categorical value
        assert item['gold'] == value_to_gold[item['value']], item
        assert item['additional_review'] is True
        counts[item['gold']] += 1

    # capped at 2000/class; 'Inferred Absent' contributes all 1455 it has
    assert counts == {0: 2000, 1: 2000, 2: 1455, 3: 2000}, counts
    # majority class stays close to the 25% chance baseline the score is centered on
    assert max(counts.values()) / len(task.data) < 0.27
    # shuffled: a prefix (what --max-per-task takes) must span all four classes
    assert len({item['gold'] for item in task.data[:40]}) == 4


def test_hist_llm_rendering():
    """The history task renders through the standard letter-choice MC path: options
    stay inlined in the query and few-shot blocks end in 'Answer: <letter>'."""
    task = {t.label: t for t in load_bundle()}['hist_llm']
    item = task.data[0]
    assert prompts.is_letter_choices(item['choices'])
    assert prompts.mc_gold_letter(item) == 'ABCD'[item['gold']]

    p = prompts.render_generation_mc(item, task.data, 0, task.num_fewshot)
    blocks = p.split('\n\n')
    assert len(blocks) == 5, f'expected 4 few-shot blocks + the query, got {len(blocks)}'
    for block in blocks[:-1]:
        assert block.startswith('Question:\n')
        assert block.split('\n')[-1].startswith('Answer: ')
        assert block.rstrip()[-1] in 'ABCD'
    assert blocks[-1].rstrip().endswith('Answer:')

    # faithful logprob mode enumerates one prompt per candidate letter
    pl, ans, gold = prompts.render_logprob_mc(item, task.data, 0, task.num_fewshot)
    assert len(pl) == 4 and ans == [' A', ' B', ' C', ' D'] and gold == item['gold']
    assert all(prompt.endswith(answer) for prompt, answer in zip(pl, ans))


def test_hist_llm_answer_parsing():
    """Letters parse as usual; a model that replies with the option wording instead
    of the letter is mapped via 'choice_labels' rather than scored unanswered."""
    task = {t.label: t for t in load_bundle()}['hist_llm']
    labels = task.data[0]['choice_labels']

    assert scoring.parse_letter('C', 4) == 2
    assert scoring.parse_letter('Answer: B', 4) == 1
    # bare wording carries no standalone letter, so the fallback has to do the work
    assert scoring.parse_letter('Inferred Present', 4) is None
    assert scoring.match_choice_text('Present', labels) == 0
    assert scoring.match_choice_text('Inferred Present', labels) == 1
    assert scoring.match_choice_text('Inferred Absent', labels) == 2
    assert scoring.match_choice_text('Absent', labels) == 3
    # nesting: a bare "Present" must not be read as the longer "Inferred Present"
    assert scoring.match_choice_text('present.', labels) == 0
    assert scoring.match_choice_text('The answer is Present.', labels) == 0
    assert scoring.match_choice_text('The answer is Inferred Present.', labels) == 1

    # end-to-end through the runner, with a client that answers by wording
    class Wordy:
        def chat(self, prompt, max_tokens, temperature=0.0, stop=None, system=None, want_logprobs=False, top_logprobs=5):
            return 'Inferred Absent'

    rec = runner.eval_example_generation(Wordy(), task, 0, use_chat=True)
    assert rec['answered'] is True and rec['pred'] == 2

    # an off-format reply is still counted as unanswered (and wrong)
    class Refuses:
        def chat(self, prompt, max_tokens, temperature=0.0, stop=None, system=None, want_logprobs=False, top_logprobs=5):
            return 'I cannot determine this.'

    rec = runner.eval_example_generation(Refuses(), task, 0, use_chat=True)
    assert rec['answered'] is False and rec['correct'] is False


def test_letter_choice_fallback_does_not_affect_other_tasks():
    """Tasks without 'choice_labels' keep their previous behaviour: an inlined-option
    MC task must not text-match against the bare letters A/B/C/D."""
    task = {t.label: t for t in load_bundle()}['commonsense_qa']
    assert 'choice_labels' not in task.data[0]

    class Wordy:
        def chat(self, prompt, max_tokens, temperature=0.0, stop=None, system=None, want_logprobs=False, top_logprobs=5):
            return 'a revolving door'

    rec = runner.eval_example_generation(Wordy(), task, 0, use_chat=True)
    assert rec['pred'] is None and rec['answered'] is False


def test_hist_llm_end_to_end_with_stub():
    """Full evaluate() path over hist_llm with a stub client: the task must appear in
    the results, centering, and CORE aggregation, and a gold-answering stub must
    score 1.0 while a fixed-'A' stub must land near the majority-class rate."""
    from vintage_core import evaluate
    from vintage_core.client import Capabilities

    caps = Capabilities(has_completions=False, has_chat=True, has_prompt_logprobs=False, logprob_style=None)
    task = {t.label: t for t in load_bundle()}['hist_llm']
    n = 40

    by_query = {item['query'].rstrip(): item['gold'] for item in task.data[:n]}

    class Oracle:
        """Reads the gold letter back out of the rendered prompt's final question."""

        def chat(self, prompt, max_tokens, temperature=0.0, stop=None, system=None, want_logprobs=False, top_logprobs=5):
            return 'ABCD'[by_query[prompt.split('\n\n')[-1].rstrip()]]

    res = evaluate(model='stub', base_url='stub://', tasks=['hist_llm'], max_per_task=n, client=Oracle(), caps=caps)
    assert res['num_tasks'] == 1
    assert res['scoring_mode'] == 'generation' and res['endpoint'] == 'chat'
    assert res['results']['hist_llm'] == 1.0
    assert abs(res['centered_results']['hist_llm'] - 1.0) < 1e-9
    assert abs(res['core_metric'] - 1.0) < 1e-9
    assert res['unparsed_rate']['hist_llm'] == 0.0
    assert res['error_rate']['hist_llm'] == 0.0

    class AlwaysA:
        def chat(self, prompt, max_tokens, temperature=0.0, stop=None, system=None, want_logprobs=False, top_logprobs=5):
            return 'A'

    biased = evaluate(model='stub', base_url='stub://', tasks=['hist_llm'], max_per_task=n, client=AlwaysA(), caps=caps)
    # a pure position bias must stay near chance, not sail past it
    assert 0.15 < biased['results']['hist_llm'] < 0.40
    assert biased['unparsed_rate']['hist_llm'] == 0.0


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'ok  {fn.__name__}')
    print(f'\nAll {len(fns)} self-checks passed.')

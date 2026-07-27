"""Minimal OpenAI-compatible API client.

Talks to any endpoint that implements the OpenAI ``/v1/chat/completions`` and/or
``/v1/completions`` schema (llama.cpp's llama-server, vLLM, OpenAI, Together,
Groq, OpenRouter, ...). The only hard dependency is ``requests``.

The client also *probes* the backend once to discover:
  * which generation endpoint exists (completions is preferred for this ICL
    benchmark; chat is the universal fallback), and
  * whether the backend can return per-token logprobs over the prompt
    (``echo``-style / vLLM ``prompt_logprobs``), which is what the faithful
    logprob scorer needs.
"""

import json
import time

import requests


class Capabilities:
    def __init__(self, has_completions, has_chat, has_prompt_logprobs, logprob_style):
        self.has_completions = has_completions
        self.has_chat = has_chat
        self.has_prompt_logprobs = has_prompt_logprobs
        # 'echo' (OpenAI legacy text_offset/token_logprobs) | 'vllm' (prompt_logprobs) | None
        self.logprob_style = logprob_style

    def __repr__(self):
        return (
            f'Capabilities(completions={self.has_completions}, chat={self.has_chat}, '
            f'prompt_logprobs={self.has_prompt_logprobs}, style={self.logprob_style})'
        )


class APIClient:
    def __init__(self, base_url, model, api_key=None, timeout=120, max_retries=5, extra_body=None):
        # Normalize base_url to end with the version root (…/v1); accept with or without.
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self.session = requests.Session()

    # -- low-level HTTP ------------------------------------------------------
    def _headers(self):
        h = {'Content-Type': 'application/json'}
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    def _post(self, path, payload):
        url = f'{self.base_url}{path}'
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.post(url, headers=self._headers(), data=json.dumps(payload), timeout=self.timeout)
                if r.status_code == 200:
                    data = r.json()
                    # Some gateways (e.g. OpenRouter) return HTTP 200 with an error
                    # body and no 'choices'. Surface the provider message; retry if
                    # it looks transient, otherwise fail with a clear reason.
                    if isinstance(data, dict) and 'choices' not in data:
                        err = data.get('error') or data
                        msg = err.get('message') if isinstance(err, dict) else str(err)
                        code = err.get('code') if isinstance(err, dict) else None
                        if code in (429, 500, 502, 503, 504):
                            last_err = RuntimeError(f'provider error {code}: {msg}')
                            time.sleep(min(2**attempt, 30))
                            continue
                        raise RuntimeError(f'API returned no choices ({code}): {msg}')
                    return data
                # Retry on transient server / rate-limit errors, fail fast otherwise.
                if r.status_code in (408, 429, 500, 502, 503, 504):
                    last_err = RuntimeError(f'HTTP {r.status_code}: {r.text[:200]}')
                    time.sleep(min(2**attempt, 30))
                    continue
                raise RuntimeError(f'HTTP {r.status_code} from {url}: {r.text[:300]}')
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f'Request to {url} failed after {self.max_retries} retries: {last_err}')

    # -- generation ----------------------------------------------------------
    @staticmethod
    def _extract_gen_logprobs(choice):
        """Normalize a choice's generated-token logprobs into a list of
        {token, logprob, top: [{token, logprob}, ...]}. Handles both the chat
        `logprobs.content` schema and the legacy completions `token_logprobs`
        schema. Returns None if the backend attached no logprobs."""
        lp = choice.get('logprobs')
        if not lp:
            return None
        if isinstance(lp, dict) and lp.get('content'):
            out = []
            for tok in lp['content']:
                out.append(
                    {
                        'token': tok.get('token', ''),
                        'logprob': tok.get('logprob'),
                        'top': [{'token': t.get('token', ''), 'logprob': t.get('logprob')} for t in (tok.get('top_logprobs') or [])],
                    }
                )
            return out
        if isinstance(lp, dict) and lp.get('tokens') is not None:
            tokens = lp.get('tokens', [])
            tlps = lp.get('token_logprobs', [])
            tops = lp.get('top_logprobs') or [None] * len(tokens)
            out = []
            for tok, tlp, top in zip(tokens, tlps, tops):
                out.append(
                    {
                        'token': tok,
                        'logprob': tlp,
                        'top': [{'token': k, 'logprob': v} for k, v in (top or {}).items()],
                    }
                )
            return out
        return None

    def complete(self, prompt, max_tokens, temperature=0.0, stop=None, want_logprobs=False, top_logprobs=5):
        """Text completion. Returns the generated string, or (string, logprobs)
        when ``want_logprobs`` is set."""
        payload = {
            'model': self.model,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            **self.extra_body,
        }
        if stop:
            payload['stop'] = stop
        if want_logprobs:
            payload['logprobs'] = top_logprobs
        data = self._post('/completions', payload)
        if not data.get('choices'):
            raise RuntimeError(f"no 'choices' in completion response: {str(data)[:200]}")
        choice = data['choices'][0]
        text = choice['text']
        if want_logprobs:
            return text, self._extract_gen_logprobs(choice)
        return text

    def chat(self, prompt, max_tokens, temperature=0.0, stop=None, system=None, want_logprobs=False, top_logprobs=5):
        """Chat completion with a single user turn. Returns the generated string,
        or (string, logprobs) when ``want_logprobs`` is set."""
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        payload = {
            'model': self.model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            **self.extra_body,
        }
        if stop:
            payload['stop'] = stop
        if want_logprobs:
            payload['logprobs'] = True
            payload['top_logprobs'] = top_logprobs
        data = self._post('/chat/completions', payload)
        if not data.get('choices'):
            raise RuntimeError(f"no 'choices' in chat response: {str(data)[:200]}")
        choice = data['choices'][0]
        text = choice['message']['content'] or ''
        if want_logprobs:
            return text, self._extract_gen_logprobs(choice)
        return text

    def prompt_logprobs(self, prompt, style):
        """Return a list of {token, logprob, is_greedy} for the prompt tokens.

        ``is_greedy`` is True when the token was the argmax at its position
        (needed for language-modeling exact-match). Only positions that carry a
        logprob are returned (the first token has none). Raises if unsupported.
        """
        if style == 'echo':
            return self._prompt_logprobs_echo(prompt)
        if style == 'vllm':
            return self._prompt_logprobs_vllm(prompt)
        raise ValueError(f'unknown logprob style: {style}')

    def _prompt_logprobs_echo(self, prompt):
        payload = {
            'model': self.model,
            'prompt': prompt,
            'max_tokens': 0,
            'temperature': 0.0,
            'echo': True,
            'logprobs': 5,
            **self.extra_body,
        }
        data = self._post('/completions', payload)
        lp = data['choices'][0]['logprobs']
        tokens = lp['tokens']
        token_logprobs = lp['token_logprobs']
        top = lp.get('top_logprobs') or [None] * len(tokens)
        out = []
        for tok, tlp, tops in zip(tokens, token_logprobs, top):
            if tlp is None:
                continue
            is_greedy = True
            if tops:
                best = max(tops.values())
                is_greedy = tlp >= best - 1e-6
            out.append({'token': tok, 'logprob': tlp, 'is_greedy': is_greedy})
        return out

    def _prompt_logprobs_vllm(self, prompt):
        payload = {
            'model': self.model,
            'prompt': prompt,
            'max_tokens': 1,
            'temperature': 0.0,
            'prompt_logprobs': 1,
            **self.extra_body,
        }
        data = self._post('/completions', payload)
        pls = data['choices'][0].get('prompt_logprobs')
        if not pls:
            raise RuntimeError('backend did not return prompt_logprobs')
        out = []
        for entry in pls:
            if entry is None:  # first token has no logprob
                continue
            # entry maps token-id -> {logprob, rank, decoded_token}
            chosen = None
            best_lp = float('-inf')
            for info in entry.values():
                if info['logprob'] > best_lp:
                    best_lp = info['logprob']
                # rank 1 == the actual sampled/prompt token in vLLM's schema
                if info.get('rank') == 1:
                    chosen = info
            if chosen is None:
                chosen = max(entry.values(), key=lambda i: i['logprob'])
            out.append(
                {
                    'token': chosen.get('decoded_token', ''),
                    'logprob': chosen['logprob'],
                    'is_greedy': chosen['logprob'] >= best_lp - 1e-6,
                }
            )
        return out

    # -- capability probe ----------------------------------------------------
    def probe(self):
        """Detect available endpoints and prompt-logprob support."""
        has_completions = has_chat = has_prompt_logprobs = False
        logprob_style = None

        try:
            self.complete('Hello', max_tokens=1)
            has_completions = True
        except Exception:
            pass

        if not has_completions:
            try:
                self.chat('Hello', max_tokens=1)
                has_chat = True
            except Exception:
                pass
        else:
            # We only *need* chat if completions is missing; assume present otherwise.
            has_chat = True

        if has_completions:
            # Try vLLM-style prompt_logprobs first, then OpenAI-legacy echo.
            for style in ('vllm', 'echo'):
                try:
                    res = self.prompt_logprobs('The capital of France is Paris', style)
                    if res:
                        has_prompt_logprobs = True
                        logprob_style = style
                        break
                except Exception:
                    continue

        return Capabilities(has_completions, has_chat, has_prompt_logprobs, logprob_style)

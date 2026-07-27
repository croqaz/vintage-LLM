"""Local HuggingFace-checkpoint backend — no API server needed.

Implements the same interface as :class:`vintage_core.client.APIClient`
(``complete`` / ``chat`` / ``prompt_logprobs`` / ``probe``) but runs a local
checkpoint (e.g. a directory with ``config.json`` + ``model.safetensors``)
in-process via ``transformers``. This supports *both* scoring modes:

* generation (chat via the checkpoint's chat template, or raw completions), and
* faithful logprob scoring, with prompt log-probs read straight from the
  model's logits — the same numbers a vLLM ``prompt_logprobs`` endpoint gives.

Requires the optional dependencies: ``pip install torch transformers``
(or ``pip install .[local]`` from this repo).

Performance notes
-----------------
Model calls are serialized behind a lock (the runner's thread pool stays, but
compute is single-stream). ``prompt_logprobs`` keeps a one-entry KV prefix
cache: the faithful MC scorer issues one call per candidate continuation, and
those calls share the whole few-shot stem, so after the first candidate each
remaining one costs only a handful of tokens of compute.
"""

import os
import threading

from .client import Capabilities

# Chunk size for the log-softmax over (positions x vocab) so long prompts do
# not materialize a huge fp32 matrix at once.
_LSM_CHUNK = 512


class LocalClient:
    """Drop-in local replacement for APIClient, backed by transformers."""

    def __init__(self, model_path, device=None, dtype=None, max_context=4096):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                'Local checkpoint support needs torch and transformers: pip install torch transformers  (or `pip install .[local]`)'
            ) from e
        self._torch = torch
        self.model_path = str(model_path)
        self.name = os.path.basename(os.path.normpath(self.model_path)) or self.model_path
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if dtype is None:
            # Respect the checkpoint dtype on GPU; on CPU fp32 is usually
            # faster than emulated bf16.
            dtype = 'auto' if self.device.type == 'cuda' else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=dtype)
        self.model.to(self.device)
        # Training checkpoints often persist use_cache=False; generation
        # without a KV cache is needlessly slow, so force it on.
        self.model.config.use_cache = True
        if getattr(self.model, 'generation_config', None) is not None:
            self.model.generation_config.use_cache = True
        self.model.eval()

        self.max_context = max_context
        self._lock = threading.Lock()
        # One-entry prefix cache shared by consecutive prompt_logprobs calls:
        # {'ids': [...], 'entries': [...], 'past': DynamicCache}
        self._plp_cache = None

    # -- probing -------------------------------------------------------------
    def probe(self):
        """A local model can do everything: completions, chat, prompt logprobs."""
        return Capabilities(True, True, True, 'local')

    # -- tokenization helpers --------------------------------------------------
    def _encode(self, text, add_bos=True):
        ids = self.tokenizer(text, add_special_tokens=False)['input_ids']
        bos = self.tokenizer.bos_token_id
        if add_bos and bos is not None:
            ids = [bos] + ids
        return ids

    def _truncate_left(self, ids):
        if self.max_context and len(ids) > self.max_context:
            ids = ids[-self.max_context :]
        return ids

    @staticmethod
    def _apply_stop(text, stop):
        if stop:
            cut = min((text.find(s) for s in stop if text.find(s) >= 0), default=-1)
            if cut >= 0:
                text = text[:cut]
        return text

    # -- generation ------------------------------------------------------------
    def _generate(self, input_ids, max_tokens, temperature, want_logprobs, top_logprobs):
        torch = self._torch
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        kwargs = {
            'max_new_tokens': max(1, int(max_tokens)),
            'do_sample': bool(temperature and temperature > 0),
            'output_scores': want_logprobs,
            'return_dict_in_generate': want_logprobs,
        }
        if kwargs['do_sample']:
            kwargs['temperature'] = float(temperature)
        with self._lock, torch.no_grad():
            out = self.model.generate(ids, **kwargs)
        if want_logprobs:
            seq = out.sequences[0][len(input_ids) :].tolist()
            lps = []
            for tid, scores in zip(seq, out.scores):
                lsm = torch.log_softmax(scores[0].float(), dim=-1)
                topv, topi = torch.topk(lsm, min(top_logprobs, lsm.numel()))
                lps.append(
                    {
                        'token': self.tokenizer.decode([tid]),
                        'logprob': float(lsm[tid]),
                        'top': [
                            {'token': self.tokenizer.decode([int(i)]), 'logprob': float(v)} for v, i in zip(topv.tolist(), topi.tolist())
                        ],
                    }
                )
            return seq, lps
        return out[0][len(input_ids) :].tolist(), None

    def complete(self, prompt, max_tokens, temperature=0.0, stop=None, want_logprobs=False, top_logprobs=5):
        """Raw-text completion (OpenAI /v1/completions semantics)."""
        ids = self._truncate_left(self._encode(prompt))
        new_ids, lps = self._generate(ids, max_tokens, temperature, want_logprobs, top_logprobs)
        text = self._apply_stop(self.tokenizer.decode(new_ids, skip_special_tokens=True), stop)
        if want_logprobs:
            return text, lps
        return text

    def chat(self, prompt, max_tokens, temperature=0.0, stop=None, system=None, want_logprobs=False, top_logprobs=5):
        """Single-user-turn chat via the checkpoint's own chat template."""
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        # Render to text and encode ourselves: the template already emits the
        # BOS token, and some tokenizer backends ignore tokenize=True here.
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = self._truncate_left(self._encode(text, add_bos=False))
        new_ids, lps = self._generate(ids, max_tokens, temperature, want_logprobs, top_logprobs)
        text = self._apply_stop(self.tokenizer.decode(new_ids, skip_special_tokens=True), stop)
        if want_logprobs:
            return text, lps
        return text

    # -- prompt log-probs --------------------------------------------------------
    def _entries_from_rows(self, rows, token_ids):
        """Turn per-position logits into {token, logprob, is_greedy} entries.
        ``rows[i]`` is the logit vector predicting ``token_ids[i]``."""
        torch = self._torch
        entries = []
        for i in range(0, len(token_ids), _LSM_CHUNK):
            r = rows[i : i + _LSM_CHUNK].float()
            lsm = torch.log_softmax(r, dim=-1)
            toks = token_ids[i : i + _LSM_CHUNK]
            idx = torch.tensor(toks, dtype=torch.long, device=lsm.device)
            lp = lsm.gather(1, idx.unsqueeze(1)).squeeze(1)
            am = lsm.argmax(dim=-1)
            for j, tid in enumerate(toks):
                entries.append(
                    {
                        'token': self.tokenizer.decode([tid]),
                        'logprob': float(lp[j]),
                        'is_greedy': bool(int(am[j]) == tid),
                    }
                )
        return entries

    def prompt_logprobs(self, prompt, style=None):
        """Per-token logprobs for the supplied prompt (first token excluded),
        matching the normalized shape APIClient returns. ``style`` is ignored;
        logits come straight from the model."""
        torch = self._torch
        ids = self._truncate_left(self._encode(prompt))
        L = len(ids)
        if L < 2:
            return []
        with self._lock:
            # Longest common prefix with the previous call (MC candidates share
            # the entire few-shot stem, so this turns N full forward passes
            # into 1 full + N-1 tiny ones).
            n = 0
            past = None
            head = []
            cache = self._plp_cache
            if cache is not None:
                prev = cache['ids']
                lim = min(len(prev), L)
                while n < lim and prev[n] == ids[n]:
                    n += 1
                if n == L:
                    # New prompt is a prefix of (or equal to) the cached one.
                    return list(cache['entries'][: L - 1])
                if n >= 16:
                    head = cache['entries'][: n - 1]
                    past = cache['past']
                    if n - 1 > 0:
                        past.crop(n - 1)
                    else:
                        past = None
                else:
                    n = 0
            start = n - 1 if n > 0 else 0
            inp = torch.tensor([ids[start:]], dtype=torch.long, device=self.device)
            try:
                with torch.no_grad():
                    out = self.model(input_ids=inp, past_key_values=past, use_cache=True)
            except Exception:
                self._plp_cache = None
                raise
            logits = out.logits[0]  # (L - start, V); row r predicts token start+r+1
            first_p = max(1, n)  # first token position we still need
            rows = logits[first_p - 1 - start : L - 1 - start]
            tail = self._entries_from_rows(rows, ids[first_p:])
            entries = head + tail
            self._plp_cache = {'ids': ids, 'entries': entries, 'past': out.past_key_values}
            return entries

    # -- convenience -----------------------------------------------------------
    def __repr__(self):
        return f'LocalClient({self.model_path!r}, device={self.device})'

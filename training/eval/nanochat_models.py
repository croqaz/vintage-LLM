"""Load Karpathy-style nanochat checkpoints behind the interface this evaluator speaks.

nanochat models are not transformers models. They ship as a bare
`model_<step>.pt` state dict plus a `meta_<step>.json` describing the
architecture, and their tokenizer is a pickled tiktoken `Encoding` rather than
a `tokenizer.json`. Nothing in `AutoModelForCausalLM` can open that.

Rather than convert the weights, this module runs the real nanochat `GPT`
module and wraps it. Conversion is not an option here: the architecture the
vintage community trains on has diverged from Llama in ways that have no HF
equivalent — value embeddings on alternating layers (ResFormer), a learned
embedding "smear" from the previous token, a mid-depth "backout" subtraction,
per-layer residual and x0 scalars, parameter-free QK norm, relu-squared MLPs
and tanh logit softcapping. Re-expressing those as a Llama config would change
the numbers we are trying to measure.

What the wrapper provides, and only this:

  * `NanochatForCausalLM` — a `PreTrainedModel` whose `forward` returns
    `CausalLMOutputWithPast`, so every scoring function in metrics.py works
    unchanged, and whose `generate` comes from `GenerationMixin`.
  * `NanochatTokenizer` — the subset of the HF tokenizer surface the evaluator
    actually calls, backed by tiktoken.

Two limits are real and are asserted rather than hidden:

  * **No attention mask.** nanochat attends causally over the whole row; it has
    no padding-mask path. Scoring is one document at a time, so that is fine,
    but batched generation left-pads. Callers must use a generation batch size
    of 1; a padded mask raises instead of quietly scoring pad tokens.
  * **No KV cache.** nanochat's cache is tied to its own Engine and to
    FlashAttention-3. The wrapper recomputes the full prefix at every decode
    step. That is slower, and it is the reason the batch-size-1 rule does not
    cost much on top.

Set `NANOCHAT_REPO` to point at a nanochat source tree if it does not live in
`MODELS/nanochat` next to the checkpoints.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
from pathlib import Path

import torch

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent

# Scalars and gates stay in fp32. Casting a weight that a matmul would cast
# anyway is exact, but these are multiplied into activations directly and are
# far too small to be worth rounding.
_KEEP_FP32 = ('resid_lambdas', 'x0_lambdas', 'smear_lambda', 'backout_lambda', 'smear_gate.weight', 've_gate.weight')


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_nanochat_repo() -> Path | None:
    """The nanochat source tree, or None. Order: env var, then the usual spots."""
    env = os.environ.get('NANOCHAT_REPO')
    candidates = [Path(env)] if env else []
    candidates += [
        PROJECT_DIR / 'MODELS' / 'nanochat',
        PROJECT_DIR.parent / 'MODELS' / 'nanochat',
        PACKAGE_DIR / 'nanochat',
    ]
    for candidate in candidates:
        if (candidate / 'nanochat' / 'gpt.py').is_file():
            return candidate.resolve()
    return None


def nanochat_parts(path: Path) -> tuple[Path, Path] | None:
    """(weights .pt, meta .json) for a nanochat checkpoint dir, or None.

    A nanochat checkpoint is `model_<step>.pt` beside `meta_<step>.json`. The
    step is zero-padded to six digits upstream but published repos rename them
    freely, so pair by suffix and fall back to the highest step.
    """
    if not path.is_dir():
        return None
    metas = {}
    for meta in path.glob('meta_*.json'):
        weights = path / f'model_{meta.stem.removeprefix("meta_")}.pt'
        if weights.is_file():
            metas[meta] = weights
    if not metas:
        return None

    def step(meta: Path) -> int:
        m = re.search(r'(\d+)', meta.stem)
        return int(m.group(1)) if m else -1

    meta = max(metas, key=step)
    return metas[meta], meta


def is_nanochat_checkpoint(path: Path) -> bool:
    return nanochat_parts(path) is not None


def resolve_nanochat_tokenizer(ckpt: Path, explicit: Path | None = None) -> Path:
    """Directory holding `tokenizer.pkl`.

    Never falls back to a sibling model's tokenizer: nanochat trains one BPE
    per run, and two 32,768-entry vocabularies built from different corpora
    disagree on almost every id.
    """

    def holds_a_vocabulary(directory: Path) -> bool:
        return (directory / 'tokenizer.pkl').is_file() or (directory / 'tokenizer.json').is_file()

    if explicit is not None:
        if not holds_a_vocabulary(explicit):
            raise FileNotFoundError(f'{explicit} has neither tokenizer.pkl nor tokenizer.json')
        return explicit.resolve()
    for candidate in (ckpt / 'tokenizer', ckpt, ckpt.parent / 'tokenizer'):
        if holds_a_vocabulary(candidate):
            return candidate.resolve()
    raise FileNotFoundError(
        f'no tokenizer.pkl or tokenizer.json found for {ckpt}. nanochat publishes the tokenizer '
        f'separately from the weights and some repos omit it entirely; pass --tokenizer DIR with '
        f'the matching one.'
    )


def read_meta(checkpoint: Path) -> dict:
    parts = nanochat_parts(checkpoint)
    if parts is None:
        raise FileNotFoundError(f'{checkpoint} is not a nanochat checkpoint')
    with open(parts[1], 'rb') as fh:
        return json.load(fh)


def nanochat_weight_bytes(checkpoint: Path) -> int:
    parts = nanochat_parts(checkpoint)
    return parts[0].stat().st_size if parts else 0


def nanochat_stage(meta: dict) -> str:
    """'base', 'sft' or 'unknown', from whatever the run happened to record.

    nanochat has no single field for this. `user_config.stage` exists on newer
    base runs; the chat SFT script is identifiable by the knobs only it writes.
    """
    user = meta.get('user_config', {})
    stage = user.get('stage') or user.get('experiment', {}).get('stage')
    if stage in ('base', 'sft', 'mid'):
        return stage
    if any(k in user for k in ('mmlu_epochs', 'gsm8k_epochs', 'chatcore_every')):
        return 'sft'
    if 'train_data' in user and str(user.get('train_data', '')).endswith('.jsonl'):
        return 'sft'
    return 'unknown'


def nanochat_lineage(checkpoint: Path) -> dict:
    """The fields `checkpoint_lineage` reports, read from nanochat's meta json.

    nanochat records a different set of things than a transformers Trainer:
    there is no trainer_state.json, no total_flos and no epoch. What it does
    have, and what is worth carrying into the report, is the step, the run
    name, the context length, the Muon matrix LR and its own validation bpb
    (which is NOT comparable to ours: different corpus, different tokenizer).
    """
    meta = read_meta(checkpoint)
    user = meta.get('user_config', {})
    model_config = meta.get('model_config', {})
    stage = nanochat_stage(meta)
    return {
        'kind': stage if stage in ('base', 'sft') else 'unknown',
        'base_model': user.get('parent_experiment_id') or user.get('init_from_checkpoint_dir'),
        'learning_rate': user.get('matrix_lr'),
        'context': model_config.get('sequence_len'),
        'global_step': meta.get('step'),
        'epoch': None,
        'tokens_seen': (user.get('experiment', {}).get('mixture_schedule', {}) or {}).get('total_tokens'),
        'tokens_per_param': None,
        'tokens_lower_bound': False,
        'runtime_minutes': (meta.get('loop_state', {}).get('total_training_time') or 0) / 60 or None,
        'runtime_segments_minutes': [],
        'budget': 'unknown',
        'note': f'nanochat run {user.get("run") or user.get("experiment_id") or "?"}',
        'optim': 'MuonAdamW',
        'lr_scheduler': user.get('branch_lr_schedule'),
        'warmup_steps': user.get('warmup_steps'),
        'stable_steps': None,
        'decay_steps': None,
        'max_steps': user.get('num_iterations'),
        'max_train_minutes': None,
        'train_seq_length': user.get('max_seq_len'),
        'tokens_per_step': user.get('total_batch_size'),
        'tokens_seen_from_steps': (user.get('total_batch_size') or 0) * (meta.get('step') or 0) or None,
        'seed': user.get('seed'),
        'config_file': None,
        'warnings': [],
        # nanochat-only, kept namespaced so nothing downstream mistakes it for ours.
        'nanochat_val_bpb': meta.get('val_bpb'),
        'nanochat_window_pattern': model_config.get('window_pattern', 'L'),
        'nanochat_stage': stage,
    }


def nanochat_identity(checkpoint: Path) -> tuple[str, int]:
    """('nanochat', parameter count) without loading weights.

    Counted from shapes in the state dict rather than derived from the config,
    because the value embeddings make the two disagree by billions.
    """
    parts = nanochat_parts(checkpoint)
    if parts is None:
        return 'nanochat', 0
    state = torch.load(parts[0], map_location='meta', mmap=True, weights_only=True)
    return 'nanochat', sum(v.numel() for v in state.values())


# ---------------------------------------------------------------------------
# The HF-shaped wrapper
# ---------------------------------------------------------------------------


def _import_nanochat(dtype: torch.dtype):
    """Import nanochat with COMPUTE_DTYPE pinned to ours.

    nanochat resolves its compute dtype once, at import, from the GPU it can
    see. It then asserts that the rotary cache matches. Setting the env var
    first is the only way to make the two agree on a card its detector has not
    heard of.
    """
    repo = find_nanochat_repo()
    if repo is None:
        raise RuntimeError(
            'nanochat source tree not found. Clone https://github.com/karpathy/nanochat into '
            'MODELS/nanochat, or set NANOCHAT_REPO=/path/to/nanochat.'
        )
    name = {torch.bfloat16: 'bfloat16', torch.float16: 'float16', torch.float32: 'float32'}.get(dtype)
    if name is None:
        raise ValueError(f'nanochat has no compute dtype for {dtype}')
    previous = os.environ.get('NANOCHAT_DTYPE')
    if 'nanochat.common' in sys.modules and previous != name:
        raise RuntimeError(
            f'nanochat was already imported with NANOCHAT_DTYPE={previous!r}; it cannot be changed to '
            f'{name!r} in the same process. Evaluate nanochat checkpoints of one dtype per run.'
        )
    os.environ['NANOCHAT_DTYPE'] = name
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from nanochat.gpt import GPT, GPTConfig

    return GPT, GPTConfig


def _match_gate_widths(model, state: dict) -> None:
    """Resize the value-embedding gates to whatever the checkpoint was trained with.

    `ve_gate_channels` is a constant in nanochat's source, and forks have moved
    it (GPT-1900 was trained at 32, the current tree hard-codes 12). It reads
    the first N channels of the residual stream, so the width is a genuine
    architectural choice rather than a saved hyperparameter, and the only
    record of it is the shape of the weight itself.
    """
    for name, module in model.named_modules():
        gate = getattr(module, 've_gate', None)
        if gate is None:
            continue
        saved = state.get(f'{name}.ve_gate.weight')
        if saved is None or saved.shape[1] == gate.in_features:
            continue
        module.ve_gate = type(gate)(saved.shape[1], saved.shape[0], bias=False, device='meta')
        module.ve_gate_channels = saved.shape[1]


def _load_gpt(checkpoint: Path, device: torch.device, dtype: torch.dtype):
    """Build the nanochat GPT and put the checkpoint's tensors into it.

    Deliberately avoids upstream's `to_empty(device) + init_weights()` dance,
    which materialises the whole model in fp32 before overwriting it: 13 GB for
    a 3.3B model, which does not fit the card we run on. `assign=True` against
    a meta-device module hands ownership straight to the loaded tensors, so the
    only copy that ever exists is the one we want.
    """
    GPT, GPTConfig = _import_nanochat(dtype)
    parts = nanochat_parts(checkpoint)
    if parts is None:
        raise FileNotFoundError(f'{checkpoint} is not a nanochat checkpoint')
    weights_path, meta_path = parts
    with open(meta_path, 'rb') as fh:
        meta = json.load(fh)

    cfg_kwargs = dict(meta['model_config'])
    cfg_kwargs.setdefault('window_pattern', 'L')  # pre-sliding-window checkpoints
    config = GPTConfig(**cfg_kwargs)
    with torch.device('meta'):
        model = GPT(config)

    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
    for key, tensor in state.items():
        if tensor.is_floating_point() and not key.endswith(_KEEP_FP32):
            state[key] = tensor.to(dtype)
        elif tensor.is_floating_point():
            state[key] = tensor.float()

    _match_gate_widths(model, state)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f'{checkpoint.name}: checkpoint has keys this nanochat build does not: {sorted(unexpected)}')

    # Older checkpoints predate parameters the current nanochat always builds.
    # Each default below is the identity, so an old model behaves exactly as it
    # did when it was trained; anything we cannot neutralise is an error.
    neutral = {
        'resid_lambdas': lambda p: torch.ones(config.n_layer),
        'x0_lambdas': lambda p: torch.zeros(config.n_layer),
        'smear_lambda': lambda p: torch.zeros(1),
        'backout_lambda': lambda p: torch.zeros(1),
        'smear_gate.weight': lambda p: torch.zeros(p.shape),
    }
    patched = []
    for key in missing:
        if key not in neutral:
            raise RuntimeError(f'{checkpoint.name}: missing weight {key!r} with no safe default')
        param = model.get_parameter(key)
        replacement = torch.nn.Parameter(neutral[key](param).float(), requires_grad=False)
        parent = model.get_submodule(key.rsplit('.', 1)[0]) if '.' in key else model
        setattr(parent, key.rsplit('.', 1)[-1], replacement)
        patched.append(key)
    if patched:
        print(f'  nanochat: pre-{"/".join(sorted(patched))} checkpoint; those terms set to identity')

    # Rotary tables are non-persistent buffers, so nothing in the checkpoint
    # filled them and they are still on meta -- which would make .to(device)
    # raise. Build them first, directly on the target device and in nanochat's
    # COMPUTE_DTYPE, which forward() asserts.
    cos, sin = model._precompute_rotary_embeddings(model.rotary_seq_len, config.n_embd // config.n_head, device=device)
    model.cos, model.sin = cos, sin
    model.to(device)
    model.eval()
    return model, config, meta


def build_nanochat_model(checkpoint: Path, device: torch.device, dtype: torch.dtype):
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.generation import GenerationMixin
    from transformers.modeling_outputs import CausalLMOutputWithPast

    class NanochatConfig(PretrainedConfig):
        model_type = 'nanochat'

    class NanochatForCausalLM(PreTrainedModel, GenerationMixin):
        """HF calling convention over a real nanochat GPT.

        The weights are never copied or reshaped; `self.gpt` IS the upstream
        module. Only the call signature is translated.
        """

        config_class = NanochatConfig
        base_model_prefix = 'gpt'
        main_input_name = 'input_ids'
        supports_gradient_checkpointing = False
        _supports_cache_class = False

        def __init__(self, config, gpt=None):
            super().__init__(config)
            self.gpt = gpt

        @classmethod
        def can_generate(cls) -> bool:
            return True

        def _init_weights(self, module):  # weights always arrive from a checkpoint
            pass

        def get_input_embeddings(self):
            return self.gpt.transformer.wte

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            # No cache: every step re-reads the whole prefix, so hand back all
            # of it and drop whatever cache object generate() offered.
            return {'input_ids': input_ids, 'attention_mask': kwargs.get('attention_mask'), 'use_cache': False}

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            output_hidden_states=False,
            past_key_values=None,
            use_cache=None,
            **kwargs,
        ):
            if attention_mask is not None and not bool(attention_mask.all()):
                raise ValueError(
                    'nanochat attention has no padding-mask path, so a padded batch would score pad '
                    'tokens as content. Use a batch size of 1 (--generation-batch-size 1).'
                )
            hidden = {}
            handle = None
            if output_hidden_states:
                # The last block's output is the deepest hidden state this
                # architecture exposes without reimplementing forward(). It is
                # taken BEFORE the backout subtraction and the final RMS norm,
                # which is stated wherever these numbers are reported.
                handle = self.gpt.transformer.h[-1].register_forward_hook(
                    lambda _m, _a, out: hidden.__setitem__('h', out[0] if isinstance(out, tuple) else out)
                )
            try:
                logits = self.gpt(input_ids)
            finally:
                if handle is not None:
                    handle.remove()

            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels[:, 1:].reshape(-1), ignore_index=-100
                )
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=None,
                hidden_states=(hidden['h'],) if 'h' in hidden else None,
            )

    gpt, gpt_config, meta = _load_gpt(checkpoint, device, dtype)
    config = NanochatConfig(
        vocab_size=gpt_config.vocab_size,
        max_position_embeddings=gpt_config.sequence_len,
        num_hidden_layers=gpt_config.n_layer,
        num_attention_heads=gpt_config.n_head,
        num_key_value_heads=gpt_config.n_kv_head,
        hidden_size=gpt_config.n_embd,
        window_pattern=gpt_config.window_pattern,
        use_cache=False,
    )
    model = NanochatForCausalLM(config, gpt)
    model.nanochat_meta = meta
    return model


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class ChatMarkers:
    """The special-token names one nanochat fork uses for conversation turns.

    nanochat's own names are not universal. Mr. Chatterbox was trained with a
    wrapper that renamed every role token and dropped the user-end marker
    entirely, so the rendering has to be read off the vocabulary rather than
    assumed. Getting this wrong does not raise: it produces a prompt the model
    was never trained on, and a quietly bad score.
    """

    def __init__(self, bos, user_start, user_end, assistant_start, assistant_end):
        self.bos = bos
        self.user_start = user_start
        self.user_end = user_end
        self.assistant_start = assistant_start
        self.assistant_end = assistant_end


# Upstream nanochat.
NANOCHAT_MARKERS = ChatMarkers('<|bos|>', '<|user_start|>', '<|user_end|>', '<|assistant_start|>', '<|assistant_end|>')
# Mr. Chatterbox's tokenizer_wrapper.py: <|endoftext|> is the document boundary
# AND the assistant stop token, and a user turn has no closing marker.
VICTORIAN_MARKERS = ChatMarkers('<|endoftext|>', '<human>', None, '<victorian>', '<|endoftext|>')


def _byte_decoder() -> dict:
    """Inverse of GPT-2's bytes_to_unicode, for reading byte-level BPE pieces."""
    printable = list(range(ord('!'), ord('~') + 1)) + list(range(ord('\xa1'), ord('\xac') + 1)) + list(range(ord('\xae'), ord('\xff') + 1))
    mapped = list(printable)
    spare = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + spare)
            spare += 1
    return {chr(code): byte for byte, code in zip(printable, mapped, strict=True)}


class HFVocabulary:
    """A `tokenizers` BPE presented with tiktoken's `Encoding` method names.

    Some nanochat forks swap the tokenizer wholesale. Mr. Chatterbox ships a
    HuggingFace `tokenizer.json` and a wrapper class that adapts it to
    nanochat's interface. Rather than teach NanochatTokenizer about two
    backends, the backend is adapted to the five methods it already calls.
    """

    def __init__(self, tokenizer, special_names):
        self._tok = tokenizer
        self.special_tokens_set = frozenset(special_names)
        self._bytes = _byte_decoder()

    @property
    def n_vocab(self):
        return self._tok.get_vocab_size(True)

    def encode_single_token(self, text):
        token_id = self._tok.token_to_id(text)
        if token_id is None:
            raise KeyError(text)
        return token_id

    def decode_single_token_bytes(self, token_id):
        piece = self._tok.id_to_token(token_id)
        if piece is None or piece in self.special_tokens_set:
            return b''
        return bytes(self._bytes[c] for c in piece)

    def encode(self, text, allowed_special=None):
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=False)


class NanochatTokenizer:
    """The slice of the HF tokenizer API this evaluator uses.

    Not a `PreTrainedTokenizer` subclass on purpose. A nanochat vocabulary is
    either a pickled tiktoken `Encoding` or, in forks, a raw `tokenizers` BPE.
    Neither carries the `tokenizer_config.json` and `special_tokens_map.json`
    that every route into the HF class hierarchy wants, so those files would
    have to be invented. Implementing the methods that are actually called is
    smaller and honest about what is supported.

    The backend is whatever object provides tiktoken's five method names; see
    HFVocabulary for the adapter.
    """

    # tiktoken's encoder is the same Rust BPE a "fast" HF tokenizer wraps, and
    # offset mapping is implemented below, so the evaluator's is_fast gate
    # (which really asks "can you give me offsets?") is satisfied.
    is_fast = True

    def __init__(self, enc, tokenizer_dir: Path, markers: 'ChatMarkers | None' = None):
        self.enc = enc
        self.tokenizer_dir = tokenizer_dir
        self.padding_side = 'left'
        specials = enc.special_tokens_set
        self.markers = markers or pick_markers(specials)
        # tiktoken happily returns b'<|bos|>' for the BOS id, so a special
        # token cannot be recognised by its bytes; keep the id set instead.
        self.special_ids = frozenset(enc.encode_single_token(s) for s in specials)
        self.bos_token = self.markers.bos if self.markers.bos in specials else None
        # nanochat has no end-of-text token in the usual sense. An assistant
        # turn ends with the fork's assistant-end marker; a base document ends
        # where the next BOS begins. Both are the right stop signal for their
        # own model, and both are what the evaluator means by "EOS".
        self.eos_token = self.markers.assistant_end if self.markers.assistant_end in specials else self.bos_token
        self.pad_token = '<|pad|>' if '<|pad|>' in specials else self.eos_token
        self.chat_template = 'nanochat' if self.markers.assistant_start in specials else None
        self.model_max_length = int(1e9)

    # -- identity -----------------------------------------------------------

    @property
    def bos_token_id(self):
        return self.enc.encode_single_token(self.bos_token) if self.bos_token else None

    @property
    def eos_token_id(self):
        return self.enc.encode_single_token(self.eos_token) if self.eos_token else None

    @property
    def pad_token_id(self):
        return self.enc.encode_single_token(self.pad_token) if self.pad_token else None

    @property
    def vocab_size(self):
        return self.enc.n_vocab

    def __len__(self):
        return self.enc.n_vocab

    # -- encode / decode ----------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = True, **_kwargs) -> list[int]:
        ids = self.enc.encode(text, allowed_special='all')
        if add_special_tokens and self.bos_token_id is not None and (not ids or ids[0] != self.bos_token_id):
            ids = [self.bos_token_id, *ids]
        return ids

    def decode(self, ids, skip_special_tokens: bool = False, **_kwargs) -> str:
        if torch.is_tensor(ids):
            ids = ids.tolist()
        ids = [int(i) for i in ids]
        if skip_special_tokens:
            ids = [i for i in ids if i not in self.special_ids]
        # Ids beyond the vocabulary cannot happen from our own sampling, but a
        # caller slicing a padded batch can hand us the pad filler.
        return self.enc.decode([i for i in ids if 0 <= i < self.enc.n_vocab])

    def batch_decode(self, sequences, **kwargs) -> list[str]:
        return [self.decode(s, **kwargs) for s in sequences]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.enc.decode([ids])
        return [self.enc.decode([int(i)]) for i in ids]

    def _char_offsets(self, text: str, ids: list[int]) -> list[tuple[int, int]]:
        """Character spans per token, the way a fast HF tokenizer reports them.

        tiktoken works in bytes, so the token spans are byte spans; the
        evaluator compares them against a character index (the length of the
        chat context). Walk the string once to build the byte->character map
        rather than re-decoding a growing prefix per token.
        """
        # Byte-level BPE can split one character across two tokens, so the map
        # must answer for EVERY byte, not just character boundaries. A token
        # that covers part of a character is reported as covering the whole
        # character, which is what a fast HF tokenizer does as well.
        char_of_byte = []
        for index, char in enumerate(text):
            char_of_byte.extend([index] * len(char.encode('utf-8')))
        n_bytes = len(char_of_byte)

        offsets = []
        position = 0
        for token in ids:
            start = position
            if token not in self.special_ids:
                position += len(self.enc.decode_single_token_bytes(token))
            # A special token contributes no source text, so it keeps an empty
            # span and is never counted as target content.
            if position == start:
                span_start = char_of_byte[start] if start < n_bytes else len(text)
                offsets.append((span_start, span_start))
            else:
                offsets.append((char_of_byte[start], char_of_byte[min(position, n_bytes) - 1] + 1))
        return offsets

    def __call__(
        self,
        text,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
        return_offsets_mapping=False,
        **_kwargs,
    ):
        from transformers import BatchEncoding

        batched = not isinstance(text, str)
        texts = [text] if not batched else list(text)
        rows = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        if truncation and max_length:
            rows = [r[:max_length] for r in rows]
        offsets = [self._char_offsets(t, r) for t, r in zip(texts, rows, strict=True)] if return_offsets_mapping else None

        masks = [[1] * len(r) for r in rows]
        if padding and len({len(r) for r in rows}) > 1:
            width = max(len(r) for r in rows)
            pad = self.pad_token_id if self.pad_token_id is not None else 0
            if self.padding_side == 'left':
                masks = [[0] * (width - len(r)) + m for r, m in zip(rows, masks, strict=True)]
                rows = [[pad] * (width - len(r)) + r for r in rows]
            else:
                masks = [m + [0] * (width - len(r)) for r, m in zip(rows, masks, strict=True)]
                rows = [r + [pad] * (width - len(r)) for r in rows]

        if offsets is not None and padding:
            raise ValueError('NanochatTokenizer will not pad and return offsets at once; the spans would be wrong')
        data = {'input_ids': rows, 'attention_mask': masks}
        if offsets is not None:
            data['offset_mapping'] = offsets
        if return_tensors == 'pt':
            data = {k: torch.tensor(v, dtype=torch.long) for k, v in data.items()}
        elif return_tensors is not None:
            raise ValueError(f'NanochatTokenizer only returns lists or "pt", not {return_tensors!r}')
        elif not batched:
            data = {k: v[0] for k, v in data.items()}
        return BatchEncoding(data)

    # -- chat ---------------------------------------------------------------

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = False, **_kwargs):
        """nanochat's own conversation rendering, in text form.

        Mirrors `RustBPETokenizer.render_conversation`: a leading BOS, a system
        message folded into the following user turn (nanochat has no system
        role), then paired start/end markers. Tool-call parts are not
        supported here; nothing in this evaluator emits them.
        """
        if self.chat_template is None:
            raise ValueError('this nanochat tokenizer has no assistant markers; it is a base-model vocabulary')
        markers = self.markers
        messages = list(messages)
        if messages and messages[0]['role'] == 'system':
            if len(messages) < 2 or messages[1]['role'] != 'user':
                raise ValueError('a nanochat system message must be followed by a user message')
            messages = [{'role': 'user', 'content': messages[0]['content'] + '\n\n' + messages[1]['content']}, *messages[2:]]

        out = [markers.bos]
        for message in messages:
            if message['role'] == 'user':
                out += [markers.user_start, message['content']]
                if markers.user_end:  # some forks close the turn, some do not
                    out.append(markers.user_end)
            elif message['role'] == 'assistant':
                out += [markers.assistant_start, message['content'], markers.assistant_end]
            else:
                raise ValueError(f'nanochat has no {message["role"]!r} role')
        if add_generation_prompt:
            out.append(markers.assistant_start)
        text = ''.join(out)
        return self.encode(text, add_special_tokens=False) if tokenize else text


def pick_markers(specials) -> ChatMarkers:
    """Choose the conversation dialect from the special tokens that exist."""
    for markers in (NANOCHAT_MARKERS, VICTORIAN_MARKERS):
        if markers.assistant_start in specials:
            return markers
    return NANOCHAT_MARKERS  # base-model vocabulary; chat_template stays None


def load_nanochat_tokenizer(tokenizer_dir: Path) -> NanochatTokenizer:
    """Load whichever vocabulary format the fork shipped.

    tiktoken pickle is upstream nanochat. A `tokenizer.json` means the run
    swapped in a HuggingFace BPE, which Mr. Chatterbox does; its own
    tokenizer_wrapper.py is the record of which special tokens map to which
    role, and ChatMarkers carries that mapping.
    """
    directory = Path(tokenizer_dir)
    pickle_path = directory / 'tokenizer.pkl'
    if pickle_path.is_file():
        with open(pickle_path, 'rb') as fh:
            return NanochatTokenizer(pickle.load(fh), directory)

    json_path = directory / 'tokenizer.json'
    if json_path.is_file():
        from tokenizers import Tokenizer

        with open(json_path, 'rb') as fh:
            specials = [t['content'] for t in json.load(fh).get('added_tokens', []) if t.get('special')]
        return NanochatTokenizer(HFVocabulary(Tokenizer.from_file(str(json_path)), specials), directory)

    raise FileNotFoundError(f'{directory} has neither tokenizer.pkl nor tokenizer.json')


def load_nanochat(checkpoint: Path, tokenizer_dir: Path, device: torch.device, dtype: torch.dtype):
    """(model, tokenizer) ready for the same code paths an HF checkpoint takes."""
    tokenizer = load_nanochat_tokenizer(tokenizer_dir)
    model = build_nanochat_model(Path(checkpoint), device, dtype)
    if tokenizer.vocab_size != model.config.vocab_size:
        raise RuntimeError(
            f'tokenizer vocab {tokenizer.vocab_size} != model vocab {model.config.vocab_size}. '
            f'nanochat trains one BPE per run; this pairing is wrong and every score would be meaningless.'
        )
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

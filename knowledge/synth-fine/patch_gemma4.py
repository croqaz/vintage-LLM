#!/usr/bin/env python3
"""Build-time patch for vLLM's native Gemma-4 attention (KV-sharing fix).

vLLM 0.25.1's ``gemma4.py`` creates ``self.k_norm`` for *every* decoder layer,
but its forward pass only applies ``k_norm`` on layers that are NOT KV-shared
(the last ``num_kv_shared_layers`` reuse an earlier layer's already-normed K).
The checkpoint therefore ships no ``k_norm.weight`` for those shared layers, and
vLLM's strict weight-loading aborts with:

    ValueError: Following weights were not initialized from checkpoint:
    {'...layers.24.self_attn.k_norm.weight', ...}

This makes the ``k_norm`` module creation conditional on the layer actually
using it — a behaviour-preserving change (the forward never touches ``k_norm``
on shared layers). Non-KV-sharing models are unaffected (the flag is False, so
``k_norm`` is created exactly as before).

Idempotent-ish: if the anchor line is absent (newer vLLM that already fixed it,
or a refactor), the script is a no-op and exits 0 so the build still succeeds.
"""

import sys

TARGET = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py'

OLD = '        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)\n'
NEW = (
    '        # [synth-fine patch] KV-shared layers (the last num_kv_shared_layers)\n'
    "        # reuse an earlier layer's already-normed K, so they neither store nor\n"
    '        # apply their own k_norm. Creating it unconditionally made vLLM expect a\n'
    "        # k_norm.weight the checkpoint doesn't have -> weight-loading failure.\n"
    '        _lidx = extract_layer_index(prefix)\n'
    "        _nkvs = getattr(config, 'num_kv_shared_layers', 0)\n"
    '        _kv_shared = _nkvs > 0 and _lidx >= config.num_hidden_layers - _nkvs\n'
    '        self.k_norm = None if _kv_shared else RMSNorm(self.head_dim, eps=config.rms_norm_eps)\n'
)


def main() -> None:
    try:
        src = open(TARGET, encoding='utf-8').read()
    except FileNotFoundError:
        print(f'[patch_gemma4] {TARGET} not found; skipping (no-op)')
        return

    if '[synth-fine patch]' in src:
        print('[patch_gemma4] already applied; skipping')
        return

    n = src.count(OLD)
    if n == 0:
        print('[patch_gemma4] anchor not found (vLLM likely already fixed this); skipping')
        return
    if n > 1:
        print(f'[patch_gemma4] WARNING: {n} matches; patching all')

    src = src.replace(OLD, NEW)
    open(TARGET, 'w', encoding='utf-8').write(src)
    print(f'[patch_gemma4] applied ({n} site) to {TARGET}')


if __name__ == '__main__':
    sys.exit(main())

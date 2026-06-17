# Embarrassingly Simple Self-Distillation Improves Code Generation

**Authors:** Ruixiang Zhang*, Richard He Bai*, Huangjie Zheng*, Navdeep Jaitly, Ronan Collobert, Yizhe Zhang* (*equal contribution)
**Affiliation:** Apple
**Date:** April 2, 2026 (arXiv:2604.01193v1, cs.CL)
**Code:** https://github.com/apple/ml-ssd

---

## TL;DR

The paper asks: *Can an LLM get better at code generation using only its own raw outputs — no verifier, no teacher model, no reinforcement learning?* The answer is **yes**. The method, **Simple Self-Distillation (SSD)**, samples solutions from the base model at a chosen decoding temperature/truncation, then fine-tunes the model on those raw, unverified samples with ordinary supervised fine-tuning (cross-entropy). SSD lifts **Qwen3-30B-Instruct from 42.4% → 55.3% pass@1 on LiveCodeBench v6** (+30% relative), with the largest gains on hard problems, and it generalizes across five models (two families, three scales, instruct and thinking variants).

The paper's deeper contribution is the *explanation*: it identifies a **precision–exploration conflict** in LLM decoding and shows SSD reshapes token distributions in a context-dependent way — suppressing distractor tails where precision matters ("locks") while preserving useful diversity where exploration matters ("forks").

---

## The Problem

As LLMs tackle harder coding tasks, high-quality supervised signal is the binding constraint. Existing approaches all carry a cost:
- **Human-written solutions** are expensive.
- **Teacher distillation** needs a stronger model and inherits the teacher's ceiling.
- **Execution-based verification / RLVR** requires test cases per problem and is operationally complex and unstable.
- **Intrinsic-reward methods** (majority voting, entropy minimization) face reward hacking and collapse under extended training.

This motivates the central question: can a model improve *without any external labeled data or verification at all*?

| Method | Dense signal | No teacher | No verifier | No privileged info |
|---|---|---|---|---|
| SFT on external data | | ✓ | ✓ | |
| GRPO | | ✓ | | ✓ |
| On-policy distillation | ✓ | | ✓ | |
| On-policy self-distillation | ✓ | ✓ | | |
| **Simple Self-Distillation (SSD, ours)** | ✓ | ✓ | ✓ | ✓ |

---

## The Method (SSD)

Three steps, requiring only a set of problem prompts and the model itself:

1. **Data synthesis.** From a *frozen* base model `p_θ`, sample N candidate solutions per prompt using training-time decoding settings (temperature `T_train`, truncation `ρ_train` = top-k/top-p). **No verification of any kind** — no execution, no test cases, no correctness filtering. In practice **N = 1 sample per prompt already suffices**. Only trivial syntactic filtering removes empty/single-line stubs.
2. **Training.** Fine-tune on these raw samples with standard supervised cross-entropy loss.
3. **Inference.** Deploy the fine-tuned model with evaluation-time decoding settings (`T_eval`, `ρ_eval`).

Note the key design point: `T_train ≠ 1`. Sampling at a *shifted* temperature and truncation is what reshapes the model — not the correctness of the data.

**Experimental setup.** Data = ~10K de-duplicated competitive-programming problems (seed subset of rSTARcoder). Five base models: Llama-3.1-8B-Instruct; Qwen3-4B-Instruct & -Thinking; Qwen3-30B-A3B-Instruct & -Thinking (MoE, 30B total / 3B active). Training with Megatron-LM on 8×B200 GPUs. Primary benchmark **LiveCodeBench v6** (Feb–May 2025), with LCB v5 as secondary; metrics pass@1, pass@5, and per-difficulty (easy/medium/hard).

---

## Key Results

### 1. SSD improves every model, most on hard problems

LiveCodeBench v6 pass@1 (base → +SSD):

| Model | Base | +SSD | Δ pass@1 |
|---|---|---|---|
| **Qwen3-30B-Instruct** | 42.4 | **55.3** | **+12.9** (+30% rel.) |
| Qwen3-4B-Instruct | 34.0 | 41.5 | +7.5 |
| Qwen3-4B-Thinking | 54.5 | 57.8 | +3.3 |
| Qwen3-30B-Thinking | 66.1 | 68.2 | +2.1 |
| Llama-3.1-8B-Instruct | 12.7 | 16.2 | +3.5 |

For Qwen3-30B-Instruct the gains concentrate on difficulty: **easy +6.5pp, medium +14.2pp, hard +15.3pp** (pass@1). Gains hold on the larger LCB v5 set too (45.8 → 54.3, +8.5pp).

### 2. SSD does not collapse diversity

Gains are often **larger at pass@5 than pass@1** — evidence that SSD preserves/improves generation diversity rather than just sharpening one mode. For Qwen3-30B-Instruct, hard-problem **pass@5 rises +23.0pp** (31.1% → 54.1%) vs +15.3pp at pass@1.

### 3. Decoding tweaks alone cannot reproduce the gains

Sweeping the base model's evaluation temperature yields only modest, flat changes (e.g., Qwen3-30B-Instruct pass@1 spans just 2.2pp across temperatures). SSD beats the *best-tuned* base model by **+11.8pp** (all problems) and **+13.3pp** on hard pass@1. This persistent gap proves SSD changes the model itself in ways no decoding configuration can replicate.

### 4. Hyperparameters: effective temperature + truncation

- Without truncation, training and evaluation temperatures **compose** through an *effective temperature* `T_eff = T_train · T_eval` (R²=0.75, quadratic peak near T_eff ≈ 1.2). Higher `T_train` makes the model more responsive to `T_eval`.
- Adding training-time truncation **raises the performance ceiling** by suppressing low-probability tails during data synthesis (best no-truncation: 49.7% pass@1).

### 5. Out-of-domain transfer

Trained only on competitive programming, the **30B models stay broadly stable** (within ~±2pp) on AIME (math), HumanEval (general code), CruxEval (code understanding), and MMLU (general knowledge; within 0.3pp). Smaller (4B) models show more uneven tradeoffs.

---

## Why SSD Works: The Precision–Exploration Conflict

The paper's mechanistic core. Code generation interleaves two kinds of decoding positions:

- **Locks** — sharply peaked distributions (syntax/semantics nearly determine the next token) with a long, low-probability **distractor tail**. They *demand precision*: commit to the dominant token, suppress the tail.
- **Forks** — distributions spread across several genuinely plausible continuations (e.g., choosing quicksort vs. insertion sort vs. a built-in). They *demand exploration*: spread mass over viable alternatives.

A single global decoding temperature `T_eval` cannot serve both: lowering it secures locks but starves forks of diversity; raising it enables fork exploration but lets distractor tails flood back at locks. The best global setting is necessarily a compromise — the **precision–exploration conflict**.

**SSD's resolution:** training on temperature-shifted, truncated samples reshapes distributions *asymmetrically* — it suppresses distractor tails most aggressively at locks (turning them into sharper "spikes") while preserving and evening out diversity at forks (turning them into broad "plateaus"). This widens the viable decoding regime, so higher-temperature decoding becomes newly effective *after* training.

The mechanism is validated three ways:
1. **Controlled toy simulation** (one fork + three locks, success computable in closed form): SSD shifts the optimal decoding temperature higher and raises success. It also shows **training and decoding are complementary** — training makes locks safe; decoding then exploits the extra room to explore forks.
2. **Real-model analysis** (Qwen3-30B-Instruct): after SSD, cumulative probability mass rises faster through top ranks (cleaner head, weaker tail), and entropy after truncation rises more strongly with temperature — more usable alternatives for exploration.
3. **Theoretical decomposition** (Eq. 4): the SSD objective splits into **support compression** (via `ρ_train` — removes diffuse tail mass), **within-support reshaping** (via `T_train` — reshapes the head, expressed through Rényi entropy of order 1/T), and **alignment to the base model**. This shows SSD is *not mere imitation* and explains why the model can become globally lower-entropy while staying more explorable where it matters. It also formally explains why decode-only tuning — constrained by the base model's fixed ranking — cannot match SSD.

### A surprising stress test: "Bad data, good results"

Pushing `T_train = 2.0` with truncation disabled produces nearly-gibberish training data (~62% of outputs contain no extractable code, often devolving into multilingual nonsense). Yet SSD **still improves the model** to 48.1% pass@1 / 64.0% pass@5 (+5.7 / +10.5pp), again concentrated on hard problems. This demonstrates the benefit comes from *how high-temperature sampling reshapes token probabilities*, not from training on correct code — though evaluation-time truncation is needed to recover precision and make the reshaping useful.

---

## Relation to Prior Work

- **Self-training / distillation:** Unlike on-policy distillation variants, SSD uses *only* temperature-shifted self-samples and plain cross-entropy — no privileged context, feedback-conditioned teachers, or auxiliary supervision.
- **Code synthetic data (STaR, ReSTEM, etc.):** Those convert self-generated outputs into supervision via correctness filtering/execution feedback; SSD trains directly on **raw, unverified** outputs.
- **Reasoning/RL & critical-token work:** Rather than asking which tokens RL should emphasize, SSD asks how far plain SFT on a model's own outputs can go without rewards/verifiers.
- **Decoding/truncation:** SSD is *not* a new decoding rule; it shows training on shifted-decoding samples alters the model so a simple fixed decoding policy becomes more effective.
- **Self-improvement without reward / unsupervised RLVR:** SSD is *not* Shannon-entropy minimization or RL on an intrinsic reward; it is better understood as **support compression + within-support reshaping**, which can lower total entropy while *increasing* useful entropy at forks.

---

## Contributions & Conclusion

1. **SSD substantially improves code-generation models using only their own unverified outputs** — no teacher, verifier, reward model, or labeled solutions.
2. It identifies the **precision–exploration conflict** as the key mechanism.
3. It supports that mechanism with aligned evidence from controlled simulation, real-model analysis, and theory.

**Takeaway:** Strong code models contain **latent capability** that can be unlocked without a verifier, teacher, or RL. SSD offers a simple, complementary post-training direction — train a model on temperature-shifted samples of itself, and ordinary fine-tuning reshapes its distributions so that fixed-policy decoding can explore good solution branches without reopening distractor tails.

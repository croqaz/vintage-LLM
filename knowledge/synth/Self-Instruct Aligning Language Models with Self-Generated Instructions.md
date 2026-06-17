# SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions

**Authors:** Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi
**Affiliations:** University of Washington, Tehran Polytechnic, Arizona State University, Johns Hopkins University, Allen Institute for AI
**Venue:** ACL 2023 (arXiv:2212.10560v2, May 2023)
**Code & data:** https://github.com/yizhongw/self-instruct

---

## TL;DR

Self-Instruct is a semi-automated framework that improves the instruction-following ability of a pretrained language model by **bootstrapping off the model's own generations** — with almost no human-labeled data. Applied to vanilla GPT-3, it produces a ~52K-instruction synthetic dataset, and finetuning GPT-3 on this data yields a **33% absolute improvement** on the SUPER-NATURALINSTRUCTIONS benchmark, putting it on par with InstructGPT₀₀₁ (which used private user data and human annotations).

---

## Problem & Motivation

Instruction-tuned LMs generalize impressively to new tasks, but they depend heavily on **human-written instruction data**, which is:
- **Costly** to collect (requires creativity to invent tasks and expertise to solve them).
- **Limited in quantity, diversity, and creativity** — human annotators tend to gravitate toward popular, classification-heavy NLP tasks, failing to cover the true variety of tasks and phrasings real users care about.

This bottleneck limits how generalizable instruction-tuned models can become. The authors ask: can a model generate its own instruction data to teach itself to follow instructions better?

---

## Method

Self-Instruct is an **iterative bootstrapping algorithm** seeded with a small pool of human-written tasks. Each generated task consists of an instruction, an optional input, and an output.

The pipeline has four steps:

1. **Instruction Generation.** Start from a pool seeded with **175 manually written tasks** (1 instruction + 1 instance each). Sample 8 instructions as in-context examples (6 human-written + 2 model-generated, to promote diversity) and prompt the LM to generate new task instructions.

2. **Classification Task Identification.** Few-shot prompt the LM to decide whether each new instruction is a classification task (small, limited output label space) or not, using 12 classification + 19 non-classification seed examples.

3. **Instance Generation.** Generate input-output instances for each instruction:
   - **Input-first approach** (non-classification tasks): generate the input fields, then the output.
   - **Output-first approach** (classification tasks): first generate the possible class labels, then condition input generation on each label. This counters the bias of input-first generation toward a single dominant label (e.g., always generating grammatical text for grammar-error detection).

4. **Filtering & Postprocessing.** Add a new instruction to the pool only if its **ROUGE-L similarity** to any existing instruction is **< 0.7**. Drop instructions with keywords LMs can't handle (image, picture, graph), duplicate instances, instances with the same input but different outputs, and instances failing heuristics (too long/short, output repeats input).

Valid tasks are added back to the pool, and the process repeats. Finally, the original LM is **finetuned** on the generated data in a standard supervised fashion, using multiple prompt templates (varying prefixes like "Task:"/"Input:"/"Output:" and line breaks) to make the model robust to format variation.

---

## The Generated Dataset (applied to GPT-3 "davinci")

| Statistic | Value |
|---|---|
| Instructions | 52,445 |
| — classification | 11,584 |
| — non-classification | 40,861 |
| Instances | 82,439 |
| — with empty input | 35,878 |
| Avg. instruction length (words) | 15.9 |
| Avg. non-empty input length (words) | 12.7 |
| Avg. output length (words) | 18.9 |

**Diversity:** Verb-noun analysis (Berkeley Neural Parser) shows a broad range of intents and formats; generated instructions have low ROUGE-L overlap with the seed tasks, indicating genuinely new tasks were created.

**Quality** (expert review of 200 sampled instructions, 1 instance each):
- 92% describe a valid task
- 79% have an appropriate input
- 58% have a correct/acceptable output
- 54% are valid across all fields

Outputs are noisier than instructions, but even imperfect generations are usually in the right format or partially correct — useful training signal.

---

## Experiments & Results

The finetuned model is called **GPT3_SELF-INST**. Baselines include vanilla LMs (T5-LM, GPT-3), publicly instruction-tuned models (T0, Tₖ-INSTRUCT, both 11B), GPT-3 finetuned on T0 / SUPERNI data, and InstructGPT (001/002/003).

### Experiment 1 — Zero-shot generalization on SUPERNI (119 unseen tasks)

| Model | Params | ROUGE-L |
|---|---|---|
| T5-LM | 11B | 25.7 |
| GPT-3 (vanilla) | 175B | 6.8 |
| T0 | 11B | 33.1 |
| GPT-3 + T0 Training | 175B | 37.9 |
| **GPT3_SELF-INST (ours)** | 175B | **39.9** |
| InstructGPT₀₀₁ | 175B | 40.8 |
| Tₖ-INSTRUCT (w/ SUPERNI) | 11B | 46.0 |
| GPT-3 + SUPERNI Training | 175B | 49.5 |
| **GPT3_SELF-INST + SUPERNI (ours)** | 175B | **51.6** |

Key findings: ① Self-Instruct boosts GPT-3 by **+33.1%**; ② it **nearly matches InstructGPT₀₀₁** without any private/human data; ③ it provides **complementary gains** even on top of the labeled SUPERNI training set.

### Experiment 2 — User-oriented instructions (252 novel, expert-written tasks)

The authors curate 252 instructions across practical domains (email, social media, productivity, entertainment, programming) with diverse formats (bullet points, tables, code, equations). Human experts rate outputs on a 4-level scale (A: valid/satisfying → D: irrelevant/invalid; inter-rater κ = 0.57).

GPT3_SELF-INST **outperforms all GPT-3 variants trained on public instruction datasets (T0, SUPERNI) by a large margin**, and comes within **~5%** of InstructGPT₀₀₁ when counting acceptable-with-minor-imperfections responses as valid. InstructGPT 002/003 remain clearly stronger.

### Ablations on data size and quality
- **Size:** Performance improves consistently with more generated data, but **plateaus after ~16K** instructions (and even earlier on SUPERNI, since the generated data is distinct from typical NLP tasks).
- **Quality:** Regenerating output fields with InstructGPT₀₀₃ (a distillation step) improves the finetuned model by **~10%**, suggesting large headroom from better supervision via human experts or stronger teacher models.

---

## Relation to Prior Work

Self-Instruct connects to but differs from several lines of research:
- **Instruction tuning** (T0, Tₖ-INSTRUCT, FLAN, InstructGPT): prior work depends on human-annotated instructions and existing NLP tasks; Self-Instruct removes the human-annotation bottleneck and goes beyond classical tasks.
- **LM-based data generation/augmentation:** unlike task-specific generation (QA, NLI), Self-Instruct is **task-agnostic** and invents new task definitions from scratch.
- **Concurrent work — Unnatural Instructions** (Honovich et al.): uses SUPERNI seeds and InstructGPT₀₀₂ (distilling an already-tuned model), whereas Self-Instruct relies only on a **vanilla** LM.
- **Self-training / knowledge distillation:** Self-Instruct is a form of distillation where source and target are the **same model**, and the distilled content is instruction-defined tasks rather than labels for a fixed target task.

---

## Contributions

1. **Self-Instruct**, a method to induce instruction-following with minimal human-labeled data.
2. Extensive instruction-tuning experiments demonstrating its effectiveness.
3. A released **52K-instruction synthetic dataset** plus 252 expert-written novel-task instructions for building and evaluating future models.

---

## Limitations & Broader Impact

- **Tail phenomena:** Gains likely skew toward frequent language/tasks seen in pretraining; the method may be brittle on uncommon, creative instructions.
- **Dependence on large models:** Relies on inductive biases of large LMs, working best at scale — a potential access barrier for those without large compute.
- **Reinforcing LM biases:** The iterative loop risks amplifying social biases; the algorithm also struggled to produce balanced labels, reflecting model priors.
- **Broader impact:** Self-Instruct brings transparency to the otherwise opaque, API-walled construction of models like InstructGPT/ChatGPT, and its core idea was quickly adopted by follow-up work (e.g., Alpaca/Taori et al., WizardLM/Xu et al., Sun et al.).

---

## Conclusion

By having a pretrained LM generate, filter, and learn from its own instruction data, Self-Instruct turns a vanilla GPT-3 into a competent instruction follower — closing most of the gap to a human-annotated commercial system at a fraction of the labeling cost. It stands as an early, influential step toward open, scalable alignment of LMs to human instructions.

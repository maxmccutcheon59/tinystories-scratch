# WRITEUP — tinystories-scratch

## Citation

Eldan, R., & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv:2305.07759.

Official paper: https://arxiv.org/abs/2305.07759  
Dataset card: https://huggingface.co/datasets/roneneldan/TinyStories

This repository is an independent educational / portfolio reimplementation by Max McCutcheon. It is **not** affiliated with Microsoft Research or the original authors.

## Method (what we implemented)

1. **Architecture:** a small decoder-only Transformer (GPT-style): token + learned position embeddings, `n_layer` blocks of causal multi-head self-attention + MLP (GELU, 4× expansion), final LayerNorm, and a weight-tied linear LM head.
2. **Objective:** next-token cross-entropy on character sequences (sliding windows of length `block_size`).
3. **Tokenizer:** character-level vocabulary built from the training texts (pad/unk specials). This differs from the paper’s GPT-Neo tokenizer with a ~10K vocab subset.
4. **Sampling:** autoregressive multinomial sampling with optional temperature and top-k.
5. **Data paths:**
   - `synthetic`: offline hand-written micro-stories (for smoke / no-network demos).
   - `tinystories`: stream a **subset** (`max_samples`) of HuggingFace TinyStories at train time.

Code lives in `tinystories_scratch/model.py`, `tokenizer.py`, and `data.py`.

## Experiment scale vs paper

| Setting | Eldan & Li 2023 (typical) | This repo (default CPU) |
|--------|---------------------------|-------------------------|
| Data | Full TinyStories train corpus | Synthetic ≤200 stories, or HF subset (e.g. 2k samples) |
| Tokenizer | GPT-Neo, ~10K tokens | Character-level |
| Architecture | GPT-Neo variants (1M–33M+, context 512) | 2–4 layers, `n_embd=64–128`, `block_size=64–128` |
| Compute | Substantial training runs | CPU-only, hundreds–low thousands of steps |
| Evaluation | Human / GPT-4 story grading, fluency | Train/val CE loss + qualitative samples only |

**We do not claim paper-matching fluency, reasoning emergence, or parameter-efficiency results.** No GPT-4 grader scores or paper tables are reproduced or invented here.

## Honest results stance

- After `configs/smoke.yaml` (≈30 CPU steps), loss should move downward; generated text is not meaningful English narrative.
- After `configs/cpu_synthetic.yaml`, expect slightly more coherent character n-grams overlapping the tiny synthetic corpus — not open-ended story quality.
- After `configs/cpu_tinystories_subset.yaml` on a few thousand stories and short CPU training, expect weak, repetitive, or broken English at best. This is expected at this scale.
- **No invented metrics:** we do not report fabricated perplexity, GPT-4 grades, or comparisons claiming parity with TinyStories-1M/3M/etc.

## Limitations

- Character LM + tiny context ≠ paper setup.
- No GPT-Neo, FlashAttention, mixed precision, multi-GPU, or instruct-tuning in v0.1.0.
- HuggingFace path needs network and the optional `datasets` extra; prefer synthetic for offline CI.
- Research / educational code — not production LLM infrastructure.

## Reproducibility

- Pin deps via `pip install -e ".[dev]"` (add `.[data]` for HF).
- Configs under `configs/`; `seed: 0` sets torch CPU RNG (full cross-platform bitwise reproducibility not guaranteed).

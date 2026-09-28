# tinystories-scratch

[![CI](https://github.com/maxmccutcheon59/tinystories-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/maxmccutcheon59/tinystories-scratch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**From-scratch PyTorch reimplementation** related to *TinyStories* (Eldan & Li, 2023) — a minimal GPT-style decoder-only language model trained and demoed on TinyStories-scale data or a tiny offline subset.

Author: **Max McCutcheon** \<MaxMcCutcheon1@outlook.com\>

This is **not** a copy of the authors’ training stack or GPT-Neo checkpoints. The Transformer (causal self-attention, MLP blocks, weight-tied LM head) and training loop are original educational code aimed at **CPU-friendly** demos ($0 GPU).

> **Honesty note:** Results here are **demo-scale** (tiny model, character tokenizer, few CPU iterations, optional small HF subset). They are **not** comparable to the paper’s 1M–33M GPT-Neo TinyStories models or reported story quality. See [WRITEUP.md](WRITEUP.md).

## Features

- Minimal GPT-style causal Transformer (original code)
- Character-level tokenizer (no external vocab files)
- Training loop with train/val loss logging + checkpointing
- Sampling script (temperature / top-k)
- Offline **synthetic** micro-stories for smoke/CPU demos (no download)
- Optional HuggingFace [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) subset download in the train script (not committed)
- Pytest coverage of tokenizer, model shapes, and data windows
- CI template: pytest + pip-audit + gitleaks

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Unit tests (no dataset download)
pytest -q

# Short smoke train (synthetic stories only)
python scripts/train.py --config configs/smoke.yaml

# Longer CPU demo (still synthetic / offline)
python scripts/train.py --config configs/cpu_synthetic.yaml

# Optional: TinyStories HF subset (needs network + datasets extra)
pip install -e ".[data]"
python scripts/train.py --config configs/cpu_tinystories_subset.yaml

# Sample from a checkpoint
python scripts/sample.py --checkpoint checkpoints/smoke.pt --prompt "Once upon a time"
```

Default device in configs is **`cpu`**. Override only if you choose to: `--device cuda`.

## Layout

```
tinystories_scratch/   # library: model, tokenizer, data, config
configs/               # smoke, cpu_synthetic, cpu_tinystories_subset
scripts/               # train.py, sample.py
tests/                 # tokenizer / model / data / config tests
WRITEUP.md             # method, citation, honest limitations
SECURITY.md
COMPLIANCE_NOTES.md
```

## Dataset attribution

When using `dataset: tinystories`, stories are streamed from Hugging Face:

- **Dataset:** [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- **Paper:** Eldan & Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*, 2023 ([arXiv:2305.07759](https://arxiv.org/abs/2305.07759))
- **License:** CDLA-Sharing-1.0 (see [COMPLIANCE_NOTES.md](COMPLIANCE_NOTES.md))

Data is downloaded at train time into `./data/` (gitignored) and is **not** committed to this repository.

## Citation (paper)

```bibtex
@article{eldan2023tinystories,
  title={TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  author={Eldan, Ronen and Li, Yuanzhi},
  journal={arXiv preprint arXiv:2305.07759},
  year={2023}
}
```

## CI

Every push runs pytest on Python 3.10 and 3.12 (CPU PyTorch) and a gitleaks secret scan. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## License

MIT — see [LICENSE](LICENSE).

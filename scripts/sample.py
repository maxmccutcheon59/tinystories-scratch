#!/usr/bin/env python3
"""Generate text from a trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinystories_scratch.model import GPT
from tinystories_scratch.tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a tiny GPT checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default=None, help="Optional output .txt path")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"])
    model = GPT(
        vocab_size=cfg["vocab_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        dropout=0.0,
        bias=cfg.get("bias", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ids = tokenizer.encode(args.prompt)
    if not ids:
        ids = tokenizer.encode("Once upon a time")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
    )
    text = tokenizer.decode(out_ids[0].tolist())
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

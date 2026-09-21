#!/usr/bin/env python3
"""Train a tiny GPT on synthetic stories or a TinyStories subset (CPU-friendly)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinystories_scratch.config import load_config
from tinystories_scratch.data import build_tokenizer_and_datasets, load_texts
from tinystories_scratch.model import GPT


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    loaders: dict[str, DataLoader],
    device: torch.device,
    eval_iters: int,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split, loader in loaders.items():
        losses = []
        it = iter(loader)
        for _ in range(eval_iters):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / max(len(losses), 1)
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny GPT (TinyStories-scale demo)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    parser.add_argument("--max-iters", type=int, default=None, help="Override max_iters")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg.device = args.device
    if args.max_iters is not None:
        cfg.max_iters = args.max_iters

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")

    print(f"Loading dataset={cfg.dataset!r} max_samples={cfg.max_samples} …")
    texts = load_texts(cfg.dataset, cfg.max_samples, cfg.data_dir, seed=cfg.seed)
    tokenizer, train_ds, val_ds = build_tokenizer_and_datasets(
        texts, cfg.block_size, val_ratio=cfg.val_ratio, seed=cfg.seed
    )
    print(f"vocab_size={tokenizer.vocab_size} train_windows={len(train_ds)} val_windows={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    loaders = {"train": train_loader, "val": val_loader}

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
        bias=cfg.bias,
    ).to(device)
    print(f"parameters={model.count_parameters():,} device={device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    data_iter = iter(train_loader)
    for step in range(1, cfg.max_iters + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.eval_interval == 0 or step == 1 or step == cfg.max_iters:
            losses = estimate_loss(model, loaders, device, cfg.eval_iters)
            print(
                f"step {step:5d}/{cfg.max_iters} "
                f"train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} "
                f"batch_loss={loss.item():.4f}"
            )

    out_path = ckpt_dir / f"{cfg.run_name}.pt"
    payload = {
        "model_state": model.state_dict(),
        "config": {
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "block_size": cfg.block_size,
            "dropout": cfg.dropout,
            "bias": cfg.bias,
            "vocab_size": tokenizer.vocab_size,
        },
        "tokenizer": tokenizer.to_dict(),
        "dataset": cfg.dataset,
        "max_samples": cfg.max_samples,
        "max_iters": cfg.max_iters,
    }
    torch.save(payload, out_path)
    meta_path = ckpt_dir / f"{cfg.run_name}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "checkpoint": str(out_path),
                "parameters": model.count_parameters(),
                "vocab_size": tokenizer.vocab_size,
                "dataset": cfg.dataset,
                "max_iters": cfg.max_iters,
                "note": "Demo-scale; not comparable to Eldan & Li 2023 reported models.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved checkpoint → {out_path}")


if __name__ == "__main__":
    main()

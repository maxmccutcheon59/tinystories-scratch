"""YAML config loading for training / sampling."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cpu"

    # Model
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    dropout: float = 0.1
    bias: bool = False

    # Data
    dataset: str = "synthetic"  # "synthetic" | "tinystories"
    data_dir: str = "data"
    max_samples: int = 2000
    val_ratio: float = 0.05

    # Optim
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_iters: int = 500
    eval_interval: int = 100
    eval_iters: int = 20
    grad_clip: float = 1.0

    # IO
    checkpoint_dir: str = "checkpoints"
    run_name: str = "cpu_demo"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config must be a mapping: {path}")
        return cls.from_dict(raw)


def load_config(path: str | Path) -> TrainConfig:
    return TrainConfig.from_yaml(path)

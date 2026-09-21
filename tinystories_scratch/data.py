"""Dataset helpers: tiny synthetic corpus + optional HuggingFace TinyStories."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer

# Hand-written child-friendly micro-stories for offline/smoke training.
# Not from the TinyStories dataset; used so tests and CPU demos need no download.
SYNTHETIC_STORIES = [
    "Once upon a time, a little cat sat on a soft mat. The cat saw a red ball and played happily.",
    "A small dog ran in the park. He found a stick and brought it to his friend. They laughed together.",
    "Lily liked to draw. She drew a big sun and a blue sky. Her mom said the picture was lovely.",
    "Tom had a tiny boat. He put it in a puddle after the rain. The boat floated and Tom smiled.",
    "Anna planted a seed. Every day she gave it water. Soon a green sprout peeked out of the soil.",
    "The moon was bright. Two owls sat on a branch and told quiet stories about the stars.",
    "Ben packed a lunch with an apple and cheese. At school he shared the apple with a new friend.",
    "A rabbit hopped across the garden. It sniffed the carrots but decided to nap under a leaf.",
    "Mia wore red boots. She jumped in every puddle on the way home and sang a silly song.",
    "The old clock ticked softly. Grandma told a story about a brave mouse who helped a lion.",
    "Sam built a tower of blocks. It fell down, so he built it again, taller than before.",
    "A bird sang outside the window. The baby clapped and tried to sing along with happy sounds.",
    "Emma found a shiny pebble by the river. She kept it in her pocket for good luck.",
    "Two friends made a fort from blankets. Inside they read books with a flashlight.",
    "The baker made warm bread. The smell filled the street and made everyone feel cozy.",
    "A kite flew high above the hill. The wind tugged the string and the child held on tight.",
    "Ned watered the flowers. Bees buzzed around the yellow ones and danced from bloom to bloom.",
    "On a rainy day, Fran painted clouds. She mixed blue and white until the paper glowed.",
    "A fox walked carefully through the snow. Soft prints followed him back to a warm den.",
    "The library was quiet. Children whispered as they chose books about dragons and dinosaurs.",
]


def expand_synthetic(n: int, seed: int = 0) -> list[str]:
    """Repeat / lightly permute synthetic stories to reach ~n samples."""
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        s = rng.choice(SYNTHETIC_STORIES)
        # Light paraphrase via optional second sentence join
        if rng.random() < 0.3:
            s = s + " " + rng.choice(SYNTHETIC_STORIES).split(".")[0] + "."
        out.append(s)
    return out[:n]


def load_tinystories_texts(
    max_samples: int,
    data_dir: str | Path,
    split: str = "train",
) -> list[str]:
    """Download/load TinyStories via HuggingFace `datasets` (not committed).

    Dataset: https://huggingface.co/datasets/roneneldan/TinyStories
    License: CDLA-Sharing-1.0 — see COMPLIANCE_NOTES.md.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Install the optional 'data' extra to use TinyStories: "
            "pip install -e \".[data]\"  (needs the `datasets` package)."
        ) from e

    cache_dir = Path(data_dir) / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Streaming avoids pulling the full multi-GB corpus when we only need a subset.
    ds = load_dataset(
        "roneneldan/TinyStories",
        split=split,
        streaming=True,
        cache_dir=str(cache_dir),
    )
    texts: list[str] = []
    for row in ds:
        text = row.get("text") or row.get("story") or ""
        text = str(text).strip()
        if text:
            texts.append(text)
        if len(texts) >= max_samples:
            break
    if not texts:
        raise RuntimeError("No TinyStories texts loaded; check network / dataset name.")
    return texts


def load_texts(
    dataset: str,
    max_samples: int,
    data_dir: str | Path,
    seed: int = 0,
) -> list[str]:
    if dataset == "synthetic":
        return expand_synthetic(max_samples, seed=seed)
    if dataset == "tinystories":
        return load_tinystories_texts(max_samples, data_dir=data_dir)
    raise ValueError(f"Unknown dataset={dataset!r}; use 'synthetic' or 'tinystories'.")


class CharLMDataset(Dataset):
    """Sliding windows over concatenated token ids for next-token prediction."""

    def __init__(
        self,
        token_ids: list[int],
        block_size: int,
    ) -> None:
        if len(token_ids) < block_size + 1:
            raise ValueError(
                f"Need at least block_size+1 tokens; got {len(token_ids)} "
                f"with block_size={block_size}"
            )
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def build_tokenizer_and_datasets(
    texts: list[str],
    block_size: int,
    val_ratio: float = 0.05,
    seed: int = 0,
) -> tuple[CharTokenizer, CharLMDataset, CharLMDataset]:
    tokenizer = CharTokenizer.from_texts(texts)
    rng = random.Random(seed)
    shuffled = list(texts)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_texts = shuffled[:n_val] if n_val else shuffled[:1]
    train_texts = shuffled[n_val:] if n_val < len(shuffled) else shuffled

    def encode_join(ts: list[str]) -> list[int]:
        ids: list[int] = []
        for t in ts:
            ids.extend(tokenizer.encode(t))
            ids.append(tokenizer.encode("\n")[0] if "\n" in tokenizer.stoi else tokenizer.pad_id)
        return ids

    train_ids = encode_join(train_texts)
    val_ids = encode_join(val_texts)
    # Ensure val has enough tokens
    while len(val_ids) < block_size + 1:
        val_ids = val_ids + encode_join(val_texts)
    while len(train_ids) < block_size + 1:
        train_ids = train_ids + encode_join(train_texts)

    return (
        tokenizer,
        CharLMDataset(train_ids, block_size),
        CharLMDataset(val_ids, block_size),
    )

"""Simple character-level tokenizer (no external vocab files)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CharTokenizer:
    """Map characters <-> integer ids. Special tokens: <pad>, <unk>."""

    stoi: dict[str, int]
    itos: dict[int, str]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharTokenizer":
        chars = sorted({c for t in texts for c in t})
        specials = ["<pad>", "<unk>"]
        vocab = specials + [c for c in chars if c not in specials]
        stoi = {ch: i for i, ch in enumerate(vocab)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str) -> list[int]:
        unk = self.unk_id
        return [self.stoi.get(c, unk) for c in text]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            ch = self.itos.get(int(i), self.unk_token)
            if ch in (self.pad_token, self.unk_token):
                continue
            out.append(ch)
        return "".join(out)

    def to_dict(self) -> dict:
        return {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        itos = {int(k): v for k, v in d["itos"].items()}
        return cls(
            stoi=dict(d["stoi"]),
            itos=itos,
            pad_token=d.get("pad_token", "<pad>"),
            unk_token=d.get("unk_token", "<unk>"),
        )

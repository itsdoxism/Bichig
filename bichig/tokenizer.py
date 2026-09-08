import json
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class CharTokenizer:
    def __init__(self, vocab: list[str]):
        ordered = []
        seen = set()
        for token in SPECIAL_TOKENS + vocab:
            if token not in seen:
                ordered.append(token)
                seen.add(token)
        self.itos = ordered
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, texts: Iterable[str]) -> "CharTokenizer":
        chars = sorted({ch for text in texts for ch in text})
        return cls(chars)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        out = []
        for idx in ids:
            token = self.itos[int(idx)]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            out.append(token)
        return "".join(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        vocab = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([token for token in vocab if token not in SPECIAL_TOKENS])

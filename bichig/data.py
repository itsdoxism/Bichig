from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


@dataclass(frozen=True)
class Pair:
    source: str
    target: str


def read_pairs(path: str | Path) -> list[Pair]:
    pairs: list[Pair] = []
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split("\t")
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_no}: expected exactly one tab-separated source/target pair")
        source, target = (part.strip() for part in parts)
        if not source or not target:
            raise ValueError(f"{path}:{line_no}: source and target must both be non-empty")
        pairs.append(Pair(source, target))
    if not pairs:
        raise ValueError(f"No training pairs found in {path}")
    return pairs


class ParallelDataset(Dataset):
    def __init__(self, pairs: list[Pair], tokenizer: CharTokenizer, max_len: int = 256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]
        src = self.tokenizer.encode(pair.source)
        tgt = self.tokenizer.encode(pair.target)
        if len(src) > self.max_len or len(tgt) > self.max_len:
            raise ValueError(
                f"Pair {index} exceeds max_len={self.max_len}: source={len(src)}, target={len(tgt)}"
            )
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def make_collate_fn(pad_id: int):
    def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]):
        srcs, tgts = zip(*batch)
        src = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=pad_id)
        tgt = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=pad_id)
        return src, tgt

    return collate

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from .infer import load_model_and_tokenizer, translate


@dataclass
class Metrics:
    samples: int
    exact_matches: int
    char_errors: int
    char_total: int

    @property
    def exact_match(self) -> float:
        return self.exact_matches / self.samples if self.samples else 0.0

    @property
    def cer(self) -> float:
        return self.char_errors / self.char_total if self.char_total else 0.0


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_no}: expected exactly one tab")
            src, tgt = parts
            pairs.append((src, tgt))
    return pairs


def evaluate(checkpoint: Path, data: Path, device: str) -> Metrics:
    model, tokenizer, cfg = load_model_and_tokenizer(checkpoint, device)
    pairs = load_pairs(data)

    exact = 0
    errors = 0
    total = 0

    for src, target in pairs:
        pred = translate(model, tokenizer, src, cfg, device)
        if pred == target:
            exact += 1
        errors += levenshtein(pred, target)
        total += max(1, len(target))

    return Metrics(len(pairs), exact, errors, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Bichig on a held-out TSV corpus")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    metrics = evaluate(args.checkpoint, args.data, args.device)
    print(f"samples: {metrics.samples}")
    print(f"exact_match: {metrics.exact_match:.4f}")
    print(f"cer: {metrics.cer:.4f}")


if __name__ == "__main__":
    main()

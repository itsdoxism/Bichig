from __future__ import annotations

import argparse
import hashlib
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def read_pairs(paths: list[Path]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for path in paths:
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            if "\t" not in raw:
                raise ValueError(f"{path}:{line_no}: expected TAB-separated source and target")
            src, tgt = raw.split("\t", 1)
            src, tgt = normalize(src), normalize(tgt)
            if not src or not tgt:
                raise ValueError(f"{path}:{line_no}: empty source or target")
            pairs.append((src, tgt))
    return pairs


def stable_key(pair: tuple[str, str], seed: int) -> str:
    payload = f"{seed}\0{pair[0]}\0{pair[1]}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_pairs(
    pairs: list[tuple[str, str]],
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    if valid_ratio < 0 or test_ratio < 0 or valid_ratio + test_ratio >= 1:
        raise ValueError("valid_ratio and test_ratio must be >= 0 and sum to < 1")

    ordered = sorted(pairs, key=lambda p: stable_key(p, seed))
    n = len(ordered)
    n_test = round(n * test_ratio)
    n_valid = round(n * valid_ratio)
    test = ordered[:n_test]
    valid = ordered[n_test : n_test + n_valid]
    train = ordered[n_test + n_valid :]
    return train, valid, test


def write_tsv(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{src}\t{tgt}\n" for src, tgt in pairs)
    path.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a clean deterministic Bichig dataset")
    parser.add_argument("inputs", nargs="+", type=Path, help="One or more UTF-8 TSV files")
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=422)
    args = parser.parse_args()

    raw = read_pairs(args.inputs)

    # Remove exact duplicate rows while preserving deterministic behavior.
    unique = sorted(set(raw))

    # Reject ambiguous supervision: the same source mapped to multiple targets.
    by_source: dict[str, set[str]] = defaultdict(set)
    for src, tgt in unique:
        by_source[src].add(tgt)
    conflicts = {src: targets for src, targets in by_source.items() if len(targets) > 1}
    if conflicts:
        preview = list(conflicts.items())[:5]
        formatted = "; ".join(f"{src!r} -> {sorted(targets)!r}" for src, targets in preview)
        raise ValueError(f"conflicting targets found for {len(conflicts)} source texts: {formatted}")

    train, valid, test = split_pairs(unique, args.valid_ratio, args.test_ratio, args.seed)
    write_tsv(args.out / "train.tsv", train)
    write_tsv(args.out / "valid.tsv", valid)
    write_tsv(args.out / "test.tsv", test)

    src_chars = Counter(ch for src, _ in unique for ch in src)
    tgt_chars = Counter(ch for _, tgt in unique for ch in tgt)

    print(f"raw rows:      {len(raw)}")
    print(f"unique rows:   {len(unique)}")
    print(f"duplicates:    {len(raw) - len(unique)}")
    print(f"train:         {len(train)}")
    print(f"valid:         {len(valid)}")
    print(f"test:          {len(test)}")
    print(f"source chars:  {len(src_chars)}")
    print(f"target chars:  {len(tgt_chars)}")
    print(f"output:        {args.out}")


if __name__ == "__main__":
    main()

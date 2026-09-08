from __future__ import annotations

import argparse
import unicodedata
from collections import Counter
from pathlib import Path

MONGOLIAN_BLOCK_START = 0x1800
MONGOLIAN_BLOCK_END = 0x18AF


def has_mongolian_script(text: str) -> bool:
    return any(MONGOLIAN_BLOCK_START <= ord(ch) <= MONGOLIAN_BLOCK_END for ch in text)


def validate(path: Path) -> int:
    errors: list[str] = []
    seen: Counter[tuple[str, str]] = Counter()
    source_to_targets: dict[str, set[str]] = {}
    count = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue

            count += 1
            if line.count("\t") != 1:
                errors.append(f"line {line_no}: expected exactly one tab")
                continue

            src, tgt = line.split("\t")
            if not src.strip() or not tgt.strip():
                errors.append(f"line {line_no}: empty source or target")
                continue

            if src != unicodedata.normalize("NFC", src):
                errors.append(f"line {line_no}: source is not NFC normalized")
            if tgt != unicodedata.normalize("NFC", tgt):
                errors.append(f"line {line_no}: target is not NFC normalized")
            if not has_mongolian_script(tgt):
                errors.append(f"line {line_no}: target has no Mongolian Unicode characters")

            pair = (src, tgt)
            seen[pair] += 1
            source_to_targets.setdefault(src, set()).add(tgt)

    for (src, tgt), n in seen.items():
        if n > 1:
            errors.append(f"duplicate pair x{n}: {src!r} -> {tgt!r}")

    for src, targets in source_to_targets.items():
        if len(targets) > 1:
            errors.append(f"conflicting targets for {src!r}: {len(targets)} variants")

    print(f"pairs: {count}")
    print(f"errors: {len(errors)}")
    for error in errors[:100]:
        print(f"- {error}")
    if len(errors) > 100:
        print(f"... and {len(errors) - 100} more")

    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Bichig TSV training data")
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    raise SystemExit(validate(args.data))


if __name__ == "__main__":
    main()

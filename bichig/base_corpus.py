import argparse
import csv
import json
import random
import re
import unicodedata
from pathlib import Path

from .base_manifest import load_manifest

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", " ").replace("\u00a0", " ")
    return WS_RE.sub(" ", text).strip()


def cyrillic_letter_ratio(text: str) -> float:
    letters = [ch for ch in text if LETTER_RE.match(ch)]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if CYRILLIC_RE.match(ch)) / len(letters)


def iter_texts(path: Path, field: str) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".text"}:
        return path.read_text(encoding="utf-8").splitlines()
    if suffix == ".jsonl":
        out = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get(field)
            if not isinstance(value, str):
                raise SystemExit(f"{path}:{lineno}: missing string field {field!r}")
            out.append(value)
        return out
    if suffix == ".csv":
        out = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if field not in (reader.fieldnames or []):
                raise SystemExit(f"{path}: missing CSV field {field!r}")
            for row in reader:
                value = row.get(field, "")
                if value:
                    out.append(value)
        return out
    raise SystemExit(f"unsupported corpus format: {path}")


def clean_texts(
    texts: list[str],
    min_chars: int = 20,
    max_chars: int = 1000,
    min_cyrillic_ratio: float = 0.75,
) -> tuple[list[str], dict]:
    seen = set()
    clean = []
    stats = {
        "raw": len(texts),
        "empty": 0,
        "too_short": 0,
        "too_long": 0,
        "low_cyrillic": 0,
        "duplicates": 0,
        "clean": 0,
    }
    for raw in texts:
        text = normalize_text(raw)
        if not text:
            stats["empty"] += 1
            continue
        if len(text) < min_chars:
            stats["too_short"] += 1
            continue
        if len(text) > max_chars:
            stats["too_long"] += 1
            continue
        if cyrillic_letter_ratio(text) < min_cyrillic_ratio:
            stats["low_cyrillic"] += 1
            continue
        key = text.casefold()
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)
        clean.append(text)
    stats["clean"] = len(clean)
    return clean, stats


def split_texts(texts: list[str], valid_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    items = list(texts)
    random.Random(seed).shuffle(items)
    if not items:
        return [], []
    if valid_ratio <= 0 or len(items) == 1:
        return items, []
    n_valid = max(1, min(round(len(items) * valid_ratio), len(items) - 1))
    valid = items[:n_valid]
    train = items[n_valid:]
    return train, valid


def write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(row + "\n" for row in rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a clean Cyrillic Mongolian corpus for Bichig Base")
    p.add_argument("inputs", nargs="*")
    p.add_argument("--manifest", help="JSON manifest of corpus sources with license metadata")
    p.add_argument("--allow-unknown-license", action="store_true")
    p.add_argument("--out", default="data/base/processed")
    p.add_argument("--field", default="text", help="JSONL/CSV text field")
    p.add_argument("--min-chars", type=int, default=20)
    p.add_argument("--max-chars", type=int, default=1000)
    p.add_argument("--min-cyrillic-ratio", type=float, default=0.75)
    p.add_argument("--valid-ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=422)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    raw = []
    per_source = {}
    source_meta = []
    input_paths = list(a.inputs)
    if a.manifest:
        for rec in load_manifest(a.manifest):
            if not rec.enabled:
                continue
            if rec.license.strip().lower() in {"", "unknown", "unspecified"} and not a.allow_unknown_license:
                raise SystemExit(f"source {rec.name!r} has unknown license; pass --allow-unknown-license to override")
            input_paths.append(rec.path)
            source_meta.append({"name": rec.name, "path": rec.path, "license": rec.license, "kind": rec.kind, "language": rec.language})
    if not input_paths:
        raise SystemExit("provide at least one input path or --manifest")
    for name in input_paths:
        path = Path(name)
        rows = iter_texts(path, a.field)
        per_source[str(path)] = len(rows)
        raw.extend(rows)
    clean, stats = clean_texts(raw, a.min_chars, a.max_chars, a.min_cyrillic_ratio)
    train, valid = split_texts(clean, a.valid_ratio, a.seed)
    out = Path(a.out)
    write_lines(out / "train.txt", train)
    write_lines(out / "valid.txt", valid)
    report = {
        "sources": per_source,
        "source_metadata": source_meta,
        **stats,
        "train": len(train),
        "valid": len(valid),
        "seed": a.seed,
        "valid_ratio": a.valid_ratio,
        "min_cyrillic_ratio": a.min_cyrillic_ratio,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Row:
    source: str
    target: str
    origin: str
    line: int


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def read_jsonl(paths: list[Path], source_key: str, target_key: str) -> list[Row]:
    rows: list[Row] = []
    for path in paths:
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            obj = json.loads(raw)
            if source_key not in obj or target_key not in obj:
                raise ValueError(f"{path}:{line_no}: missing {source_key!r} or {target_key!r}")
            src = normalize(str(obj[source_key]))
            tgt = normalize(str(obj[target_key]))
            if not src or not tgt:
                raise ValueError(f"{path}:{line_no}: empty source or target")
            rows.append(Row(src, tgt, str(path), line_no))
    return rows


def clean_rows(rows: list[Row]) -> tuple[list[tuple[str, str]], dict[str, list[str]], dict[str, int]]:
    unique_pairs = {(r.source, r.target) for r in rows}
    by_source: dict[str, set[str]] = defaultdict(set)
    for src, tgt in unique_pairs:
        by_source[src].add(tgt)

    conflicts = {
        src: sorted(targets)
        for src, targets in sorted(by_source.items())
        if len(targets) > 1
    }
    clean = sorted(
        (src, next(iter(targets)))
        for src, targets in by_source.items()
        if len(targets) == 1
    )
    stats = {
        "raw_rows": len(rows),
        "unique_pairs": len(unique_pairs),
        "exact_duplicates": len(rows) - len(unique_pairs),
        "conflicting_sources": len(conflicts),
        "conflicting_pairs": sum(len(v) for v in conflicts.values()),
        "clean_pairs": len(clean),
    }
    return clean, conflicts, stats


def write_tsv(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{s}\t{t}\n" for s, t in pairs), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Import parallel JSONL and quarantine ambiguous source mappings")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--source-key", default="cyrillic")
    p.add_argument("--target-key", default="bicig")
    p.add_argument("--out", type=Path, default=Path("data/imported.tsv"))
    p.add_argument("--conflicts", type=Path, default=Path("data/conflicts.json"))
    p.add_argument("--report", type=Path, default=Path("data/import-report.json"))
    args = p.parse_args()

    rows = read_jsonl(args.inputs, args.source_key, args.target_key)
    clean, conflicts, stats = clean_rows(rows)
    write_tsv(args.out, clean)
    args.conflicts.parent.mkdir(parents=True, exist_ok=True)
    args.conflicts.write_text(json.dumps(conflicts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"output: {args.out}")
    print(f"conflicts: {args.conflicts}")


if __name__ == "__main__":
    main()

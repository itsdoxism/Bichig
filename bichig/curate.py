import argparse
import csv
import json
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path


FIELDS = [
    "id",
    "source",
    "target",
    "status",
    "difficulty",
    "category",
    "provenance",
    "license",
    "reviewer",
    "notes",
]

ALLOWED_STATUS = {"draft", "review", "verified", "rejected"}
ALLOWED_DIFFICULTY = {"easy", "medium", "hard"}


@dataclass
class Record:
    id: str
    source: str
    target: str
    status: str = "draft"
    difficulty: str = "easy"
    category: str = "general"
    provenance: str = ""
    license: str = ""
    reviewer: str = ""
    notes: str = ""


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def read_records(path: Path) -> list[Record]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [field for field in FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Missing columns: {', '.join(missing)}")
        out = []
        for row in reader:
            out.append(Record(**{field: row.get(field, "") for field in FIELDS}))
        return out


def write_records(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def cmd_init(args: argparse.Namespace) -> None:
    path = Path(args.file)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} already exists; use --force to replace it")
    write_records(path, [])
    print(f"created {path}")


def cmd_add(args: argparse.Namespace) -> None:
    path = Path(args.file)
    records = read_records(path)
    next_id = args.id or f"b{len(records)+1:06d}"
    if any(r.id == next_id for r in records):
        raise SystemExit(f"duplicate id: {next_id}")
    if args.status not in ALLOWED_STATUS:
        raise SystemExit(f"invalid status: {args.status}")
    if args.difficulty not in ALLOWED_DIFFICULTY:
        raise SystemExit(f"invalid difficulty: {args.difficulty}")
    records.append(
        Record(
            id=next_id,
            source=normalize(args.source),
            target=normalize(args.target),
            status=args.status,
            difficulty=args.difficulty,
            category=args.category,
            provenance=args.provenance,
            license=args.license,
            reviewer=args.reviewer,
            notes=args.notes,
        )
    )
    write_records(path, records)
    print(f"added {next_id}")


def cmd_stats(args: argparse.Namespace) -> None:
    records = read_records(Path(args.file))
    summary = {
        "total": len(records),
        "status": {},
        "difficulty": {},
        "category": {},
    }
    for record in records:
        for key in ("status", "difficulty", "category"):
            bucket = summary[key]
            value = getattr(record, key) or "(empty)"
            bucket[value] = bucket.get(value, 0) + 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_export(args: argparse.Namespace) -> None:
    records = read_records(Path(args.file))
    verified = [r for r in records if r.status == "verified"]
    seen = {}
    rows = []
    for record in verified:
        source = normalize(record.source)
        target = normalize(record.target)
        if not source or not target:
            raise SystemExit(f"empty verified pair: {record.id}")
        previous = seen.get(source)
        if previous is not None and previous != target:
            raise SystemExit(f"conflicting verified target for source {source!r}")
        if previous == target:
            continue
        seen[source] = target
        rows.append((source, target))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        for source, target in rows:
            f.write(f"{source}\t{target}\n")
    print(f"exported {len(rows)} verified pairs to {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Curate Bichig parallel corpus records")
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("file")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    add = sub.add_parser("add")
    add.add_argument("file")
    add.add_argument("--id")
    add.add_argument("--source", required=True)
    add.add_argument("--target", required=True)
    add.add_argument("--status", default="draft")
    add.add_argument("--difficulty", default="easy")
    add.add_argument("--category", default="general")
    add.add_argument("--provenance", default="")
    add.add_argument("--license", default="")
    add.add_argument("--reviewer", default="")
    add.add_argument("--notes", default="")
    add.set_defaults(func=cmd_add)

    stats = sub.add_parser("stats")
    stats.add_argument("file")
    stats.set_defaults(func=cmd_stats)

    export = sub.add_parser("export")
    export.add_argument("file")
    export.add_argument("--out", default="data/seed.tsv")
    export.set_defaults(func=cmd_export)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SourceRecord:
    name: str
    path: str
    license: str
    kind: str = "text"
    language: str = "mn-Cyrl"
    enabled: bool = True
    notes: str = ""
    usage: str = "general"
    redistribute: bool = True

    @classmethod
    def from_dict(cls, row: dict):
        return cls(**row)


def load_manifest(path: str | Path) -> list[SourceRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
        raise SystemExit("manifest must be an object with a 'sources' list")
    out = []
    for idx, row in enumerate(data["sources"], start=1):
        if not isinstance(row, dict):
            raise SystemExit(f"manifest source #{idx} must be an object")
        rec = SourceRecord.from_dict(row)
        if not rec.name or not rec.path or not rec.license:
            raise SystemExit(f"manifest source #{idx} requires name/path/license")
        out.append(rec)
    return out


def write_manifest(path: str | Path, rows: list[SourceRecord]):
    payload = {"version": 1, "sources": [asdict(x) for x in rows]}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Validate/inspect a Bichig Base corpus source manifest")
    p.add_argument("manifest")
    p.add_argument("--allow-unknown-license", action="store_true")
    args = p.parse_args()
    rows = load_manifest(args.manifest)
    bad = [r.name for r in rows if r.enabled and r.license.strip().lower() in {"", "unknown", "unspecified"}]
    if bad and not args.allow_unknown_license:
        raise SystemExit("enabled sources with unknown license: " + ", ".join(bad))
    print(json.dumps({
        "sources": len(rows),
        "enabled": sum(r.enabled for r in rows),
        "licenses": sorted({r.license for r in rows if r.enabled}),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

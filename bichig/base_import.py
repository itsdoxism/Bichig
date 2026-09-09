import argparse
import csv
import json
from pathlib import Path


def read_txt(path: Path, field: str | None = None):
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            yield line


def read_jsonl(path: Path, field: str):
    with path.open('r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            value = obj
            for part in field.split('.'):
                if not isinstance(value, dict) or part not in value:
                    raise SystemExit(f'{path}:{lineno}: missing field {field!r}')
                value = value[part]
            if isinstance(value, str) and value.strip():
                yield value.strip()


def read_delimited(path: Path, field: str, delimiter: str):
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if field not in (reader.fieldnames or []):
            raise SystemExit(f'{path}: missing column {field!r}; columns={reader.fieldnames}')
        for row in reader:
            value = (row.get(field) or '').strip()
            if value:
                yield value


def infer_format(path: Path):
    suffix = path.suffix.lower()
    if suffix in {'.txt', '.text'}:
        return 'txt'
    if suffix in {'.jsonl', '.ndjson'}:
        return 'jsonl'
    if suffix == '.csv':
        return 'csv'
    if suffix in {'.tsv', '.tab'}:
        return 'tsv'
    raise SystemExit(f'cannot infer format for {path}; use --format')


def extract(path: Path, fmt: str, field: str | None):
    if fmt == 'txt':
        return list(read_txt(path))
    if not field:
        raise SystemExit(f'--field is required for {fmt}')
    if fmt == 'jsonl':
        return list(read_jsonl(path, field))
    if fmt == 'csv':
        return list(read_delimited(path, field, ','))
    if fmt == 'tsv':
        return list(read_delimited(path, field, '\t'))
    raise SystemExit(f'unsupported format: {fmt}')


def main():
    p = argparse.ArgumentParser(description='Extract text fields from source files for Bichig Base corpus building')
    p.add_argument('inputs', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--format', choices=['auto', 'txt', 'jsonl', 'csv', 'tsv'], default='auto')
    p.add_argument('--field', help='text field/column; dotted JSONL fields are supported')
    p.add_argument('--source-name', default='unknown')
    p.add_argument('--license', default='unknown')
    p.add_argument('--report')
    args = p.parse_args()

    rows = []
    per_file = {}
    for raw in args.inputs:
        path = Path(raw)
        fmt = infer_format(path) if args.format == 'auto' else args.format
        extracted = extract(path, fmt, args.field)
        rows.extend(extracted)
        per_file[str(path)] = len(extracted)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf-8')

    report = {
        'source_name': args.source_name,
        'license': args.license,
        'inputs': per_file,
        'rows_extracted': len(rows),
        'output': str(out),
    }
    report_path = Path(args.report) if args.report else out.with_suffix(out.suffix + '.report.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

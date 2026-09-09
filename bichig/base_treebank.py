import argparse
import json
import re
from pathlib import Path

TOKEN_RE = re.compile(r"\(([^()\s]+)\s+([^()]+?)\)")


def sentence_from_ptb(tree: str) -> str:
    """Best-effort reconstruction of surface text from a Penn Treebank-style tree."""
    tokens = []
    for _tag, value in TOKEN_RE.findall(tree):
        value = value.strip()
        if not value or value in {"-NONE-", "*"}:
            continue
        tokens.append(value)
    if not tokens:
        return ""
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.!?;:%)\]\}])", r"\1", text)
    text = re.sub(r"([(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+([”’])", r"\1", text)
    text = re.sub(r"([“‘])\s+", r"\1", text)
    return text.strip()


def iter_trees(text: str):
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]
                start = None


def extract_treebank(paths: list[Path]) -> list[str]:
    rows = []
    seen = set()
    for path in paths:
        content = path.read_text(encoding="utf-8", errors="replace")
        for tree in iter_trees(content):
            sentence = sentence_from_ptb(tree)
            key = sentence.casefold()
            if sentence and key not in seen:
                seen.add(key)
                rows.append(sentence)
    return rows


def main():
    p = argparse.ArgumentParser(description="Extract plain Mongolian sentences from MonTree/PTB .tbf files")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", default="data/base/raw/montree.txt")
    p.add_argument("--report", default="data/base/raw/montree-report.json")
    a = p.parse_args()
    paths = [Path(x) for x in a.inputs]
    rows = extract_treebank(paths)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(x + "\n" for x in rows), encoding="utf-8")
    report = {
        "source": "MonTree / Mongolian treebank",
        "license": "CC BY-NC 3.0",
        "doi": "10.17632/7xpzh9fv8f.3",
        "files": [str(x) for x in paths],
        "unique_sentences": len(rows),
        "usage": "non-commercial/research unless separately permitted",
        "redistribute": True,
    }
    Path(a.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

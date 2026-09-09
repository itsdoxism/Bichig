import argparse
import bz2
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .base_corpus import clean_texts, write_lines

COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
REF_RE = re.compile(r"<ref\b[^>]*>.*?</ref\s*>|<ref\b[^>]*/\s*>", re.I | re.S)
TAG_RE = re.compile(r"<[^>]+>")
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
FILE_RE = re.compile(r"\[\[(?:File|Image|Файл):.*?\]\]", re.I | re.S)
LINK_RE = re.compile(r"\[\[([^\]|]+\|)?([^\]]+)\]\]")
EXT_LINK_RE = re.compile(r"\[(?:https?://|//)[^\s\]]+\s*([^\]]*)\]")
HEADING_RE = re.compile(r"^=+\s*(.*?)\s*=+$", re.M)
TABLE_RE = re.compile(r"\{\|.*?\|\}", re.S)
LIST_PREFIX_RE = re.compile(r"^[*#:;]+\s*", re.M)
MULTI_SPACE_RE = re.compile(r"[ \t]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def strip_wikicode(text: str) -> str:
    text = html.unescape(text)
    text = COMMENT_RE.sub(" ", text)
    text = REF_RE.sub(" ", text)
    text = TABLE_RE.sub(" ", text)
    text = FILE_RE.sub(" ", text)
    for _ in range(8):
        newer = TEMPLATE_RE.sub(" ", text)
        if newer == text:
            break
        text = newer
    text = HEADING_RE.sub(r"\1", text)
    text = LINK_RE.sub(lambda m: m.group(2), text)
    text = EXT_LINK_RE.sub(lambda m: m.group(1) or " ", text)
    text = TAG_RE.sub(" ", text)
    text = LIST_PREFIX_RE.sub("", text)
    text = text.replace("'''", "").replace("''", "")
    text = text.replace("&nbsp;", " ")
    text = MULTI_SPACE_RE.sub(" ", text)
    return text


def sentence_rows(text: str) -> list[str]:
    rows = []
    for piece in SENTENCE_SPLIT_RE.split(strip_wikicode(text)):
        piece = piece.strip(" \t\r\n-|}")
        if piece:
            rows.append(piece)
    return rows


def iter_wiki_texts(path: Path):
    opener = bz2.open if path.suffix.lower() == ".bz2" else open
    with opener(path, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        for _, elem in context:
            if elem.tag.endswith("}page") or elem.tag == "page":
                ns = None
                text = None
                redirect = False
                for child in elem:
                    tag = child.tag.rsplit("}", 1)[-1]
                    if tag == "ns":
                        ns = child.text
                    elif tag == "redirect":
                        redirect = True
                    elif tag == "revision":
                        for sub in child:
                            if sub.tag.rsplit("}", 1)[-1] == "text":
                                text = sub.text
                                break
                if ns == "0" and not redirect and text:
                    yield text
                elem.clear()


def extract(path: Path, min_chars: int, max_chars: int, min_cyrillic_ratio: float):
    raw_rows = []
    pages = 0
    for text in iter_wiki_texts(path):
        pages += 1
        raw_rows.extend(sentence_rows(text))
    clean, stats = clean_texts(
        raw_rows,
        min_chars=min_chars,
        max_chars=max_chars,
        min_cyrillic_ratio=min_cyrillic_ratio,
    )
    report = {"pages": pages, "sentence_candidates": len(raw_rows), **stats}
    return clean, report


def parse_args():
    p = argparse.ArgumentParser(description="Extract clean Cyrillic Mongolian text from a Wikipedia pages-articles XML dump")
    p.add_argument("dump")
    p.add_argument("--out", default="data/base/raw/mnwiki.txt")
    p.add_argument("--report", default="data/base/raw/mnwiki-report.json")
    p.add_argument("--min-chars", type=int, default=30)
    p.add_argument("--max-chars", type=int, default=500)
    p.add_argument("--min-cyrillic-ratio", type=float, default=0.80)
    return p.parse_args()


def main():
    a = parse_args()
    rows, report = extract(Path(a.dump), a.min_chars, a.max_chars, a.min_cyrillic_ratio)
    write_lines(Path(a.out), rows)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

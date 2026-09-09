import argparse
import json
from collections import Counter
from pathlib import Path

from .base_tokenizer import HybridTokenizer, PIECE_RE
from .base_train import iter_corpus


def token_kind(token: str) -> str:
    if token == "<unk>":
        return "unk"
    if token == "<sp>":
        return "space"
    if token.startswith("<w:"):
        return "word"
    if token.startswith("<suf:"):
        return "suffix"
    if token.startswith("<ch:"):
        return "char"
    if token.startswith("<p:"):
        return "punct"
    return "special"


def analyze_tokenizer(tok: HybridTokenizer, texts, max_lines: int | None = None) -> dict:
    category_counts = Counter()
    vocab_categories = Counter(token_kind(token) for token in tok.itos)
    lines = 0
    chars = 0
    nonspace_chars = 0
    words = 0
    encoded_tokens = 0

    for text in texts:
        if max_lines is not None and lines >= max_lines:
            break
        lines += 1
        chars += len(text)
        nonspace_chars += sum(not ch.isspace() for ch in text)
        words += sum(1 for piece in PIECE_RE.findall(text) if tok._is_word(piece))
        ids = tok.encode(text, add_bos=False, add_eos=False)
        encoded_tokens += len(ids)
        for idx in ids:
            category_counts[token_kind(tok.itos[idx])] += 1

    unk = category_counts["unk"]
    report = {
        "lines": lines,
        "vocab_size": len(tok),
        "vocab_categories": dict(sorted(vocab_categories.items())),
        "characters": chars,
        "nonspace_characters": nonspace_chars,
        "words": words,
        "encoded_tokens": encoded_tokens,
        "tokens_per_word": encoded_tokens / words if words else None,
        "tokens_per_nonspace_char": encoded_tokens / nonspace_chars if nonspace_chars else None,
        "unknown_tokens": unk,
        "unknown_rate": unk / encoded_tokens if encoded_tokens else 0.0,
        "token_categories": dict(sorted(category_counts.items())),
        "learned_suffixes": list(tok._suffixes[:100]),
    }
    return report


def main():
    p = argparse.ArgumentParser(description="Measure Bichig Base tokenizer coverage and compression on a corpus")
    p.add_argument("--vocab", required=True, help="vocab.json produced by bichig-base-train")
    p.add_argument("--data", required=True, help="line-oriented text corpus")
    p.add_argument("--max-lines", type=int, help="optional cap for quick diagnostics")
    p.add_argument("--out", help="optional JSON report path")
    a = p.parse_args()

    tok = HybridTokenizer.load(a.vocab)
    report = analyze_tokenizer(tok, iter_corpus(a.data), a.max_lines)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

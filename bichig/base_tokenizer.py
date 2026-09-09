import json
import re
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<sp>"]
# Keep whitespace, runs of Unicode word characters, and punctuation/symbols separate.
PIECE_RE = re.compile(r"\s+|[\w]+|[^\w\s]", re.UNICODE)


class HybridTokenizer:
    def __init__(self, tokens: list[str]):
        vocab = []
        seen = set()
        for token in SPECIAL_TOKENS + tokens:
            if token not in seen:
                vocab.append(token)
                seen.add(token)
        self.itos = vocab
        self.stoi = {t: i for i, t in enumerate(vocab)}
        self._suffixes = sorted(
            (t[5:-1] for t in vocab if t.startswith("<suf:") and t.endswith(">")),
            key=len,
            reverse=True,
        )

    @property
    def pad_id(self): return self.stoi["<pad>"]
    @property
    def bos_id(self): return self.stoi["<bos>"]
    @property
    def eos_id(self): return self.stoi["<eos>"]
    @property
    def unk_id(self): return self.stoi["<unk>"]
    @property
    def sp_id(self): return self.stoi["<sp>"]

    def __len__(self): return len(self.itos)

    @staticmethod
    def _is_word(piece: str) -> bool:
        return bool(piece) and all(ch.isalnum() or ch == "_" for ch in piece)

    @classmethod
    def build(
        cls,
        texts: list[str],
        min_word_freq: int = 3,
        max_word_tokens: int = 8000,
        min_suffix_freq: int = 8,
        max_suffix_tokens: int = 512,
        min_suffix_len: int = 2,
        max_suffix_len: int = 5,
    ):
        word_counts = Counter()
        chars = set()
        punct = set()
        for text in texts:
            chars.update(ch for ch in text if not ch.isspace())
            for piece in PIECE_RE.findall(text):
                if piece.isspace():
                    continue
                if cls._is_word(piece):
                    word_counts[piece] += 1
                else:
                    punct.add(piece)

        words = [w for w, c in word_counts.most_common(max_word_tokens) if c >= min_word_freq]
        whole_words = set(words)

        suffix_counts = Counter()
        for word, count in word_counts.items():
            if word in whole_words or len(word) <= min_suffix_len:
                continue
            # Learn endings from corpus statistics rather than a hardcoded grammar list.
            for n in range(min_suffix_len, min(max_suffix_len, len(word) - 1) + 1):
                suffix_counts[word[-n:]] += count
        suffixes = [
            s for s, c in suffix_counts.most_common(max_suffix_tokens)
            if c >= min_suffix_freq
        ]

        word_tokens = [f"<w:{w}>" for w in words]
        suffix_tokens = [f"<suf:{s}>" for s in suffixes]
        punct_tokens = [f"<p:{p}>" for p in sorted(punct)]
        char_tokens = [f"<ch:{ch}>" for ch in sorted(chars) if ch not in punct]
        return cls(word_tokens + suffix_tokens + punct_tokens + char_tokens)

    def _encode_word(self, piece: str) -> list[int]:
        wt = f"<w:{piece}>"
        if wt in self.stoi:
            return [self.stoi[wt]]

        suffix = next((s for s in self._suffixes if piece.endswith(s) and len(piece) > len(s)), None)
        stem = piece[:-len(suffix)] if suffix else piece
        ids = [self.stoi.get(f"<ch:{ch}>", self.unk_id) for ch in stem]
        if suffix:
            ids.append(self.stoi[f"<suf:{suffix}>"])
        return ids

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = []
        for piece in PIECE_RE.findall(text):
            if piece.isspace():
                ids.append(self.sp_id)
            elif self._is_word(piece):
                ids.extend(self._encode_word(piece))
            else:
                ids.append(self.stoi.get(f"<p:{piece}>", self.stoi.get(f"<ch:{piece}>", self.unk_id)))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        out = []
        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                continue
            token = self.itos[idx]
            if token == "<sp>":
                out.append(" ")
            elif token.startswith("<w:") and token.endswith(">"):
                out.append(token[3:-1])
            elif token.startswith("<suf:") and token.endswith(">"):
                out.append(token[5:-1])
            elif token.startswith("<p:") and token.endswith(">"):
                out.append(token[3:-1])
            elif token.startswith("<ch:") and token.endswith(">"):
                out.append(token[4:-1])
            elif not skip_special:
                out.append(token)
        return "".join(out)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

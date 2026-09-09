import json
import re
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<sp>"]
WORD_RE = re.compile(r"\s+|[^\s]+", re.UNICODE)


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

    @classmethod
    def build(cls, texts: list[str], min_word_freq: int = 3, max_word_tokens: int = 8000):
        word_counts = Counter()
        chars = set()
        for text in texts:
            chars.update(text)
            for piece in WORD_RE.findall(text):
                if piece.isspace():
                    continue
                word_counts[piece] += 1
        words = [w for w, c in word_counts.most_common(max_word_tokens) if c >= min_word_freq]
        char_tokens = [f"<ch:{ch}>" for ch in sorted(chars) if not ch.isspace()]
        word_tokens = [f"<w:{w}>" for w in words]
        return cls(word_tokens + char_tokens)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = []
        for piece in WORD_RE.findall(text):
            if piece.isspace():
                ids.append(self.sp_id)
                continue
            wt = f"<w:{piece}>"
            if wt in self.stoi:
                ids.append(self.stoi[wt])
            else:
                for ch in piece:
                    ids.append(self.stoi.get(f"<ch:{ch}>", self.unk_id))
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

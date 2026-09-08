import tempfile
from pathlib import Path

import torch

from bichig.config import ModelConfig
from bichig.data import ParallelDataset, read_pairs
from bichig.model import BichigTransformer
from bichig.tokenizer import CharTokenizer


def test_tokenizer_roundtrip():
    texts = ["Монгол хэл", "ᠮᠣᠩᠭᠣᠯ"]
    tok = CharTokenizer.build(texts)
    encoded = tok.encode(texts[0], add_bos=True, add_eos=True)
    decoded = tok.decode(encoded)
    assert decoded == texts[0]


def test_model_forward_shape():
    tok = CharTokenizer.build(["абв", "ᠠᠪᠸ"])
    cfg = ModelConfig(
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_len=32,
    )
    model = BichigTransformer(len(tok), tok.pad_id, cfg)
    src = torch.tensor([[tok.bos_id, *tok.encode("абв"), tok.eos_id]], dtype=torch.long)
    tgt = torch.tensor([[tok.bos_id, *tok.encode("ᠠᠪᠸ"), tok.eos_id]], dtype=torch.long)
    logits = model(src, tgt[:, :-1])
    assert logits.shape == (1, tgt.size(1) - 1, len(tok))
    assert torch.isfinite(logits).all()


def test_parallel_dataset_reads_tsv():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "pairs.tsv"
        path.write_text("Монгол\tᠮᠣᠩᠭᠣᠯ\n", encoding="utf-8")
        pairs = read_pairs(path)
        assert len(pairs) == 1
        tok = CharTokenizer.build([pairs[0].source, pairs[0].target])
        ds = ParallelDataset(pairs, tok, max_len=32)
        src, tgt = ds[0]
        assert src[0].item() == tok.bos_id
        assert src[-1].item() == tok.eos_id
        assert tgt[0].item() == tok.bos_id
        assert tgt[-1].item() == tok.eos_id

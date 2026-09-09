import tempfile
from pathlib import Path

import torch


def test_manifest_accepts_research_metadata():
    from bichig.base_manifest import SourceRecord
    row = SourceRecord.from_dict({
        "name": "news", "path": "news.txt", "license": "unknown",
        "usage": "research-only", "redistribute": False,
    })
    assert row.usage == "research-only"
    assert row.redistribute is False


def test_treebank_sentence_reconstruction():
    from bichig.base_treebank import sentence_from_ptb, extract_treebank
    tree = "(S (NP-SBJ (PN Би)) (VP (NN сургуульд) (VB явна)) (. .))"
    assert sentence_from_ptb(tree) == "Би сургуульд явна."
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sample.tbf"
        p.write_text(tree + "\n" + tree, encoding="utf-8")
        assert extract_treebank([p]) == ["Би сургуульд явна."]


def test_base_eval_tiny_corpus_loss_is_finite():
    from bichig.base_eval import corpus_loss
    from bichig.base_model import BaseConfig, CausalBlockModel
    from bichig.base_tokenizer import HybridTokenizer
    tok = HybridTokenizer.build(["Монгол хэл сайхан.", "Монгол хүн ярьж байна."], min_word_freq=1)
    cfg = BaseConfig(d_model=32, nhead=4, layers=1, ffn=64, dropout=0.0, max_len=16)
    model = CausalBlockModel(len(tok), tok.pad_id, cfg)
    loss = corpus_loss(model, tok, ["Монгол хэл сайхан."], 16, 2, torch.device("cpu"))
    assert loss is not None and torch.isfinite(torch.tensor(loss))

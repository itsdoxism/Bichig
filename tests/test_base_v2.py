import torch
from torch import nn

from bichig.base_model import BaseConfig, CausalBlockModel
from bichig.base_tokenizer import HybridTokenizer
from bichig.base_train import LMDataset


def test_tokenizer_separates_punctuation_and_learns_suffixes():
    texts = [
        "сургуульд явна.",
        "гэрт явна.",
        "ажилд явна!",
        "номд тэмдэглэл хийв.",
    ] * 4
    tok = HybridTokenizer.build(
        texts,
        min_word_freq=99,
        min_suffix_freq=2,
        max_suffix_tokens=32,
    )
    encoded = tok.encode("сургуульд явна.")
    assert tok.decode(encoded) == "сургуульд явна."
    assert "<p:.>" in tok.itos
    assert any(t.startswith("<suf:") for t in tok.itos)


def test_lm_dataset_stride_reduces_overlapping_windows():
    ids = list(range(101))
    dense = LMDataset(ids, seq_len=10, stride=1)
    packed = LMDataset(ids, seq_len=10, stride=10)
    assert len(dense) == 91
    assert len(packed) == 10
    x, y = packed[1]
    assert x.tolist()[0] == 10
    assert y.tolist()[0] == 11


def test_base_model_initial_loss_is_sane():
    tok = HybridTokenizer.build(
        ["Монгол хэл сайхан.", "Монгол хүн ярьж байна."],
        min_word_freq=1,
    )
    cfg = BaseConfig(d_model=32, nhead=4, layers=1, ffn=64, dropout=0.0, max_len=16)
    model = CausalBlockModel(len(tok), tok.pad_id, cfg)
    ids = tok.encode("Монгол хэл сайхан.")
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, len(tok)), y.reshape(-1))
    assert loss.item() < 10.0

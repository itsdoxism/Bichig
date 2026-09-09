import torch


def test_hybrid_tokenizer_word_and_char_fallback():
    from bichig.base_tokenizer import HybridTokenizer

    tok = HybridTokenizer.build(["Монгол хэл", "Монгол сайхан"], min_word_freq=2)
    ids = tok.encode("Монгол хэл")
    assert tok.decode(ids) == "Монгол хэл"
    assert any(tok.itos[i] == "<w:Монгол>" for i in ids)
    assert any(tok.itos[i].startswith("<ch:") for i in ids)


def test_base_model_forward_shape():
    from bichig.base_model import BaseConfig, CausalBlockModel
    from bichig.base_tokenizer import HybridTokenizer

    tok = HybridTokenizer.build(["Монгол хэл", "Монгол сайхан"], min_word_freq=1)
    cfg = BaseConfig(d_model=32, nhead=4, layers=1, ffn=64, dropout=0.0, max_len=16)
    model = CausalBlockModel(len(tok), tok.pad_id, cfg)
    x = torch.tensor([tok.encode("Монгол", add_eos=False)], dtype=torch.long)
    logits = model(x)
    assert logits.shape == (1, x.size(1), len(tok))
    assert torch.isfinite(logits).all()

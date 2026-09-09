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


def test_streaming_lm_dataset_packs_without_loading_all_ids():
    from bichig.base_train import StreamingLMDataset
    from bichig.base_tokenizer import HybridTokenizer
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "corpus.txt"
        rows = ["Монгол хэл сайхан.", "Бид монголоор ярьдаг.", "Хүүхэд сургуульд явна."]
        p.write_text("\n".join(rows) + "\n", encoding="utf-8")
        tok = HybridTokenizer.build(rows, min_word_freq=1)
        ds = StreamingLMDataset(p, tok, seq_len=4)
        x, y = next(iter(ds))
        assert len(x) == 4 and len(y) == 4
        assert x[1:].tolist() == y[:-1].tolist()


def test_streaming_vocab_sample_is_deterministic():
    from bichig.base_train import sample_corpus
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "corpus.txt"
        p.write_text("".join(f"Монгол өгүүлбэр {i}.\n" for i in range(100)), encoding="utf-8")
        a = sample_corpus(p, 10, seed=422)
        b = sample_corpus(p, 10, seed=422)
        assert a == b and len(a) == 10


def test_streaming_validation_loader_does_not_require_dataset_len():
    from torch import nn
    from torch.utils.data import DataLoader
    from bichig.base_train import StreamingLMDataset, evaluate_loss
    from bichig.base_model import BaseConfig, CausalBlockModel
    from bichig.base_tokenizer import HybridTokenizer
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "corpus.txt"
        rows = ["Монгол хэл сайхан.", "Бид монголоор ярьдаг.", "Хүүхэд сургуульд явна."]
        p.write_text("\n".join(rows) + "\n", encoding="utf-8")
        tok = HybridTokenizer.build(rows, min_word_freq=1)
        ds = StreamingLMDataset(p, tok, seq_len=4)
        loader = DataLoader(ds, batch_size=2)
        cfg = BaseConfig(d_model=32, nhead=4, layers=1, ffn=64, dropout=0.0, max_len=4)
        model = CausalBlockModel(len(tok), tok.pad_id, cfg)
        loss = evaluate_loss(model, loader, nn.CrossEntropyLoss(ignore_index=tok.pad_id), torch.device("cpu"))
        assert torch.isfinite(torch.tensor(loss))


def test_base_benchmark_prefers_lower_nll_when_model_is_biased():
    from bichig.base_benchmark import run_benchmark
    from bichig.base_model import BaseConfig, CausalBlockModel
    from bichig.base_tokenizer import HybridTokenizer

    texts = ["Би номоо уншив.", "Би номын уншив."] * 4
    tok = HybridTokenizer.build(texts, min_word_freq=1)
    cfg = BaseConfig(d_model=16, nhead=4, layers=1, ffn=32, dropout=0.0, max_len=32)
    model = CausalBlockModel(len(tok), tok.pad_id, cfg)
    result = run_benchmark(
        model, tok,
        [{"category": "test", "good": "Би номоо уншив.", "bad": "Би номын уншив."}],
        torch.device("cpu"),
    )
    assert result["probes"] == 1
    assert 0.0 <= result["accuracy"] <= 1.0
    assert "margin" in result["results"][0]


def test_base_benchmark_scores_text_longer_than_context_window():
    from bichig.base_benchmark import sentence_nll
    from bichig.base_model import BaseConfig, CausalBlockModel
    from bichig.base_tokenizer import HybridTokenizer

    text = "Монгол хэл бол олон зууны түүхтэй бөгөөд хүмүүс өдөр бүр хэрэглэдэг хэл юм."
    tok = HybridTokenizer.build([text], min_word_freq=1)
    cfg = BaseConfig(d_model=16, nhead=4, layers=1, ffn=32, dropout=0.0, max_len=8)
    model = CausalBlockModel(len(tok), tok.pad_id, cfg)
    loss, count = sentence_nll(model, tok, text, torch.device("cpu"))
    assert count > cfg.max_len
    assert torch.isfinite(torch.tensor(loss))


def test_base_run_resume_adds_last_checkpoint_when_present():
    from argparse import Namespace
    from bichig.base_run import build_commands
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        run_dir = td / "run"
        run_dir.mkdir()
        (run_dir / "last.pt").write_bytes(b"x")
        a = Namespace(manifest="m.json", processed=str(td / "processed"), run_dir=str(run_dir), epochs=1,
                      batch_size=2, seq_len=8, d_model=32, nhead=4, layers=1, ffn=64,
                      vocab_lines=10, steps_per_epoch=3, resume=True, skip_corpus=True,
                      allow_unknown_license=True)
        commands = build_commands(a)
        train = commands[0]
        assert "--resume" in train and str(run_dir / "last.pt") in train
        assert "--steps-per-epoch" in train


def test_checkpoint_contains_optimizer_for_resume():
    import inspect
    import bichig.base_train as bt
    src = inspect.getsource(bt.main)
    assert '"optimizer": opt.state_dict()' in src

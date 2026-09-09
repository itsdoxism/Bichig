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
    src_ids = tok.encode("абв", add_bos=True, add_eos=True)
    tgt_ids = tok.encode("ᠠᠪᠸ", add_bos=True, add_eos=True)
    src = torch.tensor([src_ids], dtype=torch.long)
    tgt = torch.tensor([tgt_ids], dtype=torch.long)
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


def test_curation_summary_and_progress_data():
    from bichig.curate import Record, summarize

    records = [
        Record(id="b000001", source="Монгол", target="ᠮᠣᠩᠭᠣᠯ", status="verified", category="word", reviewer="human"),
        Record(id="b000002", source="хэл", target="ᠬᠡᠯᠡ", status="review", category="word"),
    ]
    summary = summarize(records)
    assert summary["total"] == 2
    assert summary["status"]["verified"] == 1
    assert summary["status"]["review"] == 1
    assert summary["category"]["word"] == 2
    assert summary["missing"]["reviewer_on_verified"] == 0


def test_curation_status_and_audit():
    from argparse import Namespace
    from bichig.curate import Record, cmd_audit, cmd_set_status, read_records, write_records

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "seed.csv"
        write_records(path, [Record(
            id="b000001",
            source="Монгол",
            target="ᠮᠣᠩᠭᠣᠯ",
            status="review",
            provenance="manual",
            license="CC0",
        )])
        cmd_set_status(Namespace(
            file=str(path), id="b000001", status="verified", reviewer="tester", notes="checked"
        ))
        records = read_records(path)
        assert records[0].status == "verified"
        assert records[0].reviewer == "tester"
        cmd_audit(Namespace(file=str(path)))


def test_prepare_split_has_no_overlap():
    from bichig.prepare_data import split_pairs

    pairs = [(f"src{i}", f"tgt{i}") for i in range(20)]
    train, valid, test = split_pairs(pairs, valid_ratio=0.2, test_ratio=0.2, seed=422)
    train_set, valid_set, test_set = set(train), set(valid), set(test)
    assert len(train) == 12
    assert len(valid) == 4
    assert len(test) == 4
    assert train_set.isdisjoint(valid_set)
    assert train_set.isdisjoint(test_set)
    assert valid_set.isdisjoint(test_set)
    assert train_set | valid_set | test_set == set(pairs)


def test_jsonl_import_quarantines_conflicting_sources():
    from bichig.import_jsonl import Row, clean_rows

    rows = [
        Row("a", "x", "one", 1),
        Row("a", "x", "one", 2),
        Row("b", "y", "one", 3),
        Row("b", "z", "two", 1),
        Row("c", "q", "two", 2),
    ]
    clean, conflicts, stats = clean_rows(rows)
    assert clean == [("a", "x"), ("c", "q")]
    assert conflicts == {"b": ["y", "z"]}
    assert stats["raw_rows"] == 5
    assert stats["exact_duplicates"] == 1
    assert stats["conflicting_sources"] == 1
    assert stats["clean_pairs"] == 2


def test_generate_stops_repetition_loop():
    tok = CharTokenizer.build(["a", "ᠠ"])
    cfg = ModelConfig(d_model=16, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32, dropout=0.0, max_len=64)
    model = BichigTransformer(len(tok), tok.pad_id, cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.output.bias[tok.stoi["ᠠ"]] = 10.0
    src = torch.tensor([tok.encode("a")], dtype=torch.long)
    out = model.generate(src, tok.bos_id, tok.eos_id, max_new_tokens=50)
    assert out.size(1) < 51


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


def test_base_corpus_clean_and_split():
    from bichig.base_corpus import clean_texts, split_texts

    rows = [
        "Монгол хэл бол Монгол Улсын албан ёсны хэл юм.",
        "Монгол хэл бол Монгол Улсын албан ёсны хэл юм.",
        "This is English only and should be filtered out.",
        "Богино",
        "Өнөөдөр бид Монгол хэлний өгүүлбэрийн бүтцийг судалж байна.",
    ]
    clean, stats = clean_texts(rows, min_chars=20, max_chars=200, min_cyrillic_ratio=0.75)
    assert len(clean) == 2
    assert stats["duplicates"] == 1
    assert stats["low_cyrillic"] == 1
    assert stats["too_short"] == 1
    train, valid = split_texts(clean, valid_ratio=0.5, seed=422)
    assert len(train) == 1 and len(valid) == 1
    assert set(train).isdisjoint(valid)


def test_wiki_markup_extraction():
    from bichig.base_wiki import sentence_rows

    raw = """'''Монгол хэл''' бол [[Монгол Улс|Монгол Улсын]] албан ёсны хэл юм.
{{Infobox language|name=Монгол}}
== Түүх ==
Монгол хэл олон зууны түүхтэй. <ref>source</ref>"""
    rows = sentence_rows(raw)
    assert "Монгол хэл бол Монгол Улсын албан ёсны хэл юм." in rows
    assert "Монгол хэл олон зууны түүхтэй." in rows
    assert not any("Infobox" in row or "<ref>" in row for row in rows)


def test_base_tokenizer_separates_punctuation_and_learns_suffixes():
    from bichig.base_tokenizer import HybridTokenizer

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
    assert any(t == "<p:.>" for t in tok.itos)
    assert any(t.startswith("<suf:") for t in tok.itos)


def test_lm_dataset_stride_reduces_overlapping_windows():
    from bichig.base_train import LMDataset

    ids = list(range(101))
    dense = LMDataset(ids, seq_len=10, stride=1)
    packed = LMDataset(ids, seq_len=10, stride=10)
    assert len(dense) == 91
    assert len(packed) == 10
    x, y = packed[1]
    assert x.tolist()[0] == 10
    assert y.tolist()[0] == 11


def test_base_model_initial_loss_is_sane():
    from torch import nn
    from bichig.base_model import BaseConfig, CausalBlockModel
    from bichig.base_tokenizer import HybridTokenizer

    tok = HybridTokenizer.build(["Монгол хэл сайхан.", "Монгол хүн ярьж байна."], min_word_freq=1)
    cfg = BaseConfig(d_model=32, nhead=4, layers=1, ffn=64, dropout=0.0, max_len=16)
    model = CausalBlockModel(len(tok), tok.pad_id, cfg)
    ids = tok.encode("Монгол хэл сайхан.")
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, len(tok)), y.reshape(-1))
    assert loss.item() < 10.0


def test_base_import_jsonl_and_tsv():
    from bichig.base_import import extract
    import json
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        j = td / "rows.jsonl"
        j.write_text(json.dumps({"meta": {"text": "Монгол хэл"}}, ensure_ascii=False) + "\n", encoding="utf-8")
        t = td / "rows.tsv"
        t.write_text("sentence\nМонгол Улс\n", encoding="utf-8")
        assert extract(j, "jsonl", "meta.text") == ["Монгол хэл"]
        assert extract(t, "tsv", "sentence") == ["Монгол Улс"]

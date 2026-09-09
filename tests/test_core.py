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

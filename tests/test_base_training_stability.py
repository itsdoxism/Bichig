import math

from bichig.base_tokenizer import HybridTokenizer
from bichig.base_tokenizer_stats import analyze_tokenizer
from bichig.base_train import learning_rate_for_update


def test_learning_rate_warmup_and_cosine_bounds():
    base = 1.0
    assert math.isclose(learning_rate_for_update(base, 0, warmup_steps=10), 0.1)
    assert math.isclose(learning_rate_for_update(base, 9, warmup_steps=10), 1.0)
    assert math.isclose(learning_rate_for_update(base, 10, warmup_steps=10, cosine_steps=100), 1.0)
    assert math.isclose(
        learning_rate_for_update(base, 110, warmup_steps=10, cosine_steps=100, min_lr_ratio=0.1),
        0.1,
        rel_tol=1e-7,
    )


def test_learning_rate_without_decay_stays_at_base_after_warmup():
    assert math.isclose(learning_rate_for_update(3e-4, 500, warmup_steps=100, cosine_steps=0), 3e-4)


def test_tokenizer_stats_reports_coverage_and_categories():
    texts = [
        "Монгол хэл сайхан.",
        "Хүүхэд сургуульд явна.",
        "Бид монголоор ярьдаг.",
    ] * 4
    tok = HybridTokenizer.build(texts, min_word_freq=2, min_suffix_freq=2)
    report = analyze_tokenizer(tok, iter(texts))
    assert report["lines"] == len(texts)
    assert report["words"] > 0
    assert report["encoded_tokens"] > 0
    assert report["tokens_per_word"] > 0
    assert 0.0 <= report["unknown_rate"] <= 1.0
    assert sum(report["token_categories"].values()) == report["encoded_tokens"]

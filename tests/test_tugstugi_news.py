import tempfile
from pathlib import Path

from bichig.base_tugstugi_news import preprocess, sentence_rows


def test_tugstugi_news_preprocess_reservoir_sample():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        raw = root / "raw"
        raw.mkdir()
        article = "Эхний өгүүлбэр нь гарчиг маягийн мэдээлэл байна. Монгол хэл олон зууны турш хөгжиж ирсэн түүхтэй. Өнөөдөр монголчууд кирилл бичгийг өргөн хэрэглэж байна. Хэлний дүрэм нь өгүүлбэрийн бүтцийг ойлгоход чухал үүрэгтэй. Үгийн залгавар нь утга болон харьцааг илэрхийлдэг. Хүүхэд олон өгүүлбэр сонссоноор хэлний хэв маягийг сурдаг. Сүүлийн өгүүлбэр нь эх сурвалжийн тэмдэглэл байж болно."
        assert len(sentence_rows(article)) == 5
        (raw / "part1.txt").write_text(article + "\n" + article.replace("Хүүхэд", "Сурагч") + "\n", encoding="utf-8")
        out = root / "sample.txt"
        report = preprocess(raw, out, max_sentences=4, seed=422)
        rows = out.read_text(encoding="utf-8").splitlines()
        assert report["sentences_seen"] == 10
        assert report["sentences_sampled"] == 4
        assert len(rows) == 4
        assert report["usage"] == "research-only"
        assert report["redistribute"] is False

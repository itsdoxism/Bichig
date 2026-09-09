import argparse
import json
import random
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

PUBLIC_KEY = "https://yadi.sk/d/z5e3MVnKvFvF6w"
YANDEX_API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


def resolve_download_url(public_key: str = PUBLIC_KEY) -> str:
    url = YANDEX_API + "?" + urllib.parse.urlencode({"public_key": public_key})
    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.loads(r.read().decode("utf-8"))
    href = payload.get("href")
    if not href:
        raise RuntimeError("Yandex API did not return a download URL")
    return href


def download_archive(out: Path, public_key: str = PUBLIC_KEY) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    href = resolve_download_url(public_key)
    with urllib.request.urlopen(href, timeout=60) as r, out.open("wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)


def extract_archive(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        ("unrar", ["unrar", "x", "-o+", str(archive), str(out_dir) + "/"]),
        ("unar", ["unar", "-f", "-o", str(out_dir), str(archive)]),
        ("bsdtar", ["bsdtar", "-xf", str(archive), "-C", str(out_dir)]),
    ]
    for binary, cmd in candidates:
        if shutil.which(binary):
            subprocess.run(cmd, check=True)
            return
    raise RuntimeError("RAR extraction needs one of: unrar, unar, bsdtar")


def sentence_rows(article: str) -> list[str]:
    article = " ".join(article.split())
    if len(article) < 150:
        return []
    parts = [x.strip() for x in SENTENCE_RE.split(article) if x.strip()]
    if len(parts) < 6:
        return []
    return parts[1:-1]


def reservoir_add(sample: list[str], value: str, seen: int, limit: int, rng: random.Random) -> None:
    if len(sample) < limit:
        sample.append(value)
    else:
        j = rng.randrange(seen)
        if j < limit:
            sample[j] = value


def preprocess(extracted_dir: Path, out: Path, max_sentences: int = 100_000, seed: int = 422) -> dict:
    files = sorted(extracted_dir.rglob("*.txt"))
    if not files:
        raise RuntimeError(f"no .txt files found under {extracted_dir}")
    rng = random.Random(seed)
    sample: list[str] = []
    seen_sentences = 0
    articles = 0
    for path in files:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                articles += 1
                for sent in sentence_rows(line):
                    seen_sentences += 1
                    reservoir_add(sample, sent, seen_sentences, max_sentences, rng)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(x + "\n" for x in sample), encoding="utf-8")
    return {
        "source": "tugstugi/mongolian-bert 700M-word online news corpus",
        "usage": "research-only",
        "license": "unknown",
        "redistribute": False,
        "files": len(files),
        "articles_seen": articles,
        "sentences_seen": seen_sentences,
        "sentences_sampled": len(sample),
        "seed": seed,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare the tugstugi 700M-word Mongolian news corpus for private Bichig research")
    p.add_argument("--archive", default="data/base/downloads/mn_news_700m.rar")
    p.add_argument("--extracted-dir", default="data/base/downloads/mn_news_700m")
    p.add_argument("--out", default="data/base/raw/tugstugi-news-research.txt")
    p.add_argument("--report", default="data/base/raw/tugstugi-news-research-report.json")
    p.add_argument("--max-sentences", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=422)
    p.add_argument("--download", action="store_true")
    p.add_argument("--extract", action="store_true")
    a = p.parse_args()
    archive = Path(a.archive); extracted = Path(a.extracted_dir)
    if a.download and not archive.exists(): download_archive(archive)
    if a.extract and not any(extracted.rglob("*.txt")):
        if not archive.exists(): raise SystemExit("archive missing; pass --download or place it at --archive")
        extract_archive(archive, extracted)
    report = preprocess(extracted, Path(a.out), a.max_sentences, a.seed)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

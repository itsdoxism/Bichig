# Bichig Base corpus sources

Bichig Base should learn Mongolian from multiple text domains rather than one repeated transcript pool.

## Recommended v1 mix

- **Mongolian Wikipedia (`mnwiki`)** — explanatory/encyclopedic Cyrillic Mongolian. Use the `pages-articles.xml.bz2` dump and `bichig-base-wiki` to extract main-namespace text. Wikimedia content is distributed under the applicable Wikipedia Creative Commons terms; preserve attribution/source metadata when publishing derived corpora.
- **Blgn94/mongolian-stt-dataset** — 97,688 Cyrillic Khalkha transcript rows across Common Voice, FLEURS and MBSpeech. The dataset card declares the combined corpus CC-BY-4.0. Use transcripts only; aggressively deduplicate because repeated recordings can share the same sentence.
- Additional dialogue, literature, instructional and casual text should be added only with clear provenance and permission/licensing.

## Why multiple domains

Speech transcripts are useful for natural sentence patterns but can overrepresent read speech and duplicate sentences. Wikipedia gives broader concepts and explanatory prose but is formal. A future chat model needs both, plus conversational material.

## Pipeline

```bash
# Wikipedia dump -> plain text
bichig-base-wiki mnwiki-pages-articles.xml.bz2 \
  --out data/base/raw/mnwiki.txt \
  --report data/base/raw/mnwiki-report.json

# Mix any licensed text sources and build deterministic train/valid files
bichig-base-corpus data/base/raw/*.txt --out data/base/processed

# Pretrain from random initialization
bichig-base-train \
  --data data/base/processed/train.txt \
  --valid-data data/base/processed/valid.txt \
  --out runs/base-v0
```

The first target is text diversity, not raw row count. Report unique cleaned lines and validation perplexity for every experiment.

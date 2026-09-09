# Bichig

**Bichig** is an OpenField project for Mongolian script conversion, beginning with a neural Cyrillic → Traditional Mongolian script model trained from scratch.

## Goals

- No pretrained model weights
- No external tokenizer model
- Train the model from random initialization
- Learn conversion from paired Cyrillic / Traditional Mongolian text
- Keep inference small enough to run locally
- Grow later into bidirectional conversion, correction, OCR, and input tools

## Bichig Model v0

The first model is a compact encoder-decoder Transformer implemented with PyTorch. It uses a Unicode character vocabulary built directly from the training corpus, which keeps the first experiments transparent and avoids hiding linguistic behavior behind a third-party tokenizer.

### Project layout

```text
bichig/
  config.py       model configuration
  tokenizer.py    corpus-built Unicode character tokenizer
  data.py         parallel dataset + batching
  model.py        encoder-decoder Transformer
  train.py        training CLI
  infer.py        inference CLI

data/
  sample.tsv      tiny format example (not a real training corpus)
```

## Dataset format

Training data is UTF-8 TSV with one pair per line:

```text
<cyrillic text>\t<traditional mongolian text>
```

Do not treat `data/sample.tsv` as linguistically authoritative; it exists only to document the file format.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Legacy/simple mode: one file with an internal validation split
bichig-train --data data/train.tsv --out runs/bichig-v0

# Preferred experiment mode: deterministic held-out splits
bichig-prepare-data data/seed.tsv --out data/processed --valid-ratio 0.10 --test-ratio 0.10
bichig-train --train-data data/processed/train.tsv --valid-data data/processed/valid.tsv --out runs/bichig-v0
bichig-eval --checkpoint runs/bichig-v0/best.pt --data data/processed/test.tsv

bichig-infer --checkpoint runs/bichig-v0/best.pt --text "Монгол хэл"
```

For a quick architecture smoke test on a CPU, lower the model dimensions and train on a small verified dataset first.

## Bichig Base (experimental)

Bichig Base is the next branch of the project: a decoder-only Mongolian language model trained from random initialization. It learns by next-token prediction rather than hardcoded question/answer pairs.

The first tokenizer is hybrid: frequent whole words become tokens, while rare or unseen words fall back to Unicode characters. This keeps the prototype robust while leaving room for morphology-aware subwords later.

```bash
bichig-base-smoke --data data/base/sample.txt
bichig-base-generate --checkpoint runs/base-smoke/best.pt --prompt "Монгол хэл"
```

The current tiny corpus is only a pipeline smoke test. Real language ability requires a much larger, cleaned Mongolian corpus.

For real pretraining, use `bichig-base-corpus` to normalize/deduplicate Cyrillic text and `bichig-base-wiki` to extract main-namespace sentences from a Mongolian Wikipedia `pages-articles` XML dump. See `data/base/SOURCES.md` for the v1 corpus plan.

## License

Not decided yet.

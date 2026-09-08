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

bichig-train --data data/train.tsv --out runs/bichig-v0
bichig-infer --checkpoint runs/bichig-v0/best.pt --text "Монгол хэл"
```

For a quick architecture smoke test on a CPU, lower the model dimensions and train on a small verified dataset first.

## Current direction

Bichig starts intentionally simple: character-level sequence-to-sequence learning. Once the dataset and evaluation pipeline are trustworthy, we can compare subword/root-aware tokenization, linguistic constraints, bidirectional training, and specialized loss functions without changing the core objective.

## License

Not decided yet.

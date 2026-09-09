# Bichig

**Bichig** is an OpenField project for Mongolian language and script AI trained from scratch.

## Goals

- No pretrained model weights
- No external tokenizer model
- Train from random initialization
- Learn Mongolian language patterns instead of hardcoded replies
- Support Cyrillic ↔ Traditional Mongolian script conversion
- Grow into correction, OCR, input tools, and eventually chat

## Bichig Script

The first branch is a compact encoder-decoder Transformer for Cyrillic → Traditional Mongolian conversion. It currently uses a Unicode character vocabulary and a reviewed parallel corpus.

```bash
bichig-prepare-data data/seed.tsv --out data/processed --valid-ratio 0.10 --test-ratio 0.10
bichig-train --train-data data/processed/train.tsv --valid-data data/processed/valid.tsv --out runs/bichig-v0
bichig-eval --checkpoint runs/bichig-v0/best.pt --data data/processed/test.tsv
bichig-infer --checkpoint runs/bichig-v0/best.pt --text "Монгол хэл"
```

## Bichig Base (experimental)

Bichig Base is a decoder-only Mongolian language model trained from random initialization. Its objective is next-token prediction: given previous text, predict what comes next. This is the foundation for future completion and chat behavior without hardcoded question/answer rules.

The first tokenizer is hybrid:

- frequent complete words become one token
- rarer words fall back to Unicode character tokens
- spaces are explicit tokens
- unknown characters have a fallback token

This is intentionally a simple v1. A later tokenizer can add morphology-aware roots and suffix subwords.

### Tiny smoke test

```bash
bichig-base-smoke --data data/base/sample.txt
bichig-base-generate --checkpoint runs/base-smoke/best.pt --prompt "Монгол хэл"
```

The included ten-line corpus is only for verifying the learning pipeline. It is far too small for useful language generation.

### Real training direction

The next milestone is a cleaned Mongolian Cyrillic corpus large enough to teach general language patterns. Bichig Base will then be evaluated with held-out next-token loss/perplexity and qualitative completions before chat or instruction tuning is attempted.

## License

Not decided yet.

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

The Base tokenizer is a hybrid v2:

- frequent complete words become one token
- punctuation is separated from words, so `Монгол`, `Монгол.`, and `Монгол,` do not become unrelated vocabulary entries
- frequent word endings are learned automatically from corpus statistics as suffix-like subwords
- rarer stems fall back to Unicode character tokens
- spaces are explicit tokens
- unknown characters have a fallback token

The suffix layer is learned from text rather than a hardcoded grammar table. Later versions can compare this baseline against a stronger morphology-aware tokenizer.

### Tiny smoke test

```bash
bichig-base-smoke --data data/base/sample.txt
bichig-base-generate --checkpoint runs/base-smoke/best.pt --prompt "Монгол хэл"
```

The included tiny corpus is only for verifying the learning pipeline. It is far too small for useful language generation.

### Cyrillic corpus pipeline

```bash
bichig-base-corpus data/base/raw/*.txt --out data/base/processed

bichig-base-train \
  --data data/base/processed/train.txt \
  --valid-data data/base/processed/valid.txt \
  --out runs/base-v0
```

Training uses packed causal windows by default (`stride = seq-len`) instead of shifting a nearly identical window by one token every sample. This makes large-corpus pretraining much less redundant; `--stride` can be lowered for deliberate overlap experiments.

Bichig Base uses small GPT-style random initialization (`std=0.02`) with tied input/output embeddings. Validation loss/perplexity selects `best.pt`.

For real pretraining, use `bichig-base-corpus` to normalize/deduplicate Cyrillic text and `bichig-base-wiki` to extract main-namespace sentences from a Mongolian Wikipedia `pages-articles` XML dump. See `data/base/SOURCES.md` for the v1 corpus plan.

### Real training direction

The current milestone is a diverse cleaned Mongolian Cyrillic corpus: encyclopedia/explanatory text, licensed speech transcripts, literature/public-domain material, and later conversational/instruction text. Base quality is evaluated with held-out next-token loss/perplexity and qualitative completions before chat tuning.

## License

Not decided yet.

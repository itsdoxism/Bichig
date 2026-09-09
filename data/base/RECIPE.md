# Bichig Base pretraining recipe v1

The first useful Bichig Base milestone is **clean Mongolian Cyrillic language modeling**, not chat tuning.

## Corpus stages

1. Extract source text without rewriting it:
   - `bichig-base-import ...`
   - `bichig-base-wiki ...`
2. Keep provenance/license reports beside each raw extraction.
3. Build a source manifest and validate it:

```bash
bichig-base-manifest data/base/manifest.json
```

4. Merge/clean/deduplicate through the manifest:

```bash
bichig-base-corpus --manifest data/base/manifest.json --out data/base/processed
```

5. Train only on `train.txt`; choose checkpoints from `valid.txt`.

```bash
bichig-base-train \
  --data data/base/processed/train.txt \
  --valid-data data/base/processed/valid.txt \
  --out runs/base-v1 \
  --seq-len 128 \
  --d-model 256 \
  --nhead 8 \
  --layers 6 \
  --ffn 1024 \
  --batch-size 32
```

## Data mix target

Do not let one source dominate. A practical early mix is:

- encyclopedic/explanatory prose
- spoken transcripts
- edited articles and educational prose
- literature/public-domain text where provenance is clear
- later, conversational and instruction data

Deduplicate globally after merging. Clip counts are not sentence diversity counts.

## What counts as progress

Track held-out validation perplexity and qualitative completion together. A lower training loss alone is not enough. Keep a fixed prompt set such as:

- `Монгол хэл бол`
- `Улаанбаатар хотын`
- `Хүүхэд сургуульд`
- `Өнөөдөр цаг агаар`

The model should gradually produce grammatical continuations before chat tuning begins.

## Source manifest rule

Every bulk source should carry `name`, `path`, `license`, `kind`, and `language` metadata. Enabled sources with `unknown`/`unspecified` license are rejected by default. Use `--allow-unknown-license` only for an explicit private experiment; do not silently mix such data into a redistributable base corpus.

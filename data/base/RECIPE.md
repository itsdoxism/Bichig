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

## Private research recipe with tugstugi news

```bash
bichig-base-tugstugi-news --download --extract --max-sentences 100000
bichig-base-corpus --manifest data/base/manifest.research.json --allow-unknown-license --out data/base/processed-research
bichig-base-train \
  --data data/base/processed-research/train.txt \
  --valid-data data/base/processed-research/valid.txt \
  --out runs/base-news-v1 \
  --streaming \
  --vocab-lines 200000 \
  --seq-len 128 \
  --d-model 256 \
  --nhead 8 \
  --layers 6 \
  --ffn 1024 \
  --batch-size 32
```

This path is intentionally marked `research-only`; the raw news archive is not redistributed by Bichig while its underlying license remains unknown.

## Large-corpus streaming

For corpora that no longer fit comfortably in RAM, use `--streaming`. Bichig learns the tokenizer from a deterministic reservoir sample (`--vocab-lines`) and then packs fixed-length token windows directly from disk. The full tokenized corpus is never materialized in memory.

```bash
bichig-base-train \
  --data data/base/processed/train.txt \
  --valid-data data/base/processed/valid.txt \
  --out runs/base-large-v1 \
  --streaming \
  --vocab-lines 200000 \
  --seq-len 256 \
  --batch-size 32
```

Use `--steps-per-epoch` for quick experiments on very large corpora. Streaming validation is supported as well.

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

## Linguistic sanity benchmark

Perplexity alone can improve while the model still mishandles Mongolian forms. Run the built-in minimal-pair benchmark after each meaningful checkpoint:

```bash
bichig-base-benchmark \
  --checkpoint runs/base-v1/best.pt \
  --out runs/base-v1/benchmark.json
```

It scores each grammatical/sensible sentence against a deliberately malformed counterpart using mean next-token NLL. Track overall accuracy, category accuracy and the NLL margin. The tiny smoke checkpoint is roughly chance-level; a real Base model should improve as corpus quality and scale increase. The suite is a sanity check, not a substitute for a reviewed linguistic benchmark.

## Source manifest rule

Every bulk source should carry `name`, `path`, `license`, `kind`, `language`, `usage`, and `redistribute` metadata. Enabled sources with `unknown`/`unspecified` license are rejected by default. Use `--allow-unknown-license` only for an explicit private experiment; do not silently mix such data into a redistributable base corpus.

## One-command research experiment

After the research manifest points at available raw text, run the whole clean → train → eval → grammar-benchmark flow with:

```bash
bichig-base-run \
  --manifest data/base/manifest.research.json \
  --processed data/base/processed-research \
  --run-dir runs/base-v1 \
  --epochs 3 \
  --steps-per-epoch 1000
```

Resume an interrupted or continued experiment with `--resume`. Bichig reuses `last.pt`, including optimizer state, continues epoch numbering, preserves the best validation metric, and regenerates `eval.json` plus `grammar.json` after training.

# Bichig dataset

Bichig learns from paired UTF-8 text rather than pretrained weights. Dataset quality therefore matters more than raw size.

## Format

Each non-empty line contains exactly one tab:

```text
<cyrillic source>\t<traditional mongolian target>
```

Keep raw reviewed pairs separate from generated splits:

```text
data/
  raw/
    verified.tsv
  processed/
    train.tsv
    valid.tsv
    test.tsv
```

## Build the splits

After adding verified pairs to one or more raw TSV files:

```bash
bichig-prepare-data data/raw/*.tsv --out data/processed
```

The builder normalizes Unicode to NFC, removes exact duplicate rows, rejects conflicting target spellings for the same source, and creates deterministic train/validation/test splits. The default split is 80/10/10 and the default seed is `422`, so rebuilding the same corpus gives the same split.

Custom ratios are supported:

```bash
bichig-prepare-data data/raw/*.tsv --out data/processed --valid-ratio 0.05 --test-ratio 0.05
```

## Rules

1. Use verified Traditional Mongolian spellings only.
2. Keep punctuation and sentence boundaries aligned between source and target.
3. Normalize text to Unicode NFC; the builder also enforces this when producing processed data.
4. Do not silently include multiple target spellings for the same Cyrillic source. Ambiguous cases should be reviewed and represented with enough context to disambiguate them.
5. Avoid duplicate sentence pairs across train/validation/test splits.
6. Keep a locked test set that is never used to tune the model.
7. Prefer complete phrases and sentences once word-level bootstrapping is stable; context is a core reason for using a neural model.
8. Record source/provenance outside the TSV if data comes from published material, and only use material you are allowed to use.

## Validation

```bash
bichig-validate-data data/processed/train.tsv
bichig-validate-data data/processed/valid.tsv
bichig-validate-data data/processed/test.tsv
```

The validator checks malformed rows, empty values, Unicode normalization, duplicate pairs, conflicting targets, and the presence of Mongolian Unicode in targets.

## Evaluation

After training:

```bash
bichig-eval --checkpoint runs/bichig-v0/best.pt --data data/processed/test.tsv
```

The first metrics are:

- **Exact match** — proportion of target strings reproduced perfectly.
- **CER** — character error rate using Levenshtein edit distance.

These are intentionally simple baseline metrics. Later versions should add word/morpheme-aware evaluation and a manually reviewed linguistic benchmark.

## First full-corpus experiment

The first larger experiment uses the 177-row Apache-2.0 `suwanpan/mongolian-script-text` dataset snapshot. The raw snapshot is kept locally under `data/raw/hf_suwanpan/`, while the reproducible import report and cleaned/split outputs can be committed independently.

```bash
bichig-import-jsonl data/raw/hf_suwanpan/*.jsonl \
  --out data/full-clean.tsv \
  --conflicts data/full-conflicts.json \
  --report data/full-import-report.json

bichig-prepare-data data/full-clean.tsv \
  --out data/full-processed \
  --valid-ratio 0.10 \
  --test-ratio 0.10 \
  --seed 422
```

For the current snapshot, 177 raw rows become 154 unique pairs. Three Cyrillic sources have conflicting Traditional-script targets; quarantining all of those mappings leaves 147 clean pairs, split into 117 train / 15 valid / 15 test.

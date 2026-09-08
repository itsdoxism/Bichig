# Bichig dataset

Bichig learns from paired UTF-8 text rather than pretrained weights. Dataset quality therefore matters more than raw size.

## Format

Each non-empty line contains exactly one tab:

```text
<cyrillic source>\t<traditional mongolian target>
```

Keep train, validation, and test data separate:

```text
data/
  train.tsv
  valid.tsv
  test.tsv
```

## Rules

1. Use verified Traditional Mongolian spellings only.
2. Keep punctuation and sentence boundaries aligned between source and target.
3. Normalize text to Unicode NFC.
4. Do not silently include multiple target spellings for the same Cyrillic source. Ambiguous cases should be reviewed and represented with context instead.
5. Avoid duplicate sentence pairs across train/validation/test splits.
6. Keep a locked test set that is never used to tune the model.
7. Prefer complete phrases and sentences once word-level bootstrapping is stable; context is a core reason for using a neural model.
8. Record source/provenance outside the TSV if data comes from published material, and only use material you are allowed to use.

## Validation

```bash
bichig-validate-data data/train.tsv
bichig-validate-data data/valid.tsv
bichig-validate-data data/test.tsv
```

The validator checks malformed rows, empty values, Unicode normalization, duplicate pairs, conflicting targets, and the presence of Mongolian Unicode in targets.

## Evaluation

After training:

```bash
bichig-eval --checkpoint runs/bichig-v0/best.pt --data data/test.tsv
```

The first metrics are:

- **Exact match** — proportion of target strings reproduced perfectly.
- **CER** — character error rate using Levenshtein edit distance.

These are intentionally simple baseline metrics. Later versions should add word/morpheme-aware evaluation and a manually reviewed linguistic benchmark.

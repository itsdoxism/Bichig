# MonTree integration

Bichig Base can use the 2026 **MonTree / Mongolian treebank** as a small, high-quality Cyrillic Mongolian slice.

- DOI: `10.17632/7xpzh9fv8f.3`
- 6,099 annotated sentences / 80,757 tokens
- Penn Treebank-style `.tbf` files
- Mendeley Data version 3
- license: CC BY-NC 3.0 on the dataset page

After downloading the dataset files locally:

```bash
bichig-base-treebank path/to/*.tbf \
  --out data/base/raw/montree.txt \
  --report data/base/raw/montree-report.json
```

Treat MonTree as a quality-oriented research/non-commercial source unless separate permission covers broader use. It should complement, not replace, large next-token corpora such as Wikipedia, transcripts, and the research-only Tugstugi news experiment.

Evaluate checkpoints with a fixed prompt suite and held-out corpus:

```bash
bichig-base-eval \
  --checkpoint runs/base-v1/best.pt \
  --data data/base/processed/valid.txt \
  --out runs/base-v1/eval.json
```

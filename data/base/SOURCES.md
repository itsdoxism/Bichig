# Bichig Base corpus sources

Bichig Base should learn Mongolian from multiple text domains rather than one repeated transcript pool.

## Recommended v1 mix

- **Mongolian Wikipedia (`mnwiki`)** — explanatory/encyclopedic Cyrillic Mongolian. Use the `pages-articles.xml.bz2` dump and `bichig-base-wiki` to extract main-namespace text. Keep attribution/source metadata with any redistributed derived corpus.
- **Mongolian speech transcripts** — useful for natural sentence patterns, but deduplicate aggressively because multiple recordings can share one transcript.
- **Open morphology/lexical resources** — useful as auxiliary diagnostics or tokenizer guidance, not as a substitute for sentence-level language-model data.
- Additional dialogue, literature, instructional and casual text should be added only with clear provenance and permission/licensing.

## Candidate source status

| Source | Role | Status |
| --- | --- | --- |
| Mongolian Wikipedia dump | core explanatory text | enabled once dump is available locally |
| Common Voice Mongolian transcripts | spoken/read sentence diversity | eligible with its declared license/provenance |
| FLEURS / other open speech transcripts | sentence diversity | verify component license before enabling |
| `tugstugi/mongolian-bert` 700M-word news archive | very large news corpus | **disabled by default** until the underlying dataset license is verified |

The existence of a public downloader is not treated as a license. Unknown-license bulk sources stay disabled in the manifest unless explicitly overridden for a private experiment.

## Pipeline

```bash
bichig-base-manifest data/base/manifest.json
bichig-base-corpus --manifest data/base/manifest.json --out data/base/processed

bichig-base-train \
  --data data/base/processed/train.txt \
  --valid-data data/base/processed/valid.txt \
  --out runs/base-v1
```

The first target is text diversity, not raw row count. Report unique cleaned lines and validation perplexity for every experiment.

# suwanpan/mongolian-script-text snapshot

This directory is the local source location for the four JSONL files that make up the 177-row Hugging Face dataset snapshot used for Bichig's first full-corpus experiment.

Source: `suwanpan/mongolian-script-text` on Hugging Face.
License declared by the dataset: Apache-2.0.
Fields: `cyrillic` and `bicig`.

The upstream repository also contains `children_songs/shine_ogloo` without a `.jsonl` extension. It is not part of the 177 rows exposed by the Hugging Face dataset loader, so it is intentionally excluded from this snapshot.

Import with:

```bash
bichig-import-jsonl data/raw/hf_suwanpan/*.jsonl \
  --out data/full-clean.tsv \
  --conflicts data/full-conflicts.json \
  --report data/full-import-report.json
```

The importer removes exact duplicate pairs and quarantines every source string that maps to more than one target instead of arbitrarily choosing a label.

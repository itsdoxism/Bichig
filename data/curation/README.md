# Corpus curation

Bichig keeps human-reviewed corpus records separate from model-ready TSV files.

## Record format

Use `bichig-curate` with a CSV file containing:

- `id` — stable record identifier
- `source` — Cyrillic Mongolian input
- `target` — Traditional Mongolian script target
- `status` — `draft`, `review`, `verified`, or `rejected`
- `difficulty` — `easy`, `medium`, or `hard`
- `category` — e.g. `word`, `suffix`, `phrase`, `sentence`, `ambiguous`, `name`
- `provenance` — where the pair came from
- `license` — permission/license note for the source material
- `reviewer` — who verified it
- `notes` — linguistic or review notes

## Workflow

```bash
bichig-curate init data/curation/seed.csv

bichig-curate add data/curation/seed.csv \
  --source "..." \
  --target "..." \
  --status review \
  --difficulty medium \
  --category sentence \
  --provenance "manual"

bichig-curate stats data/curation/seed.csv

bichig-curate export data/curation/seed.csv --out data/seed.tsv
bichig-validate-data data/seed.tsv
bichig-smoke --data data/seed.tsv
```

Only `verified` records are exported for training. The exporter refuses conflicting verified targets for the same source.

## Seed corpus coverage

Do not make the first seed corpus 100 random easy words. Deliberately cover:

1. simple common words
2. vowel harmony and common suffix behavior
3. short noun and verb phrases
4. punctuation and sentence boundaries
5. ambiguous Cyrillic forms resolved by context
6. names and loanwords, but keep these separate from the core linguistic benchmark
7. complete short sentences

For the first smoke test, a small, fully verified corpus is better than a large noisy one.

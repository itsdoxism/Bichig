# Corpus curation

Bichig keeps human-reviewed corpus records separate from model-ready TSV files.

## Seed template

The repository includes an empty ready-to-fill template:

```text
data/curation/seed.csv
```

Each row contains:

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
# Add a candidate pair
bichig-curate add data/curation/seed.csv \
  --source "..." \
  --target "..." \
  --status review \
  --difficulty medium \
  --category sentence \
  --provenance "manual"

# See detailed counts and metadata gaps
bichig-curate stats data/curation/seed.csv

# Track the first 200-pair milestone
bichig-curate progress data/curation/seed.csv --goal 200 --coverage

# Review queue
bichig-curate list data/curation/seed.csv --status review --limit 20

# Verify one record after human review
bichig-curate set-status data/curation/seed.csv b000001 verified --reviewer "name"

# Strict metadata/conflict audit
bichig-curate audit data/curation/seed.csv

# Export only verified rows for training
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

`--coverage` shows a suggested distribution across those kinds of examples. It is a guideline rather than a linguistic rule or a hard training requirement.

For the first smoke test, a small, fully verified corpus is better than a large noisy one.

## Licensing note

Do not scrape or copy dictionary databases whose terms prohibit reuse. Keep provenance and license fields explicit. A record should become `verified` only when both linguistic correctness and reuse permission are clear.

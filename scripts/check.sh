#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

printf '[1/4] Import check\n'
"$PYTHON_BIN" - <<'PY'
import bichig
from bichig.config import ModelConfig
from bichig.tokenizer import CharTokenizer
from bichig.model import BichigTransformer
print('imports: ok')
PY

printf '[2/4] Unit tests\n'
if "$PYTHON_BIN" -c 'import pytest' >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pytest -q
else
  echo 'pytest is not installed. Run: pip install pytest'
  exit 1
fi

printf '[3/4] Dataset validator smoke check\n'
if [ -f data/sample.tsv ]; then
  "$PYTHON_BIN" -m bichig.validate_data data/sample.tsv
else
  echo 'data/sample.tsv not found; skipping validator smoke check'
fi

printf '[4/4] Optional model smoke test\n'
if [ "${BICHIG_SMOKE:-0}" = "1" ]; then
  DATASET="${BICHIG_SMOKE_DATA:-data/seed.tsv}"
  if [ ! -f "$DATASET" ]; then
    echo "Smoke dataset not found: $DATASET"
    exit 1
  fi
  "$PYTHON_BIN" -m bichig.smoke --data "$DATASET"
else
  echo 'skipped (set BICHIG_SMOKE=1 to enable)'
fi

printf 'All local checks passed.\n'

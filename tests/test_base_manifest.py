import json
import tempfile
from pathlib import Path

from bichig.base_manifest import load_manifest


def test_base_manifest_loads_source_metadata():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        text = root / "rows.txt"
        text.write_text("Монгол хэлний өгүүлбэрийг зөв бичиж сурна.\n", encoding="utf-8")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"version": 1, "sources": [{
            "name": "demo",
            "path": str(text),
            "license": "CC0-1.0",
            "enabled": True
        }]}, ensure_ascii=False), encoding="utf-8")
        rows = load_manifest(manifest)
        assert rows[0].name == "demo"
        assert rows[0].license == "CC0-1.0"

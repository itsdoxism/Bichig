from bichig.base_wiki import sentence_rows


def test_wiki_markup_extraction():
    raw = """'''Монгол хэл''' бол [[Монгол Улс|Монгол Улсын]] албан ёсны хэл юм.
{{Infobox language|name=Монгол}}
== Түүх ==
Монгол хэл олон зууны түүхтэй. <ref>source</ref>"""
    rows = sentence_rows(raw)
    assert "Монгол хэл бол Монгол Улсын албан ёсны хэл юм." in rows
    assert "Монгол хэл олон зууны түүхтэй." in rows
    assert not any("Infobox" in row or "<ref>" in row for row in rows)

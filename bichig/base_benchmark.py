import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .base_generate import load_base


DEFAULT_PROBES = [
    {"category": "case", "good": "Би сургуульд явна.", "bad": "Би сургуульдийг явна."},
    {"category": "possessive", "good": "Би номоо уншив.", "bad": "Би номын уншив."},
    {"category": "plural", "good": "Оюутнууд хичээлдээ сууж байна.", "bad": "Оюутнуудууд хичээлдээ сууж байна."},
    {"category": "vowel_harmony", "good": "Би гэрээсээ гарлаа.", "bad": "Би гэроосоо гарлаа."},
    {"category": "vowel_harmony", "good": "Тэр номоос уншлаа.", "bad": "Тэр номээс уншлаа."},
    {"category": "case", "good": "Бид Улаанбаатарт амьдардаг.", "bad": "Бид Улаанбаатарыг амьдардаг."},
    {"category": "temporal", "good": "Тэр өчигдөр ирсэн.", "bad": "Тэр өчигдөр маргааш ирнэ."},
    {"category": "negation", "good": "Би өнөөдөр явахгүй.", "bad": "Би өнөөдөр явахгүйна."},
    {"category": "word_form", "good": "Монгол хэл олон зууны түүхтэй.", "bad": "Монгол хэл олон зууны түүхтэйын."},
    {"category": "copular", "good": "Монгол Улсын нийслэл бол Улаанбаатар.", "bad": "Монгол Улсын нийслэл болыг Улаанбаатар."},
]


def sentence_nll(model, tok, text: str, device: torch.device) -> tuple[float, int]:
    """Mean next-token NLL with a rolling context window for arbitrarily long text."""
    ids = tok.encode(text, add_bos=True, add_eos=True)
    if len(ids) < 2:
        return float("inf"), 0
    max_len = int(model.config.max_len)
    losses = []
    with torch.no_grad():
        for target_pos in range(1, len(ids)):
            start = max(0, target_pos - max_len)
            context = ids[start:target_pos]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x)[:, -1, :]
            y = torch.tensor([ids[target_pos]], dtype=torch.long, device=device)
            loss = nn.functional.cross_entropy(logits, y)
            losses.append(float(loss.item()))
    return sum(losses) / len(losses), len(losses)


def load_probes(path: str | None) -> list[dict]:
    if not path:
        return list(DEFAULT_PROBES)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("probes")
    if not isinstance(data, list):
        raise SystemExit("probe file must be a JSON list or an object with a 'probes' list")
    probes = []
    for i, row in enumerate(data, 1):
        if not isinstance(row, dict) or not isinstance(row.get("good"), str) or not isinstance(row.get("bad"), str):
            raise SystemExit(f"probe #{i} requires string good/bad fields")
        probes.append({"category": str(row.get("category", "other")), "good": row["good"], "bad": row["bad"]})
    return probes


def run_benchmark(model, tok, probes: list[dict], device: torch.device) -> dict:
    rows = []
    category = {}
    wins = 0
    margins = []
    for probe in probes:
        good_nll, good_tokens = sentence_nll(model, tok, probe["good"], device)
        bad_nll, bad_tokens = sentence_nll(model, tok, probe["bad"], device)
        margin = bad_nll - good_nll
        correct = margin > 0
        wins += int(correct)
        margins.append(margin)
        cat = probe["category"]
        c = category.setdefault(cat, {"correct": 0, "total": 0})
        c["correct"] += int(correct)
        c["total"] += 1
        rows.append({
            **probe,
            "good_nll": good_nll,
            "bad_nll": bad_nll,
            "margin": margin,
            "correct": correct,
            "good_tokens": good_tokens,
            "bad_tokens": bad_tokens,
        })
    for c in category.values():
        c["accuracy"] = c["correct"] / c["total"] if c["total"] else None
    return {
        "probes": len(rows),
        "correct": wins,
        "accuracy": wins / len(rows) if rows else None,
        "mean_margin": sum(margins) / len(margins) if margins else None,
        "categories": category,
        "results": rows,
    }


def main():
    p = argparse.ArgumentParser(description="Run minimal-pair Mongolian linguistic sanity checks on Bichig Base")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--probes", help="optional JSON probe suite")
    p.add_argument("--out", default="runs/base-benchmark.json")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, _ = load_base(a.checkpoint, device)
    result = run_benchmark(model, tok, load_probes(a.probes), device)
    result["checkpoint"] = a.checkpoint
    result["device"] = str(device)
    result["vocab_size"] = len(tok)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

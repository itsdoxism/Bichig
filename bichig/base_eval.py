import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .base_generate import load_base
from .base_train import LMDataset, read_corpus

DEFAULT_PROMPTS = [
    "Монгол хэл бол",
    "Улаанбаатар хотын",
    "Хүүхэд сургуульд",
    "Өнөөдөр цаг агаар",
    "Монгол Улс нь",
    "Би өнөөдөр",
]


def corpus_loss(model, tok, texts, seq_len, batch_size, device):
    ids = []
    for text in texts:
        ids.extend(tok.encode(text, add_bos=True, add_eos=True))
    if len(ids) < 2:
        return None
    effective = min(seq_len, max(1, len(ids) - 1))
    ds = LMDataset(ids, effective, stride=effective)
    if len(ds) == 0:
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
            return nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_id)
    total = 0.0
    steps = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total += loss.item()
            steps += 1
    return total / max(steps, 1)


def generate_samples(model, tok, prompts, max_new_tokens, temperature, top_k, device, seed):
    torch.manual_seed(seed)
    rows = []
    for prompt in prompts:
        ids = torch.tensor([tok.encode(prompt, add_bos=True, add_eos=False)], dtype=torch.long, device=device)
        out = model.generate(ids, max_new_tokens, tok.eos_id, temperature, top_k)
        rows.append({"prompt": prompt, "completion": tok.decode(out[0].tolist())})
    return rows


def main():
    p = argparse.ArgumentParser(description="Evaluate Bichig Base perplexity and fixed Mongolian prompt completions")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", help="Held-out plain-text corpus")
    p.add_argument("--prompts", help="One prompt per line; defaults to the built-in fixed suite")
    p.add_argument("--out", default="runs/base-eval.json")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=422)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_base(a.checkpoint, device)
    prompts = DEFAULT_PROMPTS
    if a.prompts:
        prompts = [x.strip() for x in Path(a.prompts).read_text(encoding="utf-8").splitlines() if x.strip()]
    result = {
        "checkpoint": a.checkpoint,
        "device": str(device),
        "vocab_size": len(tok),
        "prompt_samples": generate_samples(model, tok, prompts, a.max_new_tokens, a.temperature, a.top_k, device, a.seed),
    }
    if a.data:
        texts = read_corpus(a.data)
        loss = corpus_loss(model, tok, texts, cfg.max_len, a.batch_size, device)
        result["data"] = a.data
        result["lines"] = len(texts)
        result["loss"] = loss
        result["perplexity"] = math.exp(min(loss, 20)) if loss is not None else None
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

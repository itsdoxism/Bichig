import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .base_model import BaseConfig, CausalBlockModel
from .base_tokenizer import HybridTokenizer


class LMDataset(Dataset):
    def __init__(self, ids: list[int], seq_len: int, stride: int | None = None):
        self.ids = ids
        self.seq_len = seq_len
        self.stride = stride or seq_len
        if self.stride <= 0:
            raise ValueError("stride must be > 0")

    def __len__(self):
        usable = len(self.ids) - self.seq_len - 1
        return 0 if usable < 0 else usable // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.ids[start:start + self.seq_len + 1]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])


def read_corpus(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Train Bichig Base next-token model from scratch")
    p.add_argument("--data", required=True)
    p.add_argument("--valid-data")
    p.add_argument("--out", default="runs/base-v0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--stride", type=int, help="training window stride; defaults to seq-len")
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--nhead", type=int, default=6)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ffn", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--min-word-freq", type=int, default=2)
    p.add_argument("--seed", type=int, default=422)
    return p.parse_args()


def main():
    a = parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = read_corpus(a.data)
    valid_texts = read_corpus(a.valid_data) if a.valid_data else []
    tok = HybridTokenizer.build(texts, min_word_freq=a.min_word_freq)

    ids = []
    for text in texts:
        ids.extend(tok.encode(text, add_bos=True, add_eos=True))
    if len(ids) <= a.seq_len + 1:
        raise SystemExit("corpus too small for selected --seq-len")

    stride = a.stride or a.seq_len
    ds = LMDataset(ids, a.seq_len, stride=stride)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True)

    valid_ids = []
    for text in valid_texts:
        valid_ids.extend(tok.encode(text, add_bos=True, add_eos=True))
    valid_ds = LMDataset(valid_ids, a.seq_len, stride=a.seq_len) if len(valid_ids) > a.seq_len + 1 else None
    if valid_texts and valid_ds is None:
        raise SystemExit("validation corpus too small for selected --seq-len")
    valid_loader = DataLoader(valid_ds, batch_size=a.batch_size, shuffle=False) if valid_ds else None

    cfg = BaseConfig(
        d_model=a.d_model,
        nhead=a.nhead,
        layers=a.layers,
        ffn=a.ffn,
        dropout=a.dropout,
        max_len=a.seq_len,
    )
    model = CausalBlockModel(len(tok), tok.pad_id, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_id)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(out / "vocab.json")

    print(
        f"device={device} lines={len(texts)} valid_lines={len(valid_texts)} "
        f"tokens={len(ids)} windows={len(ds)} stride={stride} vocab={len(tok)} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    best_metric = float("inf")
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        avg = total / max(1, len(loader))
        val = evaluate_loss(model, valid_loader, loss_fn, device) if valid_loader else avg
        print(
            f"epoch={epoch:03d} loss={avg:.4f} ppl={math_exp(avg):.2f} "
            f"val_loss={val:.4f} val_ppl={math_exp(val):.2f}"
        )
        ckpt = {
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "vocab": tok.itos,
            "epoch": epoch,
            "loss": avg,
            "val_loss": val,
        }
        torch.save(ckpt, out / "last.pt")
        if val < best_metric:
            best_metric = val
            torch.save(ckpt, out / "best.pt")


def evaluate_loss(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    steps = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total += loss.item()
            steps += 1
    return total / max(1, steps)


def math_exp(x):
    import math
    return math.exp(min(x, 20))


if __name__ == "__main__":
    main()

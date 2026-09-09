import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .config import ModelConfig
from .data import ParallelDataset, make_collate_fn, read_pairs
from .model import BichigTransformer
from .tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Bichig from random initialization")
    p.add_argument("--data", required=True, help="UTF-8 TSV parallel corpus")
    p.add_argument("--out", default="runs/bichig-v0")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.05)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ffn", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    return p.parse_args()


def evaluate(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    steps = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            total += loss.item()
            steps += 1
    return total / max(steps, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = read_pairs(args.data)
    tokenizer = CharTokenizer.build([p.source for p in pairs] + [p.target for p in pairs])
    config = ModelConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
        dim_feedforward=args.ffn,
        dropout=args.dropout,
        max_len=args.max_len,
    )

    dataset = ParallelDataset(pairs, tokenizer, max_len=config.max_len)
    if args.val_split <= 0 or len(dataset) <= 1:
        val_size = 0
    else:
        val_size = int(len(dataset) * args.val_split)
        val_size = max(1, min(val_size, len(dataset) - 1))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    collate = make_collate_fn(tokenizer.pad_id)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate) if val_size else None

    model = BichigTransformer(len(tokenizer), tokenizer.pad_id, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id,
        label_smoothing=args.label_smoothing,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out / "vocab.json")
    best_val = float("inf")

    print(f"device={device} pairs={len(dataset)} vocab={len(tokenizer)} parameters={sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (src, tgt) in enumerate(train_loader, start=1):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()

        train_loss = running / max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, criterion, device) if val_loader else train_loss
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        checkpoint = {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "vocab": tokenizer.itos,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, out / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, out / "best.pt")


if __name__ == "__main__":
    main()

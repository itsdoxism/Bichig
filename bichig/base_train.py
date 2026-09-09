import argparse
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

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


class StreamingLMDataset(IterableDataset):
    """Pack a line-oriented corpus into fixed token windows without loading it into RAM."""
    def __init__(self, path: str | Path, tokenizer: HybridTokenizer, seq_len: int):
        super().__init__()
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        need = self.seq_len + 1
        buf: list[int] = []
        with self.path.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f):
                if lineno % num_workers != worker_id:
                    continue
                text = raw.strip()
                if not text:
                    continue
                buf.extend(self.tokenizer.encode(text, add_bos=True, add_eos=True))
                while len(buf) >= need:
                    chunk = buf[:need]
                    del buf[:self.seq_len]
                    yield torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])


def iter_corpus(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                yield text


def read_corpus(path: str) -> list[str]:
    return list(iter_corpus(path))


def sample_corpus(path: str | Path, max_lines: int, seed: int) -> list[str]:
    """Deterministic reservoir sample used only for building the tokenizer vocabulary."""
    rng = random.Random(seed)
    sample: list[str] = []
    seen = 0
    for text in iter_corpus(path):
        seen += 1
        if len(sample) < max_lines:
            sample.append(text)
            continue
        j = rng.randrange(seen)
        if j < max_lines:
            sample[j] = text
    return sample


def learning_rate_for_update(
    base_lr: float,
    update_index: int,
    warmup_steps: int = 0,
    cosine_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> float:
    """Warm up linearly, then optionally decay with cosine toward base_lr * min_lr_ratio."""
    if base_lr <= 0:
        raise ValueError("base_lr must be > 0")
    if warmup_steps < 0 or cosine_steps < 0:
        raise ValueError("schedule steps must be >= 0")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")
    if warmup_steps and update_index < warmup_steps:
        return base_lr * (update_index + 1) / warmup_steps
    if cosine_steps <= 0:
        return base_lr
    progress = min(max((update_index - warmup_steps) / cosine_steps, 0.0), 1.0)
    multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * multiplier


def set_optimizer_lr(opt, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def parse_args():
    p = argparse.ArgumentParser(description="Train Bichig Base next-token model from scratch")
    p.add_argument("--data", required=True)
    p.add_argument("--valid-data")
    p.add_argument("--out", default="runs/base-v0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimizer update")
    p.add_argument("--warmup-steps", type=int, default=0, help="optimizer updates for linear LR warmup")
    p.add_argument("--cosine-steps", type=int, default=0, help="optimizer updates for cosine LR decay; 0 keeps LR constant after warmup")
    p.add_argument("--min-lr-ratio", type=float, default=0.1, help="final cosine LR as a fraction of --lr")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--stride", type=int, help="in-memory training window stride; defaults to seq-len")
    p.add_argument("--streaming", action="store_true", help="stream/pack corpus from disk instead of holding all token IDs in RAM")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (streaming worker-shards by line number)")
    p.add_argument("--steps-per-epoch", type=int, help="optional cap for very large streaming corpora; counts micro-batches")
    p.add_argument("--vocab-lines", type=int, default=200000, help="reservoir-sampled lines used to learn tokenizer vocabulary in streaming mode")
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--nhead", type=int, default=6)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ffn", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--min-word-freq", type=int, default=2)
    p.add_argument("--seed", type=int, default=422)
    p.add_argument("--resume", help="resume model/optimizer state from an existing checkpoint")
    return p.parse_args()


def build_loaders(a, tok):
    if a.streaming:
        train_ds = StreamingLMDataset(a.data, tok, a.seq_len)
        train_loader = DataLoader(train_ds, batch_size=a.batch_size, num_workers=a.workers)
        valid_loader = None
        if a.valid_data:
            valid_ds = StreamingLMDataset(a.valid_data, tok, a.seq_len)
            valid_loader = DataLoader(valid_ds, batch_size=a.batch_size, num_workers=a.workers)
        return train_loader, valid_loader, None

    texts = read_corpus(a.data)
    ids = []
    for text in texts:
        ids.extend(tok.encode(text, add_bos=True, add_eos=True))
    if len(ids) <= a.seq_len + 1:
        raise SystemExit("corpus too small for selected --seq-len")
    stride = a.stride or a.seq_len
    ds = LMDataset(ids, a.seq_len, stride=stride)
    train_loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers)

    valid_loader = None
    if a.valid_data:
        valid_ids = []
        for text in read_corpus(a.valid_data):
            valid_ids.extend(tok.encode(text, add_bos=True, add_eos=True))
        if len(valid_ids) <= a.seq_len + 1:
            raise SystemExit("validation corpus too small for selected --seq-len")
        valid_ds = LMDataset(valid_ids, a.seq_len, stride=a.seq_len)
        valid_loader = DataLoader(valid_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers)
    return train_loader, valid_loader, {"lines": len(texts), "tokens": len(ids), "windows": len(ds), "stride": stride}


def main():
    a = parse_args()
    if a.grad_accum < 1:
        raise SystemExit("--grad-accum must be >= 1")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if a.streaming:
        vocab_texts = sample_corpus(a.data, max(1, a.vocab_lines), a.seed)
        if not vocab_texts:
            raise SystemExit("training corpus is empty")
        print(f"tokenizer_sample_lines={len(vocab_texts)} streaming=true")
    else:
        vocab_texts = read_corpus(a.data)
        if not vocab_texts:
            raise SystemExit("training corpus is empty")

    resume_ckpt = None
    if a.resume:
        resume_ckpt = torch.load(a.resume, map_location="cpu")
        tok = HybridTokenizer(resume_ckpt["vocab"])
    else:
        tok = HybridTokenizer.build(vocab_texts, min_word_freq=a.min_word_freq)
    train_loader, valid_loader, stats = build_loaders(a, tok)

    if resume_ckpt:
        cfg = BaseConfig(**resume_ckpt["config"])
        if cfg.max_len != a.seq_len:
            raise SystemExit(f"--seq-len={a.seq_len} must match resumed checkpoint max_len={cfg.max_len}")
    else:
        cfg = BaseConfig(d_model=a.d_model, nhead=a.nhead, layers=a.layers, ffn=a.ffn, dropout=a.dropout, max_len=a.seq_len)
    model = CausalBlockModel(len(tok), tok.pad_id, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    start_epoch = 1
    best_metric = float("inf")
    global_update = 0
    if resume_ckpt:
        model.load_state_dict(resume_ckpt["model"])
        if "optimizer" in resume_ckpt:
            opt.load_state_dict(resume_ckpt["optimizer"])
            for group in opt.param_groups:
                group["weight_decay"] = a.weight_decay
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        best_metric = float(resume_ckpt.get("best_val_loss", resume_ckpt.get("val_loss", float("inf"))))
        global_update = int(resume_ckpt.get("global_update", 0))
        print(f"resumed={a.resume} from_epoch={start_epoch-1} global_update={global_update} best_val_loss={best_metric:.4f}")
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_id)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(out / "vocab.json")

    detail = "streaming=true" if a.streaming else " ".join(f"{k}={v}" for k, v in stats.items())
    effective_batch = a.batch_size * a.grad_accum
    print(
        f"device={device} {detail} vocab={len(tok)} params={sum(p.numel() for p in model.parameters()):,} "
        f"grad_accum={a.grad_accum} effective_batch={effective_batch} weight_decay={a.weight_decay}"
    )

    for epoch in range(start_epoch, start_epoch + a.epochs):
        model.train()
        total = 0.0
        micro_steps = 0
        updates_this_epoch = 0
        opt.zero_grad(set_to_none=True)
        pending = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            raw_loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            (raw_loss / a.grad_accum).backward()
            total += raw_loss.item()
            micro_steps += 1
            pending += 1

            should_update = pending >= a.grad_accum
            reached_cap = bool(a.steps_per_epoch and micro_steps >= a.steps_per_epoch)
            if should_update or reached_cap:
                lr = learning_rate_for_update(a.lr, global_update, a.warmup_steps, a.cosine_steps, a.min_lr_ratio)
                set_optimizer_lr(opt, lr)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_update += 1
                updates_this_epoch += 1
                pending = 0
            if reached_cap:
                break

        if pending:
            lr = learning_rate_for_update(a.lr, global_update, a.warmup_steps, a.cosine_steps, a.min_lr_ratio)
            set_optimizer_lr(opt, lr)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_update += 1
            updates_this_epoch += 1

        if micro_steps == 0:
            raise SystemExit("no training windows produced; reduce --seq-len or provide more corpus text")
        avg = total / micro_steps
        val = evaluate_loss(model, valid_loader, loss_fn, device) if valid_loader is not None else avg
        current_lr = opt.param_groups[0]["lr"]
        print(
            f"epoch={epoch:03d} micro_steps={micro_steps} updates={updates_this_epoch} global_update={global_update} "
            f"lr={current_lr:.6g} loss={avg:.4f} ppl={math_exp(avg):.2f} "
            f"val_loss={val:.4f} val_ppl={math_exp(val):.2f}"
        )
        ckpt = {
            "format_version": 2,
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "vocab": tok.itos,
            "epoch": epoch,
            "global_update": global_update,
            "loss": avg,
            "val_loss": val,
            "best_val_loss": min(best_metric, val),
            "optimizer": opt.state_dict(),
            "training": {
                "streaming": a.streaming,
                "seq_len": a.seq_len,
                "vocab_lines": a.vocab_lines,
                "batch_size": a.batch_size,
                "grad_accum": a.grad_accum,
                "effective_batch": effective_batch,
                "base_lr": a.lr,
                "weight_decay": a.weight_decay,
                "warmup_steps": a.warmup_steps,
                "cosine_steps": a.cosine_steps,
                "min_lr_ratio": a.min_lr_ratio,
            },
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
    if steps == 0:
        return float("inf")
    return total / steps


def math_exp(x):
    return math.exp(min(x, 20))


if __name__ == "__main__":
    main()

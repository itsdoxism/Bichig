import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="Run one Bichig Base research experiment end-to-end")
    p.add_argument("--manifest", default="data/base/manifest.research.json")
    p.add_argument("--processed", default="data/base/processed-research")
    p.add_argument("--run-dir", default="runs/base-v1")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--cosine-steps", type=int, default=2000)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--ffn", type=int, default=1024)
    p.add_argument("--vocab-lines", type=int, default=200000)
    p.add_argument("--steps-per-epoch", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-corpus", action="store_true")
    p.add_argument("--allow-unknown-license", action="store_true", default=True)
    return p.parse_args()


def build_commands(a):
    py = sys.executable
    processed = Path(a.processed)
    run_dir = Path(a.run_dir)
    commands = []
    if not a.skip_corpus:
        cmd = [py, "-m", "bichig.base_corpus", "--manifest", a.manifest, "--out", str(processed)]
        if a.allow_unknown_license:
            cmd.append("--allow-unknown-license")
        commands.append(cmd)
    train = [
        py, "-m", "bichig.base_train",
        "--data", str(processed / "train.txt"),
        "--valid-data", str(processed / "valid.txt"),
        "--out", str(run_dir), "--streaming",
        "--vocab-lines", str(a.vocab_lines),
        "--seq-len", str(a.seq_len), "--d-model", str(a.d_model),
        "--nhead", str(a.nhead), "--layers", str(a.layers),
        "--ffn", str(a.ffn), "--batch-size", str(a.batch_size),
        "--grad-accum", str(a.grad_accum),
        "--lr", str(a.lr), "--weight-decay", str(a.weight_decay),
        "--warmup-steps", str(a.warmup_steps), "--cosine-steps", str(a.cosine_steps),
        "--min-lr-ratio", str(a.min_lr_ratio),
        "--epochs", str(a.epochs),
    ]
    if a.steps_per_epoch:
        train += ["--steps-per-epoch", str(a.steps_per_epoch)]
    if a.resume and (run_dir / "last.pt").exists():
        train += ["--resume", str(run_dir / "last.pt")]
    commands.append(train)
    commands.append([
        py, "-m", "bichig.base_tokenizer_stats",
        "--vocab", str(run_dir / "vocab.json"),
        "--data", str(processed / "valid.txt"),
        "--out", str(run_dir / "tokenizer.json"),
    ])
    commands.append([py, "-m", "bichig.base_eval", "--checkpoint", str(run_dir / "best.pt"), "--data", str(processed / "valid.txt"), "--out", str(run_dir / "eval.json")])
    commands.append([py, "-m", "bichig.base_benchmark", "--checkpoint", str(run_dir / "best.pt"), "--out", str(run_dir / "grammar.json")])
    return commands


def main():
    a = parse_args()
    commands = build_commands(a)
    for cmd in commands:
        run(cmd)
    summary = {
        "run_dir": a.run_dir,
        "eval": str(Path(a.run_dir) / "eval.json"),
        "grammar": str(Path(a.run_dir) / "grammar.json"),
        "tokenizer": str(Path(a.run_dir) / "tokenizer.json"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

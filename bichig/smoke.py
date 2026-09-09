import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a tiny overfit smoke test for Bichig")
    p.add_argument("--data", required=True, help="Small verified TSV corpus (recommended: 50-200 pairs)")
    p.add_argument("--out", default="runs/smoke")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--ffn", type=int, default=256)
    p.add_argument("--max-len", type=int, default=128)
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "-m", "bichig.validate_data", args.data])
    run([
        sys.executable, "-m", "bichig.train",
        "--data", args.data,
        "--out", str(out),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--val-split", "0",
        "--d-model", str(args.d_model),
        "--nhead", str(args.nhead),
        "--layers", str(args.layers),
        "--ffn", str(args.ffn),
        "--dropout", "0",
        "--label-smoothing", "0",
        "--max-len", str(args.max_len),
    ])
    run([
        sys.executable, "-m", "bichig.evaluate",
        "--checkpoint", str(out / "last.pt"),
        "--data", args.data,
    ])

    print("\nSmoke test complete.")
    print("Target: exact match should approach 100% and CER should approach 0 on this same tiny corpus.")
    print("If it cannot overfit a clean tiny corpus, debug training/model code before scaling the dataset.")


if __name__ == "__main__":
    main()

import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/base/sample.txt")
    p.add_argument("--out", default="runs/base-smoke")
    p.add_argument("--epochs", type=int, default=30)
    a = p.parse_args()
    cmd = [
        sys.executable, "-m", "bichig.base_train",
        "--data", a.data,
        "--out", a.out,
        "--epochs", str(a.epochs),
        "--batch-size", "8",
        "--seq-len", "24",
        "--d-model", "64",
        "--nhead", "4",
        "--layers", "2",
        "--ffn", "128",
        "--dropout", "0",
        "--min-word-freq", "2",
        "--lr", "0.002",
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

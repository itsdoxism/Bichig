import argparse

import torch

from .config import ModelConfig
from .model import BichigTransformer
from .tokenizer import CharTokenizer, SPECIAL_TOKENS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Bichig inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    vocab = checkpoint["vocab"]
    tokenizer = CharTokenizer([token for token in vocab if token not in SPECIAL_TOKENS])
    config = ModelConfig(**checkpoint["config"])
    model = BichigTransformer(len(tokenizer), tokenizer.pad_id, config).to(device)
    model.load_state_dict(checkpoint["model"])

    src = torch.tensor([tokenizer.encode(args.text)], dtype=torch.long, device=device)
    output = model.generate(
        src,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        max_new_tokens=args.max_new_tokens,
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()

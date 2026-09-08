import argparse
from pathlib import Path

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


def load_model_and_tokenizer(checkpoint_path: str | Path, device: str | torch.device):
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = checkpoint["vocab"]
    tokenizer = CharTokenizer([token for token in vocab if token not in SPECIAL_TOKENS])
    config = ModelConfig(**checkpoint["config"])
    model = BichigTransformer(len(tokenizer), tokenizer.pad_id, config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer, config


@torch.no_grad()
def translate(
    model,
    tokenizer,
    text: str,
    config: ModelConfig,
    device: str | torch.device,
    max_new_tokens: int | None = None,
) -> str:
    device = torch.device(device)
    src = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
    limit = max_new_tokens if max_new_tokens is not None else config.max_len
    output = model.generate(
        src,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        max_new_tokens=limit,
    )
    return tokenizer.decode(output[0].tolist())


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, device)
    print(translate(model, tokenizer, args.text, config, device, args.max_new_tokens))


if __name__ == "__main__":
    main()

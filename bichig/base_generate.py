import argparse
import torch

from .base_model import BaseConfig, CausalBlockModel
from .base_tokenizer import HybridTokenizer, SPECIAL_TOKENS


def load_base(path, device):
    ckpt = torch.load(path, map_location=device)
    tok = HybridTokenizer([t for t in ckpt["vocab"] if t not in SPECIAL_TOKENS])
    cfg = BaseConfig(**ckpt["config"])
    model = CausalBlockModel(len(tok), tok.pad_id, cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok, cfg


def main():
    p = argparse.ArgumentParser(description="Generate text with Bichig Base")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, _ = load_base(a.checkpoint, device)
    ids = torch.tensor([tok.encode(a.prompt, add_bos=True, add_eos=False)], dtype=torch.long, device=device)
    out = model.generate(ids, a.max_new_tokens, tok.eos_id, a.temperature, a.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()

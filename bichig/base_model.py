import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class BaseConfig:
    d_model: int = 256
    nhead: int = 8
    layers: int = 6
    ffn: int = 1024
    dropout: float = 0.1
    max_len: int = 256

    def to_dict(self): return asdict(self)


class CausalBlockModel(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, config: BaseConfig):
        super().__init__()
        self.config = config
        self.pad_id = pad_id
        self.token = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(config.max_len, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.ffn,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t = x.shape
        if t > self.config.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len={self.config.max_len}")
        pos = torch.arange(t, device=x.device).unsqueeze(0)
        h = self.token(x) * math.sqrt(self.config.d_model) + self.pos(pos)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        padding = x.eq(self.pad_id)
        h = self.blocks(h, mask=causal, src_key_padding_mask=padding)
        return self.lm_head(self.norm(h))

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int, eos_id: int | None = None, temperature: float = 0.8, top_k: int = 20):
        self.eval()
        for _ in range(max_new_tokens):
            x = ids[:, -self.config.max_len:]
            logits = self(x)[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
            if eos_id is not None and torch.all(next_id.squeeze(1).eq(eos_id)):
                break
        return ids

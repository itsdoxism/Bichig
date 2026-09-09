import math

import torch
from torch import nn

from .config import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class BichigTransformer(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, config: ModelConfig):
        super().__init__()
        self.config = config
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.position = PositionalEncoding(config.d_model, config.max_len, config.dropout)
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.output = nn.Linear(config.d_model, vocab_size)
        self.scale = math.sqrt(config.d_model)

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.position(self.embedding(tokens) * self.scale)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_padding = src.eq(self.pad_id)
        tgt_padding = tgt_in.eq(self.pad_id)
        tgt_mask = torch.triu(
            torch.ones(
                tgt_in.size(1), tgt_in.size(1), dtype=torch.bool, device=tgt_in.device
            ),
            diagonal=1,
        )
        hidden = self.transformer(
            self._embed(src),
            self._embed(tgt_in),
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding,
            tgt_key_padding_mask=tgt_padding,
            memory_key_padding_mask=src_padding,
        )
        return self.output(hidden)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 256,
    ) -> torch.Tensor:
        self.eval()
        generated = torch.full(
            (src.size(0), 1), bos_id, dtype=torch.long, device=src.device
        )
        for _ in range(max_new_tokens):
            logits = self(src, generated)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token.squeeze(1).eq(eos_id)):
                break

            # Guard greedy decoding against obvious collapse loops on tiny models.
            # Stop when every sample repeats the same short suffix 3 times.
            if generated.size(1) >= 13:
                collapsed = []
                for row in generated:
                    tail = row[-12:]
                    collapsed.append(
                        bool(
                            torch.equal(tail[:4], tail[4:8])
                            and torch.equal(tail[4:8], tail[8:12])
                        )
                    )
                if all(collapsed):
                    break
        return generated

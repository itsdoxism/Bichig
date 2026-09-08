from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 256

    def to_dict(self) -> dict:
        return asdict(self)

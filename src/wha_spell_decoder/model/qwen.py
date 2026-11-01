import torch
from torch import nn

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    head_dim: int | None = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True

    def __post_init__(self):
        # Some configs omit head_dim; fall back to hidden / heads.
        if self.head_dim is None:
            object.__setattr__(
                self, "head_dim", self.hidden_size // self.num_attention_heads
            )
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_heads must be a multiple of num_key_value_heads"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Qwen3Config":
        data = dict(data)
        # Newer transformers versions nest rope_theta inside rope_parameters.
        if "rope_theta" not in data and isinstance(data.get("rope_parameters"), dict):
            data["rope_theta"] = data["rope_parameters"].get(
                "rope_theta", cls.rope_theta
            )
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen3Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_activation = nn.SiLU()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        x = self.gate_activation(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return x


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin
    return q_rotated, k_rotated

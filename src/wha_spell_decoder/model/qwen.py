import configparser
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn

from wha_spell_decoder.paths import get_cache_dir


@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True

    def __post_init__(self):
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_heads must be a multiple of num_key_value_heads"
        )

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any]
    ) -> "Qwen3Config":  # Any: untyped JSON config
        cfg = dict(data)
        # Some configs omit head_dim; fall back to hidden / heads.
        if cfg.get("head_dim") is None:
            cfg["head_dim"] = cfg["hidden_size"] // cfg["num_attention_heads"]
        # Newer transformers versions nest rope_theta inside rope_parameters.
        if "rope_theta" not in cfg and isinstance(cfg.get("rope_parameters"), dict):
            cfg["rope_theta"] = cfg["rope_parameters"].get("rope_theta", cls.rope_theta)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in known})

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen3Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x) -> torch.Tensor:
        x = self.gate_activation(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return x


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


# q and k: (B, num_heads, T, head_dim)
# cos and sin: (B, T, head_dim)
def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dimension=-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dimension)
    sin = sin.unsqueeze(unsqueeze_dimension)
    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin
    return q_rotated, k_rotated


class Qwen3RotaryEmbedding(nn.Module):
    frequencies: torch.Tensor

    def __init__(self, head_dim, theta):
        super().__init__()

        self.register_buffer(
            "frequencies",
            theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim),
            persistent=False,
        )

    # Returns cos, sin for apply_rotary
    # position_ids (B, T)
    def forward(
        self, position_ids: torch.Tensor, *, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        angles = position_ids[..., None].float() * self.frequencies.float()
        cos = torch.cos(angles).to(dtype=dtype)
        sin = torch.sin(angles).to(dtype=dtype)
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        return cos, sin


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()

        self.config = config

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, bias=False
        )
        self.q_norm = Qwen3RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.k_norm = Qwen3RMSNorm(config.head_dim, config.rms_norm_eps)
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )

        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, T, hidden_size)
        *,
        cos: torch.Tensor,  # (T, head_dim) or (B, T, head_dim)
        sin: torch.Tensor,  # (T, head_dim) or (B, T, head_dim)
        attn_mask: torch.Tensor,  # (T, T)
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        q: torch.Tensor = self.q_proj(hidden_states)
        q = q.reshape(B, T, self.config.num_attention_heads, self.config.head_dim)
        q = self.q_norm(q)
        q = q.transpose(-2, -3)
        k: torch.Tensor = self.k_proj(hidden_states)
        k = k.reshape(B, T, self.config.num_key_value_heads, self.config.head_dim)
        k = self.k_norm(k)
        k = k.transpose(-2, -3)
        v: torch.Tensor = self.v_proj(hidden_states)
        v = v.reshape(B, T, self.config.num_key_value_heads, self.config.head_dim)
        v = v.transpose(-2, -3)

        q, k = apply_rotary(q, k, cos, sin)

        k = k.repeat_interleave(
            self.config.num_attention_heads // self.config.num_key_value_heads, -3
        )
        v = v.repeat_interleave(
            self.config.num_attention_heads // self.config.num_key_value_heads, -3
        )

        # (B, num_attention_heads, T_q, head_dim)
        attn_out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )

        # # (B, num_attention_heads, T_q, T_k)
        # raw_scores = q @ k.transpose(-2, -1) * (self.config.head_dim**-0.5)
        # raw_scores = raw_scores + attn_mask
        # weights = torch.softmax(raw_scores.float(), -1).to(dtype=hidden_states.dtype)
        # (B, num_attention_heads, T_q, head_dim)
        # attn_out = weights @ v

        attn_out = attn_out.transpose(-2, -3)
        attn_out = attn_out.reshape(
            B, T, self.config.num_attention_heads * self.config.head_dim
        )
        return self.o_proj(attn_out)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()

        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, T, hidden_size)
        *,
        cos: torch.Tensor,  # (T, head_dim) or (B, T, head_dim)
        sin: torch.Tensor,  # (T, head_dim) or (B, T, head_dim)
        attn_mask: torch.Tensor,  # (T, T)
    ) -> torch.Tensor:
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(x, cos=cos, sin=sin, attn_mask=attn_mask)
        hidden_states = hidden_states + x
        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)
        return hidden_states + x


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.rope = Qwen3RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ):
        """
        Exactly one of input_ids or inputs_embeds must be provided
        - attn_mask: additive mask, if not provided, will default to a causal mask
        """
        assert (input_ids is None) != (inputs_embeds is None), (
            "Exactly one of input_ids or inputs_embeds must be provided"
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        _, T, _ = inputs_embeds.shape
        position_ids = torch.arange(T, device=inputs_embeds.device)[None, :]
        cos, sin = self.rope(position_ids, dtype=inputs_embeds.dtype)
        if attn_mask is None:
            bool_attn_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=inputs_embeds.device),
                diagonal=1,
            )
            attn_mask = torch.zeros(
                T, T, dtype=inputs_embeds.dtype, device=inputs_embeds.device
            ).masked_fill(bool_attn_mask, torch.finfo(inputs_embeds.dtype).min)[
                None, None, ...
            ]

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos=cos, sin=sin, attn_mask=attn_mask)

        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    @classmethod
    def from_pretrained(cls, model_name):
        path = Path(snapshot_download(model_name, cache_dir=get_cache_dir()))
        cfg = Qwen3Config.from_json(path / "config.json")
        state = {}
        for p in path.glob("*.safetensors"):
            with safe_open(p, framework="pt") as f:
                for k in f.keys():  # noqa: SIM118 (documented method)
                    state[k] = f.get_tensor(k)
        if cfg.tie_word_embeddings:
            state["lm_head.weight"] = state["model.embed_tokens.weight"]

        model = cls(cfg)
        model.load_state_dict(state)
        return model

    def __init__(self, config: Qwen3Config):
        super().__init__()

        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_state = self.model(
            input_ids=input_ids, inputs_embeds=inputs_embeds, attn_mask=attn_mask
        )
        return self.lm_head(hidden_state)

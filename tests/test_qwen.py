from typing import TypedDict

import torch
from transformers import Qwen3Config as HFQwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention as HFAttention,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer as HFDecoderLayer,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP as HFMLP
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Model as HFModel,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding as HFRotaryEmbedding,
)
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb as hf_apply_rotary,
)

from wha_spell_decoder.model.qwen import (
    Qwen3Attention,
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3Model,
    Qwen3RotaryEmbedding,
    apply_rotary,
)


class TinyConfig(TypedDict):
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    vocab_size: int


TINY: TinyConfig = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "num_hidden_layers": 2,
    "vocab_size": 100,
}
ROPE_THETA = 1_000_000.0
hf_cfg = HFQwen3Config(
    **TINY, rope_parameters={"rope_type": "default", "rope_theta": ROPE_THETA}
)
my_cfg = Qwen3Config.from_dict({**TINY, "rope_theta": ROPE_THETA})


def test_mlp_matches_hf():
    torch.manual_seed(0)
    hf = HFMLP(hf_cfg)
    mine = Qwen3MLP(my_cfg.hidden_size, my_cfg.intermediate_size)
    mine.load_state_dict(hf.state_dict())

    x = torch.randn(2, 5, TINY["hidden_size"])
    torch.testing.assert_close(mine(x), hf(x))


def test_rotary_matches_hf():
    torch.manual_seed(0)

    hf = HFRotaryEmbedding(hf_cfg)
    mine = Qwen3RotaryEmbedding(my_cfg.head_dim, my_cfg.rope_theta)

    q = torch.randn(2, TINY["num_attention_heads"], 100, TINY["head_dim"])
    k = torch.randn(2, TINY["num_key_value_heads"], 100, TINY["head_dim"])
    position_ids = torch.arange(100)

    hf_cos, hf_sin = hf(q, position_ids)
    my_cos, my_sin = mine(position_ids, dtype=q.dtype)
    torch.testing.assert_close(hf_cos, my_cos)
    torch.testing.assert_close(hf_sin, my_sin)

    hf_q, hf_k = hf_apply_rotary(q, k, hf_cos, hf_sin, unsqueeze_dim=-3)
    my_q, my_k = apply_rotary(q, k, my_cos, my_sin)
    torch.testing.assert_close(my_q, hf_q)
    torch.testing.assert_close(my_k, hf_k)


def test_attn_matches_hf():
    torch.manual_seed(0)

    hf_attn = HFAttention(hf_cfg, 0)
    my_attn = Qwen3Attention(my_cfg)
    my_attn.load_state_dict(hf_attn.state_dict())

    x = torch.randn(2, 10, TINY["hidden_size"])
    position_ids = torch.arange(10)[None, :]

    hf_rot = HFRotaryEmbedding(hf_cfg)
    my_rot = Qwen3RotaryEmbedding(my_cfg.head_dim, my_cfg.rope_theta)
    hf_cos, hf_sin = hf_rot(x, position_ids)
    my_cos, my_sin = my_rot(position_ids, dtype=x.dtype)

    bool_attn_mask = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
    attn_mask = torch.zeros_like(bool_attn_mask, dtype=torch.float).masked_fill(
        bool_attn_mask, float("-inf")
    )[None, None, ...]

    torch.testing.assert_close(
        hf_attn(x, (hf_cos, hf_sin), attn_mask)[0],
        my_attn(x, cos=my_cos, sin=my_sin, attn_mask=attn_mask),
    )


def test_decoder_matches_hf():
    torch.manual_seed(0)

    hf_decoder = HFDecoderLayer(hf_cfg, 0)
    my_decoder = Qwen3DecoderLayer(my_cfg)
    my_decoder.load_state_dict(hf_decoder.state_dict())

    x = torch.randn(2, 10, TINY["hidden_size"])
    position_ids = torch.arange(10)[None, :]

    hf_rot = HFRotaryEmbedding(hf_cfg)
    my_rot = Qwen3RotaryEmbedding(my_cfg.head_dim, my_cfg.rope_theta)
    hf_cos, hf_sin = hf_rot(x, position_ids)
    my_cos, my_sin = my_rot(position_ids, dtype=x.dtype)

    bool_attn_mask = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
    attn_mask = torch.zeros_like(bool_attn_mask, dtype=torch.float).masked_fill(
        bool_attn_mask, float("-inf")
    )[None, None, ...]

    torch.testing.assert_close(
        hf_decoder(x, attention_mask=attn_mask, position_embeddings=(hf_cos, hf_sin)),
        my_decoder(x, cos=my_cos, sin=my_sin, attn_mask=attn_mask),
    )


def test_model_matches_hf():
    torch.manual_seed(0)

    hf_model = HFModel(hf_cfg)
    my_model = Qwen3Model(my_cfg)
    my_model.load_state_dict(hf_model.state_dict())

    input_ids = torch.randint(0, TINY["vocab_size"], (2, 10))
    hf_inputs_embeds = hf_model.embed_tokens(input_ids)
    my_inputs_embeds = my_model.embed_tokens(input_ids)

    torch.testing.assert_close(
        hf_model(input_ids=input_ids).last_hidden_state, my_model(input_ids=input_ids)
    )
    torch.testing.assert_close(
        hf_model(inputs_embeds=hf_inputs_embeds).last_hidden_state,
        my_model(inputs_embeds=my_inputs_embeds),
    )


def test_from_dict_defaults_head_dim():
    data = {k: v for k, v in TINY.items() if k != "head_dim"}
    cfg = Qwen3Config.from_dict(data)
    assert cfg.head_dim == TINY["hidden_size"] // TINY["num_attention_heads"]

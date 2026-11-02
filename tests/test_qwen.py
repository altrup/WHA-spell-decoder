import torch
from transformers import Qwen3Config as HFQwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
	Qwen3MLP as HFMLP,
	Qwen3RotaryEmbedding as HFRotaryEmbedding,
	apply_rotary_pos_emb as hf_apply_rotary
)

from wha_spell_decoder.model.qwen import Qwen3MLP, Qwen3Config, Qwen3RotaryEmbedding, apply_rotary

TINY = {
	"hidden_size": 64,
	"intermediate_size": 128,
	"num_attention_heads": 4,
	"rope_theta": 1_000_000.0,
	"num_key_value_heads": 2,
	"head_dim": 32,
	"num_hidden_layers": 2,
	"vocab_size": 100,
}
hf_cfg = HFQwen3Config(**TINY)
my_cfg = Qwen3Config.from_dict(TINY)


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
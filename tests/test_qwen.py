import torch
from transformers import Qwen3Config as HFQwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP as HFMLP, Qwen3RMSNorm as HFRMSNorm

from wha_spell_decoder.model.qwen import Qwen3MLP, Qwen3RMSNorm, Qwen3Config

TINY = dict(hidden_size=64, intermediate_size=128, num_attention_heads=4,
            num_key_value_heads=2, head_dim=32, num_hidden_layers=2, vocab_size=100)
hf_cfg = HFQwen3Config(**TINY)
my_cfg = Qwen3Config.from_dict(TINY)


def test_mlp_matches_hf():
	torch.manual_seed(0)
	hf = HFMLP(hf_cfg)
	mine = Qwen3MLP(my_cfg.hidden_size, my_cfg.intermediate_size)
	mine.load_state_dict(hf.state_dict())

	x = torch.randn(2, 5, TINY["hidden_size"])
	assert torch.allclose(mine(x), hf(x))
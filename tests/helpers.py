from typing import TypedDict

from transformers import Qwen3Config as HFQwen3Config

from wha_spell_decoder.model.qwen import Qwen3Config


class TinyConfig(TypedDict):
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    vocab_size: int
    tie_word_embeddings: bool


TINY: TinyConfig = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "num_hidden_layers": 2,
    "vocab_size": 100,
    "tie_word_embeddings": True,
}
ROPE_THETA = 1_000_000.0
hf_cfg = HFQwen3Config(
    **TINY, rope_parameters={"rope_type": "default", "rope_theta": ROPE_THETA}
)
my_cfg = Qwen3Config.from_dict({**TINY, "rope_theta": ROPE_THETA})

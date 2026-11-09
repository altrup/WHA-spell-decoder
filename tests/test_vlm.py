import math

import torch

from tests.helpers import TINY, my_cfg
from wha_spell_decoder.model.vlm import SpellVLM


def test_output_shape():
    torch.manual_seed(0)

    BATCHES = 2
    CHANNELS = 3
    IMAGE_DIM = 225
    TEXT_TOKENS = 10

    vlm = SpellVLM(my_cfg)
    image = torch.randn(BATCHES, CHANNELS, 225, 225)
    input_ids = torch.randint(0, TINY["vocab_size"], (BATCHES, TEXT_TOKENS))
    logits: torch.Tensor = vlm(image=image, input_ids=input_ids)

    output_dim = IMAGE_DIM
    for _ in range(6):
        # Each downstep halves size
        output_dim = math.ceil(output_dim / 2)
    output_tokens = output_dim**2

    assert logits.shape == (BATCHES, output_tokens + TEXT_TOKENS, my_cfg.vocab_size)

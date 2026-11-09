import math

import torch

from wha_spell_decoder.model.encoder import ImageEncoder


def test_output_shape():
    torch.manual_seed(0)

    BATCHES = 2
    CHANNELS = 3
    D_MODEL = 10
    IMAGE_DIM = 225

    encoder = ImageEncoder(d_model=D_MODEL)
    image = torch.randn(BATCHES, CHANNELS, IMAGE_DIM, IMAGE_DIM)
    encoded_image: torch.Tensor = encoder(image)

    output_dim = IMAGE_DIM
    for _ in range(6):
        # Each downstep halves size
        output_dim = math.ceil(output_dim / 2)
    output_tokens = output_dim**2

    assert (
        encoded_image.shape[0] == BATCHES
        and encoded_image.shape[-2] == output_tokens
        and encoded_image.shape[-1] == D_MODEL
    )

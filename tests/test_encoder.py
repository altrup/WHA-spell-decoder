import torch

from wha_spell_decoder.model.encoder import ImageEncoder


def test_output_shape():
    torch.manual_seed(0)

    BATCHES = 2
    CHANNELS = 3
    D_MODEL = 10

    encoder = ImageEncoder(d_model=D_MODEL)
    image = torch.randn(BATCHES, CHANNELS, 225, 225)
    encoded_image: torch.Tensor = encoder(image)

    assert encoded_image.shape[0] == BATCHES and encoded_image.shape[-1] == D_MODEL

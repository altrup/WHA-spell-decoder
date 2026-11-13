import torch

from wha_spell_decoder.data.dataset import collate


def test_collate_shape():
    batch = collate(
        [
            {
                "image": torch.randn(3, 10, 20),
                "input_ids": torch.randn(100),
                "labels": torch.randn(100),
            },
            {
                "image": torch.randn(3, 20, 15),
                "input_ids": torch.randn(30),
                "labels": torch.randn(30),
            },
        ],
        pad_id=0,
    )

    assert (
        batch["image"].shape == (2, 3, 20, 20)
        and batch["input_ids"].shape == (2, 100)
        and batch["labels"].shape == (2, 100)
    )

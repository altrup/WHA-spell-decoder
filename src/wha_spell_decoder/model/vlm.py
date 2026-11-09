from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from torch import nn
from torch.fx.experimental.migrate_gradual_types.z3_types import dim

from wha_spell_decoder.model.encoder import ImageEncoder
from wha_spell_decoder.model.qwen import Qwen3Config, Qwen3ForCausalLM
from wha_spell_decoder.paths import get_cache_dir


class SpellVLM(nn.Module):
    @classmethod
    def from_pretrained(cls, model_name):
        path = Path(snapshot_download(model_name, cache_dir=get_cache_dir()))
        cfg = Qwen3Config.from_json(path / "config.json")

        model = cls(cfg, decoder=Qwen3ForCausalLM.from_pretrained(model_name))
        return model

    def __init__(self, config: Qwen3Config, *, decoder: Qwen3ForCausalLM | None = None):
        super().__init__()

        self.encoder = ImageEncoder(d_model=config.hidden_size)
        self.decoder = decoder if decoder is not None else Qwen3ForCausalLM(config)

    def forward(
        self,
        *,
        image: torch.Tensor,  # (B, 3, H, W)
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        assert input_ids is None or inputs_embeds is None, (
            "Must provide at most 1 of either input_ids or inputs_embeds"
        )

        image_embeds: torch.Tensor = self.encoder(image)

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.decoder.model.embed_tokens(input_ids)

        if inputs_embeds is None:
            inputs_embeds = image_embeds
        else:
            inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=-2)

        return self.decoder(inputs_embeds=inputs_embeds)

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from wha_spell_decoder.model.vlm import SpellVLM
from wha_spell_decoder.paths import get_cache_dir


def load_tokenizer(model_name: str) -> Tokenizer:
    path = Path(snapshot_download(model_name, cache_dir=get_cache_dir()))
    return Tokenizer.from_file(str(path / "tokenizer.json"))


def load_model(model_name: str, *, checkpoint: dict | None = None) -> SpellVLM:
    vlm = SpellVLM.from_pretrained(model_name)
    if checkpoint is not None:
        result = vlm.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
        assert not result.unexpected_keys

    return vlm


def load_optimizer(
    optimizer: torch.optim.Optimizer, *, checkpoint: dict | None = None
) -> torch.optim.Optimizer:
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return optimizer


def save_checkpoint(
    path: Path,
    *,
    model_name: str,
    model: SpellVLM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    eval_loss: float,
    best_eval_loss: float,
):
    torch.save(
        {
            "model_name": model_name,
            "trainable_state_dict": {
                name: param.detach()
                for name, param in model.named_parameters()
                if param.requires_grad
            },
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": json.loads(json.dumps(vars(args), default=str)),
            "eval_loss": eval_loss,
            "best_eval_loss": best_eval_loss,
        },
        path,
    )

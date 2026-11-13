import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from wha_spell_decoder.consts import IGNORE_INDEX
from wha_spell_decoder.paths import get_data_dir


class Record(TypedDict):
    image: str
    kind: str
    name: str
    source: str
    model: str
    usable: str
    description: str
    justification: str
    effect: str


def preprocess_image(path: Path, *, max_pixels=1920 * 1080) -> torch.Tensor:
    """
    loads image at Path, scales down to fit max_pixels (if necessary), and converts to tensor (3, H, W)
    """
    with Image.open(path) as img:
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img).convert("RGB")

        w, h = img.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        img_tensor = torch.from_numpy(np.array(object=img))  # (H, W, 3)
        img_tensor = (
            img_tensor.permute(-1, -3, -2).float() / 255
        )  # (3, H, W) normalized to 0-1

        return img_tensor


class SpellDataset(Dataset):
    def __init__(
        self, record_files: list[Path], tokenizer: Tokenizer, *, eos_str="<|endoftext|>"
    ):
        data_dir = get_data_dir()
        assert data_dir is not None, "A data dir must be specified"
        self.data_dir = data_dir

        self.tokenizer = tokenizer

        eos_id = self.tokenizer.token_to_id(eos_str)
        assert eos_id is not None, "eos id could not be found"
        self.eos_id = eos_id

        prompt = "# Description\n"
        self.prompt_ids = torch.tensor(self.tokenizer.encode(prompt).ids)

        self.records: list[Record] = []
        for record_file in record_files:
            with open(record_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record: Record = json.loads(line)
                    self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        img_tensor = preprocess_image(self.data_dir / record["image"])

        target = "\n".join(
            [
                record["description"],
                "# Justification",
                record["justification"],
                "# Effect",
                record["effect"],
            ]
        )
        target_ids = self.tokenizer.encode(target).ids + [self.eos_id]
        target_ids_tensor = torch.tensor(target_ids)

        input_ids = torch.cat([self.prompt_ids, target_ids_tensor], dim=-1)
        labels = input_ids.clone()
        labels[: len(self.prompt_ids)] = IGNORE_INDEX

        return {
            "image": img_tensor,  # (3, H, W)
            "input_ids": input_ids,  # (L,)
            "labels": labels,  # (L,)
        }


def collate(batch: list[dict], *, pad_id: int) -> dict:
    max_h = max(item["image"].shape[1] for item in batch)
    max_w = max(item["image"].shape[2] for item in batch)
    max_L = max(len(item["input_ids"]) for item in batch)

    img_tensor = torch.stack(
        [
            F.pad(
                item["image"],
                (0, max_w - item["image"].shape[2], 0, max_h - item["image"].shape[1]),
                value=1.0,
            )
            for item in batch
        ]
    )
    input_ids = torch.stack(
        [
            F.pad(item["input_ids"], (0, max_L - len(item["input_ids"])), value=pad_id)
            for item in batch
        ]
    )
    labels = torch.stack(
        [
            F.pad(item["labels"], (0, max_L - len(item["labels"])), value=IGNORE_INDEX)
            for item in batch
        ]
    )

    return {
        "image": img_tensor,  # (B, 3, H, W)
        "input_ids": input_ids,  # (B, L)
        "labels": labels,  # (B, L)
    }

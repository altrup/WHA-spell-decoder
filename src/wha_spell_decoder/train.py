import argparse
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Subset

from wha_spell_decoder.data.dataset import SpellDataset, collate
from wha_spell_decoder.paths import get_cache_dir, get_data_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="WHA Spell Train",
        description="Trains the WHA Spell Decoder",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Load a checkpoint to train from, if not provided will start with initial values",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="What model to load as the main decoder",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Size of each batch",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.05,
        help="Fraction of data to reserve for eval",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for pytorch random numbers"
    )

    return parser


def load_tokenizer(model_name: str) -> Tokenizer:
    path = Path(snapshot_download(model_name, cache_dir=get_cache_dir()))
    return Tokenizer.from_file(str(path / "tokenizer.json"))


def main() -> None:
    data_dir = get_data_dir()
    assert data_dir is not None, "Could not find data dir"

    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.model)

    dataset = SpellDataset(
        [
            data_dir / "synth" / "records.jsonl",
            data_dir / "processed" / "records.jsonl",
        ],
        tokenizer,
    )

    name_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(dataset.records):
        name_to_indices[record["name"]].append(i)
    names = sorted(name_to_indices)
    perm = torch.randperm(
        len(names), generator=torch.Generator().manual_seed(args.seed)
    )
    shuffled_names = [names[i] for i in perm.tolist()]
    n_eval = int(args.eval_fraction * len(names))
    train_names = shuffled_names[n_eval:]
    eval_names = shuffled_names[:n_eval]
    train_set = Subset(
        dataset, [i for name in train_names for i in name_to_indices[name]]
    )
    eval_set = Subset(
        dataset, [i for name in eval_names for i in name_to_indices[name]]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate, pad_id=dataset.eos_id),
        num_workers=4,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate, pad_id=dataset.eos_id),
        num_workers=4,
    )

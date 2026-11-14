import argparse
from functools import partial
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from wha_spell_decoder.data.dataset import SpellDataset, collate, split_by_name
from wha_spell_decoder.model.checkpoint import load_model, load_tokenizer
from wha_spell_decoder.paths import get_data_dir


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


def load_data(
    tokenizer: Tokenizer, *, eval_fraction, batch_size, seed
) -> tuple[DataLoader[SpellDataset], DataLoader[SpellDataset]]:
    data_dir = get_data_dir()
    assert data_dir is not None, "Could not find data dir"

    dataset = SpellDataset(
        [
            data_dir / "synth" / "records.jsonl",
            data_dir / "processed" / "records.jsonl",
        ],
        tokenizer,
    )

    train_set, eval_set = split_by_name(dataset, eval_fraction=eval_fraction, seed=seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate, pad_id=dataset.eos_id),
        num_workers=4,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate, pad_id=dataset.eos_id),
        num_workers=4,
    )
    return train_loader, eval_loader


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if (args.model is None) != (args.checkpoint is None):
        parser.error("Exactly one of --model or --checkpoint must be provided")
    checkpoint = (
        torch.load(args.checkpoint, map_location="cpu")
        if args.checkpoint is not None
        else None
    )

    if checkpoint is not None:
        name: str = checkpoint["model_name"]
    else:
        assert args.model is not None
        name: str = args.model

    tokenizer = load_tokenizer(name)
    model = load_model(name, checkpoint=checkpoint)

    train_loader, eval_loader = load_data(
        tokenizer,
        eval_fraction=args.eval_fraction,
        batch_size=args.batch_size,
        seed=args.seed,
    )

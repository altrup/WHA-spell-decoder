import argparse
import shutil
from functools import partial
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import nn, optim
from torch.utils.data import DataLoader

from wha_spell_decoder.data.dataset import (
    DataPoint,
    SpellDataset,
    collate,
    split_by_name,
)
from wha_spell_decoder.model.checkpoint import (
    load_model,
    load_optimizer,
    load_tokenizer,
    save_checkpoint,
)
from wha_spell_decoder.paths import get_checkpoints_dir, get_data_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="WHA Spell Train",
        description="Trains the WHA Spell Decoder",
    )

    parser.add_argument(
        "--device",
        type=torch.device,
        default=None,
        help="Specify a specific device to use (otherwise defaults to GPU if possible)",
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
        "--encoder-learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for encoder's optimizer",
    )
    parser.add_argument(
        "--decoder-learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate for decoder's optimizer",
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
        "--unfreeze-decoder-epoch",
        type=int,
        default=25,
        help="Epoch at which to unfreeze decoder",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.05,
        help="Fraction of data to reserve for eval",
    )
    parser.add_argument(
        "--checkpoint-epoch-interval",
        type=int,
        default=5,
        help="Save epoch_N.pt every N epochs (latest.pt and best.pt are always kept)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for pytorch random numbers"
    )

    return parser


def load_data(
    tokenizer: Tokenizer, *, eval_fraction, batch_size, seed
) -> tuple[DataLoader[SpellDataset], DataLoader[SpellDataset]]:
    data_dir = get_data_dir()

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
        shuffle=False,
        collate_fn=partial(collate, pad_id=dataset.eos_id),
        num_workers=4,
    )
    return train_loader, eval_loader


def optimizer_params(optimizer: optim.Optimizer) -> list[torch.Tensor]:
    return [p for group in optimizer.param_groups for p in group["params"]]


def train_model(
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch: DataPoint,
    device: torch.device,
) -> float:
    """
    Returns training loss
    """
    image = batch["image"].to(device)
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    model.zero_grad()

    predictions: torch.Tensor = model(image=image, input_ids=input_ids)
    loss = F.cross_entropy(predictions[..., :-1, :].transpose(-1, -2), labels)

    loss.backward()
    nn.utils.clip_grad_norm_(optimizer_params(optimizer), 1.0)
    optimizer.step()

    return loss.item()


def eval_model(
    *,
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Returns eval loss
    """
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_batches = 0
        for batch in eval_loader:
            batch = cast(DataPoint, batch)
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            predictions: torch.Tensor = model(image=image, input_ids=input_ids)
            loss = F.cross_entropy(predictions[..., :-1, :].transpose(-1, -2), labels)

            total_loss += loss.item()
            total_batches += 1
    return total_loss / total_batches


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.device is not None:
        device: torch.device = args.device
    else:
        device = torch.accelerator.current_accelerator(True)
        if device is None:
            device = torch.device("cpu")

    if (args.model is None) == (args.checkpoint is None):
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

    # TODO Check if new args are different (if checkpoint provided)

    tokenizer = load_tokenizer(name)
    model = load_model(name, checkpoint=checkpoint).to(device)

    train_loader, eval_loader = load_data(
        tokenizer,
        eval_fraction=args.eval_fraction,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    start_epoch = checkpoint["epoch"] + 1 if checkpoint is not None else 0

    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())
    model.decoder.requires_grad_(False)
    optimizer = optim.AdamW(encoder_params, lr=args.encoder_learning_rate)
    if start_epoch > args.unfreeze_decoder_epoch:
        model.decoder.requires_grad_(True)
        optimizer.add_param_group(
            {"params": decoder_params, "lr": args.decoder_learning_rate}
        )
    optimizer = load_optimizer(optimizer, checkpoint=checkpoint)

    best_eval_loss = (
        float("inf") if checkpoint is None else checkpoint["best_eval_loss"]
    )
    for epoch in range(start_epoch, args.epochs):
        if epoch == args.unfreeze_decoder_epoch:
            model.decoder.requires_grad_(True)
            optimizer.add_param_group(
                {"params": decoder_params, "lr": args.decoder_learning_rate}
            )

        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            training_loss = train_model(
                model=model, optimizer=optimizer, batch=batch, device=device
            )
            running_loss += training_loss
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch} Step {i + 1}: Loss {running_loss / 10:.4f}")
                running_loss = 0.0
        eval_loss = eval_model(model=model, eval_loader=eval_loader, device=device)
        print(f"Epoch {epoch}: Loss {eval_loss}")

        # Save to checkpoint(s)
        checkpoint_dir = get_checkpoints_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_paths = []
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_paths.append(checkpoint_dir / "best.pt")
        if (
            args.checkpoint_epoch_interval > 0
            and epoch % args.checkpoint_epoch_interval == 0
        ):
            save_paths.append(checkpoint_dir / f"epoch{epoch}.pt")
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model_name=name,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            eval_loss=eval_loss,
            best_eval_loss=best_eval_loss,
        )
        for path in save_paths:
            shutil.copyfile(checkpoint_dir / "latest.pt", path)

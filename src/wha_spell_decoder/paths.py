# src/wha_spell_decoder/paths.py
import os
from pathlib import Path

from dotenv import load_dotenv


def find_project_root() -> Path | None:
    """Walk up from this file until a folder containing pyproject.toml is found."""
    for folder in Path(__file__).resolve().parents:
        if (folder / "pyproject.toml").exists():
            return folder
    return None


def get_checkpoints_dir(*, epoch: int, step: int) -> Path | None:
    root = find_project_root()
    if root is None:
        return None

    load_dotenv(root / ".env")  # no error if the file doesn't exist
    if env := os.environ.get("WHA_CHECKPOINTS_DIR"):
        return Path(env) / f"epoch{epoch}" / f"step{step}"
    return root / "checkpoints" / f"epoch{epoch}" / f"step{step}"


def get_cache_dir() -> Path | None:
    """
    Where downloaded models are stored, in order of priority:
    1. the WHA_CACHE_DIR environment variable, if set
    2. <project root>/.cache, if the project root can be found
    3. None, meaning huggingface_hub's default cache (~/.cache/huggingface)
    """
    root = find_project_root()
    if root is not None:
        load_dotenv(root / ".env")  # no error if the file doesn't exist
    if env := os.environ.get("WHA_CACHE_DIR"):
        return Path(env)
    return root / ".cache" if root is not None else None


def get_data_dir() -> Path | None:
    root = find_project_root()
    if root is not None:
        load_dotenv(root / ".env")  # no error if the file doesn't exist
    if env := os.environ.get("WHA_DATA_DIR"):
        return Path(env)
    return root / "data" if root is not None else None

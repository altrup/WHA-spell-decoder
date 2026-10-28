from pathlib import Path

from transformers import AutoProcessor, AutoModelForMultimodalLM

# Project root/.cache
CACHE = Path(__file__).resolve().parents[2] / ".cache"
MODEL_ID = "google/gemma-4-E2B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)

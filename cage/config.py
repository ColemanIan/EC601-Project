import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("CAGE_DATA", Path(__file__).resolve().parents[1] / "data"))
RAW = DATA_ROOT / "raw"
PROCESSED = DATA_ROOT / "processed"
CACHE = DATA_ROOT / "cache"
RUNS = Path(os.environ.get("CAGE_RUNS", DATA_ROOT.parent / "runs"))
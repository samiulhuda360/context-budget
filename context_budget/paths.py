"""Project folders. CTXB_HOME moves all of them (default: the repository root)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("CTXB_HOME", Path(__file__).resolve().parent.parent))
BENCH = ROOT / "bench"  # generated benchmark conversations (not committed; `ctxb bench` rebuilds them)
CACHE = ROOT / "cache"  # recorded model replies (committed, so CI replays offline)
EVAL = ROOT / "eval"  # evaluation results (committed)

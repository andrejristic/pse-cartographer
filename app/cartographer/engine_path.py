"""
Locate the shared PSE-A engine and put it on sys.path (C0).

PSE Cartographer does NOT fork or copy the engine — it imports the exact same `pse_a`
package that powers the PSE J-space analyzer. This module finds that package and makes
`import pse_a` work, so both apps evolve from one core.

Resolution order:
  1. $PSE_ENGINE env var (an explicit path to the folder that CONTAINS `pse_a/`),
  2. the sibling default: ../../PSE J-space/app  (relative to this file),
  3. a short upward search for a directory containing `pse_a/`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _candidates():
    env = os.environ.get("PSE_ENGINE")
    if env:
        yield Path(env)
    here = Path(__file__).resolve()
    # .../PSE Cartographer/app/cartographer/engine_path.py -> parents[2] = PSE Cartographer
    pse_root = here.parents[3]                       # .../PSE
    yield pse_root / "PSE J-space" / "app"
    # upward search fallback
    for up in here.parents:
        for name in ("PSE J-space", "PSE-J-space"):
            p = up / name / "app"
            if (p / "pse_a").is_dir():
                yield p


def ensure_engine_on_path() -> Path:
    """Insert the engine dir on sys.path and return it. Raises if not found."""
    for cand in _candidates():
        try:
            if cand and (cand / "pse_a").is_dir():
                p = str(cand)
                if p not in sys.path:
                    sys.path.insert(0, p)
                return cand
        except OSError:
            continue
    raise ImportError(
        "Could not locate the PSE-A engine (a folder containing `pse_a/`). "
        "Set the PSE_ENGINE environment variable to the PSE J-space `app/` directory, "
        "e.g. export PSE_ENGINE=\"$HOME/Desktop/PSE/PSE J-space/app\"."
    )

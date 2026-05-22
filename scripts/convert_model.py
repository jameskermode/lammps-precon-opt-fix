#!/usr/bin/env python
"""Convert the MACE foundation model to the Symmetrix JSON format.

Runs ``symmetrix_extract_mace`` (installed with the symmetrix Python package)
over the atomic numbers needed by the test structures and writes the result to
``models/mace-matpes-symmetrix.json`` — the path ``calculators.py`` expects.

Run inside the project venv:
    module load foss/2023b
    source .venv/bin/activate
    python scripts/convert_model.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
TARGET_JSON = MODELS_DIR / "mace-matpes-symmetrix.json"

# H, O, Mg, Al, Si, La — every element across the test structures.
ATOMIC_NUMBERS = [1, 8, 12, 13, 14, 57]

# Candidate locations of the MACE .model file (already on this machine).
MODEL_CANDIDATES = [
    Path.home() / "isg2026-amentum" / "MACE-matpes-pbe-omat-ft.model",
    Path.home() / ".cache" / "mace" / "MACE-matpes-pbe-omat-ft.model",
]


def find_model() -> Path:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    sys.exit(f"MACE .model not found in any of: {MODEL_CANDIDATES}")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    model_src = find_model()
    model = MODELS_DIR / model_src.name
    if not model.exists():
        print(f"Copying {model_src} -> {model}")
        shutil.copy2(model_src, model)

    if TARGET_JSON.exists():
        print(f"{TARGET_JSON} already exists — nothing to do.")
        return

    print(f"Converting {model.name} for atomic numbers {ATOMIC_NUMBERS} ...")
    subprocess.check_call(
        ["symmetrix_extract_mace", "--model", model.name,
         "--atomic-numbers", *map(str, ATOMIC_NUMBERS)],
        cwd=MODELS_DIR,
    )

    produced = sorted(MODELS_DIR.glob("MACE-matpes-pbe-omat-ft-*.json"))
    if not produced:
        sys.exit("symmetrix_extract_mace did not produce a JSON file")
    produced[-1].rename(TARGET_JSON)
    print(f"Wrote {TARGET_JSON}")


if __name__ == "__main__":
    main()

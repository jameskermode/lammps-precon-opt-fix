#!/usr/bin/env python
"""Convert the MACE foundation model to the Symmetrix JSON format.

Runs ``symmetrix_extract_mace`` (installed with the symmetrix Python package)
over the atomic numbers needed by the test structures and writes the result to
``models/mace-matpes-symmetrix.json`` — the path ``calculators.py`` expects.

Run inside the project venv, passing the path to a downloaded MACE model:
    python scripts/convert_model.py /path/to/MACE-matpes-pbe-omat-ft.model
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

MODEL_NAME = "MACE-matpes-pbe-omat-ft.model"


def find_model(cli_path: str | None) -> Path:
    """Locate the MACE .model file — an explicit path, else the mace cache."""
    candidates = []
    if cli_path:
        candidates.append(Path(cli_path).expanduser())
    candidates.append(Path.home() / ".cache" / "mace" / MODEL_NAME)
    for path in candidates:
        if path.exists():
            return path
    sys.exit(
        f"MACE .model not found (looked in: {[str(p) for p in candidates]}).\n"
        f"Pass the path to your downloaded model:\n"
        f"  python scripts/convert_model.py /path/to/{MODEL_NAME}")


def main() -> None:
    cli_path = sys.argv[1] if len(sys.argv) > 1 else None
    MODELS_DIR.mkdir(exist_ok=True)
    model_src = find_model(cli_path)
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

"""Saving and loading of per-stage validation artifacts.

Each stage writes into ``artifacts/<stage>/<structure>/``; later stages and
the test suite read these back as the parity targets.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts"


def stage_dir(stage: str, name: str) -> Path:
    """Return (creating if needed) ``artifacts/<stage>/<name>/``."""
    d = ARTIFACT_DIR / stage / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True))


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def save_sparse(path: Path, matrix: sparse.spmatrix) -> None:
    sparse.save_npz(str(path), matrix.tocsr())


def load_sparse(path: Path) -> sparse.csr_matrix:
    return sparse.load_npz(str(path))


def save_arrays(path: Path, **arrays: np.ndarray) -> None:
    np.savez_compressed(str(path), **arrays)


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(str(path)) as data:
        return {k: data[k] for k in data.files}

"""Environment-readiness checks shared by the stage test modules.

The tests need the built LAMMPS + Symmetrix environment; this module lets them
skip cleanly (instead of erroring) when ``scripts/build_lammps.sh`` /
``scripts/convert_model.py`` have not been run yet.
"""
from __future__ import annotations

import importlib

from lammps_precon.calculators import MODEL_JSON


def _can_import(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


LAMMPS_OK = _can_import("lammps")
LAMMPS_REASON = "lammps Python module not installed — run scripts/build_lammps.sh"

MACE_OK = LAMMPS_OK and _can_import("symmetrix") and MODEL_JSON.exists()
MACE_REASON = (
    "LAMMPS + Symmetrix + MACE model not ready — run scripts/build_lammps.sh "
    "and scripts/convert_model.py"
)

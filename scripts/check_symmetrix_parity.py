#!/usr/bin/env python
"""Cross-check: LAMMPS+Symmetrix vs mace-torch on the same structure.

Confirms the Symmetrix MACE pair style faithfully reproduces the mace-torch
reference evaluation (they are independent implementations of the same model,
so they agree to ~1e-4, not bit-exactly). This underpins the choice to use
LAMMPS+Symmetrix as the force engine for the validation stages.

    module load foss/2023b
    source .venv/bin/activate
    python scripts/check_symmetrix_parity.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

from lammps_precon.calculators import REPO_ROOT, symmetrix_mace_calculator
from lammps_precon.structures import mgo_supercell

MACE_MODEL = REPO_ROOT / "models" / "MACE-matpes-pbe-omat-ft.model"
ENERGY_ATOL_PER_ATOM = 1e-4  # eV/atom
FORCE_ATOL = 1e-4            # eV/A


def main() -> int:
    from mace.calculators import MACECalculator

    atoms = mgo_supercell(2)  # rattled MgO, 64 atoms, non-zero forces
    n = len(atoms)

    a1 = atoms.copy()
    a1.calc = MACECalculator(model_paths=str(MACE_MODEL), device="cpu",
                             default_dtype="float64")
    e_mace = a1.get_potential_energy()
    f_mace = a1.get_forces()

    a2 = atoms.copy()
    a2.calc = symmetrix_mace_calculator(a2)
    e_sym = a2.get_potential_energy()
    f_sym = a2.get_forces()
    if a2.calc.lmp is not None:
        a2.calc.lmp.close()

    de = abs(e_mace - e_sym)
    df = float(np.abs(f_mace - f_sym).max())
    print(f"structure          : rattled MgO 2x2x2 ({n} atoms)")
    print(f"energy  mace-torch : {e_mace:.6f} eV")
    print(f"energy  symmetrix  : {e_sym:.6f} eV")
    print(f"energy  difference : {de:.2e} eV  ({de / n:.2e} eV/atom)")
    print(f"max force difference: {df:.2e} eV/A")

    ok = (de / n) < ENERGY_ATOL_PER_ATOM and df < FORCE_ATOL
    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)  # hard exit: LAMMPSlib/Kokkos crash the teardown

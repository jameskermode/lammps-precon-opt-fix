#!/usr/bin/env python
"""Plot fmax vs. number of force calls — ASE vs the C++ LAMMPS plugin.

For each Packwood test structure, runs two fixed-cell relaxations:
  * ASE   — `PreconLBFGS` + `Exp` (the reference), forces from LAMMPS/MACE;
  * LAMMPS — the C++ plugin `min_style precon/lbfgs`.
Both convergence traces (max force component vs cumulative force evaluations)
are saved to `artifacts/figures/` and drawn as a 2x2 panel figure.

Run inside the project venv (needs the Symmetrix/MACE environment):
    python scripts/plot_convergence.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.calculators.loggingcalc import LoggingCalculator  # noqa: E402
from ase.optimize.precon import Exp, PreconLBFGS  # noqa: E402

from lammps_precon import artifacts  # noqa: E402
from lammps_precon.calculators import make_calculator  # noqa: E402
from lammps_precon.cpp_parity import run_lammps_cpp  # noqa: E402
from lammps_precon.structures import by_name  # noqa: E402

# the Packwood benchmark structures
PACKWOOD = ["Si_slab", "LaAlO3", "gamma_Al2O3", "iceVIII"]
A_EXP, C_STAB = 3.0, 0.1  # Exp preconditioner parameters (match relax.py)
MAX_STEPS = 2000


def ase_trace(structure) -> list[tuple[int, float]]:
    """ASE PreconLBFGS+Exp relaxation -> [(force_calls, fmax), ...]."""
    atoms = structure.atoms.copy()
    logging_calc = LoggingCalculator(make_calculator(atoms, structure.engine))
    atoms.calc = logging_calc
    opt = PreconLBFGS(atoms, precon=Exp(A=A_EXP, c_stab=C_STAB, solver="direct"),
                      logfile=None, use_armijo=True)
    raw: list[tuple[int, float]] = []

    def record():
        forces = atoms.get_forces()  # cached — the optimizer just evaluated it
        n_calls = sum(len(v) for v in logging_calc.fmax.values())
        raw.append((n_calls, float(np.linalg.norm(forces, axis=1).max())))

    record()                          # the initial point
    opt.attach(record, interval=1)    # one record per accepted step
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt.run(fmax=structure.fmax, steps=MAX_STEPS)
    # one (force_calls, fmax) per distinct cumulative force-call count
    return sorted(dict(raw).items())


def lammps_trace(structure) -> list[tuple[int, float]]:
    """C++ `min_style precon/lbfgs` relaxation -> [(force_calls, fmax), ...]."""
    with tempfile.TemporaryDirectory() as tmp:
        res = run_lammps_cpp(structure.atoms.copy(), structure.engine,
                             fmax=structure.fmax, maxiter=MAX_STEPS,
                             workdir=Path(tmp), trace=True)
    return [tuple(p) for p in res["trace"]]


def main() -> None:
    tracedir = artifacts.ARTIFACT_DIR / "figures"      # generated trace JSON
    tracedir.mkdir(parents=True, exist_ok=True)
    docsdir = artifacts.ARTIFACT_DIR.parent / "docs"   # committed figure
    docsdir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    for ax, name in zip(axes.flat, PACKWOOD):
        structure = by_name(name)
        natoms = len(structure.atoms)
        print(f"[plot_convergence] {name} ({natoms} atoms) ...", flush=True)
        ase_t = ase_trace(structure)
        lmp_t = lammps_trace(structure)
        artifacts.save_json(tracedir / f"{name}_convergence.json",
                            {"ase": ase_t, "lammps": lmp_t})
        print(f"    ASE    {ase_t[-1][0]:4d} force calls -> fmax {ase_t[-1][1]:.2e}")
        print(f"    LAMMPS {lmp_t[-1][0]:4d} force calls -> fmax {lmp_t[-1][1]:.2e}")

        ax.semilogy(*zip(*ase_t), "o-", ms=4, color="C0",
                    label="ASE  (PreconLBFGS + Exp)")
        ax.semilogy(*zip(*lmp_t), "s--", ms=4, color="C1",
                    label="LAMMPS  (min_style precon/lbfgs)")
        ax.axhline(structure.fmax, ls=":", color="grey", lw=1)
        ax.set_title(f"{name}  ({natoms} atoms)")
        ax.set_xlabel("number of force calls")
        ax.set_ylabel("fmax  (eV / Å)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle("Geometry-optimisation convergence: ASE vs the C++ LAMMPS "
                 "plugin — Packwood test set", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        path = docsdir / f"convergence.{ext}"
        fig.savefig(path, dpi=150)
        print(f"wrote {path}")


if __name__ == "__main__":
    try:
        main()
        code = 0
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)

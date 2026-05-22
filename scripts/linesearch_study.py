#!/usr/bin/env python
"""Line-search study — can the Python PreconLBFGS match the LAMMPS minimizer?

The C++ `min_style precon/lbfgs` converges in fewer steps than ASE's
`PreconLBFGS` on most Packwood structures. Both use an *identical*
preconditioned-LBFGS two-loop recursion, so the difference is the line search.
Reading the two implementations, the structural difference is the cap on the
per-atom displacement:

  * ASE   `PreconLBFGS(maxstep=...)`     defaults to 0.04 Angstrom
  * LAMMPS `min_modify dmax ...`         defaults to 0.10 Angstrom

ASE's cap is 2.5x more conservative, so each ASE step covers less ground. This
sweeps ASE's `maxstep` and tabulates force-call counts against the C++ plugin,
to see whether matching LAMMPS's step cap recovers its convergence rate.

Run inside the project venv (Symmetrix/MACE environment):
    python scripts/linesearch_study.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

from ase.calculators.loggingcalc import LoggingCalculator
from ase.optimize.precon import Exp, PreconLBFGS

from lammps_precon import artifacts
from lammps_precon.calculators import make_calculator
from lammps_precon.cpp_parity import run_lammps_cpp
from lammps_precon.structures import by_name

PACKWOOD = ["Si_slab", "LaAlO3", "gamma_Al2O3", "iceVIII"]
A_EXP, C_STAB = 3.0, 0.1
MAXSTEPS = [0.04, 0.1, 0.2]   # 0.04 = ASE default; 0.10 = LAMMPS dmax default


def ase_force_calls(structure, maxstep: float) -> tuple[int, int, bool]:
    """ASE PreconLBFGS+Exp at the given maxstep -> (force calls, steps, ok)."""
    atoms = structure.atoms.copy()
    logging_calc = LoggingCalculator(make_calculator(atoms, structure.engine))
    atoms.calc = logging_calc
    opt = PreconLBFGS(atoms, precon=Exp(A=A_EXP, c_stab=C_STAB, solver="direct"),
                      maxstep=maxstep, logfile=None, use_armijo=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        converged = bool(opt.run(fmax=structure.fmax, steps=2000))
    n_force = sum(len(v) for v in logging_calc.fmax.values())
    return n_force, opt.nsteps, converged


def lammps_force_calls(structure) -> tuple[int, bool]:
    """C++ min_style precon/lbfgs -> (force calls, converged)."""
    with tempfile.TemporaryDirectory() as tmp:
        res = run_lammps_cpp(structure.atoms.copy(), structure.engine,
                             fmax=structure.fmax, maxiter=2000,
                             workdir=Path(tmp))
    return res["n_force"], res["converged"]


def main() -> None:
    rows = []
    for name in PACKWOOD:
        s = by_name(name)
        print(f"[linesearch_study] {name} ({len(s.atoms)} atoms)", flush=True)
        row = {"name": name, "n_atoms": len(s.atoms)}
        for ms in MAXSTEPS:
            nf, nsteps, conv = ase_force_calls(s, ms)
            row[f"ase_maxstep_{ms}"] = nf
            print(f"    ASE  maxstep={ms:<5} {nf:4d} force calls "
                  f"({nsteps} steps, converged={conv})", flush=True)
        lnf, lconv = lammps_force_calls(s)
        row["lammps"] = lnf
        print(f"    LAMMPS  min_style precon/lbfgs  {lnf:4d} force calls "
              f"(converged={lconv})", flush=True)
        rows.append(row)

    d = artifacts.stage_dir("stage8", "_linesearch")
    artifacts.save_json(d / "linesearch_study.json", rows)

    print("\n=== force calls to the fmax tolerance ===")
    print(f"{'structure':<14}" + "".join(f"ms={m:<8}" for m in MAXSTEPS)
          + "LAMMPS")
    for r in rows:
        line = f"{r['name']:<14}"
        for m in MAXSTEPS:
            line += f"{r[f'ase_maxstep_{m}']:<11}"
        line += str(r["lammps"])
        print(line)


if __name__ == "__main__":
    try:
        main()
        code = 0
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)

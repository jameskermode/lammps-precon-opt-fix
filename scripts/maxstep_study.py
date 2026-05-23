#!/usr/bin/env python
"""Empirical study — automated / auto-tuned step cap (maxstep / dmax).

Sweeps every existing preconditioned-LBFGS variant on the Packwood structures
and asks: is there a line-search choice that converges *without* a tightly
tuned step cap?

  * ASE PreconLBFGS + Exp     x use_armijo = {True (Armijo), False (strong Wolfe)}
  * LAMMPS min_style precon/lbfgs  x min_modify line = {backtrack, quadratic, forcezero}

Each at three step caps: 0.04 (ASE default), 0.10 (LAMMPS default), and 1.00
(effectively "off" — ASE rejects maxstep > 1.0 and 1.0 Å/atom is well past any
reasonable single LBFGS step). Records force-call counts to convergence (or
to MAXEVAL for runs that don't converge) and prints a markdown table per
structure.

Run inside the project venv (Symmetrix/MACE environment):
    python scripts/maxstep_study.py
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
CAPS = [0.04, 0.10, 1.00]              # ASE-default, LAMMPS-default, "off"
A_EXP, C_STAB = 3.0, 0.1
MAX_STEPS = 500                        # plenty for any converging precon-LBFGS
MAXEVAL = 1000                         # hard ceiling for divergent runs


def ase_run(structure, use_armijo: bool, maxstep: float) -> tuple[int, bool]:
    atoms = structure.atoms.copy()
    lc = LoggingCalculator(make_calculator(atoms, structure.engine))
    atoms.calc = lc
    opt = PreconLBFGS(atoms,
                      precon=Exp(A=A_EXP, c_stab=C_STAB, solver="direct"),
                      maxstep=maxstep, use_armijo=use_armijo, logfile=None)
    converged = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converged = bool(opt.run(fmax=structure.fmax, steps=MAX_STEPS))
    except Exception:
        pass
    return sum(len(v) for v in lc.fmax.values()), converged


def lammps_run(structure, line_style: str, dmax: float) -> tuple[int, bool]:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            res = run_lammps_cpp(
                structure.atoms.copy(), structure.engine,
                fmax=structure.fmax, maxiter=MAX_STEPS, maxeval=MAXEVAL,
                workdir=Path(tmp), line_style=line_style, dmax=dmax)
        return int(res["n_force"] or -1), bool(res["converged"])
    except Exception:
        return 0, False


# (label, runner)
ROWS = [
    ("ASE     Armijo",      lambda s, c: ase_run(s, True,  c)),
    ("ASE     Wolfe",       lambda s, c: ase_run(s, False, c)),
    ("LAMMPS  backtrack",   lambda s, c: lammps_run(s, "backtrack", c)),
    ("LAMMPS  quadratic",   lambda s, c: lammps_run(s, "quadratic", c)),
    ("LAMMPS  forcezero",   lambda s, c: lammps_run(s, "forcezero", c)),
]


def fmt(n: int, ok: bool) -> str:
    return f"{n:>3d}{'' if ok else '*'}"


def main() -> None:
    out: dict = {}
    for name in PACKWOOD:
        s = by_name(name)
        print(f"\n### {name}  ({len(s.atoms)} atoms)\n", flush=True)
        print(f"| {'line search':<20} | "
              + " | ".join(f"cap={c:<4}" for c in CAPS) + " |", flush=True)
        print(f"|-{'-' * 20}-|-" + "-|-".join(["-" * 7] * len(CAPS)) + "-|",
              flush=True)
        per_struct: dict = {}
        for label, runner in ROWS:
            cells = []
            for cap in CAPS:
                print(f"  {name}  {label}  cap={cap} ...",
                      end="", flush=True)
                n, ok = runner(s, cap)
                cells.append({"n_force": n, "converged": ok})
                print(f" -> {n}{'' if ok else ' (no converge)'}",
                      flush=True)
            per_struct[label] = {f"cap={c}": cells[i]
                                 for i, c in enumerate(CAPS)}
            cols = "  | ".join(fmt(x["n_force"], x["converged"]).rjust(7)
                                for x in cells)
            print(f"| {label:<20} |  {cols} |", flush=True)
        out[name] = per_struct
        print(f"(* = did not converge within MAXEVAL={MAXEVAL})\n", flush=True)
    d = artifacts.stage_dir("stage8", "_maxstep_study")
    artifacts.save_json(d / "maxstep_study.json", out)
    print(f"\nWrote {d / 'maxstep_study.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
        code = 0
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)

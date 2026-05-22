#!/usr/bin/env python
"""Plot fmax vs. number of force calls — preconditioned vs. plain optimisers.

For each Packwood test structure, runs four fixed-cell relaxations and plots
the max force component against the cumulative number of force evaluations:

  preconditioned (converge quickly):
    * ASE     PreconLBFGS + Exp
    * LAMMPS  min_style precon/lbfgs   (the C++ plugin)
  un-preconditioned (slow — capped at UNPRECON_BUDGET force calls):
    * ASE     LBFGS
    * LAMMPS  min_style cg

The gap between the two groups is the benefit of preconditioning. Traces are
saved to `artifacts/figures/`; the figure to `docs/convergence.{png,pdf}`.

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
from ase.calculators.loggingcalc import LoggingCalculator  # noqa: E402
from ase.optimize import LBFGS  # noqa: E402
from ase.optimize.precon import Exp, PreconLBFGS  # noqa: E402

from lammps_precon import artifacts  # noqa: E402
from lammps_precon.calculators import make_calculator  # noqa: E402
from lammps_precon.cpp_parity import run_lammps_cpp  # noqa: E402
from lammps_precon.structures import by_name  # noqa: E402

PACKWOOD = ["Si_slab", "LaAlO3", "gamma_Al2O3", "iceVIII"]
A_EXP, C_STAB = 3.0, 0.1     # Exp preconditioner parameters
ASE_PRECON_MAXSTEP = 0.1     # match LAMMPS' dmax (ASE's 0.04 default is slow —
                             # see scripts/linesearch_study.py)
MAX_STEPS = 2000             # iteration cap for the (fast) preconditioned runs
UNPRECON_BUDGET = 300        # force-call cap for the (slow) plain runs


def _force_call_trace(logging_calc) -> list[tuple[int, float]]:
    """Per-force-call fmax sequence recorded by an ASE LoggingCalculator."""
    seq: list[float] = []
    for vals in logging_calc.fmax.values():
        seq.extend(vals)
    return [(i, f) for i, f in enumerate(seq, start=1)]


def _running_min(trace: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Lowest fmax reached within N force calls — a clean, monotone curve
    (the raw per-call trace zig-zags through every line-search trial)."""
    out: list[tuple[int, float]] = []
    best = float("inf")
    for n, f in trace:
        best = min(best, f)
        out.append((n, best))
    return out


def ase_trace(structure, preconditioned: bool) -> list[tuple[int, float]]:
    """ASE relaxation -> [(force_calls, fmax), ...] (one point per force call)."""
    atoms = structure.atoms.copy()
    logging_calc = LoggingCalculator(make_calculator(atoms, structure.engine))
    atoms.calc = logging_calc
    if preconditioned:
        opt = PreconLBFGS(atoms, precon=Exp(A=A_EXP, c_stab=C_STAB,
                                            solver="direct"),
                          maxstep=ASE_PRECON_MAXSTEP,
                          logfile=None, use_armijo=True)
        steps = MAX_STEPS
    else:
        opt = LBFGS(atoms, logfile=None)   # ~1 force call per step
        steps = UNPRECON_BUDGET
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt.run(fmax=structure.fmax, steps=steps)
    return _force_call_trace(logging_calc)


def lammps_trace(structure, min_style: str) -> list[tuple[int, float]]:
    """LAMMPS relaxation -> [(force_calls, fmax), ...] (one point per call)."""
    maxeval = None if min_style == "precon/lbfgs" else UNPRECON_BUDGET
    with tempfile.TemporaryDirectory() as tmp:
        res = run_lammps_cpp(structure.atoms.copy(), structure.engine,
                             fmax=structure.fmax, maxiter=MAX_STEPS,
                             workdir=Path(tmp), trace=True,
                             min_style=min_style, maxeval=maxeval)
    return [tuple(p) for p in res["trace"]]


# (label, cache key, line style, colour, trace getter)
CURVES = [
    ("ASE  PreconLBFGS + Exp", "ase_precon", "-", "C0",
     lambda s: ase_trace(s, preconditioned=True)),
    ("LAMMPS  min_style precon/lbfgs", "lammps_precon", "-", "C1",
     lambda s: lammps_trace(s, "precon/lbfgs")),
    ("ASE  LBFGS  (no precon)", "ase_plain", "--", "C3",
     lambda s: ase_trace(s, preconditioned=False)),
    ("LAMMPS  min_style cg  (no precon)", "lammps_cg", "--", "C4",
     lambda s: lammps_trace(s, "cg")),
]


def main() -> None:
    tracedir = artifacts.ARTIFACT_DIR / "figures"
    tracedir.mkdir(parents=True, exist_ok=True)
    docsdir = artifacts.ARTIFACT_DIR.parent / "docs"
    docsdir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    for ax, name in zip(axes.flat, PACKWOOD):
        structure = by_name(name)
        natoms = len(structure.atoms)
        print(f"[plot_convergence] {name} ({natoms} atoms)", flush=True)
        # one cache file per (structure, curve) — delete a file to recompute it
        for label, key, ls, colour, getter in CURVES:
            cache = tracedir / f"{name}__{key}.json"
            if cache.exists():
                trace = [tuple(p) for p in artifacts.load_json(cache)]
            else:
                trace = getter(structure)
                artifacts.save_json(cache, trace)
            curve = _running_min(trace)
            ax.semilogy(*zip(*curve), ls, color=colour, lw=1.6, label=label)
            print(f"    {label:34s} {curve[-1][0]:4d} calls -> "
                  f"fmax {curve[-1][1]:.2e}", flush=True)
        ax.axhline(structure.fmax, ls=":", color="grey", lw=1)
        ax.set_title(f"{name}  ({natoms} atoms)")
        ax.set_xlabel("number of force calls")
        ax.set_ylabel("fmax  (eV / Å)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=7)

    fig.suptitle("Geometry-optimisation convergence — preconditioned vs. "
                 "plain optimisers (Packwood test set)", fontsize=12)
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

"""Stage 5 — fixed-cell preconditioned-relaxation parity.

Runs full ``PreconLBFGS`` relaxations on LAMMPS forces, with the ``Exp``
preconditioner's linear solve done by our Stage-4 two-tier solver, and compares
the result against ASE's stock ``Exp`` preconditioned LBFGS.

Three relaxations per structure:

* ``ase``    — ASE's stock ``Exp`` (solve via ``scipy.spsolve``);
* ``direct`` — our ``TwoTierExp`` forced onto the direct tier;
* ``cg``     — our ``TwoTierExp`` forced onto the Jacobi-CG tier.

The preconditioner assembly and ``mu`` are identical across all three (Stages
2-3), so the only thing under test is the solve inside the optimisation loop.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from ase.calculators.loggingcalc import LoggingCalculator
from ase.geometry import find_mic
from ase.io import write
from ase.optimize.precon import Exp, PreconLBFGS

from . import artifacts
from .calculators import make_calculator
from .solve import DOF_THRESHOLD, solve, solve_cg, solve_direct
from .structures import TestStructure, by_name

A_DEFAULT = 3.0
C_STAB_DEFAULT = 0.1
CG_RTOL = 1e-8
# Per-atom step cap. With an Armijo-based line search (ASE's default, or
# LAMMPS' backtrack/quadratic) the Armijo condition alone keeps the relaxation
# safe — see scripts/maxstep_study.py: cap = 1.0 A converges in fewer force
# calls than ASE's 0.04 default on every Packwood structure, with no
# divergence. 1.0 is the maximum ASE allows (`maxstep > 1.0` is rejected by
# PreconLBFGS) and is effectively "no cap" for any reasonable LBFGS step.
MAXSTEP = 1.0

#: Fixed-cell test structures (gamma-Al2O3 is the variable-cell Stage-6 case).
FIXED_CELL_CASES = ["Cu_fcc", "MgO_x2", "Si_slab", "LaAlO3"]


class TwoTierExp(Exp):
    """ASE's ``Exp`` preconditioner with the linear solve routed through the
    Stage-4 two-tier solver (`lammps_precon.solve`)."""

    def __init__(self, *, A=A_DEFAULT, c_stab=C_STAB_DEFAULT,
                 solver_tier="auto", cg_rtol=CG_RTOL,
                 dof_threshold=DOF_THRESHOLD, **kwargs):
        kwargs.setdefault("solver", "direct")  # never build ASE's pyamg solver
        super().__init__(A=A, c_stab=c_stab, **kwargs)
        self.solver_tier = solver_tier
        self.cg_rtol = cg_rtol
        self.dof_threshold = dof_threshold

    def solve(self, x):
        if self.solver_tier == "direct":
            s, _ = solve_direct(self.P, x)
        elif self.solver_tier == "cg":
            s, _ = solve_cg(self.P, x, rtol=self.cg_rtol)
        else:
            s, _ = solve(self.P, x, dof_threshold=self.dof_threshold,
                         cg_rtol=self.cg_rtol)
        return s


@dataclass
class RelaxationRun:
    converged: bool
    n_steps: int
    n_force: int
    e_final: float
    fmax_final: float
    positions: np.ndarray


def run_relaxation(structure: TestStructure, precon, *,
                    steps: int = 300) -> RelaxationRun:
    """Run one PreconLBFGS relaxation of ``structure`` with ``precon``."""
    atoms = structure.atoms.copy()
    base_calc = make_calculator(atoms, structure.engine)
    logging_calc = LoggingCalculator(base_calc)
    atoms.calc = logging_calc
    try:
        opt = PreconLBFGS(atoms, precon=precon, maxstep=MAXSTEP,
                          logfile=None, use_armijo=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converged = bool(opt.run(fmax=structure.fmax, steps=steps))
        forces = atoms.get_forces()
        return RelaxationRun(
            converged=converged,
            n_steps=opt.nsteps,
            n_force=sum(len(v) for v in logging_calc.fmax.values()),
            e_final=float(atoms.get_potential_energy()),
            fmax_final=float(np.linalg.norm(forces, axis=1).max()),
            positions=atoms.get_positions(),
        )
    finally:
        lmp = getattr(base_calc, "lmp", None)
        if lmp is not None:
            try:
                lmp.close()
            except Exception:
                pass


def _rmsd(structure: TestStructure, a: np.ndarray, b: np.ndarray) -> float:
    """Minimum-image RMSD between two sets of positions."""
    delta, _ = find_mic(a - b, structure.atoms.cell, structure.atoms.pbc)
    return float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1))))


@dataclass
class RelaxationParity:
    name: str
    engine: str
    n_atoms: int
    fmax_tol: float
    ase_converged: bool
    ase_n_steps: int
    ase_n_force: int
    ase_e_final: float
    ase_fmax_final: float
    direct_converged: bool
    direct_n_force: int
    direct_energy_diff: float
    direct_rmsd: float
    cg_converged: bool
    cg_n_force: int
    cg_energy_diff: float
    cg_rmsd: float
    parity_ok: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def compare_relaxation(structure: TestStructure,
                       *, save: bool = True) -> RelaxationParity:
    """Run the Stage-5 relaxation parity check for one fixed-cell structure."""
    ase_run = run_relaxation(
        structure, Exp(A=A_DEFAULT, c_stab=C_STAB_DEFAULT, solver="direct"))
    direct_run = run_relaxation(
        structure, TwoTierExp(solver_tier="direct"))
    cg_run = run_relaxation(
        structure, TwoTierExp(solver_tier="cg", cg_rtol=CG_RTOL))

    direct_energy_diff = abs(direct_run.e_final - ase_run.e_final)
    direct_rmsd = _rmsd(structure, direct_run.positions, ase_run.positions)
    cg_energy_diff = abs(cg_run.e_final - ase_run.e_final)
    cg_rmsd = _rmsd(structure, cg_run.positions, ase_run.positions)

    # The direct tier is numerically equivalent to ASE's solve, so it must
    # reproduce the relaxation essentially exactly. The CG tier solves each
    # step only to a finite tolerance, so small line-search-level differences
    # in the force-evaluation count are acceptable.
    force_tol = max(3, int(0.25 * ase_run.n_force))
    parity_ok = (
        ase_run.converged and direct_run.converged and cg_run.converged
        and direct_energy_diff < 1e-6
        and direct_rmsd < 1e-6
        and direct_run.n_force == ase_run.n_force
        and cg_energy_diff < 1e-3
        and cg_rmsd < 1e-2
        and abs(cg_run.n_force - ase_run.n_force) <= force_tol
    )

    result = RelaxationParity(
        name=structure.name,
        engine=structure.engine,
        n_atoms=len(structure.atoms),
        fmax_tol=structure.fmax,
        ase_converged=ase_run.converged,
        ase_n_steps=ase_run.n_steps,
        ase_n_force=ase_run.n_force,
        ase_e_final=ase_run.e_final,
        ase_fmax_final=ase_run.fmax_final,
        direct_converged=direct_run.converged,
        direct_n_force=direct_run.n_force,
        direct_energy_diff=direct_energy_diff,
        direct_rmsd=direct_rmsd,
        cg_converged=cg_run.converged,
        cg_n_force=cg_run.n_force,
        cg_energy_diff=cg_energy_diff,
        cg_rmsd=cg_rmsd,
        parity_ok=parity_ok,
    )
    if save:
        d = artifacts.stage_dir("stage5", structure.name)
        artifacts.save_json(d / "summary.json", result.as_dict())
        for tag, run in [("ase", ase_run), ("direct", direct_run),
                         ("cg", cg_run)]:
            relaxed = structure.atoms.copy()
            relaxed.set_positions(run.positions)
            write(d / f"relaxed_{tag}.xyz", relaxed)
    return result


def run_all(save: bool = True) -> list[RelaxationParity]:
    """Run Stage-5 relaxation parity for every fixed-cell structure."""
    results = []
    for name in FIXED_CELL_CASES:
        structure = by_name(name)
        print(f"[stage5] {name} ({len(structure.atoms)} atoms, "
              f"{structure.engine})")
        result = compare_relaxation(structure, save=save)
        results.append(result)
        print(f"         ASE:    force_evals={result.ase_n_force:3d}  "
              f"steps={result.ase_n_steps:3d}  "
              f"E={result.ase_e_final:.6f}  converged={result.ase_converged}")
        print(f"         direct: force_evals={result.direct_n_force:3d}  "
              f"dE={result.direct_energy_diff:.1e}  "
              f"RMSD={result.direct_rmsd:.1e}")
        print(f"         cg:     force_evals={result.cg_n_force:3d}  "
              f"dE={result.cg_energy_diff:.1e}  "
              f"RMSD={result.cg_rmsd:.1e}  parity_ok={result.parity_ok}")
    return results


if __name__ == "__main__":
    import os
    import sys
    import traceback

    try:
        run_all()
        code = 0
    except Exception:
        traceback.print_exc()
        code = 1
    # Hard exit: LAMMPSlib/Kokkos crash the interpreter teardown.
    sys.stdout.flush()
    os._exit(code)

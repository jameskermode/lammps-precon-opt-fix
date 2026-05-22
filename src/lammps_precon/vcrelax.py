"""Stage 6 — variable-cell preconditioned-relaxation parity.

Variable-cell optimisation adds nine cell degrees of freedom and a cell-metric
scale ``mu_c``. The original validation project hit a confirmed bug here — the
``r_cut=None`` variable-cell bug, where ``estimate_mu`` was reached before
``r_cut`` had been resolved — so this stage checks that locus explicitly.

This module:

* confirms ``r_cut`` and ``r_NN`` are resolved *before* ``mu``/``mu_c``
  estimation, and that the cell block of ``P`` is ``mu_c * I`` (decoupled);
* runs full variable-cell ``PreconLBFGS`` relaxations (``UnitCellFilter``)
  with our two-tier solver and compares final cell, structure, energy and
  force-evaluation count against ASE's stock ``Exp``;
* exercises the cell-DOF path on gamma-Al2O3 (the designated variable-cell
  case) and on a deliberately strained cell (a significant cell change).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.calculators.loggingcalc import LoggingCalculator
from ase.filters import UnitCellFilter
from ase.io import write
from ase.optimize.precon import Exp, PreconLBFGS

from . import artifacts
from .calculators import make_calculator
from .relax import A_DEFAULT, C_STAB_DEFAULT, CG_RTOL, MAXSTEP, TwoTierExp
from .structures import gamma_al2o3, mgo_supercell

#: A symmetric strain (≈4% scale + shear) applied to create a starting cell
#: far from equilibrium, so variable-cell relaxation genuinely moves the cell.
SIGNIFICANT_STRAIN = np.array([
    [0.040, 0.015, 0.000],
    [0.015, -0.025, 0.010],
    [0.000, 0.010, 0.030],
])


def strained(atoms: Atoms, strain: np.ndarray) -> Atoms:
    """Return a copy of ``atoms`` with ``strain`` applied to cell and atoms."""
    out = atoms.copy()
    deformation = np.eye(3) + np.asarray(strain, dtype=float)
    out.set_cell(out.cell.array @ deformation.T, scale_atoms=True)
    return out


class _RcutProbe(Exp):
    """``Exp`` that records ``r_cut``/``r_NN`` at the moment ``estimate_mu``
    is entered — used to check the historical ``r_cut=None`` bug locus."""

    def estimate_mu(self, atoms, H=None):
        self.r_cut_at_estimate_mu = self.r_cut
        self.r_NN_at_estimate_mu = self.r_NN
        return super().estimate_mu(atoms, H)


def _check_bug_locus(atoms: Atoms, engine: str) -> dict:
    """Confirm r_cut/r_NN are resolved before mu estimation, and check the
    cell block of the variable-cell ``P``."""
    probe = atoms.copy()
    base_calc = make_calculator(probe, engine)
    probe.calc = base_calc
    try:
        precon = _RcutProbe(A=A_DEFAULT, c_stab=C_STAB_DEFAULT, solver="direct")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precon.make_precon(UnitCellFilter(probe))

        n_real = len(probe)
        P = precon.P.tocsr()
        cell_block = P[-9:, -9:].toarray()
        coupling = P[:3 * n_real, 3 * n_real:]
        cell_block_correct = bool(
            np.allclose(cell_block, precon.mu_c * np.eye(9))
            and coupling.nnz == 0)
        return dict(
            r_cut_resolved_before_mu=precon.r_cut_at_estimate_mu is not None,
            r_NN_resolved_before_mu=precon.r_NN_at_estimate_mu is not None,
            mu=float(precon.mu),
            mu_c=float(precon.mu_c),
            cell_block_correct=cell_block_correct,
            P_shape=list(P.shape),
        )
    finally:
        lmp = getattr(base_calc, "lmp", None)
        if lmp is not None:
            try:
                lmp.close()
            except Exception:
                pass


@dataclass
class VCRelaxRun:
    converged: bool
    n_steps: int
    n_force: int
    energy: float
    cell: np.ndarray
    positions: np.ndarray


def _run_vc_relaxation(atoms0: Atoms, engine: str, precon, *,
                       fmax: float, steps: int = 500) -> VCRelaxRun:
    """Run one variable-cell PreconLBFGS relaxation."""
    atoms = atoms0.copy()
    base_calc = make_calculator(atoms, engine)
    logging_calc = LoggingCalculator(base_calc)
    atoms.calc = logging_calc
    try:
        opt = PreconLBFGS(atoms, precon=precon, variable_cell=True,
                          maxstep=MAXSTEP, logfile=None, use_armijo=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converged = bool(opt.run(fmax=fmax, steps=steps))
        return VCRelaxRun(
            converged=converged,
            n_steps=opt.nsteps,
            n_force=sum(len(v) for v in logging_calc.fmax.values()),
            energy=float(atoms.get_potential_energy()),
            cell=atoms.cell.array.copy(),
            positions=atoms.get_positions(),
        )
    finally:
        lmp = getattr(base_calc, "lmp", None)
        if lmp is not None:
            try:
                lmp.close()
            except Exception:
                pass


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _cell_rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@dataclass
class VCRelaxParity:
    name: str
    n_atoms: int
    fmax_tol: float
    # bug-locus / assembly checks
    r_cut_resolved_before_mu: bool
    r_NN_resolved_before_mu: bool
    mu: float
    mu_c: float
    cell_block_correct: bool
    # cell change (significant-cell-change verification)
    cell_change: float
    # ASE reference
    ase_converged: bool
    ase_n_steps: int
    ase_n_force: int
    ase_energy: float
    # direct tier
    direct_converged: bool
    direct_n_force: int
    direct_energy_diff: float
    direct_rmsd: float
    direct_cell_diff: float
    # CG tier
    cg_converged: bool
    cg_n_force: int
    cg_energy_diff: float
    cg_rmsd: float
    cg_cell_diff: float
    parity_ok: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def compare_vc_relaxation(name: str, atoms0: Atoms, engine: str,
                          *, fmax: float = 1e-3,
                          save: bool = True) -> VCRelaxParity:
    """Run the Stage-6 variable-cell relaxation parity check for one case."""
    locus = _check_bug_locus(atoms0, engine)

    ase_run = _run_vc_relaxation(
        atoms0, engine, Exp(A=A_DEFAULT, c_stab=C_STAB_DEFAULT,
                            solver="direct"), fmax=fmax)
    direct_run = _run_vc_relaxation(
        atoms0, engine, TwoTierExp(solver_tier="direct"), fmax=fmax)
    cg_run = _run_vc_relaxation(
        atoms0, engine, TwoTierExp(solver_tier="cg", cg_rtol=CG_RTOL),
        fmax=fmax)

    cell_change = _cell_rel_diff(ase_run.cell, atoms0.cell.array)
    direct_energy_diff = abs(direct_run.energy - ase_run.energy)
    cg_energy_diff = abs(cg_run.energy - ase_run.energy)

    force_tol = max(3, int(0.25 * ase_run.n_force))
    parity_ok = (
        locus["r_cut_resolved_before_mu"]
        and locus["r_NN_resolved_before_mu"]
        and locus["cell_block_correct"]
        and ase_run.converged and direct_run.converged and cg_run.converged
        and direct_energy_diff < 1e-6
        and _rmsd(direct_run.positions, ase_run.positions) < 1e-5
        and _cell_rel_diff(direct_run.cell, ase_run.cell) < 1e-6
        and direct_run.n_force == ase_run.n_force
        and cg_energy_diff < 1e-3
        and _rmsd(cg_run.positions, ase_run.positions) < 1e-2
        and _cell_rel_diff(cg_run.cell, ase_run.cell) < 1e-3
        and abs(cg_run.n_force - ase_run.n_force) <= force_tol
    )

    result = VCRelaxParity(
        name=name,
        n_atoms=len(atoms0),
        fmax_tol=fmax,
        r_cut_resolved_before_mu=locus["r_cut_resolved_before_mu"],
        r_NN_resolved_before_mu=locus["r_NN_resolved_before_mu"],
        mu=locus["mu"],
        mu_c=locus["mu_c"],
        cell_block_correct=locus["cell_block_correct"],
        cell_change=cell_change,
        ase_converged=ase_run.converged,
        ase_n_steps=ase_run.n_steps,
        ase_n_force=ase_run.n_force,
        ase_energy=ase_run.energy,
        direct_converged=direct_run.converged,
        direct_n_force=direct_run.n_force,
        direct_energy_diff=direct_energy_diff,
        direct_rmsd=_rmsd(direct_run.positions, ase_run.positions),
        direct_cell_diff=_cell_rel_diff(direct_run.cell, ase_run.cell),
        cg_converged=cg_run.converged,
        cg_n_force=cg_run.n_force,
        cg_energy_diff=cg_energy_diff,
        cg_rmsd=_rmsd(cg_run.positions, ase_run.positions),
        cg_cell_diff=_cell_rel_diff(cg_run.cell, ase_run.cell),
        parity_ok=parity_ok,
    )
    if save:
        d = artifacts.stage_dir("stage6", name)
        artifacts.save_json(d / "summary.json", result.as_dict())
        for tag, run in [("ase", ase_run), ("direct", direct_run),
                         ("cg", cg_run)]:
            relaxed = atoms0.copy()
            relaxed.set_cell(run.cell)
            relaxed.set_positions(run.positions)
            write(d / f"relaxed_{tag}.xyz", relaxed)
    return result


def stage6_cases() -> list[tuple[str, Atoms, str, float]]:
    """(name, atoms, engine, fmax) for the Stage-6 cases."""
    return [
        # The designated variable-cell case, relaxed as-is.
        ("gamma_Al2O3", gamma_al2o3(), "mace", 1e-3),
        # A deliberately strained cell — a guaranteed significant cell change.
        ("MgO_x2_strained", strained(mgo_supercell(2), SIGNIFICANT_STRAIN),
         "mace", 1e-3),
    ]


def run_all(save: bool = True) -> list[VCRelaxParity]:
    """Run Stage-6 variable-cell parity for every case."""
    results = []
    for name, atoms0, engine, fmax in stage6_cases():
        print(f"[stage6] {name} ({len(atoms0)} atoms, {engine})")
        result = compare_vc_relaxation(name, atoms0, engine, fmax=fmax,
                                       save=save)
        results.append(result)
        print(f"         r_cut/r_NN resolved before mu: "
              f"{result.r_cut_resolved_before_mu}/"
              f"{result.r_NN_resolved_before_mu}  "
              f"mu={result.mu:.3f} mu_c={result.mu_c:.3f}  "
              f"cell_block_ok={result.cell_block_correct}")
        print(f"         cell change during relax = "
              f"{result.cell_change * 100:.2f}%")
        print(f"         ASE:    force_evals={result.ase_n_force:3d}  "
              f"steps={result.ase_n_steps:3d}  E={result.ase_energy:.6f}  "
              f"converged={result.ase_converged}")
        print(f"         direct: force_evals={result.direct_n_force:3d}  "
              f"dE={result.direct_energy_diff:.1e}  "
              f"RMSD={result.direct_rmsd:.1e}  "
              f"cell_diff={result.direct_cell_diff:.1e}")
        print(f"         cg:     force_evals={result.cg_n_force:3d}  "
              f"dE={result.cg_energy_diff:.1e}  "
              f"RMSD={result.cg_rmsd:.1e}  "
              f"cell_diff={result.cg_cell_diff:.1e}  "
              f"parity_ok={result.parity_ok}")
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

"""Stage 7 — scaling validation.

Confirms the implementation scales as intended:

* **flat force-eval counts** — Exp-preconditioned relaxations of the rocksalt
  MgO ``x2..x6`` supercell series take a roughly size-independent number of
  force evaluations (the ``Exp`` size-scaling signature), matching ASE where
  ASE runs; the ``x6`` case exercises the CG tier in the relaxation loop;
* **assembly cost << force-eval cost** — assembling ``P`` is far cheaper than
  one force evaluation;
* **CG iteration count / solve wall-time** scale near-linearly in N — reuses
  the Stage-4 ``cg_scaling`` study (κ bounded, O(N) solve).
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

from ase.optimize.precon import Exp

from . import artifacts
from .calculators import symmetrix_mace_calculator
from .relax import A_DEFAULT, C_STAB_DEFAULT, TwoTierExp, run_relaxation
from .solve import assemble_geometric_P, cg_scaling
from .structures import TestStructure, mgo_supercell

#: MgO supercell sizes for the relaxation-scaling series.
RELAX_SIZES = (2, 3, 4, 5, 6)
#: Largest size for which an ASE reference relaxation is run.
ASE_MAX_SIZE = 5
#: A standard relaxation tolerance for the scaling series (the flat-count
#: signature is tolerance-independent; 1e-2 keeps the series quick).
SCALING_FMAX = 1e-2


@dataclass
class RelaxScalingPoint:
    n: int
    n_atoms: int
    n_dof: int
    ours_n_force: int
    ours_n_steps: int
    ours_converged: bool
    ase_n_force: int | None
    ase_converged: bool | None
    matches_ase: bool | None
    assemble_seconds: float
    force_seconds: float
    assemble_to_force_ratio: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _mgo_structure(n: int, fmax: float) -> TestStructure:
    return TestStructure(name=f"MgO_x{n}", atoms=mgo_supercell(n),
                         engine="mace", fmax=fmax)


def _time_force_evaluation(atoms) -> float:
    """Wall time for one steady-state MACE force evaluation."""
    probe = atoms.copy()
    base_calc = symmetrix_mace_calculator(probe)
    probe.calc = base_calc
    try:
        probe.get_forces()                 # warm-up (includes LAMMPS setup)
        probe.rattle(1e-4, seed=99)        # perturb so the next call recomputes
        t0 = time.perf_counter()
        probe.get_forces()
        return time.perf_counter() - t0
    finally:
        lmp = getattr(base_calc, "lmp", None)
        if lmp is not None:
            try:
                lmp.close()
            except Exception:
                pass


def _time_assembly(atoms) -> float:
    """Wall time for one geometric assembly of the Exp preconditioner P."""
    assemble_geometric_P(atoms)            # warm-up
    t0 = time.perf_counter()
    assemble_geometric_P(atoms)
    return time.perf_counter() - t0


def relaxation_scaling(
    sizes=RELAX_SIZES,
    *,
    fmax: float = SCALING_FMAX,
    ase_max_size: int = ASE_MAX_SIZE,
    save: bool = True,
) -> list[RelaxScalingPoint]:
    """Measure Exp force-eval counts and assembly cost across MgO supercells."""
    print(f"{'n':>3} {'atoms':>7} {'n_dof':>7} {'ours_evals':>11} "
          f"{'ase_evals':>10} {'match':>6} {'asm/ms':>8} {'force/ms':>9} "
          f"{'asm/force':>10}")
    rows: list[RelaxScalingPoint] = []
    for n in sizes:
        structure = _mgo_structure(n, fmax)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ours = run_relaxation(structure, TwoTierExp(solver_tier="auto"))
            ase = None
            if n <= ase_max_size:
                ase = run_relaxation(
                    structure,
                    Exp(A=A_DEFAULT, c_stab=C_STAB_DEFAULT, solver="direct"))

        t_assemble = _time_assembly(structure.atoms)
        t_force = _time_force_evaluation(structure.atoms)
        matches = (ours.n_force == ase.n_force) if ase is not None else None

        point = RelaxScalingPoint(
            n=n,
            n_atoms=len(structure.atoms),
            n_dof=3 * len(structure.atoms),
            ours_n_force=ours.n_force,
            ours_n_steps=ours.n_steps,
            ours_converged=ours.converged,
            ase_n_force=ase.n_force if ase is not None else None,
            ase_converged=ase.converged if ase is not None else None,
            matches_ase=matches,
            assemble_seconds=t_assemble,
            force_seconds=t_force,
            assemble_to_force_ratio=t_assemble / t_force,
        )
        rows.append(point)
        print(f"{n:>3} {point.n_atoms:>7} {point.n_dof:>7} "
              f"{point.ours_n_force:>11} "
              f"{str(point.ase_n_force):>10} {str(matches):>6} "
              f"{t_assemble * 1e3:>8.1f} {t_force * 1e3:>9.1f} "
              f"{point.assemble_to_force_ratio:>10.4f}")

    if save:
        d = artifacts.stage_dir("stage7", "_relaxation")
        artifacts.save_json(d / "relaxation_scaling.json",
                            [r.as_dict() for r in rows])
    return rows


def run_all(save: bool = True):
    """Run the full Stage-7 scaling validation."""
    print("Relaxation scaling (rocksalt MgO supercells, Exp-preconditioned):")
    rows = relaxation_scaling(save=save)
    counts = [r.ours_n_force for r in rows]
    print(f"force-eval counts {counts}: min {min(counts)}, max {max(counts)}, "
          f"max/min {max(counts) / min(counts):.2f} "
          f"(atoms span {rows[0].n_atoms}->{rows[-1].n_atoms})")
    print()
    print("CG-solve scaling (rocksalt MgO supercells):")
    cg_scaling(save=save)
    return rows


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

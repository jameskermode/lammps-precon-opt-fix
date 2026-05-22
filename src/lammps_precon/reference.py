"""Stage 0 — ASE reference harness.

Runs ASE's reference ``Exp`` preconditioned LBFGS (the gold standard) on the
LAMMPS force engine and records, per structure, the artifacts that every later
stage is validated against: ``r_NN``, ``mu``/``mu_c``, the assembled sparse
``P``, the per-step solves, force-evaluation counts and the relaxed structure.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from ase.calculators.loggingcalc import LoggingCalculator
from ase.io import write
from ase.optimize.precon import Exp, PreconLBFGS

from . import artifacts
from .calculators import make_calculator
from .structures import TestStructure, reference_set

#: A=3.0, c_stab=0.1 are the ASE defaults and the values used throughout the
#: ``mace-exp-precon`` validation. ``solver="direct"`` makes the reference an
#: exact, deterministic factorisation (Stage 4a's target).
PRECON_KWARGS = dict(A=3.0, c_stab=0.1, solver="direct")


def _force_calls(logging_calc: LoggingCalculator) -> int:
    """Number of force evaluations recorded by a LoggingCalculator."""
    return sum(len(v) for v in logging_calc.fmax.values())


@dataclass
class ReferenceResult:
    name: str
    engine: str
    n_atoms: int
    r_NN: float
    r_cut: float
    mu: float
    mu_c: float | None
    P_shape: list[int]
    P_nnz: int
    P_symmetry: float
    n_force_setup: int
    n_force_total: int
    relaxed: bool
    converged: bool | None
    n_steps: int | None
    e_final: float | None
    fmax_final: float | None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def run_reference(
    structure: TestStructure,
    *,
    steps: int = 300,
    save: bool = True,
) -> ReferenceResult:
    """Run the Stage-0 reference for one structure and (optionally) save it."""
    atoms = structure.atoms.copy()
    base_calc = make_calculator(atoms, structure.engine)
    logging_calc = LoggingCalculator(base_calc)
    atoms.calc = logging_calc

    precon = Exp(**PRECON_KWARGS)

    # Record every solve P y = b performed via this preconditioner.
    solve_records: list[dict] = []
    _orig_solve = precon.solve

    def _recording_solve(b):
        y = _orig_solve(b)
        b_norm = float(np.linalg.norm(b))
        residual = float(np.linalg.norm(precon.P.dot(y) - b))
        solve_records.append({
            "index": len(solve_records),
            "b_norm": b_norm,
            "y_norm": float(np.linalg.norm(y)),
            "residual": residual,
            "rel_residual": residual / b_norm if b_norm else 0.0,
        })
        return y

    precon.solve = _recording_solve

    try:
        # make_precon resolves r_NN and r_cut, runs estimate_mu (two force
        # calls) and assembles P.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precon.make_precon(atoms)

        P0 = precon.P.tocsr().copy()
        mu_c = precon.mu_c if precon.mu_c is not None else None
        n_force_setup = _force_calls(logging_calc)

        # Frobenius norm of the asymmetric part (P should be symmetric).
        asym = (P0 - P0.T).tocoo()
        p_symmetry = float(np.sqrt((asym.data ** 2).sum())) if asym.nnz else 0.0

        # Initial-step solve P s = -g (the optimiser's first preconditioned
        # gradient); forces here are already cached so add no force call.
        f0 = atoms.get_forces().reshape(-1)
        b0 = -f0
        s0 = precon.solve(b0)

        converged: bool | None = None
        n_steps: int | None = None
        if structure.relax:
            opt = PreconLBFGS(atoms, precon=precon, logfile=None,
                              use_armijo=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converged = bool(opt.run(fmax=structure.fmax, steps=steps))
            n_steps = opt.nsteps

        forces = atoms.get_forces()
        fmax_final = float(np.linalg.norm(forces, axis=1).max())
        e_final = float(atoms.get_potential_energy())
        n_force_total = _force_calls(logging_calc)

        result = ReferenceResult(
            name=structure.name,
            engine=structure.engine,
            n_atoms=len(atoms),
            r_NN=float(precon.r_NN),
            r_cut=float(precon.r_cut),
            mu=float(precon.mu),
            mu_c=float(mu_c) if mu_c is not None else None,
            P_shape=list(P0.shape),
            P_nnz=int(P0.nnz),
            P_symmetry=p_symmetry,
            n_force_setup=n_force_setup,
            n_force_total=n_force_total,
            relaxed=structure.relax,
            converged=converged,
            n_steps=n_steps,
            e_final=e_final if structure.relax else None,
            fmax_final=fmax_final if structure.relax else None,
        )

        if save:
            d = artifacts.stage_dir("stage0", structure.name)
            artifacts.save_json(d / "summary.json", result.as_dict())
            artifacts.save_sparse(d / "P0.npz", P0)
            artifacts.save_arrays(d / "solve_initial.npz", b=b0, s=s0)
            artifacts.save_json(d / "solve_steps.json", solve_records)
            write(d / "initial.xyz", structure.atoms)
            write(d / "relaxed.xyz", atoms)
        return result
    finally:
        # Close the LAMMPS instance cleanly (LAMMPSlib can double-free at
        # interpreter teardown otherwise).
        lmp = getattr(base_calc, "lmp", None)
        if lmp is not None:
            try:
                lmp.close()
            except Exception:
                pass


def run_all(full: bool = False, save: bool = True) -> list[ReferenceResult]:
    """Run the Stage-0 reference for the whole reference set."""
    results = []
    for structure in reference_set(full=full):
        print(f"[stage0] {structure.name} "
              f"({len(structure.atoms)} atoms, {structure.engine})")
        result = run_reference(structure, save=save)
        results.append(result)
        print(f"         r_NN={result.r_NN:.4f}  mu={result.mu:.3f}  "
              f"force_evals={result.n_force_total}  "
              f"steps={result.n_steps}  converged={result.converged}")
    return results


if __name__ == "__main__":
    import argparse
    import os
    import sys
    import traceback

    parser = argparse.ArgumentParser(description="Stage 0 reference harness")
    parser.add_argument("--full", action="store_true",
                        help="relax every structure (slower)")
    args = parser.parse_args()
    try:
        run_all(full=args.full)
        code = 0
    except Exception:
        traceback.print_exc()
        code = 1
    # Hard exit: LAMMPSlib/Kokkos crash the interpreter teardown.
    sys.stdout.flush()
    os._exit(code)

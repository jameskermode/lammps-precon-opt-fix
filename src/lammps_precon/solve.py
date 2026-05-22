"""Stage 4 — two-tier sparse solver for ``P s = b``.

``P`` is SPD (a regularised graph Laplacian), so:

* **direct tier** — a sparse factorisation (scipy SuperLU here; Eigen
  ``SimplicialLDLT`` in the eventual C++ port) for small/medium systems;
* **iterative tier** — conjugate gradient with a cheap Jacobi preconditioner,
  needing only sparse matrix-vector products ``P @ v`` (no factorisation, no
  external solver dependency) for large systems.

A DOF threshold selects the tier. Both tiers are validated against ASE's
``Precon.solve`` and against each other (`compare_solve`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import artifacts
from .assembly import assemble_P
from .structures import by_name, mgo_supercell, reference_set

#: Default DOF threshold for direct -> iterative tier switching (tunable).
DOF_THRESHOLD = 5000
#: Default CG relative residual tolerance for the production solver.
CG_RTOL = 1e-10


@dataclass
class SolveInfo:
    tier: str          # "direct" or "cg"
    iterations: int    # CG iterations (0 for the direct tier)
    residual: float    # ||P s - b|| / ||b||
    converged: bool


def _rel_residual(P, s: np.ndarray, b: np.ndarray) -> float:
    b_norm = float(np.linalg.norm(b))
    if b_norm == 0.0:
        return 0.0
    return float(np.linalg.norm(P.dot(s) - b) / b_norm)


def solve_direct(P, b: np.ndarray) -> tuple[np.ndarray, SolveInfo]:
    """Direct tier — sparse LU factorisation (scipy SuperLU)."""
    b = np.asarray(b, dtype=float)
    lu = spla.splu(sp.csc_matrix(P))
    s = lu.solve(b)
    return s, SolveInfo("direct", 0, _rel_residual(P, s, b), True)


def jacobi_preconditioner(P) -> sp.spmatrix:
    """Diagonal (Jacobi) preconditioner ``M ~ P^-1`` for CG — nearly free."""
    return sp.diags(1.0 / P.diagonal())


def solve_cg(
    P,
    b: np.ndarray,
    *,
    rtol: float = CG_RTOL,
    maxiter: int | None = None,
) -> tuple[np.ndarray, SolveInfo]:
    """Iterative tier — Jacobi-preconditioned conjugate gradient.

    Uses only sparse matrix-vector products with ``P`` — the dependency-light
    path that scales to large systems.
    """
    P = sp.csr_matrix(P)
    b = np.asarray(b, dtype=float)
    M = jacobi_preconditioner(P)
    iterations = [0]

    def _count(_xk):
        iterations[0] += 1

    s, status = spla.cg(P, b, rtol=rtol, atol=0.0, maxiter=maxiter, M=M,
                        callback=_count)
    return s, SolveInfo("cg", iterations[0], _rel_residual(P, s, b),
                        status == 0)


def solve(
    P,
    b: np.ndarray,
    *,
    dof_threshold: int = DOF_THRESHOLD,
    cg_rtol: float = CG_RTOL,
) -> tuple[np.ndarray, SolveInfo]:
    """Two-tier solve of ``P s = b``: direct at/below the DOF threshold, CG above."""
    if P.shape[0] <= dof_threshold:
        return solve_direct(P, b)
    return solve_cg(P, b, rtol=cg_rtol)


# --------------------------------------------------------------------------
# Stage 4 validation
# --------------------------------------------------------------------------

@dataclass
class SolveParity:
    name: str
    n_dof: int
    condition_number: float
    # 4a — direct tier
    direct_residual: float
    rel_direct_vs_ase: float
    # 4b — iterative tier
    cg_iterations: int
    cg_converged: bool
    cg_residual: float
    rel_cg_vs_direct: float
    rel_cg_vs_ase: float
    # 4c — tier-switch consistency
    rel_tier_switch: float
    parity_ok: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _load_reference(name: str) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Load (P, b, s_ase) from the Stage-0 artifacts, generating them if absent."""
    d = artifacts.ARTIFACT_DIR / "stage0" / name
    if not (d / "P0.npz").exists() or not (d / "solve_initial.npz").exists():
        from .reference import run_reference
        run_reference(by_name(name))
    P = artifacts.load_sparse(d / "P0.npz").tocsr()
    arrays = artifacts.load_arrays(d / "solve_initial.npz")
    return P, arrays["b"], arrays["s"]


def compare_solve(name: str, *, save: bool = True) -> SolveParity:
    """Run the Stage-4 solver parity check for one structure.

    ``b`` is the structure's initial gradient and ``s_ase`` is ASE's
    ``Precon.solve`` of ``P s = b`` (both from the Stage-0 artifacts).
    """
    P, b, s_ase = _load_reference(name)
    n = P.shape[0]

    # P is small enough here for a dense symmetric eigensolve.
    eigenvalues = np.linalg.eigvalsh(P.toarray())
    condition_number = float(eigenvalues[-1] / eigenvalues[0])

    # 4a — direct tier.
    s_direct, info_direct = solve_direct(P, b)
    # 4b — iterative tier (tight tolerance so the solution matches the direct
    # solve, not just the residual).
    s_cg, info_cg = solve_cg(P, b, rtol=1e-12)
    # 4c — tier switch: force CG (threshold below n) and direct (threshold n).
    s_force_cg, _ = solve(P, b, dof_threshold=n - 1, cg_rtol=1e-12)
    s_force_direct, _ = solve(P, b, dof_threshold=n)

    norm_ase = np.linalg.norm(s_ase)
    norm_direct = np.linalg.norm(s_direct)
    rel_direct_vs_ase = float(np.linalg.norm(s_direct - s_ase) / norm_ase)
    rel_cg_vs_direct = float(np.linalg.norm(s_cg - s_direct) / norm_direct)
    rel_cg_vs_ase = float(np.linalg.norm(s_cg - s_ase) / norm_ase)
    rel_tier_switch = float(
        np.linalg.norm(s_force_cg - s_force_direct)
        / np.linalg.norm(s_force_direct))

    parity_ok = (
        info_direct.residual < 1e-10
        and rel_direct_vs_ase < 1e-8
        and info_cg.converged
        and rel_cg_vs_direct < 1e-8
        and rel_tier_switch < 1e-8
    )

    result = SolveParity(
        name=name,
        n_dof=n,
        condition_number=condition_number,
        direct_residual=info_direct.residual,
        rel_direct_vs_ase=rel_direct_vs_ase,
        cg_iterations=info_cg.iterations,
        cg_converged=info_cg.converged,
        cg_residual=info_cg.residual,
        rel_cg_vs_direct=rel_cg_vs_direct,
        rel_cg_vs_ase=rel_cg_vs_ase,
        rel_tier_switch=rel_tier_switch,
        parity_ok=parity_ok,
    )
    if save:
        d = artifacts.stage_dir("stage4", name)
        artifacts.save_json(d / "summary.json", result.as_dict())
    return result


def run_all(save: bool = True) -> list[SolveParity]:
    """Run Stage-4 solver parity for every reference structure."""
    results = []
    # Ascending DOF, so the CG iteration-count trend is easy to read.
    names = sorted({s.name for s in reference_set(full=True)},
                   key=lambda nm: len(by_name(nm).atoms))
    for name in names:
        result = compare_solve(name, save=save)
        results.append(result)
        print(f"[stage4] {name:12s} n_dof={result.n_dof:5d}  "
              f"kappa={result.condition_number:8.1f}  "
              f"direct_res={result.direct_residual:.1e}  "
              f"CG iters={result.cg_iterations:3d} "
              f"(res {result.cg_residual:.1e})")
        print(f"         direct vs ASE={result.rel_direct_vs_ase:.1e}  "
              f"CG vs direct={result.rel_cg_vs_direct:.1e}  "
              f"tier-switch={result.rel_tier_switch:.1e}  "
              f"parity_ok={result.parity_ok}")
    return results


# --------------------------------------------------------------------------
# Large-N CG scaling study
# --------------------------------------------------------------------------
#
# The iterative tier is meant to carry large systems, but the parity checks
# above all sit below the DOF threshold (so they actually take the direct
# path). This study exercises CG in its real regime.
#
# Assembling P is purely geometric (no forces, no estimate_mu — and mu is a
# global scale that does not affect the condition number), so P can be built
# for systems far larger than a force-driven harness could reach.

SCALING_SIZES = (3, 4, 6, 8, 10, 12, 14)


def assemble_geometric_P(atoms, *, mu: float = 1.0, A: float = 3.0,
                         c_stab: float = 0.1):
    """Assemble the Exp ``P`` for ``atoms`` from geometry alone (no forces)."""
    from ase.optimize.precon.neighbors import estimate_nearest_neighbour_distance
    try:
        from matscipy.neighbours import neighbour_list as nlist
    except ImportError:
        from ase.neighborlist import neighbor_list as nlist
    r_NN = float(estimate_nearest_neighbour_distance(atoms, nlist))
    r_cut = 2.0 * r_NN
    P = assemble_P(atoms, mu=mu, r_NN=r_NN, r_cut=r_cut, A=A, c_stab=c_stab,
                   neighbor_list=nlist)
    return P, r_NN, r_cut


def cg_scaling(
    sizes=SCALING_SIZES,
    *,
    cg_rtol: float = 1e-8,
    c_stab: float = 0.1,
    save: bool = True,
) -> list[dict]:
    """Measure CG iteration count, conditioning and wall-time vs system size.

    Builds rocksalt MgO supercells, assembles ``P``, and solves ``P s = b``
    for a generic (seeded random) right-hand side with the iterative tier.
    """
    import time

    from scipy.sparse.linalg import eigsh

    # mu = 1 here; lambda_min(P) = mu * c_stab exactly (Stage 2).
    lambda_min = 1.0 * c_stab

    print(f"{'n':>3} {'atoms':>7} {'n_dof':>8} {'kappa':>8} {'CG its':>7} "
          f"{'CG resid':>10} {'asm/s':>8} {'solve/s':>8} {'us/dof':>8}")
    rows: list[dict] = []
    for n in sizes:
        atoms = mgo_supercell(n)
        t0 = time.perf_counter()
        P, r_NN, _ = assemble_geometric_P(atoms, mu=1.0, c_stab=c_stab)
        t_assemble = time.perf_counter() - t0
        n_dof = P.shape[0]

        lambda_max = float(eigsh(P, k=1, which="LA",
                                 return_eigenvectors=False)[0])
        kappa = lambda_max / lambda_min

        b = np.random.default_rng(0).standard_normal(n_dof)
        t0 = time.perf_counter()
        s_cg, info = solve_cg(P, b, rtol=cg_rtol)
        t_solve = time.perf_counter() - t0

        # Cross-check against the direct tier where it is still cheap.
        rel_vs_direct = None
        if n_dof <= 6000:
            s_direct, _ = solve_direct(P, b)
            rel_vs_direct = float(np.linalg.norm(s_cg - s_direct)
                                  / np.linalg.norm(s_direct))

        rows.append(dict(
            n=n, n_atoms=len(atoms), n_dof=n_dof, r_NN=r_NN,
            condition_number=kappa, lambda_max=lambda_max,
            cg_iterations=info.iterations, cg_residual=info.residual,
            cg_converged=info.converged, rel_cg_vs_direct=rel_vs_direct,
            assemble_seconds=t_assemble, solve_seconds=t_solve,
        ))
        print(f"{n:>3} {len(atoms):>7} {n_dof:>8} {kappa:>8.1f} "
              f"{info.iterations:>7} {info.residual:>10.1e} "
              f"{t_assemble:>8.2f} {t_solve:>8.2f} "
              f"{t_solve / n_dof * 1e6:>8.1f}")

    if save:
        d = artifacts.stage_dir("stage4", "_scaling")
        artifacts.save_json(d / "cg_scaling.json", rows)
    return rows


if __name__ == "__main__":
    import os
    import sys
    import traceback

    try:
        run_all()
        print()
        print("CG scaling (rocksalt MgO supercells):")
        cg_scaling()
        code = 0
    except Exception:
        traceback.print_exc()
        code = 1
    # Hard exit in case a Stage-0 artifact had to be regenerated via LAMMPS.
    sys.stdout.flush()
    os._exit(code)

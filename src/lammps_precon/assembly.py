"""Stage 2 — P-matrix assembly parity.

Assembles the ``Exp`` preconditioner ``P`` from the LAMMPS-derived pair list
and confirms it matches ASE's ``P`` (assembled from ASE's own neighbour list)
for the same structure and the same ``mu``.

The assembly *formula* is not reimplemented: ASE's ``Exp`` assembly is reused
directly, with the LAMMPS pair list injected through the ``neighbor_list``
hook (the same drop-in mechanism ASE documents for matscipy). Since Stage 1
proved the two pair lists are identical, the two ``P`` matrices must agree to
solver precision.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.optimize.precon import Exp

from . import artifacts
from .calculators import make_calculator
from .neighbours import lammps_pairs
from .structures import TestStructure, reference_set

#: Matches the reference harness (ASE defaults, mace-exp-precon settings).
A_DEFAULT = 3.0
C_STAB_DEFAULT = 0.1


def lammps_neighbor_list(quantities: str, atoms: Atoms, cutoff: float):
    """ASE-compatible ``neighbor_list`` backed by LAMMPS.

    Drop-in replacement for ``ase.neighborlist.neighbor_list`` with the call
    signature ASE's ``get_neighbours`` uses. Returns the *full* (both
    directions) pair list within ``cutoff`` as ``(i, j, d)``.
    """
    if quantities != "ijd":
        raise ValueError(
            f"lammps_neighbor_list only supports quantities='ijd', got "
            f"{quantities!r}")
    pairs, _kind = lammps_pairs(atoms, cutoff)  # (M, 3) undirected, i < j
    if len(pairs) == 0:
        empty_i = np.empty(0, dtype=int)
        return empty_i, empty_i, np.empty(0, dtype=float)
    i = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(int)
    j = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(int)
    d = np.concatenate([pairs[:, 2], pairs[:, 2]])
    return i, j, d


def assemble_P(
    atoms: Atoms,
    *,
    mu: float,
    r_NN: float,
    r_cut: float,
    neighbor_list,
    A: float = A_DEFAULT,
    c_stab: float = C_STAB_DEFAULT,
):
    """Assemble the Exp ``P`` via ASE's assembly with fixed mu/r_NN/r_cut.

    With ``mu``, ``r_NN`` and ``r_cut`` all supplied, ``make_precon`` skips
    ``estimate_mu`` entirely and just runs ``_make_sparse_precon`` over the
    pairs returned by ``neighbor_list`` — so no force evaluations occur.
    """
    precon = Exp(A=A, c_stab=c_stab, mu=mu, r_NN=r_NN, r_cut=r_cut,
                 solver="direct")
    precon.neighbor_list = neighbor_list
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precon.make_precon(atoms)
    return precon.P.tocsr()


@dataclass
class AssemblyParity:
    name: str
    n_atoms: int
    mu: float
    r_NN: float
    r_cut: float
    P_shape: list[int]
    nnz_ase: int
    nnz_lammps: int
    pattern_match: bool
    rel_norm_diff: float
    symmetry: float
    min_eigenvalue: float
    c_stab_floor: float
    cholesky_ok: bool
    spd: bool
    parity_ok: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def compare_assembly(
    structure: TestStructure,
    *,
    A: float = A_DEFAULT,
    c_stab: float = C_STAB_DEFAULT,
    save: bool = True,
) -> AssemblyParity:
    """Run the Stage-2 assembly parity check for one structure."""
    atoms = structure.atoms.copy()
    base_calc = make_calculator(atoms, structure.engine)
    atoms.calc = base_calc
    try:
        # ASE reference: standard Exp assembly (own neighbour list + FD mu).
        precon_ref = Exp(A=A, c_stab=c_stab, solver="direct")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precon_ref.make_precon(atoms)
        P_ase = precon_ref.P.tocsr()
        mu = float(precon_ref.mu)
        r_NN = float(precon_ref.r_NN)
        r_cut = float(precon_ref.r_cut)

        # LAMMPS path: identical assembly, identical mu, LAMMPS pair list.
        P_lmp = assemble_P(atoms, mu=mu, r_NN=r_NN, r_cut=r_cut, A=A,
                           c_stab=c_stab, neighbor_list=lammps_neighbor_list)

        # Compare densely — these matrices are small (<= 1296 x 1296).
        A_dense = P_ase.toarray()
        B_dense = P_lmp.toarray()
        norm_ase = np.linalg.norm(A_dense)
        rel_norm = float(np.linalg.norm(B_dense - A_dense) / norm_ase)
        symmetry = float(np.linalg.norm(B_dense - B_dense.T))
        pattern_match = bool(np.array_equal(A_dense != 0.0, B_dense != 0.0))

        eigenvalues = np.linalg.eigvalsh(B_dense)
        min_eig = float(eigenvalues[0])
        try:
            np.linalg.cholesky(B_dense)
            cholesky_ok = True
        except np.linalg.LinAlgError:
            cholesky_ok = False
        spd = cholesky_ok and min_eig > 0.0

        parity_ok = (pattern_match and rel_norm < 1e-10
                     and symmetry < 1e-12 and spd)

        result = AssemblyParity(
            name=structure.name,
            n_atoms=len(atoms),
            mu=mu,
            r_NN=r_NN,
            r_cut=r_cut,
            P_shape=list(P_lmp.shape),
            nnz_ase=int(P_ase.nnz),
            nnz_lammps=int(P_lmp.nnz),
            pattern_match=pattern_match,
            rel_norm_diff=rel_norm,
            symmetry=symmetry,
            min_eigenvalue=min_eig,
            c_stab_floor=mu * c_stab,
            cholesky_ok=cholesky_ok,
            spd=spd,
            parity_ok=parity_ok,
        )

        if save:
            d = artifacts.stage_dir("stage2", structure.name)
            artifacts.save_json(d / "summary.json", result.as_dict())
            artifacts.save_sparse(d / "P_lammps.npz", P_lmp)
        return result
    finally:
        lmp = getattr(base_calc, "lmp", None)
        if lmp is not None:
            try:
                lmp.close()
            except Exception:
                pass


def run_all(save: bool = True) -> list[AssemblyParity]:
    """Run Stage-2 assembly parity for every reference structure."""
    results = []
    for structure in reference_set(full=True):
        print(f"[stage2] {structure.name} "
              f"({len(structure.atoms)} atoms, {structure.engine})")
        result = compare_assembly(structure, save=save)
        results.append(result)
        print(f"         nnz ase={result.nnz_ase} lammps={result.nnz_lammps}  "
              f"pattern_match={result.pattern_match}  "
              f"rel_norm={result.rel_norm_diff:.2e}  "
              f"symmetry={result.symmetry:.2e}")
        print(f"         min_eig={result.min_eigenvalue:.6f} "
              f"(c_stab floor mu*c_stab={result.c_stab_floor:.6f})  "
              f"SPD={result.spd}  parity_ok={result.parity_ok}")
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

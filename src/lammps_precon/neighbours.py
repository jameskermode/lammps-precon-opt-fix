"""Stage 1 — neighbour-list and r_NN parity between LAMMPS and ASE.

Confirms that the pair list LAMMPS produces within the preconditioner cutoff
``r_cut`` is identical to the one ASE's ``Exp`` assembly uses, and that the
nearest-neighbour distance ``r_NN`` is estimated identically. A mismatch here
is almost always a PBC / minimum-image edge case and must be resolved before
the assembly stages.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.optimize.precon.neighbors import (
    estimate_nearest_neighbour_distance,
    get_neighbours,
)

from . import artifacts
from .calculators import LAMMPS_HEADER
from .structures import TestStructure, reference_set

#: LAMMPS encodes special-bond flags in the top bits of neighbour indices.
NEIGHMASK = 0x3FFFFFFF


def ase_pairs(atoms: Atoms, r_cut: float) -> np.ndarray:
    """ASE neighbour pairs within ``r_cut`` as an ``(M, 3)`` ``(i, j, d)`` array.

    ASE returns a full list (both ``i->j`` and ``j->i``); we keep one direction
    (``i < j``) so each geometric contact appears exactly once.
    """
    i, j, d, _ = get_neighbours(atoms, r_cut)
    keep = i < j
    return _as_pair_array(i[keep], j[keep], d[keep])


def lammps_pairs(atoms: Atoms, r_cut: float) -> tuple[np.ndarray, str]:
    """LAMMPS neighbour pairs within ``r_cut`` via ``pair_style zero``.

    Returns ``((M, 3) array of (i, j, d), kind)`` where ``kind`` is "half" or
    "full" describing the raw LAMMPS list. Indices are ASE atom indices
    (LAMMPS tags minus one — the ``sort 0`` header keeps the orders aligned).
    """
    from ase.calculators.lammpslib import LAMMPSlib

    species = sorted(set(atoms.get_chemical_symbols()))
    calc = LAMMPSlib(
        lmpcmds=[f"pair_style zero {r_cut}", "pair_coeff * *"],
        atom_types={s: k + 1 for k, s in enumerate(species)},
        keep_alive=True,
        log_file="log.lammps",
        lammps_header=LAMMPS_HEADER,
    )
    probe = atoms.copy()
    probe.calc = calc
    try:
        probe.get_potential_energy()  # builds the box + neighbour list
        lmp = calc.lmp
        # Extract local+ghost atom data (explicit nelem so ghosts are included).
        nall = int(lmp.extract_global("nlocal")) + \
            int(lmp.extract_global("nghost"))
        x = lmp.numpy.extract_atom("x", nelem=nall)    # (nall, 3) with ghosts
        tag = lmp.numpy.extract_atom("id", nelem=nall)  # (nall,) 1-based IDs
        idx = lmp.find_pair_neighlist("zero")
        if idx < 0:
            raise RuntimeError("LAMMPS 'zero' pair neighbour list not found")
        nl = lmp.numpy.get_neighlist(idx)

        rows: list[tuple[int, int, float]] = []
        for ii in range(len(nl)):
            entry = nl[ii]
            iatom, neighs = entry[0], entry[-1]
            ai = int(tag[iatom]) - 1
            for raw in np.asarray(neighs):
                jatom = int(raw) & NEIGHMASK
                aj = int(tag[jatom]) - 1
                if ai == aj:
                    continue  # self-image, excluded like ASE
                d = float(np.linalg.norm(x[iatom] - x[jatom]))
                if d <= r_cut:
                    rows.append((min(ai, aj), max(ai, aj), d))
    finally:
        lmp = getattr(calc, "lmp", None)
        if lmp is not None:
            try:
                lmp.close()
            except Exception:
                pass

    if not rows:
        return np.empty((0, 3)), "half"
    i = np.array([r[0] for r in rows])
    j = np.array([r[1] for r in rows])
    d = np.array([r[2] for r in rows])
    pairs = _as_pair_array(i, j, d)
    # A full list contains every contact twice (identical i, j and distance).
    kind = "full" if _looks_doubled(pairs) else "half"
    if kind == "full":
        pairs = pairs[::2]
    return pairs, kind


def _as_pair_array(i, j, d) -> np.ndarray:
    """Stack and sort pairs by (i, j, d) so two lists can be compared rowwise."""
    pairs = np.column_stack([np.asarray(i, float),
                             np.asarray(j, float),
                             np.asarray(d, float)])
    order = np.lexsort((pairs[:, 2], pairs[:, 1], pairs[:, 0]))
    return pairs[order]


def _looks_doubled(pairs: np.ndarray) -> bool:
    """True if every contact appears as an adjacent identical (i, j, d) pair."""
    if len(pairs) < 2 or len(pairs) % 2 != 0:
        return False
    a, b = pairs[0::2], pairs[1::2]
    return bool(np.all(a[:, :2] == b[:, :2])
                and np.allclose(a[:, 2], b[:, 2], rtol=0, atol=1e-9))


def r_NN_from_pairs(pairs: np.ndarray, n_atoms: int) -> float:
    """Nearest-neighbour distance: max over atoms of the per-atom min distance."""
    best = np.full(n_atoms, np.inf)
    for i, j, d in pairs:
        i, j = int(i), int(j)
        best[i] = min(best[i], d)
        best[j] = min(best[j], d)
    return float(best.max())


@dataclass
class NeighbourParity:
    name: str
    n_atoms: int
    r_cut: float
    r_NN_ase: float
    r_NN_lammps: float
    r_NN_diff: float
    n_pairs_ase: int
    n_pairs_lammps: int
    lammps_list_kind: str
    pairs_match: bool
    max_distance_diff: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def compare(atoms: Atoms, name: str = "structure") -> NeighbourParity:
    """Run the Stage-1 neighbour-list / r_NN parity check for one structure."""
    r_NN_ase = float(estimate_nearest_neighbour_distance(atoms))
    r_cut = 2.0 * r_NN_ase

    ase = ase_pairs(atoms, r_cut)
    lmp, kind = lammps_pairs(atoms, r_cut)

    same_count = len(ase) == len(lmp)
    if same_count and len(ase):
        same_pairs = bool(np.array_equal(ase[:, :2], lmp[:, :2]))
        max_dd = float(np.abs(ase[:, 2] - lmp[:, 2]).max())
    elif same_count:  # both empty
        same_pairs, max_dd = True, 0.0
    else:
        same_pairs, max_dd = False, float("inf")

    pairs_match = same_count and same_pairs and max_dd < 1e-8

    r_NN_lammps = (r_NN_from_pairs(lmp, len(atoms)) if len(lmp)
                   else float("inf"))

    return NeighbourParity(
        name=name,
        n_atoms=len(atoms),
        r_cut=r_cut,
        r_NN_ase=r_NN_ase,
        r_NN_lammps=r_NN_lammps,
        r_NN_diff=abs(r_NN_ase - r_NN_lammps),
        n_pairs_ase=len(ase),
        n_pairs_lammps=len(lmp),
        lammps_list_kind=kind,
        pairs_match=pairs_match,
        max_distance_diff=max_dd,
    )


def run_all(save: bool = True) -> list[NeighbourParity]:
    """Run Stage-1 parity for every reference structure."""
    results = []
    for structure in reference_set(full=True):
        print(f"[stage1] {structure.name} ({len(structure.atoms)} atoms)")
        result = compare(structure.atoms, structure.name)
        results.append(result)
        print(f"         r_NN ase={result.r_NN_ase:.6f} "
              f"lammps={result.r_NN_lammps:.6f} "
              f"(diff {result.r_NN_diff:.2e})  "
              f"pairs ase={result.n_pairs_ase} lammps={result.n_pairs_lammps} "
              f"[{result.lammps_list_kind}]  "
              f"match={result.pairs_match} maxdd={result.max_distance_diff:.2e}")
        if save:
            d = artifacts.stage_dir("stage1", structure.name)
            artifacts.save_json(d / "summary.json", result.as_dict())
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

"""Test structures for Exp-preconditioner validation.

The Packwood benchmark structures (copied into ``structures/`` from the
``mace-exp-precon`` validation project) plus generated rocksalt MgO supercells
and a classical-EAM Cu cell. See ``spec.md`` Stage 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ase import Atoms
from ase.build import bulk
from ase.io import read

REPO_ROOT = Path(__file__).resolve().parents[2]
STRUCTURE_DIR = REPO_ROOT / "structures"

# Atomic numbers that must be present in the Symmetrix MACE model so it can
# evaluate every test structure: H, O, Mg, Al, Si, La.
MACE_ATOMIC_NUMBERS = (1, 8, 12, 13, 14, 57)


@dataclass
class TestStructure:
    """A test structure plus the metadata the reference harness needs."""

    name: str
    atoms: Atoms
    engine: str  # "mace" (Symmetrix) or "eam" (classical Cu)
    fmax: float = 1e-3
    variable_cell: bool = False
    relax: bool = True  # whether the first pass runs a full relaxation


def _read_packwood(filename: str) -> Atoms:
    atoms = read(STRUCTURE_DIR / filename)
    if isinstance(atoms, list):  # extxyz can yield a list
        atoms = atoms[-1]
    return atoms


def si_slab() -> Atoms:
    """Packwood Si slab test configuration (160 atoms)."""
    return _read_packwood("Si_slab.xyz")


def laalo3() -> Atoms:
    """Packwood LaAlO3 crack configuration (220 atoms, triclinic)."""
    return _read_packwood("LaAlO3.xyz")


def gamma_al2o3() -> Atoms:
    """Packwood gamma-Al2O3 configuration (106 atoms, variable-cell case)."""
    return _read_packwood("gamma_Al2O3.xyz")


def ice_viii() -> Atoms:
    """Packwood ice VIII configuration (432 atoms)."""
    return _read_packwood("iceVIII.xyz")


def mgo_supercell(n: int, rattle: float = 0.05, seed: int = 1) -> Atoms:
    """Rocksalt MgO ``n x n x n`` conventional-cell supercell, lightly rattled.

    Rattling gives non-zero forces so the structure is a genuine relaxation
    test; it is harmless for the (purely geometric) neighbour-list parity check.
    """
    atoms = bulk("MgO", crystalstructure="rocksalt", a=4.21, cubic=True)
    atoms = atoms * (n, n, n)
    if rattle:
        atoms.rattle(stdev=rattle, seed=seed)
    return atoms


def cu_fcc(n: int = 3, rattle: float = 0.05, seed: int = 1) -> Atoms:
    """Classical-EAM Cu fcc ``n x n x n`` supercell, rattled (108 atoms for n=3)."""
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True) * (n, n, n)
    if rattle:
        atoms.rattle(stdev=rattle, seed=seed)
    return atoms


def reference_set(full: bool = False) -> list[TestStructure]:
    """The Stage-0 reference structure set.

    With ``full=False`` (default) the slower large structures are recorded
    only (r_NN / mu / P, no relaxation) to keep the first pass quick; the
    relaxation-parity headline cases (Si slab, Cu/EAM, MgO x2) are relaxed.
    With ``full=True`` everything is relaxed.
    """
    items = [
        TestStructure("Si_slab", si_slab(), "mace", fmax=1e-3),
        TestStructure("Cu_fcc", cu_fcc(), "eam", fmax=1e-4),
        TestStructure("MgO_x2", mgo_supercell(2), "mace", fmax=1e-3),
        TestStructure("MgO_x3", mgo_supercell(3), "mace", fmax=1e-3,
                      relax=full),
        TestStructure("LaAlO3", laalo3(), "mace", fmax=1e-3, relax=full),
        TestStructure("gamma_Al2O3", gamma_al2o3(), "mace", fmax=1e-3,
                      relax=full),
        TestStructure("iceVIII", ice_viii(), "mace", fmax=1e-3, relax=full),
    ]
    return items


def by_name(name: str) -> TestStructure:
    for s in reference_set(full=True):
        if s.name == name:
            return s
    raise KeyError(f"unknown test structure: {name!r}")

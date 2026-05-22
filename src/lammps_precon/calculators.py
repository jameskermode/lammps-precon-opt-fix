"""LAMMPS-backed ASE calculators — the force engine for every stage.

A single force engine (LAMMPS, driven through ASE's ``LAMMPSlib``) is used on
both sides of every parity check, so the variable under test is the
preconditioner code and never the forces. Two potentials are provided:

* ``symmetrix_mace_calculator`` — ``pair_style symmetrix/mace`` with the MACE
  foundation model (covers Si / MgO / oxides / ice).
* ``eam_calculator`` — classical ``pair_style eam`` for the Cu test cell,
  used to confirm the preconditioner is potential-agnostic (Stage 3).
"""
from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.calculators.lammpslib import LAMMPSlib

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_JSON = REPO_ROOT / "models" / "mace-matpes-symmetrix.json"
LAMMPS_DIR = REPO_ROOT / "lammps-symmetrix" / "lammps"

# LAMMPS header shared by all calculators. ``sort 0`` keeps LAMMPS' atom order
# equal to the ASE order, so atom tags map straight back to ASE indices — this
# is what makes the Stage-1 neighbour-list comparison well defined.
LAMMPS_HEADER = [
    "units metal",
    "atom_style atomic",
    "atom_modify map array sort 0 0.0",
]


def _atom_types(atoms: Atoms) -> tuple[dict[str, int], list[str]]:
    """Map element symbols to LAMMPS atom types (1-based, sorted by symbol)."""
    species = sorted(set(atoms.get_chemical_symbols()))
    atom_types = {sym: i + 1 for i, sym in enumerate(species)}
    return atom_types, species


def eam_potential_path(name: str = "Cu_u3.eam") -> Path:
    """Locate an EAM potential file shipped with the cloned LAMMPS source."""
    path = LAMMPS_DIR / "potentials" / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — has scripts/build_lammps.sh been run?")
    return path


def symmetrix_mace_calculator(
    atoms: Atoms,
    model_json: Path | str = MODEL_JSON,
    log_file: str = "log.lammps",
) -> LAMMPSlib:
    """Build a ``pair_style symmetrix/mace`` LAMMPSlib calculator for ``atoms``."""
    model_json = Path(model_json)
    if not model_json.exists():
        raise FileNotFoundError(
            f"{model_json} not found — run scripts/convert_model.py first.")
    atom_types, species = _atom_types(atoms)
    elements = " ".join(species)
    return LAMMPSlib(
        lmpcmds=[
            "pair_style symmetrix/mace",
            f"pair_coeff * * {model_json} {elements}",
        ],
        atom_types=atom_types,
        keep_alive=True,
        log_file=log_file,
        lammps_header=LAMMPS_HEADER,
    )


def eam_calculator(
    atoms: Atoms,
    potential: Path | str | None = None,
    log_file: str = "log.lammps",
) -> LAMMPSlib:
    """Build a classical ``pair_style eam`` LAMMPSlib calculator for ``atoms``."""
    potential = Path(potential) if potential else eam_potential_path()
    atom_types, _ = _atom_types(atoms)
    return LAMMPSlib(
        lmpcmds=[
            "pair_style eam",
            f"pair_coeff * * {potential}",
        ],
        atom_types=atom_types,
        keep_alive=True,
        log_file=log_file,
        lammps_header=LAMMPS_HEADER,
    )


def make_calculator(atoms: Atoms, engine: str, **kwargs) -> LAMMPSlib:
    """Dispatch to the calculator for ``engine`` ("mace" or "eam")."""
    if engine == "mace":
        return symmetrix_mace_calculator(atoms, **kwargs)
    if engine == "eam":
        return eam_calculator(atoms, **kwargs)
    raise ValueError(f"unknown engine: {engine!r}")

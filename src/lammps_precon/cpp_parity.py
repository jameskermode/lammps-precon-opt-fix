"""Stage 8a — validate the C++ plugin against the Python reference.

Drives the C++ `fix precon/exp` + `min_style precon/lbfgs` plugin through the
`lammps` Python module and compares, per structure:

* the assembled P  -> the Python `assembly.py` P (at the C++'s own mu/r_NN);
* the estimated mu -> the Stage-3 reference;
* the relaxation   -> the Stage-5 reference (final energy, convergence).
"""
from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import ase.io
import numpy as np
import scipy.io
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.neighborlist import neighbor_list as ase_neighbor_list

from . import artifacts
from .assembly import assemble_P
from .calculators import MODEL_JSON, REPO_ROOT, eam_potential_path
from .structures import by_name

PLUGIN_SO = REPO_ROOT / "cpp" / "build" / "liblammps_precon.so"


def _pair_commands(atoms: Atoms, engine: str):
    species = sorted(set(atoms.get_chemical_symbols()))
    if engine == "mace":
        return species, [
            "pair_style symmetrix/mace",
            f"pair_coeff * * {MODEL_JSON} {' '.join(species)}",
        ]
    if engine == "eam":
        return species, [
            "pair_style eam",
            f"pair_coeff * * {eam_potential_path()}",
        ]
    raise ValueError(f"unknown engine: {engine!r}")


def _parse_minimize_log(logfile: Path) -> dict:
    text = Path(logfile).read_text(errors="ignore")
    n_steps = n_force = None
    m = re.search(r"force evaluations\s*=\s*(\d+)\s+(\d+)", text)
    if m:
        n_steps, n_force = int(m.group(1)), int(m.group(2))
    stop = ""
    m = re.search(r"Stopping criterion\s*=\s*(.+)", text)
    if m:
        stop = m.group(1).strip()
    return dict(n_steps=n_steps, n_force=n_force, stop_reason=stop,
                converged=stop in ("force tolerance", "energy tolerance"))


def run_lammps_cpp(
    atoms: Atoms,
    engine: str,
    *,
    fmax: float,
    maxiter: int,
    workdir: Path,
    dump_prefix: str | None = None,
) -> dict:
    """Run a `min_style precon/lbfgs` relaxation in LAMMPS via the C++ plugin."""
    from lammps import lammps

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    species, pair_cmds = _pair_commands(atoms, engine)

    datafile = workdir / "structure.data"
    ase.io.write(datafile, atoms, format="lammps-data",
                 specorder=species, atom_style="atomic")
    logfile = workdir / "log.lammps"

    lmp = lammps(cmdargs=["-log", str(logfile), "-screen", "none", "-nocite"])
    mass_cmds = [f"mass {k + 1} {atomic_masses[atomic_numbers[s]]}"
                 for k, s in enumerate(species)]
    fix_cmd = "fix pc all precon/exp"
    if dump_prefix:
        fix_cmd += f" dump {dump_prefix}"
    commands = [
        f"plugin load {PLUGIN_SO}",
        "units metal",
        "atom_style atomic",
        "atom_modify map array sort 0 0.0",
        "boundary p p p",
        f"read_data {datafile}",
        *mass_cmds,
        *pair_cmds,
        "min_style precon/lbfgs",
        fix_cmd,
        "min_modify norm max",
        f"minimize 0.0 {fmax} {maxiter} {max(400, 200 * maxiter)}",
    ]
    lmp.commands_string("\n".join(commands))

    energy = float(lmp.get_thermo("pe"))
    natoms = lmp.get_natoms()
    positions = np.array(lmp.gather_atoms("x", 1, 3)).reshape(natoms, 3)
    lmp.close()

    result = dict(energy=energy, positions=positions, n_atoms=natoms)
    result.update(_parse_minimize_log(logfile))
    return result


@dataclass
class Stage8Parity:
    name: str
    engine: str
    n_atoms: int
    # assembly parity (C++ P vs Python assemble_P at the C++'s mu/r_NN/r_cut)
    P_shape: list[int]
    pattern_match: bool
    P_rel_norm: float
    # mu parity (C++ vs the Stage-3 reference)
    mu_cpp: float
    mu_ref: float
    mu_rel_diff: float
    # relaxation parity (C++ vs the Stage-5 reference)
    converged: bool
    n_force_cpp: int
    n_force_ref: int
    energy_cpp: float
    energy_ref: float
    energy_diff: float
    parity_ok: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _stage_summary(stage: str, name: str) -> dict | None:
    path = artifacts.ARTIFACT_DIR / stage / name / "summary.json"
    return artifacts.load_json(path) if path.exists() else None


def validate_structure(name: str, *, save: bool = True) -> Stage8Parity:
    """Run the C++ plugin on one structure and compare to the Python reference."""
    structure = by_name(name)
    atoms = structure.atoms.copy()

    # Ensure the Python references exist (cheap if already on disk).
    if _stage_summary("stage3", name) is None:
        from .mu import compare_mu
        compare_mu(structure)
    if _stage_summary("stage5", name) is None:
        from .relax import compare_relaxation
        compare_relaxation(structure)

    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "precon"
        run = run_lammps_cpp(atoms, structure.engine, fmax=structure.fmax,
                             maxiter=2000, workdir=tmp, dump_prefix=str(prefix))
        meta = json.loads((prefix.with_suffix(".json")).read_text())
        P_cpp = scipy.io.mmread(str(prefix.with_suffix(".mtx"))).tocsr()

    # --- assembly parity: Python P at the C++'s own mu / r_NN / r_cut --------
    P_py = assemble_P(atoms, mu=meta["mu"], r_NN=meta["r_NN"],
                      r_cut=meta["r_cut"], neighbor_list=ase_neighbor_list)
    A = P_cpp.toarray()
    B = P_py.toarray()
    pattern_match = bool(np.array_equal(A != 0.0, B != 0.0))
    P_rel_norm = float(np.linalg.norm(A - B) / np.linalg.norm(B))

    # --- mu parity ----------------------------------------------------------
    mu_ref = float(_stage_summary("stage3", name)["mu_fd"])
    mu_cpp = float(meta["mu"])
    mu_rel_diff = abs(mu_cpp - mu_ref) / abs(mu_ref)

    # --- relaxation parity --------------------------------------------------
    s5 = _stage_summary("stage5", name)
    energy_ref = float(s5["ase_e_final"])
    n_force_ref = int(s5["ase_n_force"])
    energy_diff = abs(run["energy"] - energy_ref)

    parity_ok = (
        pattern_match
        and P_rel_norm < 1e-8
        and mu_rel_diff < 2e-2
        and bool(run["converged"])
        and energy_diff < 1e-3
        and 0 < run["n_force"] <= 3 * n_force_ref
    )

    result = Stage8Parity(
        name=name,
        engine=structure.engine,
        n_atoms=run["n_atoms"],
        P_shape=list(A.shape),
        pattern_match=pattern_match,
        P_rel_norm=P_rel_norm,
        mu_cpp=mu_cpp,
        mu_ref=mu_ref,
        mu_rel_diff=mu_rel_diff,
        converged=bool(run["converged"]),
        n_force_cpp=int(run["n_force"]),
        n_force_ref=n_force_ref,
        energy_cpp=float(run["energy"]),
        energy_ref=energy_ref,
        energy_diff=energy_diff,
        parity_ok=parity_ok,
    )
    if save:
        d = artifacts.stage_dir("stage8", name)
        artifacts.save_json(d / "summary.json", result.as_dict())
    return result


# Structures with both Stage-3 and Stage-5 references; LaAlO3 exercises the
# triclinic (rotated-frame) path.
STAGE8_CASES = ["Cu_fcc", "MgO_x2", "Si_slab", "LaAlO3"]


def run_all(save: bool = True) -> list[Stage8Parity]:
    results = []
    for name in STAGE8_CASES:
        print(f"[stage8] {name}")
        r = validate_structure(name, save=save)
        results.append(r)
        print(f"         assembly: pattern_match={r.pattern_match} "
              f"P_rel_norm={r.P_rel_norm:.2e}")
        print(f"         mu: cpp={r.mu_cpp:.6f} ref={r.mu_ref:.6f} "
              f"(rel diff {r.mu_rel_diff:.2e})")
        print(f"         relax: converged={r.converged} "
              f"force_evals cpp={r.n_force_cpp} ref={r.n_force_ref}  "
              f"E cpp={r.energy_cpp:.6f} ref={r.energy_ref:.6f} "
              f"(dE {r.energy_diff:.2e})")
        print(f"         parity_ok={r.parity_ok}")
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
    sys.stdout.flush()
    os._exit(code)

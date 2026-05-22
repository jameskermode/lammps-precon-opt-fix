"""Stage 8c — variable-cell relaxation validation of the C++ plugin.

Runs the C++ `min_style precon/lbfgs` together with `fix box/relax` (LAMMPS'
variable-cell minimize mechanism) and confirms it reproduces the Stage-6
Python reference (`vcrelax.py`): the relaxation reaches the same minimum —
final energy and cell volume — and genuinely moves the cell.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import ase.io
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

from . import artifacts
from .cpp_parity import PLUGIN_SO, _pair_commands, _parse_minimize_log
from .vcrelax import stage6_cases


def run_lammps_vc(atoms: Atoms, engine: str, *, fmax: float,
                  workdir: Path) -> dict:
    """Run a variable-cell `min_style precon/lbfgs` relaxation (fix box/relax)."""
    from lammps import lammps

    workdir = Path(workdir)
    species, pair_cmds = _pair_commands(atoms, engine)
    datafile = workdir / "structure.data"
    ase.io.write(datafile, atoms, format="lammps-data",
                 specorder=species, atom_style="atomic")
    logfile = workdir / "log.lammps"

    lmp = lammps(cmdargs=["-log", str(logfile), "-screen", "none", "-nocite"])
    mass_cmds = [f"mass {k + 1} {atomic_masses[atomic_numbers[s]]}"
                 for k, s in enumerate(species)]
    commands = [
        f"plugin load {PLUGIN_SO}",
        "units metal",
        "atom_style atomic",
        "atom_modify map array sort 0 0.0",
        "boundary p p p",
        f"read_data {datafile}",
        "change_box all triclinic",  # enable shear DOF for `box/relax tri`
        *mass_cmds,
        *pair_cmds,
        "min_style precon/lbfgs",
        "fix pc all precon/exp",
        "fix br all box/relax tri 0.0 vmax 0.01",  # full variable cell, P=0
        "min_modify norm max",
        f"minimize 0.0 {fmax} 3000 600000",
    ]
    lmp.commands_string("\n".join(commands))
    energy = float(lmp.get_thermo("pe"))
    volume = float(lmp.get_thermo("vol"))
    lmp.close()

    result = dict(energy=energy, volume=volume)
    result.update(_parse_minimize_log(logfile))
    return result


@dataclass
class VCParity:
    name: str
    converged: bool
    energy_cpp: float
    energy_ref: float
    energy_diff: float
    volume_cpp: float
    volume_ref: float
    volume_rel_diff: float
    cell_change: float          # |V_final - V_initial| / V_initial
    n_force: int
    parity_ok: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def validate_vc(name: str, atoms0: Atoms, engine: str, fmax: float,
                *, save: bool = True) -> VCParity:
    """Run the C++ variable-cell relaxation and compare to the Stage-6 reference."""
    with tempfile.TemporaryDirectory() as tmp:
        run = run_lammps_vc(atoms0, engine, fmax=fmax, workdir=Path(tmp))

    s6_dir = artifacts.ARTIFACT_DIR / "stage6" / name
    summary = artifacts.load_json(s6_dir / "summary.json")
    energy_ref = float(summary["ase_energy"])
    ref_relaxed = ase.io.read(s6_dir / "relaxed_ase.xyz")
    volume_ref = float(ref_relaxed.get_volume())
    volume_init = float(atoms0.get_volume())

    energy_diff = abs(run["energy"] - energy_ref)
    volume_rel_diff = abs(run["volume"] - volume_ref) / volume_ref
    cell_change = abs(run["volume"] - volume_init) / volume_init

    parity_ok = (bool(run["converged"]) and energy_diff < 1e-3
                 and volume_rel_diff < 2e-3)

    result = VCParity(
        name=name,
        converged=bool(run["converged"]),
        energy_cpp=float(run["energy"]),
        energy_ref=energy_ref,
        energy_diff=energy_diff,
        volume_cpp=float(run["volume"]),
        volume_ref=volume_ref,
        volume_rel_diff=volume_rel_diff,
        cell_change=cell_change,
        n_force=int(run["n_force"]) if run["n_force"] else -1,
        parity_ok=parity_ok,
    )
    if save:
        d = artifacts.stage_dir("stage8", name + "_vc")
        artifacts.save_json(d / "summary.json", result.as_dict())
    return result


def run_all() -> list[VCParity]:
    results = []
    for name, atoms0, engine, fmax in stage6_cases():
        print(f"[stage8c] {name} ({len(atoms0)} atoms, {engine})")
        r = validate_vc(name, atoms0, engine, fmax)
        results.append(r)
        print(f"         converged={r.converged}  force_evals={r.n_force}  "
              f"cell change={r.cell_change * 100:.2f}%")
        print(f"         E   cpp={r.energy_cpp:.6f} ref={r.energy_ref:.6f} "
              f"(dE {r.energy_diff:.2e})")
        print(f"         vol cpp={r.volume_cpp:.4f} ref={r.volume_ref:.4f} "
              f"(rel diff {r.volume_rel_diff:.2e})  parity_ok={r.parity_ok}")
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

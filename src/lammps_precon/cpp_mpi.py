"""Stage 8b — MPI / domain-decomposition validation of the C++ plugin.

Runs the `lmp` binary under ``mpirun`` on 1, 2 and 4 ranks and confirms the
domain-decomposed `min_style precon/lbfgs` reproduces the single-rank result
(final energy and force-evaluation count) — i.e. the distributed Jacobi-CG and
the MPI-aware LBFGS are correct.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import ase.io
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

from .calculators import MODEL_JSON, REPO_ROOT, eam_potential_path
from .cpp_parity import PLUGIN_SO
from .structures import by_name

VENV_LIB = REPO_ROOT / ".venv" / "lib"


def _lmp_binary() -> Path:
    """Locate the LAMMPS `lmp` executable in a build directory under lammps/."""
    lammps_dir = REPO_ROOT / "lammps-symmetrix" / "lammps"
    # match any build dir name: `build`, `build-ci`, `build-HSW-none`, ...
    builds = sorted(lammps_dir.glob("build*/lmp"))
    if not builds:
        raise FileNotFoundError(
            f"LAMMPS 'lmp' binary not found under {lammps_dir}/build*/")
    return builds[0]


def _pair_block(atoms: Atoms, engine: str) -> tuple[list[str], list[str]]:
    species = sorted(set(atoms.get_chemical_symbols()))
    if engine == "mace":
        pair = ["pair_style symmetrix/mace",
                f"pair_coeff * * {MODEL_JSON} {' '.join(species)}"]
    elif engine == "eam":
        pair = ["pair_style eam", f"pair_coeff * * {eam_potential_path()}"]
    else:
        raise ValueError(engine)
    masses = [f"mass {k + 1} {atomic_masses[atomic_numbers[s]]}"
              for k, s in enumerate(species)]
    return species, masses + pair


@dataclass
class MpiRun:
    nprocs: int
    energy: float
    n_steps: int
    n_force: int
    converged: bool


def run_lammps_mpi(atoms: Atoms, engine: str, *, fmax: float, nprocs: int,
                   workdir: Path) -> MpiRun:
    """Run one `min_style precon/lbfgs` relaxation under ``mpirun -np nprocs``."""
    workdir = Path(workdir)
    species, pair_cmds = _pair_block(atoms, engine)
    datafile = workdir / "structure.data"
    ase.io.write(datafile, atoms, format="lammps-data",
                 specorder=species, atom_style="atomic")

    script = workdir / f"in.np{nprocs}"
    screen = workdir / f"screen.np{nprocs}"
    script.write_text("\n".join([
        f"plugin load {PLUGIN_SO}",
        "units metal",
        "atom_style atomic",
        "atom_modify map array",
        "boundary p p p",
        f"read_data {datafile}",
        *pair_cmds,
        "min_style precon/lbfgs",
        "fix pc all precon/exp",
        "min_modify norm max",
        f"minimize 0.0 {fmax} 2000 400000",
        "variable efinal equal pe",
        'print "RESULT_PE ${efinal}"',
    ]) + "\n")

    cmd = (f"module load foss/2023b 2>/dev/null; "
           f"export LD_LIBRARY_PATH={VENV_LIB}:$LD_LIBRARY_PATH; "
           f"mpirun -np {nprocs} {_lmp_binary()} -in {script} "
           f"-screen {screen} -log none")
    # Scrub inherited MPI/PMIx variables: if the calling process already
    # MPI_Init'd (e.g. an earlier in-process `lammps()`), a nested mpirun
    # would otherwise think it is part of that job.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OMPI_", "PMIX_", "PMI_", "OPAL_", "MPIR_"))}
    subprocess.run(["bash", "-lc", cmd], check=True, capture_output=True,
                   env=env)

    text = screen.read_text(errors="ignore")
    m = re.search(r"RESULT_PE\s+(\S+)", text)
    energy = float(m.group(1))
    m = re.search(r"force evaluations\s*=\s*(\d+)\s+(\d+)", text)
    n_steps, n_force = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
    m = re.search(r"Stopping criterion\s*=\s*(.+)", text)
    stop = m.group(1).strip() if m else ""
    return MpiRun(nprocs=nprocs, energy=energy, n_steps=n_steps,
                  n_force=n_force,
                  converged=stop in ("force tolerance", "energy tolerance"))


@dataclass
class MpiParity:
    name: str
    engine: str
    runs: dict          # nprocs -> MpiRun
    max_energy_diff: float
    max_force_diff: int
    all_converged: bool
    parity_ok: bool


def compare_ranks(name: str, ranks=(1, 2, 4)) -> MpiParity:
    """Relax one structure at each rank count and compare to the 1-rank run."""
    structure = by_name(name)
    runs: dict[int, MpiRun] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for n in ranks:
            runs[n] = run_lammps_mpi(structure.atoms.copy(), structure.engine,
                                     fmax=structure.fmax, nprocs=n,
                                     workdir=Path(tmp))
    serial = runs[ranks[0]]
    max_de = max(abs(runs[n].energy - serial.energy) for n in ranks)
    max_df = max(abs(runs[n].n_force - serial.n_force) for n in ranks)
    all_conv = all(runs[n].converged for n in ranks)
    parity_ok = all_conv and max_de < 1e-5 and max_df <= 2
    return MpiParity(name=name, engine=structure.engine, runs=runs,
                     max_energy_diff=max_de, max_force_diff=max_df,
                     all_converged=all_conv, parity_ok=parity_ok)


MPI_CASES = ["Cu_fcc", "MgO_x2", "Si_slab"]


def run_all() -> list[MpiParity]:
    results = []
    for name in MPI_CASES:
        print(f"[stage8b] {name}")
        r = compare_ranks(name)
        results.append(r)
        for n, run in sorted(r.runs.items()):
            print(f"         np={n}: E={run.energy:.6f}  "
                  f"force_evals={run.n_force}  steps={run.n_steps}  "
                  f"converged={run.converged}")
        print(f"         max dE vs serial={r.max_energy_diff:.2e}  "
              f"max d(force_evals)={r.max_force_diff}  "
              f"parity_ok={r.parity_ok}")
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

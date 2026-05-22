"""CI smoke tests — exercise the C++ plugin on the classical-EAM path.

Unlike the Stage-8 test files, which gate the whole module on
``envcheck.MACE_OK`` (Symmetrix + the MACE foundation model), these gate only
on the plugin being built and the ``lammps`` Python module being importable —
so they run in GitHub Actions CI without the 266 MB MACE model.

They cover the plugin end-to-end on the Cu/EAM case: assembly + mu +
relaxation (Stage 8a), MPI domain decomposition (8b), and variable cell (8c).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import envcheck
from lammps_precon.cpp_mpi import compare_ranks
from lammps_precon.cpp_parity import PLUGIN_SO, validate_structure
from lammps_precon.cpp_vcrelax import run_lammps_vc
from lammps_precon.structures import cu_fcc

pytestmark = pytest.mark.skipif(
    not (envcheck.LAMMPS_OK and PLUGIN_SO.exists()),
    reason=("C++ plugin not built or the lammps Python module is unavailable "
            "— build LAMMPS (PLUGIN package) and cpp/scripts/build_plugin.sh"))


def test_eam_assembly_mu_relax():
    """Stage 8a — C++ assembly/mu match the Python reference; relaxation converges."""
    r = validate_structure("Cu_fcc")
    print(f"\nCu/EAM: P rel-norm {r.P_rel_norm:.1e}, mu {r.mu_cpp:.4f}, "
          f"relax dE {r.energy_diff:.1e} eV in {r.n_force_cpp} force evals")
    assert r.pattern_match
    assert r.P_rel_norm < 1e-8
    assert r.converged
    assert r.parity_ok


def test_eam_mpi():
    """Stage 8b — 1/2/4 MPI ranks reproduce the serial relaxation."""
    r = compare_ranks("Cu_fcc")
    print(f"\nCu/EAM MPI: max dE vs serial {r.max_energy_diff:.1e}, "
          f"max d(force_evals) {r.max_force_diff}")
    assert r.all_converged
    assert r.parity_ok


def test_eam_variable_cell():
    """Stage 8c — variable-cell relaxation of a strained Cu cell converges."""
    atoms = cu_fcc()
    atoms.set_cell(atoms.cell * 1.02, scale_atoms=True)  # 2 % isotropic strain
    volume_init = atoms.get_volume()
    with tempfile.TemporaryDirectory() as tmp:
        run = run_lammps_vc(atoms, "eam", fmax=1e-3, workdir=Path(tmp))
    cell_change = abs(run["volume"] - volume_init) / volume_init
    print(f"\nCu/EAM variable cell: converged={run['converged']}, "
          f"cell volume change {cell_change * 100:.2f}%")
    assert run["converged"]
    assert cell_change > 5e-3  # the cell DOF genuinely relaxed

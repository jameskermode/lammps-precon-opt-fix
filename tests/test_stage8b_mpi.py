"""Stage 8b — MPI / domain-decomposition parity checks.

The domain-decomposed `min_style precon/lbfgs` (distributed Jacobi-CG +
MPI-aware LBFGS) must reproduce the single-rank relaxation when run on 2 and 4
MPI ranks: same final energy, same force-evaluation count.
"""
from __future__ import annotations

import pytest

import envcheck
from lammps_precon.cpp_mpi import MPI_CASES, compare_ranks
from lammps_precon.cpp_parity import PLUGIN_SO

pytestmark = pytest.mark.skipif(
    not (envcheck.MACE_OK and PLUGIN_SO.exists()),
    reason=("C++ plugin not built or environment not ready — run "
            "scripts/build_lammps.sh and cpp/scripts/build_plugin.sh"))


@pytest.fixture(scope="module", params=MPI_CASES)
def parity(request):
    name = request.param
    return name, compare_ranks(name)


def test_all_ranks_converge(parity):
    name, r = parity
    for n, run in sorted(r.runs.items()):
        assert run.converged, f"{name}: relaxation on {n} rank(s) did not converge"


def test_energy_matches_serial(parity):
    name, r = parity
    print(f"\n{name}: energies "
          f"{ {n: round(run.energy, 6) for n, run in sorted(r.runs.items())} }; "
          f"max dE vs serial {r.max_energy_diff:.2e}")
    assert r.max_energy_diff < 1e-5, (
        f"{name}: MPI final energy differs from serial by "
        f"{r.max_energy_diff:.2e} eV")


def test_force_eval_count_matches_serial(parity):
    name, r = parity
    print(f"{name}: force evals "
          f"{ {n: run.n_force for n, run in sorted(r.runs.items())} }")
    assert r.max_force_diff <= 2, (
        f"{name}: MPI force-eval count differs from serial by "
        f"{r.max_force_diff}")


def test_parity_ok(parity):
    name, r = parity
    assert r.parity_ok, f"{name}: Stage 8b MPI parity failed"

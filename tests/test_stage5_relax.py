"""Stage 5 — fixed-cell preconditioned-relaxation parity checks.

Full ``PreconLBFGS`` relaxations driven on LAMMPS forces with our two-tier
``Exp`` solver must reproduce ASE's stock ``Exp`` relaxation: same final
energy and structure, matching force-evaluation counts, same convergence.
"""
from __future__ import annotations

import pytest

import envcheck
from lammps_precon.relax import FIXED_CELL_CASES, compare_relaxation
from lammps_precon.structures import by_name

pytestmark = pytest.mark.skipif(not envcheck.MACE_OK, reason=envcheck.MACE_REASON)


@pytest.fixture(scope="module", params=FIXED_CELL_CASES)
def parity(request):
    name = request.param
    return name, compare_relaxation(by_name(name))


def test_all_relaxations_converge(parity):
    name, res = parity
    assert res.ase_converged, f"{name}: ASE reference relaxation did not converge"
    assert res.direct_converged, f"{name}: direct-tier relaxation did not converge"
    assert res.cg_converged, f"{name}: CG-tier relaxation did not converge"


def test_direct_tier_reproduces_ase(parity):
    name, res = parity
    # The direct tier is numerically equivalent to ASE's solve, so the whole
    # relaxation must reproduce ASE essentially exactly.
    assert res.direct_n_force == res.ase_n_force, (
        f"{name}: direct {res.direct_n_force} vs ASE {res.ase_n_force} "
        "force evaluations")
    assert res.direct_energy_diff < 1e-6, f"{name}: dE={res.direct_energy_diff:.2e}"
    assert res.direct_rmsd < 1e-6, f"{name}: RMSD={res.direct_rmsd:.2e}"


def test_cg_tier_matches_ase(parity):
    name, res = parity
    print(f"\n{name}: force_evals ASE={res.ase_n_force} "
          f"direct={res.direct_n_force} cg={res.cg_n_force}; "
          f"cg dE={res.cg_energy_diff:.2e} RMSD={res.cg_rmsd:.2e}")
    assert res.cg_energy_diff < 1e-3, (
        f"{name}: CG final energy off by {res.cg_energy_diff:.2e} eV")
    assert res.cg_rmsd < 1e-2, f"{name}: CG final structure RMSD {res.cg_rmsd:.2e}"
    force_tol = max(3, int(0.25 * res.ase_n_force))
    assert abs(res.cg_n_force - res.ase_n_force) <= force_tol, (
        f"{name}: CG {res.cg_n_force} vs ASE {res.ase_n_force} force evals")


def test_force_eval_count_is_sane(parity):
    name, res = parity
    # Where ASE Exp converges, ours converges, with a sane force-eval count.
    assert 1 <= res.ase_n_force < 200, (
        f"{name}: implausible force-eval count {res.ase_n_force}")


def test_parity_ok(parity):
    name, res = parity
    assert res.parity_ok, f"{name}: Stage 5 relaxation parity failed ({res})"

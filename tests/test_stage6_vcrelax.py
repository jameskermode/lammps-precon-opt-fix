"""Stage 6 — variable-cell preconditioned-relaxation parity checks.

The bug-prone path: cell DOF, the cell-metric ``mu_c``, and the historical
``r_cut=None`` bug locus. Variable-cell ``PreconLBFGS`` relaxations with our
two-tier solver must reproduce ASE's stock ``Exp`` — final cell, structure,
energy and force-evaluation count.
"""
from __future__ import annotations

import pytest

import envcheck
from lammps_precon.vcrelax import compare_vc_relaxation, stage6_cases

pytestmark = pytest.mark.skipif(not envcheck.MACE_OK, reason=envcheck.MACE_REASON)

CASES = stage6_cases()


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def parity(request):
    name, atoms0, engine, fmax = request.param
    return name, compare_vc_relaxation(name, atoms0, engine, fmax=fmax)


def test_rcut_resolved_before_mu_estimation(parity):
    """The historical r_cut=None variable-cell bug locus."""
    name, res = parity
    assert res.r_cut_resolved_before_mu, (
        f"{name}: r_cut still None when estimate_mu was entered")
    assert res.r_NN_resolved_before_mu, (
        f"{name}: r_NN still None when estimate_mu was entered")


def test_cell_block_is_mu_c_identity(parity):
    name, res = parity
    assert res.cell_block_correct, (
        f"{name}: cell block of P is not mu_c * I (decoupled)")
    assert res.mu_c >= 1.0, f"{name}: mu_c={res.mu_c} below the 1.0 floor"


def test_cell_dof_genuinely_exercised(parity):
    name, res = parity
    print(f"\n{name}: cell change during relaxation = "
          f"{res.cell_change * 100:.2f}%")
    assert res.cell_change > 5e-3, (
        f"{name}: cell barely moved ({res.cell_change * 100:.2f}%) — the "
        "cell-DOF path is not genuinely exercised")


def test_all_relaxations_converge(parity):
    name, res = parity
    assert res.ase_converged, f"{name}: ASE variable-cell relaxation failed"
    assert res.direct_converged, f"{name}: direct-tier relaxation failed"
    assert res.cg_converged, f"{name}: CG-tier relaxation failed"


def test_direct_tier_reproduces_ase(parity):
    name, res = parity
    assert res.direct_n_force == res.ase_n_force, (
        f"{name}: direct {res.direct_n_force} vs ASE {res.ase_n_force} "
        "force evaluations")
    assert res.direct_energy_diff < 1e-6, f"{name}: dE={res.direct_energy_diff:.2e}"
    assert res.direct_rmsd < 1e-5, f"{name}: RMSD={res.direct_rmsd:.2e}"
    assert res.direct_cell_diff < 1e-6, (
        f"{name}: cell diff={res.direct_cell_diff:.2e}")


def test_cg_tier_matches_ase(parity):
    name, res = parity
    print(f"{name}: force_evals ASE={res.ase_n_force} direct={res.direct_n_force} "
          f"cg={res.cg_n_force}; cg dE={res.cg_energy_diff:.2e} "
          f"RMSD={res.cg_rmsd:.2e} cell_diff={res.cg_cell_diff:.2e}")
    assert res.cg_energy_diff < 1e-3, f"{name}: cg dE={res.cg_energy_diff:.2e}"
    assert res.cg_rmsd < 1e-2, f"{name}: cg RMSD={res.cg_rmsd:.2e}"
    assert res.cg_cell_diff < 1e-3, f"{name}: cg cell diff={res.cg_cell_diff:.2e}"
    force_tol = max(3, int(0.25 * res.ase_n_force))
    assert abs(res.cg_n_force - res.ase_n_force) <= force_tol, (
        f"{name}: cg {res.cg_n_force} vs ASE {res.ase_n_force} force evals")


def test_parity_ok(parity):
    name, res = parity
    assert res.parity_ok, f"{name}: Stage 6 variable-cell parity failed ({res})"

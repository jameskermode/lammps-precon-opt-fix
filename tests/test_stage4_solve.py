"""Stage 4 — two-tier sparse-solver parity checks.

Validates the ``P s = b`` solve, both tiers, against ASE's ``Precon.solve``:

* 4a — the direct tier is an exact factorisation (residual at machine
  precision) and matches ASE's solve;
* 4b — Jacobi-preconditioned CG converges to the same ``s`` as the direct
  solve, with a low iteration count (well-conditioned SPD);
* 4c — the direct and iterative tiers agree at the DOF threshold.
"""
from __future__ import annotations

import pytest

import envcheck
from lammps_precon.solve import cg_scaling, compare_solve

pytestmark = pytest.mark.skipif(not envcheck.MACE_OK, reason=envcheck.MACE_REASON)

# Ascending DOF — lets the CG iteration-count trend be read across the suite.
CASES = ["MgO_x2", "gamma_Al2O3", "Cu_fcc", "Si_slab", "MgO_x3", "LaAlO3",
         "iceVIII"]


@pytest.fixture(scope="module", params=CASES)
def parity(request):
    name = request.param
    return name, compare_solve(name)


def test_direct_residual_at_machine_precision(parity):
    name, res = parity
    assert res.direct_residual < 1e-10, (
        f"{name}: direct-solve residual {res.direct_residual:.2e} too large")


def test_direct_matches_ase(parity):
    name, res = parity
    assert res.rel_direct_vs_ase < 1e-8, (
        f"{name}: direct solve vs ASE = {res.rel_direct_vs_ase:.2e}")


def test_cg_converges(parity):
    name, res = parity
    print(f"\n{name}: n_dof={res.n_dof}  kappa={res.condition_number:.1f}  "
          f"CG iters={res.cg_iterations}  CG residual={res.cg_residual:.2e}")
    assert res.cg_converged, f"{name}: CG did not converge"


def test_cg_matches_direct(parity):
    name, res = parity
    assert res.rel_cg_vs_direct < 1e-8, (
        f"{name}: CG vs direct = {res.rel_cg_vs_direct:.2e}")
    assert res.rel_cg_vs_ase < 1e-8, (
        f"{name}: CG vs ASE = {res.rel_cg_vs_ase:.2e}")


def test_cg_iteration_count_is_low(parity):
    name, res = parity
    # A well-conditioned SPD system: CG must converge in far fewer than n
    # iterations (the trivial CG bound).
    assert res.cg_iterations < res.n_dof, (
        f"{name}: CG took {res.cg_iterations} iters for n_dof={res.n_dof}")


def test_tier_switch_consistent(parity):
    name, res = parity
    assert res.rel_tier_switch < 1e-8, (
        f"{name}: direct/iterative tiers disagree at the threshold "
        f"({res.rel_tier_switch:.2e})")


def test_parity_ok(parity):
    name, res = parity
    assert res.parity_ok, f"{name}: Stage 4 solver parity failed ({res})"


def test_cg_scales_to_large_systems():
    """CG iteration count and conditioning must stay bounded as N grows.

    Assembling P is purely geometric, so this reaches DOF counts (~40k) well
    above the direct tier's practical range — the iterative tier's real regime.
    """
    rows = cg_scaling(sizes=(4, 8, 12), save=False)  # ~1.5k -> ~41k DOF
    assert all(r["cg_converged"] for r in rows), "CG failed to converge"
    iterations = [r["cg_iterations"] for r in rows]
    kappa = [r["condition_number"] for r in rows]
    n_dof = [r["n_dof"] for r in rows]
    print(f"\nCG scaling: n_dof={n_dof} iters={iterations} "
          f"kappa={[round(k, 1) for k in kappa]}")
    # lambda_min is fixed at mu*c_stab and lambda_max saturates with the
    # coordination number, so the condition number stays bounded.
    assert max(kappa) < 300, f"condition number grew unexpectedly: {kappa}"
    # ~27x more DOF must cost far less than 27x the iterations (sub-linear).
    assert iterations[-1] < 3 * iterations[0], (
        f"CG iteration count scaled poorly: {iterations} for n_dof={n_dof}")
    assert max(iterations) < 120

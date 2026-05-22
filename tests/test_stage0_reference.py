"""Stage 0 — ASE reference harness parity checks.

Confirms the reference ``Exp`` preconditioned LBFGS runs on the LAMMPS force
engine and produces sane gold-standard artifacts.
"""
from __future__ import annotations

import pytest

import envcheck
from lammps_precon.reference import run_reference
from lammps_precon.structures import by_name

pytestmark = pytest.mark.skipif(not envcheck.MACE_OK, reason=envcheck.MACE_REASON)

# Headline cases: Cu/EAM (classical, instant), MgO x2 (small MACE), Si slab
# (the VALIDATION_REPORT reference case).
RELAX_CASES = ["Cu_fcc", "MgO_x2", "Si_slab"]


@pytest.fixture(scope="module")
def references() -> dict:
    return {name: run_reference(by_name(name)) for name in RELAX_CASES}


@pytest.mark.parametrize("name", RELAX_CASES)
def test_reference_artifacts_are_sane(references, name):
    r = references[name]
    n = r.n_atoms
    assert r.r_NN > 0.5, f"{name}: implausible r_NN={r.r_NN}"
    assert r.r_cut == pytest.approx(2.0 * r.r_NN)
    assert r.mu >= 1.0, f"{name}: mu={r.mu} should be capped at >= 1.0"
    # P is (3N x 3N), symmetric, and graph-Laplacian-sparse (not dense).
    assert r.P_shape == [3 * n, 3 * n]
    assert r.P_symmetry < 1e-8, f"{name}: P not symmetric ({r.P_symmetry})"
    assert 3 * n <= r.P_nnz < (3 * n) ** 2


@pytest.mark.parametrize("name", RELAX_CASES)
def test_reference_relaxation_converges(references, name):
    r = references[name]
    assert r.relaxed and r.converged, f"{name}: relaxation did not converge"
    s = by_name(name)
    assert r.fmax_final < s.fmax * 1.5, f"{name}: fmax_final={r.fmax_final}"
    assert r.n_steps is not None and r.n_steps > 0


def test_si_slab_force_eval_count_in_ballpark(references):
    """VALIDATION_REPORT reports ~33-40 Exp force-evals for the Si slab.

    Symmetrix is not bit-identical to mace-torch, so an exact match is not
    expected — only that the count is in the same ballpark (a wildly different
    number would indicate a broken preconditioner path).
    """
    r = references["Si_slab"]
    print(f"\nSi slab: {r.n_force_total} force evals "
          f"({r.n_force_setup} setup + {r.n_force_total - r.n_force_setup} "
          f"relaxation), {r.n_steps} LBFGS steps")
    assert 15 <= r.n_force_total <= 90, (
        f"Si slab force-eval count {r.n_force_total} far outside the "
        "~33-40 ballpark — preconditioner likely misbehaving")

"""Stage 2 — P-matrix assembly parity checks.

The ``P`` assembled from the LAMMPS pair list must match ASE's ``P`` for the
same structure and the same ``mu``: identical sparsity pattern, relative
Frobenius difference < 1e-10, symmetric, and SPD with smallest eigenvalue
equal to the ``mu * c_stab`` floor.
"""
from __future__ import annotations

import pytest

import envcheck
from lammps_precon.assembly import compare_assembly
from lammps_precon.structures import by_name

pytestmark = pytest.mark.skipif(not envcheck.MACE_OK, reason=envcheck.MACE_REASON)

# Single-element (Si, Cu), two-element (MgO) and multi-element (LaAlO3 — the
# case the validation project found canonicalisation-type bugs in).
CASES = ["Cu_fcc", "MgO_x2", "Si_slab", "LaAlO3"]


@pytest.fixture(scope="module", params=CASES)
def parity(request):
    name = request.param
    return name, compare_assembly(by_name(name))


def test_sparsity_pattern_matches(parity):
    name, res = parity
    assert res.nnz_ase == res.nnz_lammps, (
        f"{name}: nnz differ — ASE {res.nnz_ase}, LAMMPS {res.nnz_lammps}")
    assert res.pattern_match, f"{name}: P sparsity patterns differ"


def test_relative_norm_below_tolerance(parity):
    name, res = parity
    print(f"\n{name}: rel-norm(P_lammps - P_ase) = {res.rel_norm_diff:.2e}")
    assert res.rel_norm_diff < 1e-10, (
        f"{name}: relative norm difference {res.rel_norm_diff:.2e} >= 1e-10")


def test_P_is_symmetric(parity):
    name, res = parity
    assert res.symmetry < 1e-12, (
        f"{name}: norm(P - P^T) = {res.symmetry:.2e} >= 1e-12")


def test_P_is_spd(parity):
    name, res = parity
    print(f"{name}: min eigenvalue = {res.min_eigenvalue:.8f}, "
          f"mu*c_stab floor = {res.c_stab_floor:.8f}")
    assert res.cholesky_ok, f"{name}: Cholesky factorisation failed"
    assert res.min_eigenvalue > 0.0, f"{name}: P not positive definite"
    # The Exp P is a graph Laplacian plus mu*c_stab on the diagonal, so its
    # smallest eigenvalue must equal that stabilisation floor.
    assert res.min_eigenvalue == pytest.approx(res.c_stab_floor, rel=1e-6), (
        f"{name}: min eigenvalue {res.min_eigenvalue} != mu*c_stab "
        f"{res.c_stab_floor}")


def test_parity_ok(parity):
    name, res = parity
    assert res.parity_ok, f"{name}: Stage 2 assembly parity failed ({res})"

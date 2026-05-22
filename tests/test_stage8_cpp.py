"""Stage 8a — C++ LAMMPS-native plugin parity checks.

Drives the C++ `fix precon/exp` + `min_style precon/lbfgs` plugin and checks it
against the validated Python implementation: the assembled P, the estimated mu,
and the relaxation (convergence + final energy).
"""
from __future__ import annotations

import pytest

import envcheck
from lammps_precon.cpp_parity import PLUGIN_SO, STAGE8_CASES, validate_structure

pytestmark = pytest.mark.skipif(
    not (envcheck.MACE_OK and PLUGIN_SO.exists()),
    reason=("C++ plugin not built or environment not ready — run "
            "scripts/build_lammps.sh and cpp/scripts/build_plugin.sh"))


@pytest.fixture(scope="module", params=STAGE8_CASES)
def parity(request):
    name = request.param
    return name, validate_structure(name)


def test_assembly_matches_python(parity):
    name, r = parity
    assert r.pattern_match, f"{name}: C++ P sparsity pattern differs"
    assert r.P_rel_norm < 1e-8, (
        f"{name}: C++ vs Python P relative norm {r.P_rel_norm:.2e}")


def test_mu_matches_reference(parity):
    name, r = parity
    print(f"\n{name}: mu cpp={r.mu_cpp:.8f} ref={r.mu_ref:.8f} "
          f"(rel diff {r.mu_rel_diff:.2e})")
    assert r.mu_rel_diff < 2e-2, (
        f"{name}: C++ mu {r.mu_cpp} vs reference {r.mu_ref}")


def test_relaxation_converges(parity):
    name, r = parity
    assert r.converged, f"{name}: C++ min_style precon/lbfgs did not converge"


def test_relaxation_reaches_same_minimum(parity):
    name, r = parity
    print(f"{name}: final energy cpp={r.energy_cpp:.6f} "
          f"ref={r.energy_ref:.6f} (diff {r.energy_diff:.2e} eV)")
    assert r.energy_diff < 1e-3, (
        f"{name}: C++ final energy off by {r.energy_diff:.2e} eV")


def test_force_eval_count_is_sane(parity):
    name, r = parity
    print(f"{name}: force evals cpp={r.n_force_cpp} ref={r.n_force_ref}")
    # The C++ line search differs from ASE's, so an exact count match is not
    # expected — only that it is in the same ballpark (no blow-up / no bug).
    assert 0 < r.n_force_cpp <= 3 * r.n_force_ref, (
        f"{name}: C++ force-eval count {r.n_force_cpp} implausible "
        f"(reference {r.n_force_ref})")


def test_parity_ok(parity):
    name, r = parity
    assert r.parity_ok, f"{name}: Stage 8a C++ parity failed ({r})"

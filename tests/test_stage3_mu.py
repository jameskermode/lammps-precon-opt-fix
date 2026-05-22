"""Stage 3 — mu-estimation parity checks.

The independent finite-difference mu-estimator must reproduce ASE's
`estimate_mu` on LAMMPS forces: same probe displacement, exactly two probe
force evaluations, and matching mu. Exercised with both the MACE potential
and a classical EAM potential to confirm the probe is potential-agnostic.
"""
from __future__ import annotations

import pytest

import envcheck
from lammps_precon.mu import compare_mu
from lammps_precon.structures import by_name

pytestmark = pytest.mark.skipif(not envcheck.MACE_OK, reason=envcheck.MACE_REASON)

# Cu_fcc exercises the classical EAM path; the rest exercise MACE. Si,
# gamma_Al2O3 and iceVIII have mu > 1 (uncapped — a non-trivial value check).
CASES = ["Cu_fcc", "Si_slab", "MgO_x2", "LaAlO3", "gamma_Al2O3", "iceVIII"]


@pytest.fixture(scope="module", params=CASES)
def parity(request):
    name = request.param
    return name, compare_mu(by_name(name))


def test_two_probe_force_evaluations(parity):
    name, res = parity
    assert res.n_probes_ase == 2, (
        f"{name}: estimate_mu used {res.n_probes_ase} force evals, expected 2")


def test_probe_displacement_matches(parity):
    name, res = parity
    print(f"\n{name}: probe displacement match = {res.displacement_match:.2e} A")
    assert res.displacement_match < 1e-12, (
        f"{name}: FD displacement differs by {res.displacement_match:.2e} A")


def test_mu_matches_ase(parity):
    name, res = parity
    print(f"{name}: mu ase={res.mu_ase:.8f} fd={res.mu_fd:.8f} "
          f"(rel diff {res.mu_rel_diff:.2e}, raw fd {res.mu_raw_fd:.6f})")
    assert res.mu_rel_diff < 1e-10, (
        f"{name}: mu mismatch — ASE {res.mu_ase}, FD {res.mu_fd}")


def test_mu_is_physical(parity):
    name, res = parity
    assert res.mu_fd >= 1.0, f"{name}: mu={res.mu_fd} below the 1.0 floor"


def test_parity_ok(parity):
    name, res = parity
    assert res.parity_ok, f"{name}: Stage 3 mu parity failed ({res})"


def test_uncapped_case_is_nontrivial():
    """At least one MACE case must have mu > 1 so the value check is real."""
    _, res = "Si_slab", compare_mu(by_name("Si_slab"), save=False)
    assert not res.mu_capped and res.mu_fd > 1.0, (
        "expected Si_slab mu > 1 (uncapped) for a non-trivial parity check")

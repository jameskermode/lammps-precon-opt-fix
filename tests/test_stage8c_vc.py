"""Stage 8c — variable-cell relaxation parity checks.

The C++ `min_style precon/lbfgs` driving `fix box/relax` (variable cell) must
reach the Stage-6 Python reference minimum: same final energy and cell volume,
with the cell DOF genuinely exercised.
"""
from __future__ import annotations

import pytest

import envcheck
from lammps_precon.cpp_parity import PLUGIN_SO
from lammps_precon.cpp_vcrelax import validate_vc
from lammps_precon.vcrelax import stage6_cases

pytestmark = pytest.mark.skipif(
    not (envcheck.MACE_OK and PLUGIN_SO.exists()),
    reason=("C++ plugin not built or environment not ready — run "
            "scripts/build_lammps.sh and cpp/scripts/build_plugin.sh"))

CASES = stage6_cases()


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def parity(request):
    name, atoms0, engine, fmax = request.param
    return name, validate_vc(name, atoms0, engine, fmax)


def test_converges(parity):
    name, r = parity
    assert r.converged, f"{name}: variable-cell relaxation did not converge"


def test_energy_matches_stage6(parity):
    name, r = parity
    print(f"\n{name}: final energy cpp={r.energy_cpp:.6f} "
          f"ref={r.energy_ref:.6f} (diff {r.energy_diff:.2e} eV)")
    assert r.energy_diff < 1e-3, (
        f"{name}: C++ variable-cell energy off by {r.energy_diff:.2e} eV")


def test_volume_matches_stage6(parity):
    name, r = parity
    print(f"{name}: final volume cpp={r.volume_cpp:.4f} "
          f"ref={r.volume_ref:.4f} (rel diff {r.volume_rel_diff:.2e})")
    assert r.volume_rel_diff < 2e-3, (
        f"{name}: C++ relaxed cell volume off by {r.volume_rel_diff:.2e}")


def test_cell_dof_exercised(parity):
    name, r = parity
    print(f"{name}: cell volume change {r.cell_change * 100:.2f}%")
    assert r.cell_change > 1e-2, (
        f"{name}: cell barely moved ({r.cell_change:.2e}) — cell DOF not "
        f"exercised")


def test_parity_ok(parity):
    name, r = parity
    assert r.parity_ok, f"{name}: Stage 8c variable-cell parity failed"

"""Stage 7 — scaling-validation checks.

Confirms the Exp size-scaling signature: preconditioned relaxations of the
rocksalt MgO supercell series take a roughly size-independent number of force
evaluations (matching ASE), and preconditioner assembly costs far less than a
force evaluation. (CG-solve scaling is covered by ``test_stage4_solve.py``.)
"""
from __future__ import annotations

import pytest

import envcheck
from lammps_precon.scaling import relaxation_scaling

pytestmark = pytest.mark.skipif(not envcheck.MACE_OK, reason=envcheck.MACE_REASON)


@pytest.fixture(scope="module")
def scaling():
    # x2..x4 (64..512 atoms) keeps the test quick; the full x2..x6 series is
    # exercised by `python -m lammps_precon.scaling`.
    return relaxation_scaling(sizes=(2, 3, 4), fmax=1e-2, save=False)


def test_all_relaxations_converge(scaling):
    for p in scaling:
        assert p.ours_converged, f"MgO_x{p.n}: our relaxation did not converge"
        if p.ase_converged is not None:
            assert p.ase_converged, f"MgO_x{p.n}: ASE relaxation did not converge"


def test_force_eval_counts_match_ase(scaling):
    for p in scaling:
        if p.matches_ase is not None:
            assert p.matches_ase, (
                f"MgO_x{p.n}: ours {p.ours_n_force} vs ASE {p.ase_n_force} "
                "force evaluations")


def test_force_eval_count_stays_flat(scaling):
    counts = [p.ours_n_force for p in scaling]
    print(f"\nforce-eval counts vs atoms: "
          f"{[(p.n_atoms, p.ours_n_force) for p in scaling]}")
    # ~8x more atoms must not scale the count up — Exp keeps it ~flat.
    assert max(counts) <= 2 * min(counts), (
        f"force-eval count is not flat with size: {counts}")


def test_assembly_cost_well_below_force_cost(scaling):
    for p in scaling:
        print(f"MgO_x{p.n}: assembly {p.assemble_seconds * 1e3:.2f} ms, "
              f"force {p.force_seconds * 1e3:.1f} ms, "
              f"ratio {p.assemble_to_force_ratio:.4f}")
        assert p.assemble_to_force_ratio < 0.05, (
            f"MgO_x{p.n}: assembly is {p.assemble_to_force_ratio:.3f} of the "
            "force-evaluation cost")

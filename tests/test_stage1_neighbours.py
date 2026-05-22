"""Stage 1 — neighbour-list and r_NN parity checks.

For each test structure the LAMMPS pair list within the preconditioner cutoff
must match ASE's exactly (same pairs, same distances), and r_NN must agree.
Only ``pair_style zero`` is used here, so this needs LAMMPS but not Symmetrix.
"""
from __future__ import annotations

import pytest

import envcheck
from lammps_precon.neighbours import compare
from lammps_precon.structures import cu_fcc, laalo3, mgo_supercell, si_slab

pytestmark = pytest.mark.skipif(not envcheck.LAMMPS_OK,
                                reason=envcheck.LAMMPS_REASON)

CASES = {
    "Cu_fcc": lambda: cu_fcc(),
    "MgO_x2": lambda: mgo_supercell(2),
    "Si_slab": si_slab,        # orthorhombic slab with vacuum
    "LaAlO3": laalo3,          # triclinic — exercises the Prism transform
}


@pytest.fixture(scope="module", params=sorted(CASES))
def parity(request):
    name = request.param
    return name, compare(CASES[name](), name)


def test_pair_list_is_nonempty(parity):
    name, res = parity
    assert res.n_pairs_ase > 0, f"{name}: ASE found no pairs"


def test_pair_counts_match(parity):
    name, res = parity
    assert res.n_pairs_ase == res.n_pairs_lammps, (
        f"{name}: {res.n_pairs_ase} ASE pairs vs "
        f"{res.n_pairs_lammps} LAMMPS pairs ({res.lammps_list_kind} list)")


def test_pairs_and_distances_match(parity):
    name, res = parity
    print(f"\n{name}: {res.n_pairs_ase} pairs, "
          f"max distance diff {res.max_distance_diff:.2e} A")
    assert res.pairs_match, (
        f"{name}: LAMMPS/ASE pair lists differ "
        f"(max distance diff {res.max_distance_diff:.2e} A)")
    assert res.max_distance_diff < 1e-8


def test_r_NN_matches(parity):
    name, res = parity
    print(f"{name}: r_NN ase={res.r_NN_ase:.8f} "
          f"lammps={res.r_NN_lammps:.8f} (diff {res.r_NN_diff:.2e})")
    assert res.r_NN_diff < 1e-8, f"{name}: r_NN mismatch {res.r_NN_diff:.2e}"

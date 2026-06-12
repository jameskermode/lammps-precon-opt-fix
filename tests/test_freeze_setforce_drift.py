"""Regression test: ``fix setforce 0 0 0`` atoms must not drift.

The ``Exp`` preconditioner couples every atom to its spatial neighbours, so the
preconditioned search direction ``h`` on a pinned atom is generally non-zero
*even though its force is exactly zero*. Before the 2026-06-12 fix in
``min_precon_lbfgs.cpp`` the line search stepped along that direction and
DRIFTED the pinned atoms (a ~10 Å blow-up in the case below), contaminating the
LBFGS ``Y_s`` history with spurious frozen-atom components.

The fix zeros ``h`` on any atom whose force is identically ``(0, 0, 0)`` — the
exact-zero signature that ``setforce``'s ``post_force`` leaves behind. This
test pins the lowest-z third of a rattled Cu/EAM cell with ``fix freeze ...
setforce 0 0 0``, relaxes the free atoms, and asserts the pinned atoms have not
moved. Reverting the fix turns the frozen drift from 0 into ~10 Å (red).

Uses EAM (no MACE model required); needs only LAMMPS + the built plugin.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import envcheck
from lammps_precon.cpp_parity import PLUGIN_SO

pytestmark = pytest.mark.skipif(
    not (envcheck.LAMMPS_OK and PLUGIN_SO.exists()),
    reason=("LAMMPS or the C++ plugin not built — run scripts/build_lammps.sh "
            "and cpp/scripts/build_plugin.sh"))


@pytest.fixture(scope="module")
def frozen_relax():
    """Relax a Cu/EAM cell with its lowest-z third pinned by setforce 0 0 0.

    Returns ``(frozen_idx, init_pos, final_pos, frozen_force_max)`` with
    positions gathered from LAMMPS itself (id order, sorting off) so the
    before/after comparison is free of any ASE-vs-LAMMPS wrapping mismatch.
    """
    import ase.io
    from ase.data import atomic_masses, atomic_numbers
    from lammps import lammps

    from lammps_precon.calculators import eam_potential_path
    from lammps_precon.structures import by_name

    atoms = by_name("Cu_fcc").atoms.copy()
    species = sorted(set(atoms.get_chemical_symbols()))

    # Pin the lowest-z third. With sorting off, LAMMPS atom ids follow the
    # read_data order, which is the ASE atom order, so id = index + 1.
    z = atoms.positions[:, 2]
    frozen = np.where(z <= np.percentile(z, 33))[0]
    frozen_ids = " ".join(str(i + 1) for i in frozen)

    with tempfile.TemporaryDirectory() as tmp:
        datafile = Path(tmp) / "structure.data"
        ase.io.write(datafile, atoms, format="lammps-data",
                     specorder=species, atom_style="atomic")

        lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
        setup = [
            f"plugin load {PLUGIN_SO}",
            "units metal",
            "atom_style atomic",
            "atom_modify map array sort 0 0.0",
            "boundary p p p",
            f"read_data {datafile}",
            f"mass 1 {atomic_masses[atomic_numbers[species[0]]]}",
            "pair_style eam",
            f"pair_coeff * * {eam_potential_path()}",
            "min_style precon/lbfgs",
            "fix pc all precon/exp",
            "min_modify norm max",
            f"group frozen id {frozen_ids}",
            "fix fr frozen setforce 0.0 0.0 0.0",
            "run 0",
        ]
        lmp.commands_string("\n".join(setup))
        nat = lmp.get_natoms()
        init = np.array(lmp.gather_atoms("x", 1, 3)).reshape(nat, 3)
        force = np.array(lmp.gather_atoms("f", 1, 3)).reshape(nat, 3)
        frozen_force_max = float(np.abs(force[frozen]).max())

        # Enough iterations for the free atoms to relax appreciably; without
        # the fix the run diverges well before this.
        lmp.commands_string("minimize 0.0 1e-4 50 500")
        final = np.array(lmp.gather_atoms("x", 1, 3)).reshape(nat, 3)
        lmp.close()

    return frozen, init, final, frozen_force_max


def test_setforce_makes_forces_exactly_zero(frozen_relax):
    """Sanity check on the premise: setforce zeroes the pinned forces."""
    frozen, _init, _final, frozen_force_max = frozen_relax
    assert len(frozen) > 0
    assert frozen_force_max == 0.0, (
        f"pinned-atom force not exactly zero ({frozen_force_max:.2e}); the "
        "exact-zero detection in the fix relies on this")


def test_pinned_atoms_do_not_drift(frozen_relax):
    """The regression: pinned atoms must not move during relaxation."""
    frozen, init, final, _ = frozen_relax
    drift = np.abs(final[frozen] - init[frozen]).max()
    assert drift < 1e-8, (
        f"pinned atoms drifted by {drift:.3e} Å — the preconditioned search "
        "direction is leaking onto setforce-constrained atoms")


def test_free_atoms_actually_relaxed(frozen_relax):
    """Guard against a trivial pass: the free atoms must have moved."""
    frozen, init, final, _ = frozen_relax
    nat = len(init)
    free = np.setdiff1d(np.arange(nat), frozen)
    moved = np.abs(final[free] - init[free]).max()
    # A real relaxation moves the free atoms by ~0.1 Å; a divergent run (the
    # un-fixed failure mode) moves them by several Å. Require non-trivial but
    # bounded motion so this test fails loudly on a blow-up too.
    assert 1e-3 < moved < 1.0, (
        f"free-atom motion {moved:.3e} Å is implausible (expected a normal "
        "relaxation, not a no-op or a blow-up)")

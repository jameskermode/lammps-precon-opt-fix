"""Tests for the energy-free fixed-step mode (``min_modify precon_fixed_step``).

``min_modify precon_fixed_step on`` makes ``min_style precon/lbfgs`` accept the
preconditioned LBFGS step at ``alpha = min(1, dmax/|h|_inf)`` with NO energy
accept/reject test (``linemin_fixed`` in ``cpp/src/min_precon_lbfgs.cpp``).
Rationale: energy-difference line searches stall at
``fmax ~ sqrt(ulp(|E_total|))``, so the fixed step targets the small-force tail
where the Exp-preconditioned step is Newton-like. It is intentionally NOT
stable far from the minimum, hence the intended *hybrid* usage exercised here:
a default-Armijo ``minimize`` first, then ``precon_fixed_step on`` plus a
fresh ``minimize`` to a tight force tolerance.

Uses the rattled 3x3x3 Cu/EAM cell (108 atoms) the other classical-path tests
use — no MACE model required; needs only LAMMPS + the built plugin.
"""
from __future__ import annotations

import numpy as np
import pytest

import envcheck
from lammps_precon.cpp_parity import PLUGIN_SO

pytestmark = pytest.mark.skipif(
    not (envcheck.LAMMPS_OK and PLUGIN_SO.exists()),
    reason=("LAMMPS or the C++ plugin not built — run scripts/build_lammps.sh "
            "and cpp/scripts/build_plugin.sh"))

# Tight tolerance for the fixed-step tail (eV/A, per-atom max norm). For this
# ~-380 eV cell the energy-difference floor sqrt(ulp(|E|)) is ~2e-7 eV/A, so
# the pure-Armijo reference can still reach this too, allowing the
# hybrid-vs-Armijo energy comparison at identical ftol.
FTOL = 1e-6
# Loose tolerance for the Armijo first stage of the hybrid path.
FTOL_STAGE1 = 1e-3


@pytest.fixture(scope="module")
def cu_datafile(tmp_path_factory):
    """Write the rattled Cu/EAM cell once; return (datafile, species)."""
    import ase.io

    from lammps_precon.structures import by_name

    atoms = by_name("Cu_fcc").atoms.copy()
    species = sorted(set(atoms.get_chemical_symbols()))
    datafile = tmp_path_factory.mktemp("fixed_step") / "structure.data"
    ase.io.write(datafile, atoms, format="lammps-data",
                 specorder=species, atom_style="atomic")
    return datafile, species


def _new_lammps(datafile, species):
    """Fresh serial LAMMPS with the plugin loaded and precon/lbfgs set up."""
    from ase.data import atomic_masses, atomic_numbers
    from lammps import lammps

    from lammps_precon.calculators import eam_potential_path

    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    lmp.commands_string("\n".join([
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
    ]))
    return lmp


def _fmax(lmp):
    """Max per-atom force norm — matches ``min_modify norm max``."""
    nat = lmp.get_natoms()
    f = np.array(lmp.gather_atoms("f", 1, 3)).reshape(nat, 3)
    return float(np.linalg.norm(f, axis=1).max())


@pytest.fixture(scope="module")
def hybrid_relax(cu_datafile):
    """Armijo to 1e-3, then precon_fixed_step on + fresh minimize to FTOL."""
    datafile, species = cu_datafile
    lmp = _new_lammps(datafile, species)
    lmp.commands_string(f"minimize 0.0 {FTOL_STAGE1} 200 2000")
    fmax_stage1 = _fmax(lmp)
    lmp.commands_string("min_modify precon_fixed_step on")
    lmp.commands_string(f"minimize 0.0 {FTOL} 500 5000")
    fmax = _fmax(lmp)
    energy = float(lmp.get_thermo("pe"))
    nat = lmp.get_natoms()
    lmp.close()
    return {"fmax_stage1": fmax_stage1, "fmax": fmax,
            "energy": energy, "natoms": nat}


@pytest.fixture(scope="module")
def armijo_relax(cu_datafile):
    """Pure default-Armijo relaxation of the same start to the same FTOL."""
    datafile, species = cu_datafile
    lmp = _new_lammps(datafile, species)
    lmp.commands_string(f"minimize 0.0 {FTOL} 500 5000")
    fmax = _fmax(lmp)
    energy = float(lmp.get_thermo("pe"))
    lmp.close()
    return {"fmax": fmax, "energy": energy}


def test_keyword_accepted(cu_datafile):
    """``min_modify precon_fixed_step on`` parses and a minimize runs."""
    datafile, species = cu_datafile
    lmp = _new_lammps(datafile, species)
    lmp.commands_string("min_modify precon_fixed_step on")
    # Short minimize just to prove linemin_fixed is exercised without error.
    lmp.commands_string("minimize 0.0 1e-2 20 200")
    assert lmp.get_natoms() == 108  # instance still alive and sane
    lmp.close()


def test_hybrid_reaches_tight_ftol(hybrid_relax):
    """The fixed-step tail converges to a tight force tolerance."""
    # Premise: stage 1 genuinely stopped at the loose tolerance, so the tight
    # tolerance is reached by the fixed-step stage, not by Armijo alone.
    assert hybrid_relax["fmax_stage1"] <= FTOL_STAGE1
    assert hybrid_relax["fmax"] <= FTOL, (
        f"hybrid fixed-step relaxation stalled at fmax = "
        f"{hybrid_relax['fmax']:.3e} eV/A (ftol {FTOL:.0e})")


def test_hybrid_energy_matches_pure_armijo(hybrid_relax, armijo_relax):
    """Both paths land in the same minimum: energies agree per atom."""
    assert armijo_relax["fmax"] <= FTOL  # reference itself converged
    de = abs(hybrid_relax["energy"] - armijo_relax["energy"])
    de_per_atom = de / hybrid_relax["natoms"]
    assert de_per_atom < 1e-8, (
        f"hybrid vs pure-Armijo energy differs by {de_per_atom:.3e} eV/atom "
        f"(total {de:.3e} eV) — the fixed step converged elsewhere")


def test_off_switch_restores_armijo(cu_datafile):
    """``precon_fixed_step off`` after ``on`` restores Armijo behaviour."""
    datafile, species = cu_datafile
    lmp = _new_lammps(datafile, species)
    lmp.commands_string("min_modify precon_fixed_step on")
    lmp.commands_string("min_modify precon_fixed_step off")
    # With the flag off, init() must fall back to the Armijo linemin; the
    # minimize therefore runs the default energy-checked path without error.
    lmp.commands_string(f"minimize 0.0 {FTOL_STAGE1} 200 2000")
    assert _fmax(lmp) <= FTOL_STAGE1
    lmp.close()

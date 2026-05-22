"""Stage 3 — mu-estimation parity.

ASE estimates the global scale ``mu`` with a two-force-call finite-difference
probe (`SparsePrecon.estimate_mu`):

    [dE(p + v) - dE(p)] . v  =  mu <P1 v, v>

with a deterministic sinusoidal perturbation ``v``. This module reimplements
that probe independently (`estimate_mu_fd`) and validates it against ASE's
`estimate_mu`, both driven on LAMMPS forces. The independent implementation is
the universal mu-setter the eventual C++ port needs; reimplementing it now and
checking parity confirms the FD convention and displacement are understood.

The probe is potential-agnostic — it is exercised here with both the Symmetrix
MACE potential and a classical EAM potential.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_properties
from ase.optimize.precon import Exp
from ase.utils import longsum

from . import artifacts
from .assembly import assemble_P, lammps_neighbor_list
from .calculators import make_calculator
from .structures import TestStructure, reference_set

A_DEFAULT = 3.0
C_STAB_DEFAULT = 0.1


class _ProbeRecorder(Calculator):
    """Calculator wrapper that records the positions of each force evaluation.

    Used to capture exactly where ASE's ``estimate_mu`` probes the potential,
    so the finite-difference displacement can be compared directly.
    """

    implemented_properties = all_properties
    name = "ProbeRecorder"

    def __init__(self, calculator):
        Calculator.__init__(self)
        self.calculator = calculator
        self.probe_positions: list[np.ndarray] = []

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.probe_positions.append(atoms.positions.copy())
        self.results = {p: self.calculator.get_property(p, atoms)
                        for p in properties}


def fd_perturbation(positions: np.ndarray, r_NN: float) -> np.ndarray:
    """The deterministic sinusoidal FD displacement ``v`` ASE's probe uses.

    ``v[:, i] = 1e-2 * r_NN * sin(p[:, i] / L_i)`` with ``L_i`` the extent of
    the structure along axis ``i`` (the component is left flat if ``L_i == 0``).
    """
    H = 1e-2 * r_NN * np.eye(3)
    components = [positions[:, 0].copy(),
                  positions[:, 1].copy(),
                  positions[:, 2].copy()]
    for i in range(3):
        extent = positions[:, i].max() - positions[:, i].min()
        if extent == 0.0:
            H[i, i] = 0.0
        else:
            components[i] = np.sin(components[i] / extent)
    return np.dot(H, components).T


def estimate_mu_fd(
    atoms: Atoms,
    *,
    r_NN: float,
    r_cut: float,
    A: float = A_DEFAULT,
    c_stab: float = C_STAB_DEFAULT,
) -> tuple[float, dict]:
    """Independent two-force-call finite-difference estimate of ``mu``.

    Reimplements ASE's `SparsePrecon.estimate_mu` (fixed-cell path): builds the
    perturbation, evaluates the gradient at ``p`` and ``p + v``, and solves
    ``mu = sum(LHS) / sum(RHS)`` with ``LHS = [dE(p+v) - dE(p)] . v`` and
    ``RHS = (P1 v) . v`` for the mu=1 preconditioner ``P1``. The P1 assembly is
    the Stage-2-validated one, fed the LAMMPS pair list.
    """
    p = atoms.get_positions()
    v = fd_perturbation(p, r_NN)
    v1 = v.reshape(-1)

    dE_p = -atoms.get_forces().reshape(-1)
    atoms_v = atoms.copy()
    atoms_v.calc = atoms.calc
    atoms_v.set_positions(p + v)
    dE_p_plus_v = -atoms_v.get_forces().reshape(-1)

    LHS = (dE_p_plus_v - dE_p) * v1

    P1 = assemble_P(atoms, mu=1.0, r_NN=r_NN, r_cut=r_cut, A=A, c_stab=c_stab,
                    neighbor_list=lammps_neighbor_list)
    RHS = P1.dot(v1) * v1

    mu_raw = float(longsum(LHS) / longsum(RHS))
    mu = max(mu_raw, 1.0)  # ASE caps mu at 1.0
    record = dict(
        v=v,
        LHS=LHS,
        RHS=RHS,
        lhs_sum=float(longsum(LHS)),
        rhs_sum=float(longsum(RHS)),
        mu_raw=mu_raw,
    )
    return mu, record


@dataclass
class MuParity:
    name: str
    engine: str
    n_atoms: int
    r_NN: float
    r_cut: float
    mu_ase: float
    mu_fd: float
    mu_raw_fd: float
    mu_capped: bool
    mu_abs_diff: float
    mu_rel_diff: float
    n_probes_ase: int
    displacement_match: float
    parity_ok: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def compare_mu(
    structure: TestStructure,
    *,
    A: float = A_DEFAULT,
    c_stab: float = C_STAB_DEFAULT,
    save: bool = True,
) -> MuParity:
    """Run the Stage-3 mu-estimation parity check for one structure."""
    atoms = structure.atoms.copy()
    base_calc = make_calculator(atoms, structure.engine)
    recorder = _ProbeRecorder(base_calc)
    atoms.calc = recorder
    try:
        # ASE reference: estimate_mu via the standard make_precon path.
        precon = Exp(A=A, c_stab=c_stab, solver="direct")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precon.make_precon(atoms)
        mu_ase = float(precon.mu)
        r_NN = float(precon.r_NN)
        r_cut = float(precon.r_cut)

        # The probes ASE used (estimate_mu does exactly two force evaluations;
        # _make_sparse_precon does none).
        n_probes = len(recorder.probe_positions)
        if n_probes >= 2:
            v_ase = recorder.probe_positions[1] - recorder.probe_positions[0]
        else:
            v_ase = None

        # Independent finite-difference estimate on the same LAMMPS forces.
        mu_fd, record = estimate_mu_fd(atoms, r_NN=r_NN, r_cut=r_cut,
                                       A=A, c_stab=c_stab)
        v_fd = record["v"]

        disp_match = (float(np.abs(v_ase - v_fd).max())
                      if v_ase is not None else float("inf"))
        abs_diff = abs(mu_ase - mu_fd)
        rel_diff = abs_diff / abs(mu_ase)
        parity_ok = (rel_diff < 1e-10 and disp_match < 1e-12
                     and n_probes == 2)

        result = MuParity(
            name=structure.name,
            engine=structure.engine,
            n_atoms=len(atoms),
            r_NN=r_NN,
            r_cut=r_cut,
            mu_ase=mu_ase,
            mu_fd=mu_fd,
            mu_raw_fd=record["mu_raw"],
            mu_capped=record["mu_raw"] < 1.0,
            mu_abs_diff=abs_diff,
            mu_rel_diff=rel_diff,
            n_probes_ase=n_probes,
            displacement_match=disp_match,
            parity_ok=parity_ok,
        )

        if save:
            d = artifacts.stage_dir("stage3", structure.name)
            artifacts.save_json(d / "summary.json", result.as_dict())
            arrays = dict(v_fd=v_fd, LHS=record["LHS"], RHS=record["RHS"])
            if v_ase is not None:
                arrays["v_ase"] = v_ase
            artifacts.save_arrays(d / "probe.npz", **arrays)
        return result
    finally:
        lmp = getattr(base_calc, "lmp", None)
        if lmp is not None:
            try:
                lmp.close()
            except Exception:
                pass


def run_all(save: bool = True) -> list[MuParity]:
    """Run Stage-3 mu-estimation parity for every reference structure."""
    results = []
    for structure in reference_set(full=True):
        print(f"[stage3] {structure.name} "
              f"({len(structure.atoms)} atoms, {structure.engine})")
        result = compare_mu(structure, save=save)
        results.append(result)
        capped = " (raw < 1, capped)" if result.mu_capped else ""
        print(f"         mu ase={result.mu_ase:.8f} fd={result.mu_fd:.8f}  "
              f"rel_diff={result.mu_rel_diff:.2e}  "
              f"mu_raw_fd={result.mu_raw_fd:.6f}{capped}")
        print(f"         probes={result.n_probes_ase}  "
              f"displacement_match={result.displacement_match:.2e}  "
              f"parity_ok={result.parity_ok}")
    return results


if __name__ == "__main__":
    import os
    import sys
    import traceback

    try:
        run_all()
        code = 0
    except Exception:
        traceback.print_exc()
        code = 1
    # Hard exit: LAMMPSlib/Kokkos crash the interpreter teardown.
    sys.stdout.flush()
    os._exit(code)

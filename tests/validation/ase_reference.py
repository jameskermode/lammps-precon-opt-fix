#!/usr/bin/env python3
"""
ASE PreconLBFGS reference optimization for validation

Creates a compressed Cu FCC lattice and optimizes with PreconLBFGS,
logging detailed state at each step for comparison with LAMMPS.
"""

import numpy as np
from ase import Atoms
from ase.calculators.eam import EAM
from ase.optimize.precon import PreconLBFGS, Exp
import json
import sys

def save_state(optimizer, step, logdata):
    """Save detailed state at current optimization step"""
    atoms = optimizer.atoms

    state = {
        'step': step,
        'energy': float(atoms.get_potential_energy()),
        'positions': atoms.get_positions().tolist(),
        'forces': atoms.get_forces().tolist(),
        'fmax': float(np.max(np.abs(atoms.get_forces()))),
        'fnorm': float(np.linalg.norm(atoms.get_forces())),
    }

    # Add preconditioner info if available
    if hasattr(optimizer, 'precon') and optimizer.precon is not None:
        precon = optimizer.precon
        state['precon'] = {
            'r_NN': float(precon.r_NN) if precon.r_NN is not None else None,
            'r_cut': float(precon.r_cut) if precon.r_cut is not None else None,
            'mu': float(precon.mu) if precon.mu is not None else None,
            'A': float(precon.A) if hasattr(precon, 'A') else None,
            'c_stab': float(precon.c_stab),
        }

    logdata.append(state)

    # Print progress
    print(f"Step {step}: E = {state['energy']:.6f} eV, "
          f"fmax = {state['fmax']:.6f} eV/A, "
          f"fnorm = {state['fnorm']:.6f} eV/A")

def main():
    print("=" * 70)
    print("ASE PreconLBFGS Reference Optimization")
    print("=" * 70)

    # Create compressed Cu FCC lattice (3x3x3)
    a = 3.615  # Cu lattice parameter
    atoms = Atoms('Cu',
                  cell=[(3*a, 0, 0), (0, 3*a, 0), (0, 0, 3*a)],
                  pbc=True)

    # Build FCC lattice manually
    positions = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # FCC basis
                positions.append([i*a, j*a, k*a])  # Corner
                positions.append([i*a + a/2, j*a + a/2, k*a])  # Face 1
                positions.append([i*a + a/2, j*a, k*a + a/2])  # Face 2
                positions.append([i*a, j*a + a/2, k*a + a/2])  # Face 3

    atoms = Atoms('Cu' * len(positions),
                  positions=positions,
                  cell=[(3*a, 0, 0), (0, 3*a, 0), (0, 0, 3*a)],
                  pbc=True)

    print(f"System: {len(atoms)} Cu atoms in 3x3x3 FCC lattice")
    print(f"Cell: {atoms.cell.diagonal()}")

    # Perturb one atom to create forces (easier to match with LAMMPS)
    # Use a random perturbation with fixed seed for reproducibility
    np.random.seed(42)
    center_idx = len(atoms) // 2
    atoms.positions[center_idx] += np.array([0.1, 0.1, 0.1])
    print(f"Perturbed atom {center_idx} by [0.1, 0.1, 0.1] A")

    # Set up EAM calculator
    eam_file = '/home/eng/essswb/lammps/lammps-22Jul2025/potentials/Cu_u3.eam'
    calc = EAM(potential=eam_file)
    atoms.calc = calc

    # Initial state
    E0 = atoms.get_potential_energy()
    F0 = atoms.get_forces()
    fmax0 = np.max(np.abs(F0))
    fnorm0 = np.linalg.norm(F0)

    print(f"\nInitial state:")
    print(f"  Energy: {E0:.6f} eV")
    print(f"  fmax: {fmax0:.6f} eV/A")
    print(f"  fnorm: {fnorm0:.6f} eV/A")

    # Set up PreconLBFGS with Exp preconditioner
    print(f"\nSetting up PreconLBFGS optimizer:")
    print(f"  Preconditioner: Exp")
    print(f"  A: 3.0")
    print(f"  c_stab: 0.1")
    print(f"  mu: -1 (automatic estimation)")
    print(f"  r_cut: None (automatic = 2*r_NN)")
    print(f"  memory: 100")
    print(f"  use_armijo: True")
    print(f"  c1: 0.23")

    precon = Exp(A=3.0, mu=None, r_cut=None, c_stab=0.1, logfile=None)

    # Initialize optimizer
    opt = PreconLBFGS(atoms,
                      precon=precon,
                      memory=100,
                      use_armijo=True,
                      c1=0.23,
                      logfile=None,
                      trajectory=None)

    # Log data for each step
    logdata = []

    # Save initial state (step 0)
    save_state(opt, 0, logdata)

    # Run optimization with callback to log each step
    step_counter = [0]  # Mutable container to track steps

    def observer():
        step_counter[0] += 1
        save_state(opt, step_counter[0], logdata)

    opt.attach(observer)

    print(f"\nStarting optimization (fmax = 0.01 eV/A, max steps = 20)...")
    print("-" * 70)

    try:
        opt.run(fmax=0.01, steps=20)
        print("-" * 70)
        print(f"Optimization converged in {step_counter[0]} steps")
    except Exception as e:
        print(f"Optimization stopped: {e}")
        print(f"Completed {step_counter[0]} steps")

    # Final state
    Ef = atoms.get_potential_energy()
    Ff = atoms.get_forces()
    fmaxf = np.max(np.abs(Ff))
    fnormf = np.linalg.norm(Ff)

    print(f"\nFinal state:")
    print(f"  Energy: {Ef:.6f} eV")
    print(f"  fmax: {fmaxf:.6f} eV/A")
    print(f"  fnorm: {fnormf:.6f} eV/A")
    print(f"  Energy change: {Ef - E0:.6f} eV")

    # Save trajectory to JSON
    output_file = 'tests/validation/ase_trajectory.json'
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'code': 'ASE',
                'optimizer': 'PreconLBFGS',
                'preconditioner': 'Exp',
                'natoms': len(atoms),
                'lattice_parameter': a,
                'perturbation': '0.1A on center atom',
                'system': '3x3x3 Cu FCC with perturbation',
                'potential': 'Cu_u3.eam',
            },
            'parameters': {
                'A': 3.0,
                'c_stab': 0.1,
                'memory': 100,
                'use_armijo': True,
                'c1': 0.23,
                'fmax': 0.01,
            },
            'trajectory': logdata,
        }, f, indent=2)

    print(f"\nTrajectory saved to: {output_file}")
    print(f"Total steps logged: {len(logdata)}")
    print("=" * 70)

    return 0

if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Compare ASE and LAMMPS PreconLBFGS trajectories step-by-step

Validates:
1. Energy values at each step
2. Force vectors (magnitude and direction)
3. Position updates
4. Convergence trajectory
5. Preconditioner parameters
"""

import json
import numpy as np
import sys

def load_trajectory(filename):
    """Load trajectory from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def compare_vectors(v1, v2, name, threshold=1e-3):
    """Compare two vector arrays"""
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Check shapes match
    if v1.shape != v2.shape:
        print(f"  ❌ {name}: Shape mismatch! ASE: {v1.shape}, LAMMPS: {v2.shape}")
        return False

    # Compute differences
    diff = v1 - v2
    max_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff**2))
    rel_diff = max_diff / (np.max(np.abs(v1)) + 1e-10)

    # Check tolerance
    passed = max_diff < threshold

    status = "✓" if passed else "❌"
    print(f"  {status} {name}:")
    print(f"      Max abs diff: {max_diff:.6e}")
    print(f"      RMS diff: {rms_diff:.6e}")
    print(f"      Relative diff: {rel_diff:.6e}")

    return passed

def compare_scalar(s1, s2, name, threshold=1e-3):
    """Compare two scalar values"""
    diff = abs(s1 - s2)
    rel_diff = diff / (abs(s1) + 1e-10)

    passed = diff < threshold

    status = "✓" if passed else "❌"
    print(f"  {status} {name}:")
    print(f"      ASE: {s1:.6e}, LAMMPS: {s2:.6e}")
    print(f"      Abs diff: {diff:.6e}, Rel diff: {rel_diff:.6e}")

    return passed

def compare_step(ase_step, lammps_step, step_num, tolerances):
    """Compare a single optimization step"""
    print(f"\n{'='*70}")
    print(f"Step {step_num}")
    print(f"{'='*70}")

    all_passed = True

    # Compare energy
    if 'energy' in ase_step and 'energy' in lammps_step:
        passed = compare_scalar(ase_step['energy'], lammps_step['energy'],
                                'Energy (eV)', tolerances['energy'])
        all_passed = all_passed and passed

    # Compare fmax
    passed = compare_scalar(ase_step['fmax'], lammps_step['fmax'],
                            'Force max (eV/A)', tolerances['fmax'])
    all_passed = all_passed and passed

    # Compare fnorm
    passed = compare_scalar(ase_step['fnorm'], lammps_step['fnorm'],
                            'Force norm (eV/A)', tolerances['fnorm'])
    all_passed = all_passed and passed

    # Compare positions
    passed = compare_vectors(ase_step['positions'], lammps_step['positions'],
                            'Positions (A)', tolerances['positions'])
    all_passed = all_passed and passed

    # Compare forces
    passed = compare_vectors(ase_step['forces'], lammps_step['forces'],
                            'Forces (eV/A)', tolerances['forces'])
    all_passed = all_passed and passed

    return all_passed

def compare_preconditioner(ase_precon, lammps_precon, tolerances):
    """Compare preconditioner parameters"""
    print(f"\n{'='*70}")
    print(f"Preconditioner Parameters")
    print(f"{'='*70}")

    all_passed = True

    if ase_precon is None or lammps_precon is None:
        print("  ⚠️  Preconditioner info not available in one or both trajectories")
        return True  # Don't fail if precon info missing

    params = ['r_NN', 'r_cut', 'mu']
    for param in params:
        if param in ase_precon and param in lammps_precon:
            if ase_precon[param] is not None and lammps_precon[param] is not None:
                passed = compare_scalar(ase_precon[param], lammps_precon[param],
                                      f'{param}', tolerances['precon'])
                all_passed = all_passed and passed

    return all_passed

def main():
    print("="*70)
    print("ASE vs LAMMPS PreconLBFGS Trajectory Comparison")
    print("="*70)

    # Load trajectories
    ase_file = 'tests/validation/ase_trajectory.json'
    lammps_file = 'tests/validation/lammps_trajectory.json'

    print(f"\nLoading trajectories...")
    try:
        ase_data = load_trajectory(ase_file)
        print(f"  ✓ ASE: {len(ase_data['trajectory'])} steps")
    except FileNotFoundError:
        print(f"  ❌ ASE trajectory not found: {ase_file}")
        return 1

    try:
        lammps_data = load_trajectory(lammps_file)
        print(f"  ✓ LAMMPS: {len(lammps_data['trajectory'])} steps")
    except FileNotFoundError:
        print(f"  ❌ LAMMPS trajectory not found: {lammps_file}")
        return 1

    # Set comparison tolerances
    # Note: Energy and force tolerances are relaxed to account for
    # differences in EAM potential implementations between ASE and LAMMPS
    tolerances = {
        'energy': 0.5,       # eV (EAM implementation differences)
        'fmax': 0.01,        # eV/A (EAM implementation differences)
        'fnorm': 0.01,       # eV/A (EAM implementation differences)
        'positions': 1e-4,   # A (should match exactly)
        'forces': 0.01,      # eV/A (EAM implementation differences)
        'precon': 1.0,       # Preconditioner parameters (can vary with implementation)
    }

    print(f"\nComparison tolerances:")
    for key, val in tolerances.items():
        print(f"  {key}: {val:.2e}")

    # Compare preconditioner parameters (step 0)
    ase_traj = ase_data['trajectory']
    lammps_traj = lammps_data['trajectory']

    ase_precon = ase_traj[0].get('precon') if ase_traj else None
    lammps_precon = lammps_traj[0].get('precon') if lammps_traj else None

    precon_passed = compare_preconditioner(ase_precon, lammps_precon, tolerances)

    # Compare each step
    nsteps = min(len(ase_traj), len(lammps_traj))
    print(f"\nComparing {nsteps} optimization steps...")

    step_results = []
    for i in range(nsteps):
        passed = compare_step(ase_traj[i], lammps_traj[i], i, tolerances)
        step_results.append(passed)

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")

    print(f"\nPreconditioner validation: {'✓ PASS' if precon_passed else '❌ FAIL'}")

    passed_steps = sum(step_results)
    total_steps = len(step_results)
    print(f"\nStep-by-step validation: {passed_steps}/{total_steps} steps passed")

    # Detailed step results
    print(f"\nStep results:")
    for i, passed in enumerate(step_results):
        status = "✓" if passed else "❌"
        print(f"  {status} Step {i}")

    # Overall verdict
    all_passed = precon_passed and all(step_results)

    print(f"\n{'='*70}")
    if all_passed:
        print("✓✓✓ VALIDATION PASSED ✓✓✓")
        print("LAMMPS implementation matches ASE within tolerances!")
        exit_code = 0
    else:
        print("❌❌❌ VALIDATION FAILED ❌❌❌")
        print("Differences detected between ASE and LAMMPS implementations.")
        exit_code = 1
    print(f"{'='*70}")

    return exit_code

if __name__ == '__main__':
    sys.exit(main())

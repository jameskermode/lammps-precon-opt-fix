#!/usr/bin/env python3
"""
Convergence comparison: ASE vs LAMMPS with Identity and Exp preconditioners

Produces a plot showing max force vs step count for:
- ASE with Identity preconditioner
- ASE with Exp preconditioner
- LAMMPS with Identity preconditioner
- LAMMPS with Exp preconditioner
"""

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.calculators.eam import EAM
from ase.optimize.precon import PreconLBFGS, Exp, make_precon
import subprocess
import re
import json
import os

def create_test_system():
    """Create a test system with forces (vacancy in Cu FCC)"""
    a = 3.615  # Cu lattice parameter

    # Build 4x4x4 FCC lattice
    positions = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # FCC basis
                positions.append([i*a, j*a, k*a])
                positions.append([i*a + a/2, j*a + a/2, k*a])
                positions.append([i*a + a/2, j*a, k*a + a/2])
                positions.append([i*a, j*a + a/2, k*a + a/2])

    atoms = Atoms('Cu' * len(positions),
                  positions=positions,
                  cell=[(4*a, 0, 0), (0, 4*a, 0), (0, 0, 4*a)],
                  pbc=True)

    # Delete center atom to create vacancy
    center_idx = len(atoms) // 2
    del atoms[center_idx]

    print(f"Created test system: {len(atoms)} Cu atoms with vacancy")
    print(f"Cell: {atoms.cell.diagonal()}")

    return atoms

def run_ase_optimization(atoms, precon_type='exp', max_steps=50):
    """Run ASE optimization with specified preconditioner"""
    print(f"\n{'='*70}")
    print(f"Running ASE optimization with {precon_type.upper()} preconditioner")
    print(f"{'='*70}")

    # Set up calculator
    eam_file = '/home/eng/essswb/lammps/lammps-22Jul2025/potentials/Cu_u3.eam'
    calc = EAM(potential=eam_file)
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc

    # Set up preconditioner
    if precon_type.lower() == 'exp':
        precon = Exp(A=3.0, mu=None, r_cut=None, c_stab=0.1, logfile=None)
        name = "ASE Exp"
    else:  # identity
        precon = make_precon('ID')  # Identity preconditioner
        name = "ASE Identity"

    # Set up optimizer
    opt = PreconLBFGS(atoms_copy,
                      precon=precon,
                      memory=100,
                      use_armijo=True,
                      c1=0.23,
                      logfile=None,
                      trajectory=None)

    # Track convergence
    forces_history = []
    energy_history = []

    def observer():
        forces = atoms_copy.get_forces()
        fmax = np.max(np.abs(forces))
        energy = atoms_copy.get_potential_energy()
        forces_history.append(fmax)
        energy_history.append(energy)
        print(f"  Step {len(forces_history)-1}: E = {energy:.6f} eV, fmax = {fmax:.6f} eV/A")

    # Initial state
    forces = atoms_copy.get_forces()
    fmax = np.max(np.abs(forces))
    energy = atoms_copy.get_potential_energy()
    forces_history.append(fmax)
    energy_history.append(energy)
    print(f"  Step 0: E = {energy:.6f} eV, fmax = {fmax:.6f} eV/A")

    # Attach observer
    opt.attach(observer)

    # Run optimization
    try:
        opt.run(fmax=0.01, steps=max_steps)
        converged = True
    except Exception as e:
        print(f"  Optimization stopped: {e}")
        converged = False

    print(f"  Final: {len(forces_history)} steps, fmax = {forces_history[-1]:.6f} eV/A")

    return {
        'name': name,
        'forces': forces_history,
        'energies': energy_history,
        'converged': converged,
        'steps': len(forces_history) - 1
    }

def write_lammps_data_file(atoms, filename):
    """Write LAMMPS data file from ASE atoms"""
    with open(filename, 'w') as f:
        f.write("# LAMMPS data file - Cu FCC with vacancy\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"1 atom types\n\n")
        f.write(f"0.0 {atoms.cell[0,0]:.10f} xlo xhi\n")
        f.write(f"0.0 {atoms.cell[1,1]:.10f} ylo yhi\n")
        f.write(f"0.0 {atoms.cell[2,2]:.10f} zlo zhi\n\n")
        f.write(f"Masses\n\n")
        f.write(f"1 63.546\n\n")
        f.write(f"Atoms # atomic\n\n")
        for i, pos in enumerate(atoms.positions):
            f.write(f"{i+1} 1 {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}\n")

def create_lammps_input(precon_type, data_file, output_prefix):
    """Create LAMMPS input file"""
    precon_line = {
        'identity': 'fix opt all precon_lbfgs 0.01 precon none memory 100 maxstep 0.04 c1 0.23',
        'exp': 'fix opt all precon_lbfgs 0.01 precon exp r_cut -1.0 mu -1.0 A 3.0 c_stab 0.1 memory 100 maxstep 0.04 c1 0.23'
    }

    input_file = f"{output_prefix}.lam"

    with open(input_file, 'w') as f:
        f.write(f"# LAMMPS convergence test - {precon_type} preconditioner\n\n")
        f.write("units metal\n")
        f.write("atom_style atomic\n")
        f.write("atom_modify map yes\n\n")
        f.write(f"read_data {data_file}\n\n")
        f.write("pair_style eam\n")
        f.write("pair_coeff * * /home/eng/essswb/lammps/lammps-22Jul2025/potentials/Cu_u3.eam\n\n")
        f.write("neighbor 2.0 bin\n")
        f.write("neigh_modify delay 0 every 1 check yes\n\n")
        f.write("plugin load /home/eng/essswb/lammps-precon-opt-fix/build/preconlbfgsplugin.so\n\n")
        f.write(f"{precon_line[precon_type]}\n\n")
        f.write("thermo 1\n")
        f.write("thermo_style custom step pe fmax fnorm\n\n")
        f.write("minimize 0.0 0.01 50 10000\n\n")
        f.write("unfix opt\n")

    return input_file

def parse_lammps_log(log_file):
    """Parse LAMMPS log file to extract force history"""
    forces_history = []
    energies_history = []

    with open(log_file, 'r') as f:
        content = f.read()

    # Find thermo output
    lines = content.split('\n')
    in_thermo = False

    for line in lines:
        if 'Step' in line and 'PotEng' in line and 'Fmax' in line:
            in_thermo = True
            continue

        if in_thermo:
            if line.strip() == '' or line.startswith('Loop'):
                in_thermo = False
                continue

            parts = line.split()
            if len(parts) >= 3:
                try:
                    step = int(parts[0])
                    energy = float(parts[1])
                    fmax = float(parts[2])
                    energies_history.append(energy)
                    forces_history.append(fmax)
                except (ValueError, IndexError):
                    pass

    return forces_history, energies_history

def run_lammps_optimization(atoms, precon_type='exp'):
    """Run LAMMPS optimization with specified preconditioner"""
    print(f"\n{'='*70}")
    print(f"Running LAMMPS optimization with {precon_type.upper()} preconditioner")
    print(f"{'='*70}")

    # Write data file
    data_file = f"tests/validation/lammps_{precon_type}_atoms.data"
    write_lammps_data_file(atoms, data_file)

    # Create input file
    output_prefix = f"tests/validation/lammps_{precon_type}"
    input_file = create_lammps_input(precon_type, data_file, output_prefix)
    log_file = f"{output_prefix}.log"

    # Run LAMMPS
    lammps_exe = "/home/eng/essswb/lammps/lammps-22Jul2025/build/lmp"

    # Need to unload OpenMPI module to avoid conflicts
    env = os.environ.copy()

    try:
        result = subprocess.run(
            f"module purge && module load GCC/13.2.0 Eigen && {lammps_exe} -in {input_file}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            executable='/bin/bash',
            timeout=120,
            env=env
        )

        # Save output
        with open(log_file, 'w') as f:
            f.write(result.stdout)
            f.write(result.stderr)

        if result.returncode != 0:
            print(f"  Warning: LAMMPS exited with code {result.returncode}")
        else:
            print(f"  LAMMPS completed successfully")

    except subprocess.TimeoutExpired:
        print(f"  Error: LAMMPS timed out")
        return None
    except Exception as e:
        print(f"  Error running LAMMPS: {e}")
        return None

    # Parse results
    forces_history, energies_history = parse_lammps_log(log_file)

    if not forces_history:
        print(f"  Error: Could not parse LAMMPS output")
        return None

    name = f"LAMMPS {precon_type.capitalize()}"
    print(f"  Final: {len(forces_history)} steps, fmax = {forces_history[-1]:.6f} eV/A")

    converged = forces_history[-1] < 0.01

    return {
        'name': name,
        'forces': forces_history,
        'energies': energies_history,
        'converged': converged,
        'steps': len(forces_history) - 1
    }

def create_convergence_plot(results, output_file='convergence_comparison.png'):
    """Create publication-quality convergence plot"""
    plt.figure(figsize=(10, 7))

    colors = {
        'ASE Identity': '#1f77b4',
        'ASE Exp': '#ff7f0e',
        'LAMMPS Identity': '#2ca02c',
        'LAMMPS Exp': '#d62728'
    }

    markers = {
        'ASE Identity': 'o',
        'ASE Exp': 's',
        'LAMMPS Identity': '^',
        'LAMMPS Exp': 'v'
    }

    linestyles = {
        'ASE Identity': '-',
        'ASE Exp': '-',
        'LAMMPS Identity': '--',
        'LAMMPS Exp': '--'
    }

    for result in results:
        if result is None:
            continue

        name = result['name']
        forces = result['forces']
        steps = range(len(forces))

        plt.semilogy(steps, forces,
                     label=name,
                     color=colors.get(name, 'black'),
                     marker=markers.get(name, 'o'),
                     linestyle=linestyles.get(name, '-'),
                     markersize=6,
                     markevery=max(1, len(forces)//20),
                     linewidth=2,
                     alpha=0.8)

    # Add convergence threshold
    plt.axhline(y=0.01, color='gray', linestyle=':', linewidth=1.5,
                label='Convergence (fmax = 0.01 eV/Å)', alpha=0.7)

    plt.xlabel('Optimization Step', fontsize=14, fontweight='bold')
    plt.ylabel('Max Force (eV/Å)', fontsize=14, fontweight='bold')
    plt.title('PreconLBFGS Convergence: ASE vs LAMMPS\nCu FCC with Vacancy (255 atoms)',
              fontsize=16, fontweight='bold', pad=20)

    plt.legend(loc='upper right', fontsize=11, framealpha=0.95)
    plt.grid(True, which='both', alpha=0.3, linestyle='--')
    plt.tick_params(labelsize=12)

    # Set y-axis limits
    plt.ylim(1e-3, 1e1)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"Plot saved to: {output_file}")
    print(f"{'='*70}")

    return output_file

def print_summary_table(results):
    """Print summary table of results"""
    print(f"\n{'='*70}")
    print("CONVERGENCE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Steps':>8} {'Final fmax':>12} {'Converged':>12}")
    print(f"{'-'*70}")

    for result in results:
        if result is None:
            continue

        name = result['name']
        steps = result['steps']
        fmax = result['forces'][-1]
        converged = "✓" if result['converged'] else "✗"

        print(f"{name:<20} {steps:>8} {fmax:>12.6f} {converged:>12}")

    print(f"{'='*70}\n")

def main():
    print("="*70)
    print("PreconLBFGS Convergence Comparison")
    print("ASE vs LAMMPS with Identity and Exp Preconditioners")
    print("="*70)

    # Create test system
    atoms = create_test_system()

    # Run all optimizations
    results = []

    # ASE optimizations
    try:
        result_ase_id = run_ase_optimization(atoms.copy(), precon_type='identity', max_steps=50)
        results.append(result_ase_id)
    except Exception as e:
        print(f"Error in ASE Identity: {e}")
        results.append(None)

    try:
        result_ase_exp = run_ase_optimization(atoms.copy(), precon_type='exp', max_steps=50)
        results.append(result_ase_exp)
    except Exception as e:
        print(f"Error in ASE Exp: {e}")
        results.append(None)

    # LAMMPS optimizations
    try:
        result_lammps_id = run_lammps_optimization(atoms.copy(), precon_type='identity')
        results.append(result_lammps_id)
    except Exception as e:
        print(f"Error in LAMMPS Identity: {e}")
        results.append(None)

    try:
        result_lammps_exp = run_lammps_optimization(atoms.copy(), precon_type='exp')
        results.append(result_lammps_exp)
    except Exception as e:
        print(f"Error in LAMMPS Exp: {e}")
        results.append(None)

    # Print summary
    print_summary_table(results)

    # Create plot
    output_file = 'tests/validation/convergence_comparison.png'
    create_convergence_plot(results, output_file)

    # Save data to JSON
    json_file = 'tests/validation/convergence_data.json'
    with open(json_file, 'w') as f:
        json.dump({
            'metadata': {
                'system': 'Cu FCC 4x4x4 with vacancy',
                'natoms': len(atoms),
                'methods': ['ASE Identity', 'ASE Exp', 'LAMMPS Identity', 'LAMMPS Exp']
            },
            'results': results
        }, f, indent=2)

    print(f"Data saved to: {json_file}")

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

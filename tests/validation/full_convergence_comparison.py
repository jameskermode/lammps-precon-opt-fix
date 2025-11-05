#!/usr/bin/env python3
"""
Comprehensive convergence comparison: ASE vs LAMMPS with Identity and Exp preconditioners
Plots max force vs iteration on semilogy scale
"""
import numpy as np
from ase import Atoms
from ase.calculators.eam import EAM
from ase.optimize.precon import PreconLBFGS, Exp
import subprocess
import json
import matplotlib.pyplot as plt
import os

print("="*80)
print("Full Convergence Comparison: ASE vs LAMMPS")
print("="*80)

# Create identical system (2047-atom Cu with vacancy - 8x8x8 supercell)
a = 3.615
n_cells = 8  # 8x8x8 supercell (2x bigger in each dimension than original 4x4x4)
positions = []
for i in range(n_cells):
    for j in range(n_cells):
        for k in range(n_cells):
            for basis in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                pos = [a*(i+basis[0]), a*(j+basis[1]), a*(k+basis[2])]
                positions.append(pos)

n_atoms = n_cells**3 * 4
atoms = Atoms('Cu' * n_atoms, positions=positions,
              cell=[(n_cells*a, 0, 0), (0, n_cells*a, 0), (0, 0, n_cells*a)], pbc=True)
center_idx = len(atoms) // 2
del atoms[center_idx]

atoms.calc = EAM(potential='/home/eng/essswb/lammps/lammps-22Jul2025/potentials/Cu_u3.eam')

results = {}

# ============================================================================
# 1. ASE with Identity Preconditioner
# ============================================================================
print("\n" + "="*80)
print("1. ASE PreconLBFGS - Identity Preconditioner")
print("="*80)

# Reset atoms
atoms.positions[:] = np.array(positions)[:-1]  # Exclude deleted atom
atoms.calc = EAM(potential='/home/eng/essswb/lammps/lammps-22Jul2025/potentials/Cu_u3.eam')

logdata_ase_id = []
def observer_ase_id():
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    fmax = np.max(np.abs(f))
    step = len(logdata_ase_id)
    logdata_ase_id.append({'step': step, 'fmax': fmax})
    print(f"  Step {step}: fmax = {fmax:.6e} eV/A")

opt = PreconLBFGS(atoms, precon=None, memory=100, use_armijo=True, c1=0.01, maxstep=0.2)
opt.attach(observer_ase_id)
opt.run(fmax=0.0001, steps=100)

results['ase_identity'] = logdata_ase_id
print(f"ASE Identity: Converged in {len(logdata_ase_id)} steps to fmax={logdata_ase_id[-1]['fmax']:.6e}")

# ============================================================================
# 2. ASE with Exp Preconditioner
# ============================================================================
print("\n" + "="*80)
print("2. ASE PreconLBFGS - Exponential Preconditioner")
print("="*80)

# Reset atoms
atoms.positions[:] = np.array(positions)[:-1]
atoms.calc = EAM(potential='/home/eng/essswb/lammps/lammps-22Jul2025/potentials/Cu_u3.eam')

logdata_ase_exp = []
def observer_ase_exp():
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    fmax = np.max(np.abs(f))
    step = len(logdata_ase_exp)
    logdata_ase_exp.append({'step': step, 'fmax': fmax})
    print(f"  Step {step}: fmax = {fmax:.6e} eV/A")

precon = Exp(A=3.0, mu=None, r_cut=None, c_stab=0.1, logfile=None)
opt = PreconLBFGS(atoms, precon=precon, memory=100, use_armijo=True, c1=0.01, maxstep=0.2)
opt.attach(observer_ase_exp)
opt.run(fmax=0.0001, steps=100)

results['ase_exp'] = logdata_ase_exp
print(f"ASE Exp: Converged in {len(logdata_ase_exp)} steps to fmax={logdata_ase_exp[-1]['fmax']:.6e}")

# ============================================================================
# Write LAMMPS data file for both LAMMPS tests
# ============================================================================
print("\n" + "="*80)
print("Creating LAMMPS data file")
print("="*80)

# Reset atoms to initial positions
atoms.positions[:] = np.array(positions)[:-1]

# Write LAMMPS data file
import os
from ase.io import write
data_file = os.path.join(os.path.dirname(__file__), 'lammps_comparison_atoms.data')
write(data_file, atoms, format='lammps-data')
print(f"Created {data_file}")

# ============================================================================
# 3. LAMMPS with Identity Preconditioner
# ============================================================================
print("\n" + "="*80)
print("3. LAMMPS PreconLBFGS - Identity Preconditioner")
print("="*80)

# Create LAMMPS input
lammps_id_input = f"""units metal
atom_style atomic
atom_modify map yes
read_data {data_file}
pair_style eam
pair_coeff * * /home/eng/essswb/lammps/lammps-22Jul2025/potentials/Cu_u3.eam
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
plugin load /home/eng/essswb/lammps-precon-opt-fix/build/preconlbfgsplugin.so
fix opt all precon_lbfgs 0.0001 precon none memory 100 maxstep 0.2 c1 0.01
thermo 1
thermo_style custom step pe fmax fnorm
minimize 0.0 0.0001 100 10000
unfix opt
"""

with open('/tmp/lammps_id_test.lam', 'w') as f:
    f.write(lammps_id_input)

result = subprocess.run(
    "module purge && module load GCC/13.2.0 Eigen && " +
    "/home/eng/essswb/lammps/lammps-22Jul2025/build/lmp -in /tmp/lammps_id_test.lam",
    shell=True, capture_output=True, text=True, executable='/bin/bash', timeout=120
)

# Parse output - look for lines with numeric step, energy, fmax, fnorm
logdata_lammps_id = []
for line in result.stdout.split('\n'):
    parts = line.split()
    if len(parts) >= 4:
        try:
            step = int(parts[0])
            pe = float(parts[1])
            fmax = float(parts[2])
            fnorm = float(parts[3])
            logdata_lammps_id.append({'step': step, 'fmax': fmax})
            print(f"  Step {step}: fmax = {fmax:.6e} eV/A")
        except (ValueError, IndexError):
            # Skip header lines and non-numeric data
            pass

results['lammps_identity'] = logdata_lammps_id
if logdata_lammps_id:
    print(f"LAMMPS Identity: {len(logdata_lammps_id)} steps, final fmax={logdata_lammps_id[-1]['fmax']:.6e}")

# ============================================================================
# 4. LAMMPS with Exp Preconditioner
# ============================================================================
print("\n" + "="*80)
print("4. LAMMPS PreconLBFGS - Exponential Preconditioner")
print("="*80)

lammps_exp_input = f"""units metal
atom_style atomic
atom_modify map yes
read_data {data_file}
pair_style eam
pair_coeff * * /home/eng/essswb/lammps/lammps-22Jul2025/potentials/Cu_u3.eam
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
plugin load /home/eng/essswb/lammps-precon-opt-fix/build/preconlbfgsplugin.so
fix opt all precon_lbfgs 0.0001 precon exp memory 100 maxstep 0.2 c1 0.01
thermo 1
thermo_style custom step pe fmax fnorm
minimize 0.0 0.0001 100 10000
unfix opt
"""

with open('/tmp/lammps_exp_test.lam', 'w') as f:
    f.write(lammps_exp_input)

result = subprocess.run(
    "module purge && module load GCC/13.2.0 Eigen && " +
    "/home/eng/essswb/lammps/lammps-22Jul2025/build/lmp -in /tmp/lammps_exp_test.lam",
    shell=True, capture_output=True, text=True, executable='/bin/bash', timeout=120
)

# Parse output - look for lines with numeric step, energy, fmax, fnorm
logdata_lammps_exp = []
for line in result.stdout.split('\n'):
    parts = line.split()
    if len(parts) >= 4:
        try:
            step = int(parts[0])
            pe = float(parts[1])
            fmax = float(parts[2])
            fnorm = float(parts[3])
            logdata_lammps_exp.append({'step': step, 'fmax': fmax})
            print(f"  Step {step}: fmax = {fmax:.6e} eV/A")
        except (ValueError, IndexError):
            # Skip header lines and non-numeric data
            pass

results['lammps_exp'] = logdata_lammps_exp
if logdata_lammps_exp:
    print(f"LAMMPS Exp: {len(logdata_lammps_exp)} steps, final fmax={logdata_lammps_exp[-1]['fmax']:.6e}")

# ============================================================================
# Create Plot
# ============================================================================
print("\n" + "="*80)
print("Creating Convergence Plot")
print("="*80)

fig, ax = plt.subplots(figsize=(10, 7))

# Plot all curves
if results['ase_identity']:
    steps = [d['step'] for d in results['ase_identity']]
    fmax = [d['fmax'] for d in results['ase_identity']]
    ax.semilogy(steps, fmax, 'o-', label='ASE Identity', linewidth=2, markersize=8, color='#1f77b4')

if results['ase_exp']:
    steps = [d['step'] for d in results['ase_exp']]
    fmax = [d['fmax'] for d in results['ase_exp']]
    ax.semilogy(steps, fmax, 's-', label='ASE Exp', linewidth=2, markersize=8, color='#ff7f0e')

if results['lammps_identity']:
    steps = [d['step'] for d in results['lammps_identity']]
    fmax = [d['fmax'] for d in results['lammps_identity']]
    ax.semilogy(steps, fmax, '^-', label='LAMMPS Identity', linewidth=2, markersize=8, color='#2ca02c')

if results['lammps_exp']:
    steps = [d['step'] for d in results['lammps_exp']]
    fmax = [d['fmax'] for d in results['lammps_exp']]
    ax.semilogy(steps, fmax, 'd-', label='LAMMPS Exp', linewidth=2, markersize=8, color='#d62728')

# Target line
ax.axhline(y=0.001, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (0.001 eV/Å)')

# Formatting
ax.set_xlabel('Optimization Step', fontsize=14, fontweight='bold')
ax.set_ylabel('Max Force (eV/Å)', fontsize=14, fontweight='bold')
ax.set_title('Convergence Comparison: ASE vs LAMMPS PreconLBFGS\n2047-atom Cu with Vacancy (8x8x8)',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, which='both')
ax.tick_params(labelsize=11)

# Set reasonable y-axis limits
ax.set_ylim(5e-4, 2e-1)

plt.tight_layout()

# Save plot
output_path = os.path.join(os.path.dirname(__file__), 'full_convergence_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Save data
data_json_path = os.path.join(os.path.dirname(__file__), 'full_convergence_data.json')
with open(data_json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Data saved to: {data_json_path}")

# ============================================================================
# Summary Table
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Method':<20} {'Steps':>8} {'Final fmax':>15} {'Status':>10}")
print("-"*80)

for name, label in [('ase_identity', 'ASE Identity'), 
                     ('ase_exp', 'ASE Exp'),
                     ('lammps_identity', 'LAMMPS Identity'),
                     ('lammps_exp', 'LAMMPS Exp')]:
    if results[name]:
        steps = len(results[name])
        final_fmax = results[name][-1]['fmax']
        status = 'Converged' if final_fmax < 0.001 else 'Running'
        print(f"{label:<20} {steps:>8} {final_fmax:>15.6e} {status:>10}")
    else:
        print(f"{label:<20} {'N/A':>8} {'N/A':>15} {'Failed':>10}")

print("="*80)
print(f"\nPlot saved to: {output_path}")
print(f"View with: display {output_path}")

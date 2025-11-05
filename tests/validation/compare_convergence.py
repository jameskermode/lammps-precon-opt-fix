#!/usr/bin/env python3
import numpy as np
from ase import Atoms
from ase.calculators.eam import EAM
from ase.optimize.precon import PreconLBFGS
import subprocess

# Create identical system
a = 3.615
positions = []
for i in range(4):
    for j in range(4):
        for k in range(4):
            for basis in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                pos = [a*(i+basis[0]), a*(j+basis[1]), a*(k+basis[2])]
                positions.append(pos)

atoms = Atoms('Cu' * 256, positions=positions, 
              cell=[(4*a, 0, 0), (0, 4*a, 0), (0, 0, 4*a)], pbc=True)
center_idx = len(atoms) // 2
del atoms[center_idx]

atoms.calc = EAM(potential='/home/eng/essswb/lammps/lammps-22Jul2025/potentials/Cu_u3.eam')

# Run ASE
print("=" * 60)
print("ASE PreconLBFGS (Identity preconditioner)")
print("=" * 60)
logdata = []

def observer():
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    fmax = np.max(np.abs(f))
    fnorm = np.linalg.norm(f)
    step = len(logdata)
    logdata.append({'step': step, 'energy': e, 'fmax': fmax, 'fnorm': fnorm})
    print(f"Step {step}: E = {e:.10f}, fmax = {fmax:.10f}, fnorm = {fnorm:.10f}")

opt = PreconLBFGS(atoms, precon=None, memory=100, use_armijo=True, c1=0.001)
opt.attach(observer)
opt.run(fmax=0.01, steps=20)

print(f"\nASE Final: E = {atoms.get_potential_energy():.10f}, fmax = {np.max(np.abs(atoms.get_forces())):.10f}")
print(f"ASE Steps: {len(logdata)}")

# Run LAMMPS
print("\n" + "=" * 60)
print("LAMMPS PreconLBFGS (Identity preconditioner)")
print("=" * 60)

result = subprocess.run(
    "module purge && module load GCC/13.2.0 Eigen && " +
    "/home/eng/essswb/lammps/lammps-22Jul2025/build/lmp -in tests/linesearch_debug/test_fix2.lam",
    shell=True, capture_output=True, text=True, executable='/bin/bash', timeout=60
)

# Parse LAMMPS output
for line in result.stdout.split('\n'):
    if 'PreconLBFGS: step' in line or 'Step' in line or 'Force' in line or 'Iterations' in line:
        print(line)

print("\n" + "=" * 60)
print("Comparison Summary")
print("=" * 60)
print(f"ASE:    {len(logdata)} steps to convergence")
print(f"LAMMPS: Check output above")

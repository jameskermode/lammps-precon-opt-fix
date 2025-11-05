#!/usr/bin/env python3
import numpy as np
from ase import Atoms
from ase.calculators.eam import EAM
from ase.optimize.precon import PreconLBFGS, Exp

# Create 4x4x4 FCC Cu with vacancy (255 atoms) - same as LAMMPS test
a = 3.615  # Cu lattice constant
positions = []
for i in range(4):
    for j in range(4):
        for k in range(4):
            for basis in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                pos = [a*(i+basis[0]), a*(j+basis[1]), a*(k+basis[2])]
                positions.append(pos)

atoms = Atoms('Cu' * 256, positions=positions, 
              cell=[(4*a, 0, 0), (0, 4*a, 0), (0, 0, 4*a)], pbc=True)

# Create vacancy at center
center_idx = len(atoms) // 2
del atoms[center_idx]

# Set up calculator
atoms.calc = EAM(potential='/home/eng/essswb/lammps/lammps-22Jul2025/potentials/Cu_u3.eam')

# Initial state
e0 = atoms.get_potential_energy()
f0 = atoms.get_forces()
fmax0 = np.max(np.abs(f0))
print(f"Initial: E = {e0:.10f} eV, fmax = {fmax0:.10f} eV/A")

# Run with identity preconditioner (same as LAMMPS test)
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

print(f"\nFinal: E = {atoms.get_potential_energy():.10f}, fmax = {np.max(np.abs(atoms.get_forces())):.10f}")
print(f"Total steps: {len(logdata)}")

#!/usr/bin/env python3
import numpy as np
from ase import Atoms
from ase.calculators.eam import EAM
from ase.io import read, write

# Create system
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
print(f"Deleting atom at index {center_idx}")
del atoms[center_idx]

atoms.calc = EAM(potential='/home/eng/essswb/lammps/lammps-22Jul2025/potentials/Cu_u3.eam')

e = atoms.get_potential_energy()
f = atoms.get_forces()
fmax = np.max(np.abs(f))

print(f"ASE Initial:")
print(f"  Atoms: {len(atoms)}")
print(f"  Energy: {e:.10f} eV")
print(f"  Fmax: {fmax:.10f} eV/A")

# Check what LAMMPS has
print(f"\nLAMMPS data file check:")
with open('tests/validation/lammps_identity_atoms.data', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines[:20]):
        print(f"{i}: {line.rstrip()}")

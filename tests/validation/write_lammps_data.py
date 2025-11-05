#!/usr/bin/env python3
"""
Create a LAMMPS data file from ASE atoms object
This ensures identical starting configurations between ASE and LAMMPS
"""

import numpy as np
from ase import Atoms

# Create same system as in ase_reference.py
a = 3.615  # Cu lattice parameter

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

# Perturb center atom (same as ASE reference)
np.random.seed(42)
center_idx = len(atoms) // 2
atoms.positions[center_idx] += np.array([0.1, 0.1, 0.1])

print(f"Created {len(atoms)} atoms")
print(f"Cell: {atoms.cell.diagonal()}")
print(f"Perturbed atom {center_idx} at position: {atoms.positions[center_idx]}")

# Write LAMMPS data file
with open('tests/validation/atoms.data', 'w') as f:
    f.write("# LAMMPS data file written from ASE\n")
    f.write(f"# 3x3x3 Cu FCC with perturbed atom {center_idx}\n")
    f.write(f"\n")
    f.write(f"{len(atoms)} atoms\n")
    f.write(f"1 atom types\n")
    f.write(f"\n")
    f.write(f"0.0 {atoms.cell[0,0]:.10f} xlo xhi\n")
    f.write(f"0.0 {atoms.cell[1,1]:.10f} ylo yhi\n")
    f.write(f"0.0 {atoms.cell[2,2]:.10f} zlo zhi\n")
    f.write(f"\n")
    f.write(f"Masses\n")
    f.write(f"\n")
    f.write(f"1 63.546\n")
    f.write(f"\n")
    f.write(f"Atoms # atomic\n")
    f.write(f"\n")
    for i, pos in enumerate(atoms.positions):
        f.write(f"{i+1} 1 {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}\n")

print(f"Wrote LAMMPS data file: tests/validation/atoms.data")

#!/usr/bin/env python3
"""
Generate reference data from ASE PreconLBFGS for validation

This script runs ASE PreconLBFGS on test structures and saves:
- Preconditioner matrices
- Optimized structures
- Mu values
- Optimization trajectories

Output is saved in tests/reference_data/
"""

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize.precon import PreconLBFGS, Exp
from ase.io import write
import os

# Create output directory
output_dir = "../tests/reference_data"
os.makedirs(output_dir, exist_ok=True)

print("Generating reference data from ASE PreconLBFGS...")
print("=" * 60)

# ============================================================================
# Test Case 1: Bulk Al 2x2x2 (for preconditioner assembly tests)
# ============================================================================
print("\n1. Bulk Al 2x2x2 - Preconditioner assembly")
print("-" * 60)

atoms = bulk('Al', 'fcc', a=3.994) * (2, 2, 2)
atoms.calc = EMT()

# Build preconditioner
precon = Exp(A=3.0, r_cut=4.0, mu=1.0)
precon.make_precon(atoms)

# Save preconditioner matrix as dense array
P = precon.asarray()
np.save(f"{output_dir}/al_bulk_precon_matrix.npy", P)
print(f"  Saved: al_bulk_precon_matrix.npy, shape={P.shape}")

# Save structure
write(f"{output_dir}/al_bulk.xyz", atoms)
print(f"  Saved: al_bulk.xyz, natoms={len(atoms)}")

# Save forces
forces = atoms.get_forces()
np.save(f"{output_dir}/al_bulk_forces.npy", forces)
print(f"  Saved: al_bulk_forces.npy")

# ============================================================================
# Test Case 2: Bulk Al rattled (for perturbed structure tests)
# ============================================================================
print("\n2. Bulk Al 2x2x2 - Rattled")
print("-" * 60)

atoms_rattled = bulk('Al', 'fcc', a=3.994) * (2, 2, 2)
atoms_rattled.rattle(stdev=0.1, seed=7)
atoms_rattled.calc = EMT()

precon_rattled = Exp(A=3.0, r_cut=4.0, mu=1.0)
precon_rattled.make_precon(atoms_rattled)

P_rattled = precon_rattled.asarray()
np.save(f"{output_dir}/al_bulk_rattled_precon_matrix.npy", P_rattled)
write(f"{output_dir}/al_bulk_rattled.xyz", atoms_rattled)
print(f"  Saved: al_bulk_rattled structures")

# ============================================================================
# Test Case 3: Mu estimation for Al bulk
# ============================================================================
print("\n3. Mu estimation - Al bulk")
print("-" * 60)

atoms_mu = bulk('Al', 'fcc', a=3.994) * (2, 2, 2)
atoms_mu.calc = EMT()

precon_auto = Exp(A=3.0, r_cut=4.0, mu=None)  # Auto-estimate mu
precon_auto.make_precon(atoms_mu)
mu_estimated = precon_auto.mu

np.save(f"{output_dir}/al_bulk_mu.npy", np.array([mu_estimated]))
print(f"  Estimated mu = {mu_estimated:.6f}")
print(f"  Saved: al_bulk_mu.npy")

# ============================================================================
# Test Case 4: Mu estimation for Cu bulk
# ============================================================================
print("\n4. Mu estimation - Cu bulk")
print("-" * 60)

atoms_cu = bulk('Cu', 'fcc', a=3.615) * (2, 2, 2)
atoms_cu.calc = EMT()

precon_cu = Exp(A=3.0, r_cut=4.0, mu=None)
precon_cu.make_precon(atoms_cu)
mu_cu = precon_cu.mu

np.save(f"{output_dir}/cu_bulk_mu.npy", np.array([mu_cu]))
print(f"  Estimated mu = {mu_cu:.6f}")
print(f"  Saved: cu_bulk_mu.npy")

# ============================================================================
# Test Case 5: PreconLBFGS optimization - compressed Cu
# ============================================================================
print("\n5. PreconLBFGS optimization - Compressed Cu")
print("-" * 60)

# Create compressed structure
atoms_opt = bulk('Cu', 'fcc', a=3.615) * (2, 2, 2)
s = atoms_opt.get_scaled_positions()
s[:, 0] *= 0.995  # Compress by 0.5%
atoms_opt.set_scaled_positions(s)
atoms_opt.calc = EMT()

# Save initial structure
write(f"{output_dir}/cu_bulk_compressed.xyz", atoms_opt)
print(f"  Initial energy: {atoms_opt.get_potential_energy():.6f} eV")

# Optimize with Exp preconditioner
precon_opt = Exp(A=3.0, mu=None)  # Auto mu
opt = PreconLBFGS(atoms_opt, precon=precon_opt, use_armijo=True, logfile='-')
opt.run(fmax=0.01)

print(f"  Converged in {opt.get_number_of_steps()} steps")
print(f"  Final energy: {atoms_opt.get_potential_energy():.6f} eV")

# Save optimized structure
write(f"{output_dir}/cu_bulk_optimized_with_precon.xyz", atoms_opt)

# Save optimization info
opt_info = {
    'nsteps': opt.get_number_of_steps(),
    'final_energy': atoms_opt.get_potential_energy(),
    'final_fmax': np.abs(atoms_opt.get_forces()).max()
}
np.save(f"{output_dir}/cu_bulk_opt_info_precon.npy", opt_info)
print(f"  Saved: cu_bulk_optimized_with_precon.xyz")

# ============================================================================
# Test Case 6: PreconLBFGS optimization - identity preconditioner (baseline)
# ============================================================================
print("\n6. PreconLBFGS optimization - Identity precon (baseline)")
print("-" * 60)

atoms_opt_id = bulk('Cu', 'fcc', a=3.615) * (2, 2, 2)
s = atoms_opt_id.get_scaled_positions()
s[:, 0] *= 0.995
atoms_opt_id.set_scaled_positions(s)
atoms_opt_id.calc = EMT()

# Optimize with no preconditioner (identity)
opt_id = PreconLBFGS(atoms_opt_id, precon=None, use_armijo=True, logfile='-')
opt_id.run(fmax=0.01)

print(f"  Converged in {opt_id.get_number_of_steps()} steps")
print(f"  Final energy: {atoms_opt_id.get_potential_energy():.6f} eV")

write(f"{output_dir}/cu_bulk_optimized_identity.xyz", atoms_opt_id)

opt_info_id = {
    'nsteps': opt_id.get_number_of_steps(),
    'final_energy': atoms_opt_id.get_potential_energy(),
    'final_fmax': np.abs(atoms_opt_id.get_forces()).max()
}
np.save(f"{output_dir}/cu_bulk_opt_info_identity.npy", opt_info_id)

# Compare
speedup = opt_info_id['nsteps'] / opt_info['nsteps']
print(f"\n  Speedup: {speedup:.2f}x (Exp vs Identity)")
print(f"  Saved: cu_bulk_optimized_identity.xyz")

# ============================================================================
# Test Case 7: r_NN values for different structures
# ============================================================================
print("\n7. r_NN values for various structures")
print("-" * 60)

structures = {
    'Al_fcc': bulk('Al', 'fcc', a=3.994) * (2, 2, 2),
    'Cu_fcc': bulk('Cu', 'fcc', a=3.615) * (2, 2, 2),
}

r_NN_values = {}
for name, atoms_struct in structures.items():
    atoms_struct.calc = EMT()
    precon_temp = Exp(A=3.0)
    precon_temp.make_precon(atoms_struct)
    r_NN_values[name] = precon_temp.r_NN
    print(f"  {name}: r_NN = {precon_temp.r_NN:.6f} Angstrom")

np.save(f"{output_dir}/r_NN_values.npy", r_NN_values)
print(f"  Saved: r_NN_values.npy")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("Reference data generation complete!")
print(f"Files saved in: {output_dir}/")
print("\nGenerated files:")
print("  - al_bulk_precon_matrix.npy")
print("  - al_bulk.xyz, al_bulk_forces.npy")
print("  - al_bulk_rattled_precon_matrix.npy, al_bulk_rattled.xyz")
print("  - al_bulk_mu.npy, cu_bulk_mu.npy")
print("  - cu_bulk_compressed.xyz")
print("  - cu_bulk_optimized_with_precon.xyz")
print("  - cu_bulk_optimized_identity.xyz")
print("  - cu_bulk_opt_info_precon.npy, cu_bulk_opt_info_identity.npy")
print("  - r_NN_values.npy")
print("\nThese files will be used to validate the C++ implementation.")
print("=" * 60)

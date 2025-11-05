#!/usr/bin/env python3
import subprocess, os

def run_test(c1, maxstep, precon, debug=0):
    precon_args = 'precon none' if precon == 'identity' else 'precon exp r_cut -1.0 mu -1.0 A 3.0 c_stab 0.1'
    content = f"""units metal
atom_style atomic
atom_modify map yes
read_data tests/validation/lammps_identity_atoms.data
pair_style eam
pair_coeff * * /home/eng/essswb/lammps/lammps-22Jul2025/potentials/Cu_u3.eam
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
plugin load /home/eng/essswb/lammps-precon-opt-fix/build/preconlbfgsplugin.so
fix opt all precon_lbfgs 0.01 {precon_args} memory 100 maxstep {maxstep} c1 {c1} linesearch_debug {debug}
thermo 1
thermo_style custom step pe fmax fnorm
minimize 0.0 0.01 20 1000
unfix opt
"""
    name = f"test_c1_{c1}_ms_{maxstep}_{precon}"
    inp_file = f"tests/linesearch_debug/{name}.lam"
    with open(inp_file, 'w') as f: f.write(content)
    
    lammps = "/home/eng/essswb/lammps/lammps-22Jul2025/build/lmp"
    result = subprocess.run(f"module purge && module load GCC/13.2.0 Eigen && {lammps} -in {inp_file}",
                          shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          executable='/bin/bash', timeout=60)
    
    log_file = f"tests/linesearch_debug/{name}.log"
    with open(log_file, 'w') as f: f.write(result.stdout + result.stderr)
    
    # Parse result
    fmax, steps = None, 0
    for line in result.stdout.split('\n'):
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            try: steps, fmax = int(parts[0]), float(parts[2])
            except: pass
    
    converged = fmax is not None and fmax < 0.01
    return converged, steps, fmax, log_file

# Test different parameters
print("Testing line search parameters...")
print("="*70)

tests = [
    (0.23, 0.04, 'identity'),
    (0.1, 0.04, 'identity'),
    (0.01, 0.04, 'identity'),
    (0.001, 0.04, 'identity'),
    (0.0001, 0.04, 'identity'),
    (0.23, 0.02, 'identity'),
    (0.01, 0.02, 'identity'),
    (0.001, 0.02, 'identity'),
    (0.0001, 0.02, 'identity'),
    (0.01, 0.01, 'identity'),
    (0.001, 0.01, 'identity'),
    (0.0001, 0.01, 'identity'),
]

for c1, maxstep, precon in tests:
    try:
        conv, steps, fmax, log = run_test(c1, maxstep, precon)
        status = "✓" if conv else "✗"
        fmax_str = f"{fmax:.6f}" if fmax else "N/A"
        print(f"{status} c1={c1:<7} ms={maxstep:<6} {precon:<8} steps={steps:<3} fmax={fmax_str}")
    except Exception as e:
        print(f"✗ c1={c1:<7} ms={maxstep:<6} {precon:<8} ERROR: {e}")

print("="*70)

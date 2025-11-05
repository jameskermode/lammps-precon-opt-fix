#!/usr/bin/env python3
"""
Line search parameter sweep to debug convergence failures

Tests various combinations of:
- c1 (Armijo condition parameter)
- maxstep (maximum step size)
- preconditioner (identity vs exp)
"""

import subprocess
import os
import re
import json
import numpy as np

def create_lammps_input(output_dir, test_name, c1, maxstep, precon_type, debug=1):
    """Create LAMMPS input file with specified parameters"""

    precon_args = {
        'identity': 'precon none',
        'exp': 'precon exp r_cut -1.0 mu -1.0 A 3.0 c_stab 0.1'
    }

    input_content = f"""# Line search debug test: {test_name}
# Parameters: c1={c1}, maxstep={maxstep}, precon={precon_type}

units metal
atom_style atomic
atom_modify map yes

# Read vacancy system
read_data tests/validation/lammps_identity_atoms.data

pair_style eam
pair_coeff * * /home/eng/essswb/lammps/lammps-22Jul2025/potentials/Cu_u3.eam

neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

plugin load /home/eng/essswb/lammps-precon-opt-fix/build/preconlbfgsplugin.so

# PreconLBFGS with debug enabled
fix opt all precon_lbfgs 0.01 {precon_args[precon_type]} \\
    memory 100 maxstep {maxstep} c1 {c1} linesearch_debug {debug}

thermo 1
thermo_style custom step pe fmax fnorm

minimize 0.0 0.01 20 1000

unfix opt
"""

    filename = f"{output_dir}/{test_name}.lam"
    with open(filename, 'w') as f:
        f.write(input_content)

    return filename

def run_lammps_test(input_file, output_dir, test_name):
    """Run LAMMPS with the given input file"""
    log_file = f"{output_dir}/{test_name}.log"
    lammps_exe = "/home/eng/essswb/lammps/lammps-22Jul2025/build/lmp"

    try:
        result = subprocess.run(
            f"module purge && module load GCC/13.2.0 Eigen && {lammps_exe} -in {input_file}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            executable='/bin/bash',
            timeout=60
        )

        with open(log_file, 'w') as f:
            f.write(result.stdout)
            f.write(result.stderr)

        return log_file, result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {test_name}")
        return log_file, False
    except Exception as e:
        print(f"  ERROR: {test_name}: {e}")
        return log_file, False

def parse_lammps_log(log_file):
    """Parse LAMMPS log to extract convergence info"""
    with open(log_file, 'r') as f:
        content = f.read()

    # Extract line search debug info
    ls_debug_pattern = r'\[LS DEBUG\](.*?)(?=\[LS DEBUG\]|\Z)'
    ls_matches = re.findall(ls_debug_pattern, content, re.DOTALL)

    # Extract final convergence status
    lines = content.split('\n')

    converged = False
    final_fmax = None
    final_energy = None
    steps = 0

    in_thermo = False
    for line in lines:
        if 'Step' in line and 'PotEng' in line and 'Fmax' in line:
            in_thermo = True
            continue

        if in_thermo and line.strip():
            if line.startswith('Loop'):
                in_thermo = False
                continue

            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    steps = int(parts[0])
                    final_energy = float(parts[1])
                    final_fmax = float(parts[2])
                except (ValueError, IndexError):
                    pass

    if final_fmax is not None and final_fmax < 0.01:
        converged = True

    # Extract line search details
    ls_iterations = []
    for match in ls_matches:
        # Parse alpha values from each line search
        alpha_matches = re.findall(r'iter\s+(\d+):\s+alpha\s+=\s+([\d.eE+-]+)', match)
        if alpha_matches:
            alphas = [(int(m[0]), float(m[1])) for m in alpha_matches]
            ls_iterations.append(alphas)

    # Check for specific failure modes
    line_search_failed = 'Line search failed' in content or 'alpha < a_min' in content
    not_descent = 'Not a descent direction' in content

    return {
        'converged': converged,
        'final_fmax': final_fmax,
        'final_energy': final_energy,
        'steps': steps,
        'line_search_failed': line_search_failed,
        'not_descent_direction': not_descent,
        'ls_iterations': ls_iterations,
        'num_line_searches': len(ls_iterations)
    }

def main():
    print("="*80)
    print("LAMMPS PreconLBFGS Line Search Parameter Sweep")
    print("="*80)

    # Create output directory
    output_dir = "tests/linesearch_debug"
    os.makedirs(output_dir, exist_ok=True)

    # Parameter ranges to test
    c1_values = [0.23, 0.1, 0.01, 0.001, 0.0001]
    maxstep_values = [0.04, 0.02, 0.01, 0.005]
    precon_types = ['identity', 'exp']

    results = []

    test_count = 0
    total_tests = len(c1_values) * len(maxstep_values) * len(precon_types)

    print(f"\nRunning {total_tests} tests...\n")

    for precon in precon_types:
        for c1 in c1_values:
            for maxstep in maxstep_values:
                test_count += 1
                test_name = f"test_c1_{c1}_maxstep_{maxstep}_{precon}"

                print(f"[{test_count}/{total_tests}] Testing c1={c1}, maxstep={maxstep}, precon={precon}...")

                # Create input file
                input_file = create_lammps_input(output_dir, test_name, c1, maxstep, precon, debug=1)

                # Run LAMMPS
                log_file, success = run_lammps_test(input_file, output_dir, test_name)

                # Parse results
                if success:
                    result = parse_lammps_log(log_file)
                    result['c1'] = c1
                    result['maxstep'] = maxstep
                    result['precon'] = precon
                    result['test_name'] = test_name
                    results.append(result)

                    status = "✓ CONVERGED" if result['converged'] else "✗ FAILED"
                    fmax_str = f"{result['final_fmax']:.6f}" if result['final_fmax'] is not None else "N/A"
                    print(f"  {status}: fmax={fmax_str}, steps={result['steps']}")
                else:
                    print(f"  ✗ RUN FAILED")

    # Save results to JSON
    results_file = f"{output_dir}/sweep_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    # Analyze results
    converged_results = [r for r in results if r['converged']]
    failed_results = [r for r in results if not r['converged']]

    print(f"Total tests: {len(results)}")
    print(f"Converged: {len(converged_results)} ({100*len(converged_results)/len(results):.1f}%)")
    print(f"Failed: {len(failed_results)} ({100*len(failed_results)/len(results):.1f}%)")

    if converged_results:
        print(f"\n{'='*80}")
        print("SUCCESSFUL PARAMETER COMBINATIONS")
        print(f"{'='*80}")
        print(f"{'Precon':<10} {'c1':<8} {'maxstep':<10} {'Steps':<8} {'Final fmax':<12}")
        print(f"{'-'*80}")

        for r in converged_results:
            print(f"{r['precon']:<10} {r['c1']:<8.4f} {r['maxstep']:<10.4f} "
                  f"{r['steps']:<8} {r['final_fmax']:<12.6f}")

    if failed_results:
        print(f"\n{'='*80}")
        print("FAILED PARAMETER COMBINATIONS")
        print(f"{'='*80}")
        print(f"{'Precon':<10} {'c1':<8} {'maxstep':<10} {'Steps':<8} {'Reason':<20}")
        print(f"{'-'*80}")

        for r in failed_results:
            reason = "Line search failed" if r['line_search_failed'] else \
                     "Not descent" if r['not_descent_direction'] else \
                     "Unknown"
            print(f"{r['precon']:<10} {r['c1']:<8.4f} {r['maxstep']:<10.4f} "
                  f"{r['steps']:<8} {reason:<20}")

    # Find best parameters
    if converged_results:
        print(f"\n{'='*80}")
        print("RECOMMENDED PARAMETERS")
        print(f"{'='*80}\n")

        # Group by preconditioner
        for precon in precon_types:
            precon_results = [r for r in converged_results if r['precon'] == precon]
            if precon_results:
                # Find combination with fewest steps
                best = min(precon_results, key=lambda r: r['steps'])
                print(f"{precon.upper()} preconditioner:")
                print(f"  c1 = {best['c1']}")
                print(f"  maxstep = {best['maxstep']}")
                print(f"  Converged in {best['steps']} steps")
                print(f"  Final fmax = {best['final_fmax']:.6f} eV/A")
                print()

    print(f"Results saved to: {results_file}")
    print(f"Log files in: {output_dir}/")
    print("="*80)

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

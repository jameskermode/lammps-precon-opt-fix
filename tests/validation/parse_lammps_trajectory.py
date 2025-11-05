#!/usr/bin/env python3
"""
Parse LAMMPS dump file and log to JSON format matching ASE output
"""

import json
import re
import numpy as np

def parse_lammps_dump(dump_file):
    """Parse LAMMPS dump file into trajectory data"""
    trajectory = []

    with open(dump_file, 'r') as f:
        lines = f.readlines()

    i = 0
    step = 0
    while i < len(lines):
        if lines[i].startswith('ITEM: TIMESTEP'):
            # Parse timestep
            timestep = int(lines[i+1].strip())
            i += 2

            # Parse number of atoms
            if lines[i].startswith('ITEM: NUMBER OF ATOMS'):
                natoms = int(lines[i+1].strip())
                i += 2

            # Skip box bounds
            if lines[i].startswith('ITEM: BOX BOUNDS'):
                i += 4  # Skip 3 lines of box bounds + header

            # Parse atoms header
            if lines[i].startswith('ITEM: ATOMS'):
                header = lines[i].strip().split()[2:]  # Skip "ITEM: ATOMS"
                i += 1

                # Parse atom data
                atoms_data = []
                for j in range(natoms):
                    atoms_data.append(lines[i+j].strip().split())
                i += natoms

                # Extract data based on header
                id_idx = header.index('id')
                x_idx = header.index('x')
                y_idx = header.index('y')
                z_idx = header.index('z')
                fx_idx = header.index('fx')
                fy_idx = header.index('fy')
                fz_idx = header.index('fz')

                # Sort by atom ID
                atoms_data.sort(key=lambda x: int(x[id_idx]))

                positions = []
                forces = []
                for atom in atoms_data:
                    positions.append([
                        float(atom[x_idx]),
                        float(atom[y_idx]),
                        float(atom[z_idx])
                    ])
                    forces.append([
                        float(atom[fx_idx]),
                        float(atom[fy_idx]),
                        float(atom[fz_idx])
                    ])

                positions = np.array(positions)
                forces = np.array(forces)

                # Compute metrics
                fmax = float(np.max(np.abs(forces)))
                fnorm = float(np.linalg.norm(forces))

                frame = {
                    'step': step,
                    'timestep': timestep,
                    'positions': positions.tolist(),
                    'forces': forces.tolist(),
                    'fmax': fmax,
                    'fnorm': fnorm,
                }

                trajectory.append(frame)
                step += 1
        else:
            i += 1

    return trajectory

def parse_lammps_log(log_file):
    """Parse LAMMPS log file to extract energy and preconditioner info"""
    energies = []
    precon_info = {}

    with open(log_file, 'r') as f:
        content = f.read()

    # Extract preconditioner info
    match = re.search(r'Preconditioner built: r_cut=([\d.]+), r_NN=([\d.]+), mu=([\d.]+)', content)
    if match:
        precon_info = {
            'r_cut': float(match.group(1)),
            'r_NN': float(match.group(2)),
            'mu': float(match.group(3)),
        }

    # Extract thermo data
    lines = content.split('\n')
    in_thermo = False
    for line in lines:
        if 'Step' in line and 'PotEng' in line:
            in_thermo = True
            continue
        if in_thermo:
            if line.strip() and not line.startswith('Loop'):
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    try:
                        energy = float(parts[1])
                        energies.append(energy)
                    except ValueError:
                        pass
            if line.startswith('Loop'):
                in_thermo = False

    return energies, precon_info

def main():
    print("Parsing LAMMPS trajectory...")

    dump_file = 'tests/validation/lammps_trajectory.dump'
    log_file = 'tests/validation/lammps_validation.log'

    # Parse dump file
    trajectory = parse_lammps_dump(dump_file)
    print(f"Parsed {len(trajectory)} frames from dump file")

    # Parse log file for energies and precon info
    energies, precon_info = parse_lammps_log(log_file)
    print(f"Parsed {len(energies)} energy values from log file")

    # Add energies to trajectory
    for i, frame in enumerate(trajectory):
        if i < len(energies):
            frame['energy'] = energies[i]
        if i == 0:
            frame['precon'] = precon_info

    # Save to JSON
    output = {
        'metadata': {
            'code': 'LAMMPS',
            'optimizer': 'PreconLBFGS',
            'preconditioner': 'Exp',
            'natoms': len(trajectory[0]['positions']) if trajectory else 0,
            'lattice_parameter': 3.615,
            'compression': 0.99,
            'system': '3x3x3 Cu FCC',
            'potential': 'Cu_u3.eam',
        },
        'parameters': {
            'A': 3.0,
            'c_stab': 0.1,
            'memory': 100,
            'use_armijo': True,
            'c1': 0.23,
            'fmax': 0.01,
        },
        'trajectory': trajectory,
    }

    output_file = 'tests/validation/lammps_trajectory.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved trajectory to: {output_file}")
    print(f"Total steps: {len(trajectory)}")

if __name__ == '__main__':
    main()

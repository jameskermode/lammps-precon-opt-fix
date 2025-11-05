#!/bin/bash
# Master validation script: Compare ASE and LAMMPS PreconLBFGS implementations

set -e  # Exit on error

echo "========================================================================"
echo "PreconLBFGS Validation: ASE vs LAMMPS"
echo "========================================================================"

# Check if we're in the right directory
if [ ! -d "tests/validation" ]; then
    echo "Error: Must run from project root directory"
    exit 1
fi

# Create validation directory if needed
mkdir -p tests/validation

echo ""
echo "Step 1: Running ASE reference optimization..."
echo "------------------------------------------------------------------------"
cd tests/validation
python3 ../../tests/validation/ase_reference.py
cd ../..

if [ ! -f "tests/validation/ase_trajectory.json" ]; then
    echo "Error: ASE trajectory not generated"
    exit 1
fi

echo ""
echo "Step 2: Running LAMMPS validation optimization..."
echo "------------------------------------------------------------------------"
/home/eng/essswb/lammps/lammps-22Jul2025/build/lmp -in tests/validation/lammps_validation.lam \
    > tests/validation/lammps_validation.log 2>&1

if [ ! -f "tests/validation/lammps_trajectory.dump" ]; then
    echo "Error: LAMMPS trajectory not generated"
    exit 1
fi

echo ""
echo "Step 3: Parsing LAMMPS trajectory..."
echo "------------------------------------------------------------------------"
python3 tests/validation/parse_lammps_trajectory.py

if [ ! -f "tests/validation/lammps_trajectory.json" ]; then
    echo "Error: Failed to parse LAMMPS trajectory"
    exit 1
fi

echo ""
echo "Step 4: Comparing trajectories..."
echo "------------------------------------------------------------------------"
python3 tests/validation/compare_trajectories.py

exit_code=$?

echo ""
echo "========================================================================"
if [ $exit_code -eq 0 ]; then
    echo "✓✓✓ VALIDATION COMPLETE: PASS ✓✓✓"
else
    echo "❌❌❌ VALIDATION COMPLETE: FAIL ❌❌❌"
fi
echo "========================================================================"

exit $exit_code

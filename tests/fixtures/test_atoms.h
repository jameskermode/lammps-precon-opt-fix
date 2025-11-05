/**
 * Test fixtures for atom configurations
 *
 * Provides factory functions to create standard test structures:
 * - Bulk crystals (Al, Cu FCC)
 * - Perturbed/rattled structures
 * - Structures with constraints
 */

#ifndef TEST_ATOMS_H
#define TEST_ATOMS_H

#include <vector>
#include <array>
#include <string>
#include <Eigen/Dense>

namespace TestFixtures {

/**
 * Atomic structure for testing
 */
struct TestAtoms {
    int natoms;                              // Number of atoms
    std::vector<std::array<double, 3>> positions;  // Atom positions [natoms][3]
    std::vector<int> types;                  // Atom types
    std::array<double, 3> box_lo;            // Box lower bounds
    std::array<double, 3> box_hi;            // Box upper bounds
    std::array<bool, 3> pbc;                 // Periodic boundary conditions
    std::vector<int> fixed_atoms;            // Indices of fixed atoms
    double lattice_constant;                 // For reference

    TestAtoms() : natoms(0), box_lo({0,0,0}), box_hi({0,0,0}),
                  pbc({true, true, true}), lattice_constant(0.0) {}
};

/**
 * Create bulk Al FCC structure (2x2x2 supercell)
 * Lattice constant: 3.994 Angstrom (relaxed EMT value)
 */
TestAtoms create_bulk_Al_2x2x2();

/**
 * Create bulk Cu FCC structure (2x2x2 supercell)
 * Lattice constant: 3.615 Angstrom (relaxed EMT value)
 */
TestAtoms create_bulk_Cu_2x2x2();

/**
 * Create compressed Cu bulk (for optimization tests)
 * @param scale Scaling factor (e.g., 0.995 = 0.5% compression)
 */
TestAtoms create_bulk_Cu_compressed(double scale = 0.995);

/**
 * Create rattled version of structure
 * @param base Base structure
 * @param stdev Standard deviation of displacement (Angstrom)
 * @param seed Random seed for reproducibility
 */
TestAtoms create_rattled(const TestAtoms& base, double stdev, int seed = 42);

/**
 * Add fixed atoms constraint
 * @param atoms Structure to modify
 * @param indices Atom indices to fix (0-based)
 */
void add_fixed_atoms(TestAtoms& atoms, const std::vector<int>& indices);

/**
 * Compute distance between two positions with minimum image convention
 */
std::array<double, 3> minimum_image(
    const std::array<double, 3>& r1,
    const std::array<double, 3>& r2,
    const std::array<double, 3>& box_lo,
    const std::array<double, 3>& box_hi,
    const std::array<bool, 3>& pbc
);

/**
 * Compute distance with PBC
 */
double compute_distance(
    const std::array<double, 3>& r1,
    const std::array<double, 3>& r2,
    const std::array<double, 3>& box_lo,
    const std::array<double, 3>& box_hi,
    const std::array<bool, 3>& pbc
);

/**
 * Save structure to XYZ file for debugging
 */
void write_xyz(const TestAtoms& atoms, const std::string& filename);

/**
 * Load structure from XYZ file
 */
TestAtoms read_xyz(const std::string& filename);

} // namespace TestFixtures

#endif // TEST_ATOMS_H

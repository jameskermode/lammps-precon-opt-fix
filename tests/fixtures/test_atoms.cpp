/**
 * Implementation of test atom fixtures
 */

#include "test_atoms.h"
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace TestFixtures {

TestAtoms create_bulk_Al_2x2x2() {
    TestAtoms atoms;
    atoms.lattice_constant = 3.994;  // Angstrom (EMT relaxed)

    // FCC conventional cell: 4 atoms
    // Base positions in fractional coordinates
    std::vector<std::array<double, 3>> fcc_base = {
        {0.0, 0.0, 0.0},
        {0.5, 0.5, 0.0},
        {0.5, 0.0, 0.5},
        {0.0, 0.5, 0.5}
    };

    // Create 2x2x2 supercell
    double a = atoms.lattice_constant;
    atoms.natoms = 32;  // 4 * 2 * 2 * 2
    atoms.positions.reserve(atoms.natoms);
    atoms.types.resize(atoms.natoms, 1);  // All type 1 (Al)

    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            for (int iz = 0; iz < 2; iz++) {
                for (const auto& base : fcc_base) {
                    std::array<double, 3> pos = {
                        (base[0] + ix) * a,
                        (base[1] + iy) * a,
                        (base[2] + iz) * a
                    };
                    atoms.positions.push_back(pos);
                }
            }
        }
    }

    // Box dimensions
    atoms.box_lo = {0.0, 0.0, 0.0};
    atoms.box_hi = {2*a, 2*a, 2*a};
    atoms.pbc = {true, true, true};

    return atoms;
}

TestAtoms create_bulk_Cu_2x2x2() {
    TestAtoms atoms;
    atoms.lattice_constant = 3.615;  // Angstrom (EMT relaxed)

    // FCC conventional cell: 4 atoms
    std::vector<std::array<double, 3>> fcc_base = {
        {0.0, 0.0, 0.0},
        {0.5, 0.5, 0.0},
        {0.5, 0.0, 0.5},
        {0.0, 0.5, 0.5}
    };

    // Create 2x2x2 supercell
    double a = atoms.lattice_constant;
    atoms.natoms = 32;
    atoms.positions.reserve(atoms.natoms);
    atoms.types.resize(atoms.natoms, 1);  // All type 1 (Cu)

    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            for (int iz = 0; iz < 2; iz++) {
                for (const auto& base : fcc_base) {
                    std::array<double, 3> pos = {
                        (base[0] + ix) * a,
                        (base[1] + iy) * a,
                        (base[2] + iz) * a
                    };
                    atoms.positions.push_back(pos);
                }
            }
        }
    }

    atoms.box_lo = {0.0, 0.0, 0.0};
    atoms.box_hi = {2*a, 2*a, 2*a};
    atoms.pbc = {true, true, true};

    return atoms;
}

TestAtoms create_bulk_Cu_compressed(double scale) {
    TestAtoms atoms = create_bulk_Cu_2x2x2();

    // Scale positions and box
    for (auto& pos : atoms.positions) {
        pos[0] *= scale;
        pos[1] *= scale;
        pos[2] *= scale;
    }

    atoms.box_hi[0] *= scale;
    atoms.box_hi[1] *= scale;
    atoms.box_hi[2] *= scale;

    atoms.lattice_constant *= scale;

    return atoms;
}

TestAtoms create_rattled(const TestAtoms& base, double stdev, int seed) {
    TestAtoms atoms = base;  // Copy

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, stdev);

    for (auto& pos : atoms.positions) {
        pos[0] += dist(rng);
        pos[1] += dist(rng);
        pos[2] += dist(rng);

        // Wrap into box
        for (int d = 0; d < 3; d++) {
            if (atoms.pbc[d]) {
                double L = atoms.box_hi[d] - atoms.box_lo[d];
                while (pos[d] < atoms.box_lo[d]) pos[d] += L;
                while (pos[d] >= atoms.box_hi[d]) pos[d] -= L;
            }
        }
    }

    return atoms;
}

void add_fixed_atoms(TestAtoms& atoms, const std::vector<int>& indices) {
    atoms.fixed_atoms = indices;
}

std::array<double, 3> minimum_image(
    const std::array<double, 3>& r1,
    const std::array<double, 3>& r2,
    const std::array<double, 3>& box_lo,
    const std::array<double, 3>& box_hi,
    const std::array<bool, 3>& pbc
) {
    std::array<double, 3> dr;

    for (int d = 0; d < 3; d++) {
        dr[d] = r2[d] - r1[d];

        if (pbc[d]) {
            double L = box_hi[d] - box_lo[d];
            if (dr[d] > 0.5 * L) dr[d] -= L;
            if (dr[d] < -0.5 * L) dr[d] += L;
        }
    }

    return dr;
}

double compute_distance(
    const std::array<double, 3>& r1,
    const std::array<double, 3>& r2,
    const std::array<double, 3>& box_lo,
    const std::array<double, 3>& box_hi,
    const std::array<bool, 3>& pbc
) {
    auto dr = minimum_image(r1, r2, box_lo, box_hi, pbc);
    return std::sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
}

void write_xyz(const TestAtoms& atoms, const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    out << atoms.natoms << "\n";
    out << "Lattice=\"" << atoms.box_hi[0] << " 0 0 0 "
        << atoms.box_hi[1] << " 0 0 0 " << atoms.box_hi[2] << "\" ";
    out << "Properties=species:S:1:pos:R:3\n";

    for (int i = 0; i < atoms.natoms; i++) {
        out << "Al " << atoms.positions[i][0] << " "
            << atoms.positions[i][1] << " "
            << atoms.positions[i][2] << "\n";
    }
}

TestAtoms read_xyz(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    TestAtoms atoms;

    // Read number of atoms
    in >> atoms.natoms;
    in.ignore(1000, '\n');

    // Read comment line (contains box info)
    std::string comment;
    std::getline(in, comment);

    // Parse positions
    atoms.positions.resize(atoms.natoms);
    atoms.types.resize(atoms.natoms, 1);

    for (int i = 0; i < atoms.natoms; i++) {
        std::string element;
        in >> element >> atoms.positions[i][0]
           >> atoms.positions[i][1] >> atoms.positions[i][2];
    }

    return atoms;
}

} // namespace TestFixtures

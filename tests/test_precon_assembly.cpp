/**
 * Unit tests for preconditioner matrix assembly
 *
 * Tests the PreconExp class:
 * 1. Matrix is symmetric positive definite
 * 2. Correct size (3N x 3N)
 * 3. Handles perturbed structures
 * 4. Handles fixed atoms correctly
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "test_atoms.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <fstream>

using Catch::Approx;
using namespace TestFixtures;

// Forward declarations (will be implemented in src/)
// These are placeholder signatures for the preconditioner class
namespace PreconLBFGS {

struct PreconExp {
    double r_cut;
    double r_NN;
    double mu;
    double A;
    double c_stab;

    PreconExp(double r_cut_ = -1.0, double mu_ = 1.0, double A_ = 3.0, double c_stab_ = 0.1)
        : r_cut(r_cut_), r_NN(-1.0), mu(mu_), A(A_), c_stab(c_stab_) {}

    // Build preconditioner matrix from test atoms
    // void make_precon(const TestAtoms& atoms);

    // Get dense matrix representation for testing
    // Eigen::MatrixXd get_matrix_dense() const;

    // Apply preconditioner: solve P*x = rhs
    // Eigen::VectorXd solve(const Eigen::VectorXd& rhs) const;
};

} // namespace PreconLBFGS

// Helper functions for testing

bool is_symmetric(const Eigen::MatrixXd& M, double tol = 1e-6) {
    if (M.rows() != M.cols()) return false;
    return (M - M.transpose()).cwiseAbs().maxCoeff() < tol;
}

bool is_positive_definite(const Eigen::MatrixXd& M, double tol = 1e-6) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
    if (es.info() != Eigen::Success) return false;

    // All eigenvalues should be positive
    return (es.eigenvalues().array() > tol).all();
}

Eigen::MatrixXd load_reference_matrix(const std::string& filename) {
    // TODO: Implement numpy .npy file loader
    // For now, return placeholder
    throw std::runtime_error("Reference data loading not yet implemented");
}

// ============================================================================
// TEST SUITE: Preconditioner Assembly
// ============================================================================

TEST_CASE("Preconditioner assembly - bulk Al", "[precon][assembly]") {
    auto atoms = create_bulk_Al_2x2x2();

    SECTION("Matrix dimensions correct") {
        // PreconExp precon(/*r_cut=*/4.0, /*mu=*/1.0, /*A=*/3.0);
        // precon.make_precon(atoms);
        // auto P = precon.get_matrix_dense();

        // int expected_size = 3 * atoms.natoms;  // 3N x 3N
        // REQUIRE(P.rows() == expected_size);
        // REQUIRE(P.cols() == expected_size);

        // Placeholder: test will be implemented when PreconExp is ready
        WARN("Test skipped: PreconExp not yet implemented");
        REQUIRE(atoms.natoms == 32);  // Verify test fixture works
    }

    SECTION("Matrix is symmetric") {
        // PreconExp precon(/*r_cut=*/4.0, /*mu=*/1.0, /*A=*/3.0);
        // precon.make_precon(atoms);
        // auto P = precon.get_matrix_dense();

        // REQUIRE(is_symmetric(P, 1e-6));

        WARN("Test skipped: PreconExp not yet implemented");
        REQUIRE(atoms.natoms == 32);
    }

    SECTION("Matrix is positive definite") {
        // PreconExp precon(/*r_cut=*/4.0, /*mu=*/1.0, /*A=*/3.0);
        // precon.make_precon(atoms);
        // auto P = precon.get_matrix_dense();

        // REQUIRE(is_positive_definite(P, 1e-6));

        WARN("Test skipped: PreconExp not yet implemented");
        REQUIRE(atoms.natoms == 32);
    }

    SECTION("Matrix matches ASE reference") {
        // PreconExp precon(/*r_cut=*/4.0, /*mu=*/1.0, /*A=*/3.0);
        // precon.make_precon(atoms);
        // auto P = precon.get_matrix_dense();

        // auto P_ref = load_reference_matrix("reference_data/al_bulk_precon_matrix.npy");

        // double max_diff = (P - P_ref).cwiseAbs().maxCoeff();
        // REQUIRE(max_diff < 1e-4);

        WARN("Test skipped: Reference data not yet generated");
        REQUIRE(atoms.natoms == 32);
    }
}

TEST_CASE("Preconditioner assembly - rattled structure", "[precon][assembly]") {
    auto atoms_perfect = create_bulk_Al_2x2x2();
    // Use smaller displacement to ensure max < 1.0 Å
    auto atoms_rattled = create_rattled(atoms_perfect, 0.05, 7);

    SECTION("Rattled structure still produces SPD matrix") {
        // PreconExp precon(/*r_cut=*/4.0, /*mu=*/1.0, /*A=*/3.0);
        // precon.make_precon(atoms_rattled);
        // auto P = precon.get_matrix_dense();

        // REQUIRE(is_symmetric(P, 1e-6));
        // REQUIRE(is_positive_definite(P, 1e-6));

        WARN("Test skipped: PreconExp not yet implemented");
        REQUIRE(atoms_rattled.natoms == 32);

        // Verify rattling worked (use minimum image convention for PBC)
        double max_displacement = 0.0;
        for (int i = 0; i < atoms_perfect.natoms; i++) {
            double dx = atoms_rattled.positions[i][0] - atoms_perfect.positions[i][0];
            double dy = atoms_rattled.positions[i][1] - atoms_perfect.positions[i][1];
            double dz = atoms_rattled.positions[i][2] - atoms_perfect.positions[i][2];

            // Apply minimum image convention for periodic boundaries
            for (int d = 0; d < 3; d++) {
                double delta = (d == 0) ? dx : (d == 1) ? dy : dz;
                if (atoms_rattled.pbc[d]) {
                    double L = atoms_rattled.box_hi[d] - atoms_rattled.box_lo[d];
                    if (delta > L/2) delta -= L;
                    if (delta < -L/2) delta += L;
                }
                if (d == 0) dx = delta;
                else if (d == 1) dy = delta;
                else dz = delta;
            }

            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            max_displacement = std::max(max_displacement, dist);
        }
        REQUIRE(max_displacement > 0.01);  // Atoms were actually displaced
        REQUIRE(max_displacement < 1.0);   // But not too much
    }
}

TEST_CASE("Preconditioner assembly - fixed atoms", "[precon][assembly][constraints]") {
    auto atoms = create_bulk_Al_2x2x2();
    add_fixed_atoms(atoms, {0, 1});  // Fix first two atoms

    SECTION("Fixed atom rows/columns are identity") {
        // PreconExp precon(/*r_cut=*/4.0, /*mu=*/1.0, /*A=*/3.0);
        // precon.make_precon(atoms);
        // auto P = precon.get_matrix_dense();

        // // Check that rows/cols for fixed atoms are identity
        // for (int fixed_idx : atoms.fixed_atoms) {
        //     for (int d = 0; d < 3; d++) {
        //         int idx = 3 * fixed_idx + d;
        //
        //         // Diagonal should be 1
        //         REQUIRE(P(idx, idx) == Approx(1.0));
        //
        //         // Off-diagonals in this row/col should be 0
        //         for (int j = 0; j < P.cols(); j++) {
        //             if (j != idx) {
        //                 REQUIRE(std::abs(P(idx, j)) < 1e-10);
        //                 REQUIRE(std::abs(P(j, idx)) < 1e-10);
        //             }
        //         }
        //     }
        // }

        WARN("Test skipped: PreconExp not yet implemented");
        REQUIRE(atoms.fixed_atoms.size() == 2);
    }

    SECTION("Matrix still SPD with fixed atoms") {
        // PreconExp precon(/*r_cut=*/4.0, /*mu=*/1.0, /*A=*/3.0);
        // precon.make_precon(atoms);
        // auto P = precon.get_matrix_dense();

        // REQUIRE(is_symmetric(P, 1e-6));
        // REQUIRE(is_positive_definite(P, 1e-6));

        WARN("Test skipped: PreconExp not yet implemented");
        REQUIRE(atoms.fixed_atoms.size() == 2);
    }
}

TEST_CASE("Apply preconditioner to forces", "[precon][apply]") {
    auto atoms = create_bulk_Cu_compressed(0.995);

    SECTION("Preconditioned forces have reasonable magnitude") {
        // Mock forces (would come from calculator)
        Eigen::VectorXd forces = Eigen::VectorXd::Random(3 * atoms.natoms);

        // PreconExp precon(/*r_cut=*/4.0, /*mu=*/1.0, /*A=*/3.0);
        // precon.make_precon(atoms);
        // Eigen::VectorXd precon_forces = precon.solve(forces);

        // // Preconditioned forces should exist and have same size
        // REQUIRE(precon_forces.size() == forces.size());

        // // For compressed structure, preconditioner should not amplify forces too much
        // double force_norm = forces.norm();
        // double precon_force_norm = precon_forces.norm();
        // REQUIRE(precon_force_norm < 10.0 * force_norm);  // Reasonable bound

        WARN("Test skipped: PreconExp not yet implemented");
        REQUIRE(forces.size() == 3 * atoms.natoms);
    }

    SECTION("Fixed atom forces remain zero") {
        add_fixed_atoms(atoms, {0, 5, 10});

        Eigen::VectorXd forces = Eigen::VectorXd::Random(3 * atoms.natoms);

        // Set forces on fixed atoms to zero
        for (int idx : atoms.fixed_atoms) {
            forces[3*idx] = 0.0;
            forces[3*idx + 1] = 0.0;
            forces[3*idx + 2] = 0.0;
        }

        // PreconExp precon(/*r_cut=*/4.0, /*mu=*/1.0, /*A=*/3.0);
        // precon.make_precon(atoms);
        // Eigen::VectorXd precon_forces = precon.solve(forces);

        // // Preconditioned forces on fixed atoms should still be zero
        // for (int idx : atoms.fixed_atoms) {
        //     REQUIRE(std::abs(precon_forces[3*idx]) < 1e-10);
        //     REQUIRE(std::abs(precon_forces[3*idx + 1]) < 1e-10);
        //     REQUIRE(std::abs(precon_forces[3*idx + 2]) < 1e-10);
        // }

        WARN("Test skipped: PreconExp not yet implemented");
        REQUIRE(atoms.fixed_atoms.size() == 3);
    }
}

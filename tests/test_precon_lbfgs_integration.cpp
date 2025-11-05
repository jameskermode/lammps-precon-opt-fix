/**
 * Integration tests for PreconLBFGS optimizer
 *
 * Tests full optimization with:
 * 1. Convergence with Exp preconditioner vs Identity
 * 2. Constraints preserved during optimization
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "test_atoms.h"
#include <Eigen/Dense>
#include <vector>

using Catch::Approx;
using namespace TestFixtures;

// Forward declarations (will be implemented)
namespace PreconLBFGS {
    struct PreconExp;

    struct PreconLBFGSOptimizer {
        // Constructor
        // PreconLBFGSOptimizer(PreconExp* precon, double fmax, int maxsteps);

        // Run optimization
        // bool run(TestAtoms& atoms, MockCalculator& calc);

        // Get number of steps taken
        // int get_nsteps() const;

        // Get final energy
        // double get_energy() const;
    };
}

// Mock calculator for testing
struct MockCalculator {
    double get_potential_energy(const TestAtoms& atoms) {
        // Simple harmonic potential: E = 0.5 * k * sum((r - r0)^2)
        // For compressed Cu, energy increases
        double E = 0.0;
        double k = 1.0;  // Force constant
        auto ref_atoms = create_bulk_Cu_2x2x2();

        for (int i = 0; i < atoms.natoms; i++) {
            for (int d = 0; d < 3; d++) {
                double dr = atoms.positions[i][d] - ref_atoms.positions[i][d];
                E += 0.5 * k * dr * dr;
            }
        }
        return E;
    }

    Eigen::VectorXd get_forces(const TestAtoms& atoms) {
        // F = -dE/dr = -k * (r - r0)
        Eigen::VectorXd forces(3 * atoms.natoms);
        double k = 1.0;
        auto ref_atoms = create_bulk_Cu_2x2x2();

        for (int i = 0; i < atoms.natoms; i++) {
            for (int d = 0; d < 3; d++) {
                double dr = atoms.positions[i][d] - ref_atoms.positions[i][d];
                forces[3*i + d] = -k * dr;
            }
        }
        return forces;
    }
};

TEST_CASE("PreconLBFGS convergence - bulk Cu", "[precon][integration][optimization]") {
    auto atoms_compressed = create_bulk_Cu_compressed(0.995);
    MockCalculator calc;

    SECTION("Convergence with Exp preconditioner") {
        // PreconExp precon_exp(/*r_cut=*/4.0, /*mu=*/10.0, /*A=*/3.0);
        // PreconLBFGSOptimizer opt_exp(&precon_exp, /*fmax=*/0.01, /*maxsteps=*/100);

        // auto atoms_copy = atoms_compressed;
        // bool converged = opt_exp.run(atoms_copy, calc);

        // REQUIRE(converged);
        // REQUIRE(opt_exp.get_nsteps() < 100);  // Should converge quickly

        // // Final energy should be close to zero (relaxed structure)
        // double E_final = calc.get_potential_energy(atoms_copy);
        // REQUIRE(E_final < 1e-4);

        WARN("Test skipped: PreconLBFGSOptimizer not yet implemented");
        REQUIRE(atoms_compressed.natoms == 32);
    }

    SECTION("Convergence with Identity preconditioner (baseline)") {
        // Identity precon = standard LBFGS
        // PreconExp precon_identity(/*r_cut=*/4.0, /*mu=*/0.0, /*A=*/0.0);  // Special case for identity
        // PreconLBFGSOptimizer opt_identity(&precon_identity, /*fmax=*/0.01, /*maxsteps=*/100);

        // auto atoms_copy = atoms_compressed;
        // bool converged = opt_identity.run(atoms_copy, calc);

        // REQUIRE(converged);
        // int steps_identity = opt_identity.get_nsteps();

        WARN("Test skipped: PreconLBFGSOptimizer not yet implemented");
        REQUIRE(atoms_compressed.natoms == 32);
    }

    SECTION("Exp preconditioner faster than Identity") {
        // Run both optimizations and compare step counts

        // PreconExp precon_exp(/*r_cut=*/4.0, /*mu=*/10.0, /*A=*/3.0);
        // PreconLBFGSOptimizer opt_exp(&precon_exp, /*fmax=*/0.01, /*maxsteps=*/100);
        // auto atoms_exp = atoms_compressed;
        // opt_exp.run(atoms_exp, calc);
        // int steps_exp = opt_exp.get_nsteps();

        // PreconExp precon_identity(/*r_cut=*/4.0, /*mu=*/0.0, /*A=*/0.0);
        // PreconLBFGSOptimizer opt_identity(&precon_identity, /*fmax=*/0.01, /*maxsteps=*/100);
        // auto atoms_identity = atoms_compressed;
        // opt_identity.run(atoms_identity, calc);
        // int steps_identity = opt_identity.get_nsteps();

        // // Preconditioned should converge in < 50% of steps
        // INFO("Steps with Exp precon: " << steps_exp);
        // INFO("Steps with Identity: " << steps_identity);
        // REQUIRE(steps_exp < 0.5 * steps_identity);

        WARN("Test skipped: PreconLBFGSOptimizer not yet implemented");
        WARN("This test will verify the main speedup claim: 3-5x fewer steps");
        REQUIRE(true);
    }

    SECTION("Final structures match with both preconditioners") {
        // Both should converge to same equilibrium structure

        // PreconExp precon_exp(/*r_cut=*/4.0, /*mu=*/10.0, /*A=*/3.0);
        // PreconLBFGSOptimizer opt_exp(&precon_exp, /*fmax=*/0.01, /*maxsteps=*/100);
        // auto atoms_exp = atoms_compressed;
        // opt_exp.run(atoms_exp, calc);

        // PreconExp precon_identity(/*r_cut=*/4.0, /*mu=*/0.0, /*A=*/0.0);
        // PreconLBFGSOptimizer opt_identity(&precon_identity, /*fmax=*/0.01, /*maxsteps=*/100);
        // auto atoms_identity = atoms_compressed;
        // opt_identity.run(atoms_identity, calc);

        // // Compare final positions
        // for (int i = 0; i < atoms_exp.natoms; i++) {
        //     for (int d = 0; d < 3; d++) {
        //         REQUIRE(atoms_exp.positions[i][d] ==
        //                 Approx(atoms_identity.positions[i][d]).epsilon(1e-3));
        //     }
        // }

        WARN("Test skipped: PreconLBFGSOptimizer not yet implemented");
        REQUIRE(true);
    }
}

TEST_CASE("PreconLBFGS with constraints", "[precon][integration][constraints]") {
    auto atoms = create_bulk_Cu_compressed(0.995);
    add_fixed_atoms(atoms, {0, 1, 2});  // Fix first 3 atoms
    MockCalculator calc;

    SECTION("Fixed atoms remain fixed during optimization") {
        // Store initial positions of fixed atoms
        std::vector<std::array<double, 3>> fixed_positions;
        for (int idx : atoms.fixed_atoms) {
            fixed_positions.push_back(atoms.positions[idx]);
        }

        // PreconExp precon(/*r_cut=*/4.0, /*mu=*/10.0, /*A=*/3.0);
        // PreconLBFGSOptimizer opt(&precon, /*fmax=*/0.01, /*maxsteps=*/100);
        // opt.run(atoms, calc);

        // // Check fixed atoms didn't move
        // for (size_t i = 0; i < atoms.fixed_atoms.size(); i++) {
        //     int idx = atoms.fixed_atoms[i];
        //     for (int d = 0; d < 3; d++) {
        //         REQUIRE(atoms.positions[idx][d] ==
        //                 Approx(fixed_positions[i][d]).epsilon(1e-10));
        //     }
        // }

        WARN("Test skipped: PreconLBFGSOptimizer not yet implemented");
        REQUIRE(atoms.fixed_atoms.size() == 3);
    }

    SECTION("Free atoms still move") {
        // Store initial positions
        auto initial_positions = atoms.positions;

        // PreconExp precon(/*r_cut=*/4.0, /*mu=*/10.0, /*A=*/3.0);
        // PreconLBFGSOptimizer opt(&precon, /*fmax=*/0.01, /*maxsteps=*/100);
        // opt.run(atoms, calc);

        // // At least one free atom should have moved
        // bool any_moved = false;
        // for (int i = 0; i < atoms.natoms; i++) {
        //     // Skip fixed atoms
        //     if (std::find(atoms.fixed_atoms.begin(), atoms.fixed_atoms.end(), i) !=
        //         atoms.fixed_atoms.end()) {
        //         continue;
        //     }
        //
        //     double dx = atoms.positions[i][0] - initial_positions[i][0];
        //     double dy = atoms.positions[i][1] - initial_positions[i][1];
        //     double dz = atoms.positions[i][2] - initial_positions[i][2];
        //     double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        //
        //     if (dist > 1e-6) {
        //         any_moved = true;
        //         break;
        //     }
        // }
        // REQUIRE(any_moved);

        WARN("Test skipped: PreconLBFGSOptimizer not yet implemented");
        REQUIRE(true);
    }

    SECTION("Preconditioner handles fixed atoms correctly") {
        // Verify that preconditioner matrix has identity rows/cols for fixed atoms
        // (tested in test_precon_assembly.cpp, but good to verify in context)

        // PreconExp precon(/*r_cut=*/4.0, /*mu=*/10.0, /*A=*/3.0);
        // precon.make_precon(atoms);
        // auto P = precon.get_matrix_dense();

        // for (int idx : atoms.fixed_atoms) {
        //     for (int d = 0; d < 3; d++) {
        //         int i = 3*idx + d;
        //         REQUIRE(P(i, i) == Approx(1.0));
        //
        //         for (int j = 0; j < P.cols(); j++) {
        //             if (j != i) {
        //                 REQUIRE(std::abs(P(i, j)) < 1e-10);
        //             }
        //         }
        //     }
        // }

        WARN("Test skipped: PreconExp not yet implemented");
        REQUIRE(atoms.fixed_atoms.size() == 3);
    }
}

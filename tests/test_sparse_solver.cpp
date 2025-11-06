/**
 * Unit tests for sparse solver utilities
 *
 * Tests Eigen3 sparse matrix wrapper and solver
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

using Catch::Approx;

// Helper functions for sparse solver testing

TEST_CASE("Sparse matrix assembly from triplets", "[sparse][eigen]") {

    SECTION("Simple 3x3 matrix") {
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.emplace_back(0, 0, 2.0);
        triplets.emplace_back(0, 1, -1.0);
        triplets.emplace_back(1, 0, -1.0);
        triplets.emplace_back(1, 1, 2.0);
        triplets.emplace_back(1, 2, -1.0);
        triplets.emplace_back(2, 1, -1.0);
        triplets.emplace_back(2, 2, 2.0);

        Eigen::SparseMatrix<double> A(3, 3);
        A.setFromTriplets(triplets.begin(), triplets.end());

        REQUIRE(A.nonZeros() == 7);
        REQUIRE(A.coeff(0, 0) == 2.0);
        REQUIRE(A.coeff(0, 1) == -1.0);
        REQUIRE(A.coeff(1, 1) == 2.0);
    }

    SECTION("Matrix is symmetric") {
        std::vector<Eigen::Triplet<double>> triplets;
        int n = 5;

        // Create symmetric tridiagonal matrix
        for (int i = 0; i < n; i++) {
            triplets.emplace_back(i, i, 2.0);
            if (i > 0) {
                triplets.emplace_back(i, i-1, -1.0);
                triplets.emplace_back(i-1, i, -1.0);
            }
        }

        Eigen::SparseMatrix<double> A(n, n);
        A.setFromTriplets(triplets.begin(), triplets.end());

        // Convert to dense for easier checking
        Eigen::MatrixXd A_dense = A;
        REQUIRE((A_dense - A_dense.transpose()).norm() < 1e-10);
    }
}

TEST_CASE("Sparse direct solver - SimplicialLDLT", "[sparse][solver]") {

    SECTION("Solve simple SPD system") {
        // Create SPD matrix: A = [2 -1; -1 2]
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.emplace_back(0, 0, 2.0);
        triplets.emplace_back(0, 1, -1.0);
        triplets.emplace_back(1, 0, -1.0);
        triplets.emplace_back(1, 1, 2.0);

        Eigen::SparseMatrix<double> A(2, 2);
        A.setFromTriplets(triplets.begin(), triplets.end());

        // Right-hand side
        Eigen::VectorXd b(2);
        b << 1.0, 0.0;

        // Solve A*x = b
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);

        REQUIRE(solver.info() == Eigen::Success);

        Eigen::VectorXd x = solver.solve(b);

        REQUIRE(solver.info() == Eigen::Success);

        // Check solution
        REQUIRE(x(0) == Approx(2.0/3.0).epsilon(1e-10));
        REQUIRE(x(1) == Approx(1.0/3.0).epsilon(1e-10));

        // Verify: A*x = b
        Eigen::VectorXd residual = A * x - b;
        REQUIRE(residual.norm() < 1e-10);
    }

    SECTION("Solve larger system") {
        int n = 10;
        std::vector<Eigen::Triplet<double>> triplets;

        // Symmetric tridiagonal
        for (int i = 0; i < n; i++) {
            triplets.emplace_back(i, i, 2.0);
            if (i > 0) {
                triplets.emplace_back(i, i-1, -1.0);
                triplets.emplace_back(i-1, i, -1.0);
            }
        }

        Eigen::SparseMatrix<double> A(n, n);
        A.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::VectorXd b = Eigen::VectorXd::Random(n);

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        Eigen::VectorXd x = solver.solve(b);

        // Verify solution
        Eigen::VectorXd residual = A * x - b;
        REQUIRE(residual.norm() < 1e-8);
    }
}

TEST_CASE("Sparse matrix operations", "[sparse][operations]") {

    SECTION("Matrix-vector multiplication") {
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.emplace_back(0, 0, 2.0);
        triplets.emplace_back(0, 1, -1.0);
        triplets.emplace_back(1, 0, -1.0);
        triplets.emplace_back(1, 1, 2.0);

        Eigen::SparseMatrix<double> A(2, 2);
        A.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::VectorXd x(2);
        x << 1.0, 2.0;

        Eigen::VectorXd y = A * x;

        REQUIRE(y(0) == Approx(0.0));  // 2*1 + (-1)*2 = 0
        REQUIRE(y(1) == Approx(3.0));  // (-1)*1 + 2*2 = 3
    }

    SECTION("Convert sparse to dense") {
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.emplace_back(0, 0, 1.0);
        triplets.emplace_back(1, 1, 2.0);
        triplets.emplace_back(2, 2, 3.0);

        Eigen::SparseMatrix<double> A_sparse(3, 3);
        A_sparse.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::MatrixXd A_dense = A_sparse;

        REQUIRE(A_dense(0, 0) == 1.0);
        REQUIRE(A_dense(1, 1) == 2.0);
        REQUIRE(A_dense(2, 2) == 3.0);
        REQUIRE(A_dense(0, 1) == 0.0);
    }
}

TEST_CASE("Preconditioner matrix properties", "[sparse][preconditioner]") {

    SECTION("Diagonal stabilization") {
        int n = 5;
        double mu = 10.0;
        double c_stab = 0.1;

        std::vector<Eigen::Triplet<double>> triplets;

        // Simple preconditioner-like matrix
        // For SPD: diagonal must dominate (Gershgorin circle theorem)
        for (int i = 0; i < n; i++) {
            // Diagonal: must be > sum of off-diagonals for SPD
            triplets.emplace_back(i, i, mu * (c_stab + 1.0));

            // Off-diagonal coupling (negative, small enough for SPD)
            if (i > 0) {
                triplets.emplace_back(i, i-1, -mu * 0.1);
            }
            if (i < n-1) {
                triplets.emplace_back(i, i+1, -mu * 0.1);
            }
        }

        Eigen::SparseMatrix<double> P(n, n);
        P.setFromTriplets(triplets.begin(), triplets.end());

        // Check positive definiteness
        Eigen::MatrixXd P_dense = P;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(P_dense);

        INFO("Minimum eigenvalue: " << es.eigenvalues().minCoeff());
        REQUIRE(es.eigenvalues().minCoeff() > 0);  // Positive definite
    }
}

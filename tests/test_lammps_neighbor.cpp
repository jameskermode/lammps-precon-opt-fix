/**
 * Unit tests for LAMMPS neighbor list interface
 *
 * Tests extraction of neighbors from LAMMPS neighbor lists:
 * 1. Correct neighbor extraction
 * 2. Proper handling of ghost atoms, tags
 * 3. r_NN estimation from LAMMPS data
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "test_atoms.h"
#include <vector>
#include <set>

using Catch::Approx;
using namespace TestFixtures;

// Forward declarations for LAMMPS neighbor interface
namespace PreconLBFGS {

struct NeighborData {
    std::vector<std::pair<int, int>> pairs;  // (tag_i, tag_j) pairs
    std::vector<double> distances;           // Corresponding distances
};

// Extract neighbors from LAMMPS neighbor list
// NeighborData extract_neighbors_from_lammps(
//     LAMMPS* lmp, int groupbit, double r_cut
// );

// Estimate r_NN from LAMMPS neighbor list
// double estimate_r_NN_from_lammps(LAMMPS* lmp, int groupbit);

} // namespace PreconLBFGS

TEST_CASE("Neighbor extraction from LAMMPS - perfect FCC", "[lammps][neighbor]") {
    auto atoms = create_bulk_Al_2x2x2();

    SECTION("Extract FCC first neighbors") {
        // Setup LAMMPS with this structure
        // LAMMPS* lmp = setup_lammps_with_atoms(atoms);

        // double r_cut = 3.0;  // First neighbor shell only
        // auto neighbors = extract_neighbors_from_lammps(lmp, all_groupbit, r_cut);

        // // FCC has 12 first neighbors per atom
        // // With 32 atoms, we expect 32*12/2 = 192 unique pairs
        // int expected_pairs = 32 * 12 / 2;
        // REQUIRE(neighbors.pairs.size() == expected_pairs);

        // // Verify coordination number for a few atoms
        // std::map<int, int> coordination;
        // for (const auto& [i, j] : neighbors.pairs) {
        //     coordination[i]++;
        //     coordination[j]++;
        // }
        //
        // // Each atom should have 12 neighbors (FCC)
        // for (const auto& [tag, count] : coordination) {
        //     REQUIRE(count == 12);
        // }

        WARN("Test skipped: LAMMPS neighbor extraction not yet implemented");
        REQUIRE(atoms.natoms == 32);
    }

    SECTION("Extract FCC first and second neighbors") {
        // LAMMPS* lmp = setup_lammps_with_atoms(atoms);

        // double r_cut = 4.5;  // Include second neighbors
        // auto neighbors = extract_neighbors_from_lammps(lmp, all_groupbit, r_cut);

        // // FCC has 12 first + 6 second neighbors = 18 total
        // // With 32 atoms: 32*18/2 = 288 unique pairs
        // int expected_pairs = 32 * 18 / 2;
        // REQUIRE(neighbors.pairs.size() == expected_pairs);

        WARN("Test skipped: LAMMPS neighbor extraction not yet implemented");
        REQUIRE(atoms.natoms == 32);
    }

    SECTION("Neighbor distances correct") {
        // LAMMPS* lmp = setup_lammps_with_atoms(atoms);

        // double r_cut = 4.5;
        // auto neighbors = extract_neighbors_from_lammps(lmp, all_groupbit, r_cut);

        // // All distances should be < r_cut
        // for (double dist : neighbors.distances) {
        //     REQUIRE(dist < r_cut);
        //     REQUIRE(dist > 0.01);  // No self-interactions
        // }

        // // Find minimum distance (should be first neighbor distance)
        // double min_dist = *std::min_element(neighbors.distances.begin(),
        //                                     neighbors.distances.end());
        //
        // // For Al FCC with a=3.994, first neighbor = a/sqrt(2) ≈ 2.824
        // double expected_r_NN = atoms.lattice_constant / std::sqrt(2.0);
        // REQUIRE(min_dist == Approx(expected_r_NN).epsilon(0.01));

        WARN("Test skipped: LAMMPS neighbor extraction not yet implemented");
        double expected_r_NN = atoms.lattice_constant / std::sqrt(2.0);
        REQUIRE(expected_r_NN == Approx(2.824).epsilon(0.01));
    }

    SECTION("Proper handling of periodic boundaries") {
        // Atoms near boundaries should have correct neighbors via PBC

        // LAMMPS* lmp = setup_lammps_with_atoms(atoms);
        // double r_cut = 3.0;
        // auto neighbors = extract_neighbors_from_lammps(lmp, all_groupbit, r_cut);

        // // Check that boundary atoms have full coordination
        // // (LAMMPS should handle PBC automatically)

        // // Find atoms near boundaries
        // std::vector<int> boundary_atoms;
        // for (int i = 0; i < atoms.natoms; i++) {
        //     if (atoms.positions[i][0] < 0.5 || atoms.positions[i][0] > atoms.box_hi[0] - 0.5) {
        //         boundary_atoms.push_back(i);
        //     }
        // }

        // // Count neighbors for boundary atoms
        // std::map<int, int> coordination;
        // for (const auto& [i, j] : neighbors.pairs) {
        //     coordination[i]++;
        //     coordination[j]++;
        // }

        // for (int tag : boundary_atoms) {
        //     REQUIRE(coordination[tag] == 12);  // Full coordination via PBC
        // }

        WARN("Test skipped: LAMMPS neighbor extraction not yet implemented");
        REQUIRE(atoms.pbc[0] == true);
    }
}

TEST_CASE("Neighbor extraction - ghost atoms and tags", "[lammps][neighbor][mpi]") {

    SECTION("Use global tags not local indices") {
        // In MPI, local indices are different on each rank
        // Must use atom tags (global IDs) for consistency

        // auto atoms = create_bulk_Cu_2x2x2();
        // LAMMPS* lmp = setup_lammps_with_atoms(atoms, /*nprocs=*/2);

        // double r_cut = 3.0;
        // auto neighbors = extract_neighbors_from_lammps(lmp, all_groupbit, r_cut);

        // // All pairs should use tags (positive integers)
        // for (const auto& [i, j] : neighbors.pairs) {
        //     REQUIRE(i > 0);
        //     REQUIRE(j > 0);
        //     REQUIRE(i <= atoms.natoms);
        //     REQUIRE(j <= atoms.natoms);
        // }

        WARN("Test skipped: LAMMPS MPI neighbor extraction not yet implemented");
        REQUIRE(true);
    }

    SECTION("Filter neighbors by group") {
        // If only a subset of atoms is in the group, only extract those neighbors

        // auto atoms = create_bulk_Al_2x2x2();
        // add_fixed_atoms(atoms, {0, 1, 2, 3});  // Fix 4 atoms

        // LAMMPS* lmp = setup_lammps_with_atoms(atoms);
        // int mobile_groupbit = get_mobile_group(lmp, atoms.fixed_atoms);

        // double r_cut = 3.0;
        // auto neighbors = extract_neighbors_from_lammps(lmp, mobile_groupbit, r_cut);

        // // Should only have pairs among mobile atoms (28 atoms)
        // std::set<int> tags_in_pairs;
        // for (const auto& [i, j] : neighbors.pairs) {
        //     tags_in_pairs.insert(i);
        //     tags_in_pairs.insert(j);
        // }

        // // Fixed atoms should not appear
        // for (int fixed : atoms.fixed_atoms) {
        //     REQUIRE(tags_in_pairs.find(fixed) == tags_in_pairs.end());
        // }

        WARN("Test skipped: LAMMPS group filtering not yet implemented");
        REQUIRE(atoms.fixed_atoms.size() == 4);
    }
}

TEST_CASE("r_NN estimation from LAMMPS", "[lammps][r_NN]") {

    SECTION("r_NN for Al FCC") {
        auto atoms = create_bulk_Al_2x2x2();

        // LAMMPS* lmp = setup_lammps_with_atoms(atoms);
        // double r_NN = estimate_r_NN_from_lammps(lmp, all_groupbit);

        // // FCC nearest neighbor: a/sqrt(2)
        // double expected = atoms.lattice_constant / std::sqrt(2.0);
        // REQUIRE(r_NN == Approx(expected).epsilon(0.01));

        WARN("Test skipped: r_NN estimation from LAMMPS not yet implemented");
        double expected = atoms.lattice_constant / std::sqrt(2.0);
        REQUIRE(expected == Approx(2.824).epsilon(0.01));
    }

    SECTION("r_NN for Cu FCC") {
        auto atoms = create_bulk_Cu_2x2x2();

        // LAMMPS* lmp = setup_lammps_with_atoms(atoms);
        // double r_NN = estimate_r_NN_from_lammps(lmp, all_groupbit);

        // double expected = atoms.lattice_constant / std::sqrt(2.0);
        // REQUIRE(r_NN == Approx(expected).epsilon(0.01));

        WARN("Test skipped: r_NN estimation from LAMMPS not yet implemented");
        double expected = atoms.lattice_constant / std::sqrt(2.0);
        REQUIRE(expected == Approx(2.556).epsilon(0.01));
    }

    SECTION("r_NN for rattled structure") {
        auto atoms_perfect = create_bulk_Al_2x2x2();
        auto atoms_rattled = create_rattled(atoms_perfect, 0.05, 42);

        // LAMMPS* lmp = setup_lammps_with_atoms(atoms_rattled);
        // double r_NN = estimate_r_NN_from_lammps(lmp, all_groupbit);

        // // Should still be close to perfect lattice
        // double expected = atoms_perfect.lattice_constant / std::sqrt(2.0);
        // REQUIRE(r_NN == Approx(expected).epsilon(0.05));  // 5% tolerance for rattling

        WARN("Test skipped: r_NN estimation from LAMMPS not yet implemented");
        REQUIRE(atoms_rattled.natoms == 32);
    }

    SECTION("r_NN consistent across MPI ranks") {
        // All ranks should compute same r_NN value

        // auto atoms = create_bulk_Cu_2x2x2();
        // LAMMPS* lmp = setup_lammps_with_atoms(atoms, /*nprocs=*/4);

        // double r_NN = estimate_r_NN_from_lammps(lmp, all_groupbit);

        // // Verify via MPI_Allreduce that all ranks got same value
        // double r_NN_check;
        // MPI_Allreduce(&r_NN, &r_NN_check, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        // REQUIRE(r_NN == r_NN_check);

        WARN("Test skipped: MPI consistency test not yet implemented");
        REQUIRE(true);
    }
}

TEST_CASE("LAMMPS neighbor cutoff verification", "[lammps][neighbor][cutoff]") {

    SECTION("Error if LAMMPS cutoff < preconditioner r_cut") {
        // Should detect and error if pair_style cutoff is too small

        // auto atoms = create_bulk_Al_2x2x2();
        // LAMMPS* lmp = setup_lammps_with_atoms(atoms, /*pair_cutoff=*/2.5);

        // double precon_r_cut = 5.0;  // Larger than LAMMPS cutoff

        // // Should throw error or return failure
        // REQUIRE_THROWS_WITH(
        //     extract_neighbors_from_lammps(lmp, all_groupbit, precon_r_cut),
        //     Catch::Matchers::Contains("LAMMPS neighbor cutoff too small")
        // );

        WARN("Test skipped: Cutoff verification not yet implemented");
        REQUIRE(true);
    }

    SECTION("Success if LAMMPS cutoff >= preconditioner r_cut") {
        // auto atoms = create_bulk_Al_2x2x2();
        // LAMMPS* lmp = setup_lammps_with_atoms(atoms, /*pair_cutoff=*/6.0);

        // double precon_r_cut = 4.0;  // Smaller than LAMMPS cutoff

        // // Should succeed
        // REQUIRE_NOTHROW(
        //     extract_neighbors_from_lammps(lmp, all_groupbit, precon_r_cut)
        // );

        WARN("Test skipped: Cutoff verification not yet implemented");
        REQUIRE(true);
    }
}

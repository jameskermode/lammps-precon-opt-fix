cmake_minimum_required(VERSION 3.16)

# Download and setup Catch2
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.0
)
FetchContent_MakeAvailable(Catch2)

# Find LAMMPS for tests
find_package(LAMMPS REQUIRED)
find_package(MPI REQUIRED)

# Test executable
add_executable(precon_tests
    main.cpp
    fixtures/test_atoms.cpp
    test_precon_assembly.cpp
    test_mu_estimate.cpp
    test_precon_lbfgs_integration.cpp
    test_lammps_neighbor.cpp
    test_sparse_solver.cpp
    test_precon_matrix_direct.cpp
    # Include implementation files needed by tests
    ../src/precon_exp.cpp
)

target_include_directories(precon_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${LAMMPS_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS}
    fixtures
)

target_link_libraries(precon_tests PRIVATE
    Catch2::Catch2WithMain
    Eigen3::Eigen
    ${MPI_CXX_LIBRARIES}
    /home/eng/essswb/lammps/lammps-22Jul2025/build/liblammps.a
    gomp
    jpeg
    png
    z
)

# Add LAMMPS include directories
target_compile_definitions(precon_tests PRIVATE
    LAMMPS_LIB_MPI
)

# Register tests with CTest
include(CTest)
include(Catch)
catch_discover_tests(precon_tests)

# Helper to run tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS precon_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

cmake_minimum_required(VERSION 3.16)
project(lammps_precon_lbfgs VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_PLUGIN "Build LAMMPS plugin" ON)

# Add cmake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(MPI REQUIRED)
if(BUILD_PLUGIN)
    find_package(LAMMPS REQUIRED)
endif()

# Plugin source files
if(BUILD_PLUGIN)
    add_library(preconlbfgsplugin SHARED
        src/fix_precon_lbfgs.cpp
        src/precon_exp.cpp
        src/plugin.cpp
    )

    target_include_directories(preconlbfgsplugin PRIVATE
        ${LAMMPS_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${MPI_CXX_INCLUDE_DIRS}
    )

    target_link_libraries(preconlbfgsplugin PRIVATE
        ${LAMMPS_LIBRARIES}
        Eigen3::Eigen
        ${MPI_CXX_LIBRARIES}
    )

    set_target_properties(preconlbfgsplugin PROPERTIES
        PREFIX ""
        OUTPUT_NAME "preconlbfgsplugin"
    )
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
if(BUILD_PLUGIN)
    install(TARGETS preconlbfgsplugin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()

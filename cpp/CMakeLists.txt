# Stage 8 — C++ LAMMPS-native Exp preconditioner plugin.
#
# Builds liblammps_precon.so, a runtime plugin (loaded via `plugin load`)
# providing `fix precon/exp` and `min_style precon/lbfgs`. A plugin links no
# LAMMPS library — its undefined symbols resolve against the LAMMPS process
# at load time — so only the LAMMPS *headers* (+ MPI + Eigen) are needed.
cmake_minimum_required(VERSION 3.20)
project(lammps_precon VERSION 0.1 LANGUAGES CXX)

# C++20 — must match the Symmetrix LAMMPS build, whose headers then resolve
# `fmt::` to std::format (a C++17 plugin would use the bundled fmt and hit an
# ABI mismatch on Error::_all etc.).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- LAMMPS headers (the Symmetrix build) -----------------------------------
get_filename_component(_default_lammps
  "${CMAKE_SOURCE_DIR}/../lammps-symmetrix/lammps/src" ABSOLUTE)
set(LAMMPS_SOURCE_DIR "${_default_lammps}" CACHE PATH "LAMMPS src/ directory")
if(NOT EXISTS "${LAMMPS_SOURCE_DIR}/lammps.h")
  message(FATAL_ERROR "LAMMPS headers not found at ${LAMMPS_SOURCE_DIR} "
                      "(set -D LAMMPS_SOURCE_DIR=...)")
endif()

# --- dependencies -----------------------------------------------------------
# MPI only — the preconditioner is dependency-light: pair-list assembly + a
# hand-written distributed Jacobi-CG (no external linear-algebra library).
set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI REQUIRED COMPONENTS CXX)

# LAMMPS C++ interface — headers only. LAMMPS_SMALLBIG must match the LAMMPS
# build's integer-size setting (the default).
add_library(lammps_iface INTERFACE)
target_include_directories(lammps_iface INTERFACE "${LAMMPS_SOURCE_DIR}")
target_compile_definitions(lammps_iface INTERFACE -DLAMMPS_SMALLBIG)
target_link_libraries(lammps_iface INTERFACE MPI::MPI_CXX)

# --- the plugin -------------------------------------------------------------
add_library(lammps_precon MODULE
  src/fix_precon_exp.cpp
  src/min_precon_lbfgs.cpp
  src/plugin.cpp)
target_link_libraries(lammps_precon PRIVATE lammps_iface)
set_target_properties(lammps_precon PROPERTIES
  PREFIX "lib" SUFFIX ".so" LINK_FLAGS "-rdynamic")

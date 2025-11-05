# FindLAMMPS.cmake - Find LAMMPS installation
#
# This module defines:
#   LAMMPS_FOUND - True if LAMMPS is found
#   LAMMPS_INCLUDE_DIRS - Include directories for LAMMPS
#   LAMMPS_LIBRARIES - Libraries to link against
#   LAMMPS_VERSION - LAMMPS version

# Look for LAMMPS in common locations
find_path(LAMMPS_INCLUDE_DIR
    NAMES lammps.h
    PATHS
        ${LAMMPS_SOURCE_DIR}
        $ENV{LAMMPS_SOURCE_DIR}
        $ENV{LAMMPS_DIR}/src
        ${LAMMPS_DIR}/src
        /usr/local/include/lammps
        /usr/include/lammps
    PATH_SUFFIXES LAMMPS
)

find_library(LAMMPS_LIBRARY
    NAMES lammps liblammps
    PATHS
        # Build directories (CI and local builds)
        ${LAMMPS_INCLUDE_DIR}/../build
        ${LAMMPS_SOURCE_DIR}/../build
        $ENV{LAMMPS_SOURCE_DIR}/../build
        $ENV{LAMMPS_DIR}/build
        ${LAMMPS_DIR}/build
        # Source directories (for plugins that link directly)
        $ENV{LAMMPS_DIR}/src
        ${LAMMPS_DIR}/src
        # System installation paths
        /usr/local/lib
        /usr/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAMMPS
    REQUIRED_VARS LAMMPS_INCLUDE_DIR
)

if(LAMMPS_FOUND)
    set(LAMMPS_INCLUDE_DIRS "${LAMMPS_INCLUDE_DIR}")
    if(LAMMPS_LIBRARY)
        set(LAMMPS_LIBRARIES "${LAMMPS_LIBRARY}")
        message(STATUS "Found LAMMPS library: ${LAMMPS_LIBRARY}")
    else()
        # Try to construct library path from source/build directory
        if(LAMMPS_SOURCE_DIR)
            get_filename_component(LAMMPS_BUILD_DIR "${LAMMPS_SOURCE_DIR}/../build" ABSOLUTE)
            if(EXISTS "${LAMMPS_BUILD_DIR}/liblammps.so")
                set(LAMMPS_LIBRARIES "${LAMMPS_BUILD_DIR}/liblammps.so")
                message(STATUS "Using LAMMPS library from build: ${LAMMPS_LIBRARIES}")
            elseif(EXISTS "${LAMMPS_BUILD_DIR}/liblammps.a")
                set(LAMMPS_LIBRARIES "${LAMMPS_BUILD_DIR}/liblammps.a")
                message(STATUS "Using LAMMPS library from build: ${LAMMPS_LIBRARIES}")
            else()
                message(STATUS "LAMMPS library not found (OK for plugin-only builds)")
            endif()
        else()
            message(STATUS "LAMMPS library not found (OK for plugin-only builds)")
        endif()
    endif()
endif()

mark_as_advanced(LAMMPS_INCLUDE_DIR LAMMPS_LIBRARY)

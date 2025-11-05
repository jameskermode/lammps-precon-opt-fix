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
        $ENV{LAMMPS_DIR}/src
        ${LAMMPS_DIR}/src
        ~/lammps/lammps-22Jul2025/src
        /usr/local/include/lammps
        /usr/include/lammps
    PATH_SUFFIXES LAMMPS
)

find_library(LAMMPS_LIBRARY
    NAMES lammps liblammps
    PATHS
        $ENV{LAMMPS_DIR}/src
        ${LAMMPS_DIR}/src
        ~/lammps/lammps-22Jul2025/src
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
    endif()
endif()

mark_as_advanced(LAMMPS_INCLUDE_DIR LAMMPS_LIBRARY)

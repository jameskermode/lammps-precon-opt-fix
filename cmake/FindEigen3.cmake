# FindEigen3.cmake - Find Eigen3 library
#
# This module defines:
#   EIGEN3_FOUND - True if Eigen3 is found
#   EIGEN3_INCLUDE_DIRS - Include directories for Eigen3
#   Eigen3::Eigen - Imported target for Eigen3

find_path(EIGEN3_INCLUDE_DIR
    NAMES Eigen/Core
    PATHS
        /usr/include/eigen3
        /usr/local/include/eigen3
        /opt/eigen3/include
        $ENV{EIGEN3_ROOT}/include
        ${EIGEN3_ROOT}/include
    PATH_SUFFIXES eigen3
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen3
    REQUIRED_VARS EIGEN3_INCLUDE_DIR
)

if(EIGEN3_FOUND AND NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
    )
    set(EIGEN3_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")
endif()

mark_as_advanced(EIGEN3_INCLUDE_DIR)

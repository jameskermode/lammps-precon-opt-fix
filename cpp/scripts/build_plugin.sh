#!/usr/bin/env bash
# Build the Stage-8 C++ Exp-preconditioner plugin (liblammps_precon.so).
#
#   bash cpp/scripts/build_plugin.sh          # configure + build
#   bash cpp/scripts/build_plugin.sh clean    # remove the build directory
#
# Built with foss/2023b (GCC 13.2.0) to ABI-match the Symmetrix LAMMPS build.
set -euo pipefail

CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$CPP_DIR/build"

if [[ "${1:-}" == "clean" ]]; then
    rm -rf "$BUILD_DIR"
    echo "Removed $BUILD_DIR"
    exit 0
fi

# Ensure the module system is available, then load the toolchain.
if ! type module &>/dev/null; then
    for f in /etc/profile.d/*lmod*.sh /etc/profile.d/*modules*.sh; do
        [[ -f "$f" ]] && source "$f" && break
    done
fi
module purge
module load foss/2023b CMake/3.27.6

echo "Compiler: $(g++ --version | head -1)"
echo "CMake:    $(cmake --version | head -1)"

cmake -B "$BUILD_DIR" -S "$CPP_DIR" -D CMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j "$(nproc --all 2>/dev/null || echo 4)"

echo ""
echo "Built: $BUILD_DIR/liblammps_precon.so"

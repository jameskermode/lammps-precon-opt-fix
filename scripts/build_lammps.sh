#!/usr/bin/env bash
# Build LAMMPS + Symmetrix (MACE) with the PLUGIN and MANYBODY (EAM) packages
# and set up the uv Python environment.
#
# This is an HPC *convenience* script: it assumes the Lmod module system (see
# the "Toolchain" section below). On a machine without Lmod, follow the generic
# build in README.md instead. CPU-only by default — the test systems are small
# geometry-optimisation cells; set FORCE_CPU=0 to auto-detect a GPU.
#
# Usage:
#   bash scripts/build_lammps.sh              # build + set up the Python env
#   bash scripts/build_lammps.sh clean        # remove the local build directory
#   FORCE_CPU=0 bash scripts/build_lammps.sh  # auto-detect and use a GPU
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Module versions (foss/2023b uses GCC 13.2.0, which CUDA 12.9.1 supports;
# GCC 13.3.0 from foss/2024a has AMX header issues with nvcc)
FOSS_MODULE="foss/2023b"
CMAKE_MODULE="CMake/3.27.6"
CUDA_MODULE="CUDA/12.9.1"

# ── Toolchain (HPC / Lmod — local setup) ─────────────────────────────────────
# This convenience script targets an HPC environment with the Lmod module
# system; the `module load` calls below provide GCC 13.2, OpenMPI and CMake.
# On a machine without Lmod, remove this section and the `module` calls and
# provide a C++20 compiler, MPI and CMake >= 3.20 by your own means — the
# generic build in README.md does exactly that.
if ! type module &>/dev/null; then
    for f in /etc/profile.d/*lmod*.sh /etc/profile.d/*modules*.sh; do
        if [[ -f "$f" ]]; then
            source "$f"
            break
        fi
    done
fi

if ! type module &>/dev/null; then
    echo "ERROR: Lmod 'module' command not found. This is an HPC convenience"
    echo "       script — for a generic build follow the steps in README.md."
    exit 1
fi

# ── GPU detection (CPU-only by default for this project) ────────────────────
GPU_DETECTED=false
KOKKOS_GPU_ARCH=""

if [[ "${FORCE_CPU:-1}" == "1" ]]; then
    echo "FORCE_CPU=1 — building CPU-only (set FORCE_CPU=0 to enable GPU)."
elif command -v nvidia-smi &>/dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
    if [[ -n "$COMPUTE_CAP" ]]; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
        case "$COMPUTE_CAP" in
            7.0) KOKKOS_GPU_ARCH="VOLTA70" ;;
            7.5) KOKKOS_GPU_ARCH="TURING75" ;;
            8.0) KOKKOS_GPU_ARCH="AMPERE80" ;;
            8.6) KOKKOS_GPU_ARCH="AMPERE86" ;;
            8.9) KOKKOS_GPU_ARCH="ADA89" ;;
            9.0) KOKKOS_GPU_ARCH="HOPPER90" ;;
            *)
                echo "WARNING: Unknown compute capability $COMPUTE_CAP — building CPU-only."
                COMPUTE_CAP=""
                ;;
        esac
        if [[ -n "$KOKKOS_GPU_ARCH" ]]; then
            GPU_DETECTED=true
            echo "Detected GPU: $GPU_NAME (compute $COMPUTE_CAP → Kokkos $KOKKOS_GPU_ARCH)"
        fi
    fi
fi

if [[ "$GPU_DETECTED" == false ]]; then
    echo "Building CPU-only."
fi

# ── Auto-detect CPU architecture for Kokkos ─────────────────────────────────
KOKKOS_CPU_ARCH=""
if lscpu 2>/dev/null | grep -q "Flags.*avx512"; then
    KOKKOS_CPU_ARCH="SKX"
elif lscpu 2>/dev/null | grep -q "Flags.*avx2"; then
    KOKKOS_CPU_ARCH="HSW"
fi
if [[ -n "$KOKKOS_CPU_ARCH" ]]; then
    echo "Detected CPU arch: Kokkos $KOKKOS_CPU_ARCH"
else
    echo "WARNING: Could not detect CPU arch, using Kokkos_ARCH_NATIVE=ON"
fi

# ── Arch-specific paths (so a shared filesystem works across mixed nodes) ────
ARCH_KEY="${KOKKOS_CPU_ARCH:-NATIVE}-${KOKKOS_GPU_ARCH:-none}"
VENV_DIR="$REPO_DIR/.venv-${ARCH_KEY}"
BASE_DIR="$REPO_DIR/lammps-symmetrix"
LAMMPS_DIR="$BASE_DIR/lammps"
SYMMETRIX_DIR="$BASE_DIR/symmetrix"
BUILD_DIR="$LAMMPS_DIR/build-${ARCH_KEY}"

# ── Handle arguments ─────────────────────────────────────────────────────────
if [[ "${1:-}" == "clean" ]]; then
    echo "Removing build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
    echo "Done. Re-run without 'clean' to rebuild."
    exit 0
fi

# ── Load modules ─────────────────────────────────────────────────────────────
echo ""
echo "=== Loading modules ==="
module purge
module load "$FOSS_MODULE"
module load "$CMAKE_MODULE"

if [[ "$GPU_DETECTED" == true ]]; then
    module load "$CUDA_MODULE"
fi

echo "Compiler: $(gcc --version | head -1)"
echo "MPI:      $(mpirun --version 2>&1 | head -1)"
echo "CMake:    $(cmake --version | head -1)"

# ── Ensure source trees are available ────────────────────────────────────────
mkdir -p "$BASE_DIR"
if [[ ! -d "$SYMMETRIX_DIR" ]]; then
    echo "Cloning symmetrix (recursive)..."
    git clone --recursive https://github.com/wcwitt/symmetrix "$SYMMETRIX_DIR"
fi

echo ""
echo "=== Downloading sources ==="
if [[ ! -f "$LAMMPS_DIR/CMakeLists.txt" ]]; then
    echo "Cloning LAMMPS (release branch)..."
    git clone -b release https://github.com/lammps/lammps "$LAMMPS_DIR"
else
    echo "LAMMPS directory already exists, skipping clone."
fi

# ── Patch LAMMPS with symmetrix ──────────────────────────────────────────────
echo ""
echo "=== Patching LAMMPS with symmetrix ==="
if grep -q "libsymmetrix" "$LAMMPS_DIR/cmake/CMakeLists.txt" 2>/dev/null; then
    echo "Symmetrix patch already applied, skipping."
else
    cd "$SYMMETRIX_DIR/pair_symmetrix"
    ./install.sh "$LAMMPS_DIR"
    cd "$REPO_DIR"
fi

# ── CMake configure ──────────────────────────────────────────────────────────
echo ""
echo "=== Configuring LAMMPS ==="

CMAKE_FLAGS=(
    -D CMAKE_BUILD_TYPE=Release
    -D CMAKE_CXX_STANDARD=20
    -D CMAKE_CXX_STANDARD_REQUIRED=ON
    -D BUILD_SHARED_LIBS=ON
    -D BUILD_OMP=ON
    -D BUILD_MPI=ON
    -D PKG_MANYBODY=ON
    -D PKG_KOKKOS=ON
    -D PKG_PYTHON=ON
    -D PKG_PLUGIN=ON
    -D Kokkos_ENABLE_SERIAL=ON
    -D Kokkos_ENABLE_OPENMP=ON
    -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON
    -D SYMMETRIX_KOKKOS=ON
)

if [[ -n "$KOKKOS_CPU_ARCH" ]]; then
    CMAKE_FLAGS+=(-D "Kokkos_ARCH_${KOKKOS_CPU_ARCH}=ON")
else
    CMAKE_FLAGS+=(-D Kokkos_ARCH_NATIVE=ON)
fi

if [[ "$GPU_DETECTED" == true ]]; then
    CMAKE_FLAGS+=(
        -D CMAKE_CXX_COMPILER="$LAMMPS_DIR/lib/kokkos/bin/nvcc_wrapper"
        -D Kokkos_ENABLE_CUDA=ON
        -D "Kokkos_ARCH_${KOKKOS_GPU_ARCH}=ON"
        -D SYMMETRIX_SPHERICART_CUDA=ON
    )
fi

cmake -B "$BUILD_DIR" "${CMAKE_FLAGS[@]}" "$LAMMPS_DIR/cmake"

# ── Build ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Building LAMMPS ==="
NPROCS="${NPROCS:-$(nproc --all 2>/dev/null || nproc 2>/dev/null || echo 8)}"
echo "Building with $NPROCS parallel jobs..."
cmake --build "$BUILD_DIR" -j "$NPROCS"

# ── Set up Python environment ────────────────────────────────────────────────
echo ""
echo "=== Setting up Python environment ($ARCH_KEY) ==="

# Install uv if not available
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$REPO_DIR"

# Use arch-specific venv; set CC/CXX for packages needing C compilation
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
export CC=gcc CXX=g++

if [[ "$GPU_DETECTED" == true ]]; then
    echo "Installing Python packages (with CUDA extras)..."
    uv sync --extra cuda
else
    echo "Installing Python packages..."
    uv sync
fi

# Install LAMMPS Python interface
echo "Installing LAMMPS Python interface..."
LAMMPS_VERSION_FILE="$LAMMPS_DIR/src/version.h" uv pip install --python "$VENV_DIR/bin/python" "$LAMMPS_DIR/python"

# Build and install the symmetrix Python package
PY_TAG=$("$VENV_DIR/bin/python" -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
echo "Building symmetrix wheel for $PY_TAG (this takes ~2 min)..."
WHL_DIR="$BUILD_DIR/symmetrix-wheel"
mkdir -p "$WHL_DIR"
CC=gcc CXX=g++ uv build --wheel "$SYMMETRIX_DIR/symmetrix" --out-dir "$WHL_DIR"
SYMMETRIX_WHL=$(find "$WHL_DIR" -name "symmetrix-*-${PY_TAG}-*.whl" | head -1)
uv pip install --python "$VENV_DIR/bin/python" "$SYMMETRIX_WHL"

# Install lmp binary and shared libs into the venv
echo "Installing lmp binary and shared libs into venv..."
mkdir -p "$VENV_DIR/lib"
find "$BUILD_DIR" -maxdepth 1 -name "*.so*" -exec cp -P {} "$VENV_DIR/lib/" \;

# Bundle GCC 13.2 libstdc++/libgcc so the venv runs without module load at runtime.
GCC_LIBSTDCXX=$(g++ -print-file-name=libstdc++.so.6 2>/dev/null)
if [[ -n "$GCC_LIBSTDCXX" && -f "$GCC_LIBSTDCXX" ]]; then
    cp -P "$GCC_LIBSTDCXX"* "$VENV_DIR/lib/" 2>/dev/null || true
    GCC_LIBGCC=$(g++ -print-file-name=libgcc_s.so.1 2>/dev/null)
    [[ -f "$GCC_LIBGCC" ]] && cp -P "$GCC_LIBGCC"* "$VENV_DIR/lib/" 2>/dev/null || true
    GCC_LIBGOMP=$(g++ -print-file-name=libgomp.so.1 2>/dev/null)
    [[ -f "$GCC_LIBGOMP" ]] && cp -P "$GCC_LIBGOMP"* "$VENV_DIR/lib/" 2>/dev/null || true
    echo "Bundled libstdc++/libgcc/libgomp from $(dirname "$GCC_LIBSTDCXX")"
fi

# Create lmp wrapper that sets LD_LIBRARY_PATH
cp "$BUILD_DIR/lmp" "$VENV_DIR/lib/lmp.real"
cat > "$VENV_DIR/bin/lmp" <<'WRAPPER'
#!/usr/bin/env bash
VENV="$(cd "$(dirname "$0")/.." && pwd)"
export LD_LIBRARY_PATH="$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$VENV/lib/lmp.real" "$@"
WRAPPER
chmod +x "$VENV_DIR/bin/lmp"

# Patch venv activate scripts to set LD_LIBRARY_PATH (bash + fish)
ACTIVATE="$VENV_DIR/bin/activate"
if ! grep -q "lammps_precon LD_LIBRARY_PATH" "$ACTIVATE" 2>/dev/null; then
    cat >> "$ACTIVATE" <<'ACTIVATE_PATCH'

# lammps_precon LD_LIBRARY_PATH — make liblammps.so / libsymmetrix.so findable
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ACTIVATE_PATCH
fi
ACTIVATE_FISH="$VENV_DIR/bin/activate.fish"
if [[ -f "$ACTIVATE_FISH" ]] && ! grep -q "lammps_precon LD_LIBRARY_PATH" "$ACTIVATE_FISH" 2>/dev/null; then
    cat >> "$ACTIVATE_FISH" <<'ACTIVATE_FISH_PATCH'

# lammps_precon LD_LIBRARY_PATH — make liblammps.so / libsymmetrix.so findable
set -gx LD_LIBRARY_PATH "$VIRTUAL_ENV/lib" $LD_LIBRARY_PATH
ACTIVATE_FISH_PATCH
fi

# Convenience symlink: .venv -> arch-specific venv
ln -sfn "$(basename "$VENV_DIR")" "$REPO_DIR/.venv"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Build complete ($ARCH_KEY) ==="
echo ""
echo "GPU support: $([[ "$GPU_DETECTED" == true ]] && echo "YES ($KOKKOS_GPU_ARCH)" || echo "NO (CPU-only)")"
echo ""
echo "Usage:"
echo "  module load $FOSS_MODULE"
echo "  source .venv/bin/activate"
echo "  lmp -h"
echo "  python -c 'import lammps; print(lammps.lammps().version)'"

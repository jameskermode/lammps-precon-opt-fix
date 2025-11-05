# CI/CD Setup Documentation

## Overview

This project uses GitHub Actions for continuous integration. The CI workflow automatically:

1. **Builds LAMMPS** (with caching to speed up subsequent runs)
2. **Compiles the PreconLBFGS plugin**
3. **Runs unit tests** (C++ Catch2 tests)
4. **Runs integration tests** (LAMMPS input scripts)
5. **Runs MPI tests** (2 and 4 ranks)

## Workflow File

Location: `.github/workflows/ci.yml`

## Caching Strategy

### LAMMPS Build Cache

The LAMMPS build is cached to avoid expensive recompilation on every CI run.

**Cache Key**: `${{ runner.os }}-lammps-${{ env.LAMMPS_VERSION }}-precon`

Where `LAMMPS_VERSION` is the commit hash of the LAMMPS stable branch. This means:
- Cache is automatically invalidated when LAMMPS stable is updated
- Same OS and LAMMPS version will reuse cached build
- Typical cache time: ~10 minutes for LAMMPS build vs ~1 minute when cached

**Cache Contents**:
- `lammps/build/` - Compiled LAMMPS binaries
- `lammps/src/` - LAMMPS source headers (needed for plugin compilation)

## Test Stages

### 1. Unit Tests

Runs C++ unit tests using Catch2 framework.

**Command**: `./build/tests/precon_tests --reporter compact --success`

**Tests**:
- Sparse matrix operations
- Preconditioner formula verification
- Matrix properties (SPD, symmetry)
- Parameter validation

**Success Criteria**: All unit tests pass (20/22 currently passing)

### 2. Integration Tests

Runs LAMMPS input scripts to verify plugin functionality.

**Tests**:
- `test_identity.lam` - Identity preconditioner (baseline)
- `test_precon.lam` - Exp preconditioner with auto mu estimation
- `test_auto_mu.lam` - Auto mu estimation
- `test_fixed.lam` - Fixed atom constraints

**Success Criteria**: Each test completes with "Loop time" in output

### 3. MPI Tests

Verifies parallel execution with multiple MPI ranks.

**Tests**:
- 2 ranks: `mpirun -np 2 lmp -in test_precon.lam`
- 4 ranks: `mpirun -np 4 lmp -in test_precon.lam`

**Success Criteria**: Tests complete without MPI errors

## Artifacts

Test logs are uploaded as artifacts for debugging:

**Artifact name**: `test-logs`

**Contents**: All `*.log` files from `tests/lammps/`

## Local Testing

To test the CI workflow locally (requires Docker):

```bash
# Install act (https://github.com/nektos/act)
# Ubuntu/Debian:
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run the CI workflow locally
act push
```

## Troubleshooting

### LAMMPS Build Fails

Check that dependencies are installed:
- libeigen3-dev
- libopenmpi-dev
- libfftw3-dev
- libjpeg-dev, libpng-dev, zlib1g-dev

### Plugin Build Fails

Ensure LAMMPS_SOURCE_DIR is correctly set:
```bash
cmake .. -D LAMMPS_SOURCE_DIR=/path/to/lammps/src
```

### Unit Tests Fail

Run locally with verbose output:
```bash
cd build
./tests/precon_tests -s  # Show all assertions
```

### Integration Tests Fail

Check test logs in artifacts:
1. Go to GitHub Actions run
2. Download "test-logs" artifact
3. Inspect individual `*.log` files

## Updating the Workflow

### Adding a New Test

1. Add test input file to `tests/lammps/`
2. Add test step to `.github/workflows/ci.yml`:

```yaml
- name: Run integration test - My New Test
  run: |
    cd tests/lammps
    ${GITHUB_WORKSPACE}/lammps/build/lmp -in test_mytest.lam > test_mytest.log 2>&1

    if grep -q "Loop time" test_mytest.log; then
      echo "✓ My test passed"
    else
      echo "✗ My test failed"
      cat test_mytest.log
      exit 1
    fi
```

### Changing LAMMPS Packages

Modify the cmake command in the "Clone and build LAMMPS" step:

```yaml
cmake ../cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_MPI=yes \
  -D PKG_MANYBODY=yes \
  -D PKG_MY_NEW_PACKAGE=yes  # Add packages here
```

**Important**: Changing packages will invalidate the LAMMPS cache and trigger a rebuild.

### Cache Management

To force a cache rebuild:
1. Update the cache key suffix in `.github/workflows/ci.yml`
2. Change `-precon` to `-precon-v2` (or similar)

GitHub automatically manages cache storage with:
- 10 GB total cache limit per repository
- 7 day eviction policy for unused caches

## Performance

Typical CI run times:

| Stage | First Run (no cache) | Subsequent (cached) |
|-------|---------------------|---------------------|
| LAMMPS build | ~8 minutes | ~30 seconds (cache restore) |
| Plugin build | ~1 minute | ~1 minute |
| Unit tests | ~5 seconds | ~5 seconds |
| Integration tests | ~30 seconds | ~30 seconds |
| **Total** | **~10 minutes** | **~2 minutes** |

## Status Badge

The README includes a CI status badge:

```markdown
[![CI](https://github.com/jameskermode/lammps-precon-opt-fix/workflows/CI/badge.svg)](https://github.com/jameskermode/lammps-precon-opt-fix/actions)
```

Badge colors:
- 🟢 Green: All tests passing
- 🔴 Red: One or more tests failing
- 🟡 Yellow: Tests running or workflow disabled

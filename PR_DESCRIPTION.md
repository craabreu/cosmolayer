# Add CI Testing for COSMOSAC Reference Package

## Summary

This PR adds automated installation and testing of the [COSMOSAC reference implementation](https://github.com/usnistgov/COSMOSAC) across all CI platforms (Windows, Linux, macOS). This enables testing of `cosmolayer` against the reference COSMO-SAC implementation to ensure accuracy and compatibility.

## Changes

### Installation Scripts

Added platform-specific installation scripts for the COSMOSAC reference package:

- **`devtools/scripts/install_cosmosac.sh`** - Bash script for Linux/macOS
- **`devtools/scripts/install_cosmosac.ps1`** - PowerShell script for Windows

Both scripts:
- Clone the COSMOSAC repository with submodules
- Default to commit `21dd92b` (cCOSMO v1.0.3) for reproducible builds
- Auto-detect package metadata (pyproject.toml/setup.py) using native shell operations
- Support custom versions via `COSMOSAC_GIT_REF` environment variable
- Install via pip with `--no-deps` flag to avoid dependency conflicts

### CI Workflow Updates

Updated all three CI workflows to install and verify COSMOSAC:

- **Linux** (`.github/workflows/Linux.yaml`)
- **macOS** (`.github/workflows/MacOS.yaml`)
- **Windows** (`.github/workflows/Windows.yaml`)

Each workflow now:
1. Installs COSMOSAC reference package after main package installation
2. Runs quick import verification: `python -c "import cCOSMO"`
3. Executes full test suite including COSMOSAC installation tests

### Test Environment Dependencies

Updated `devtools/conda-envs/test_env.yaml` to include build dependencies required for compiling COSMOSAC's C++ extensions:

- `cmake` - Build system generator
- `ninja` - Fast build tool
- `c-compiler` - C compiler
- `cxx-compiler` - C++ compiler

### COSMOSAC Installation Tests

Added `cosmolayer/tests/test_cosmosac_installation.py` with three test functions:

1. **`test_cCOSMO_imported()`** - Verifies the cCOSMO package can be imported
2. **`test_cCOSMO_has_basic_structure()`** - Checks module has expected attributes
3. **`test_cCOSMO_key_classes()`** - Validates presence of key classes:
   - `COSMO1` (main COSMO-SAC calculation class)
   - `VirginiaTechProfileDatabase` (profile database class)

## Technical Details

### Package Import Name

The COSMOSAC reference package is imported as **`cCOSMO`**, not `COSMOSAC`. This follows the convention used in the [official notebooks](https://github.com/usnistgov/COSMOSAC/blob/master/COSMO-SAC.ipynb).

### Version Pinning

The installation scripts default to commit `21dd92b` which:
- Builds cCOSMO version 1.0.3
- Uses modern `pyproject.toml` build system
- Is compatible with current CMake versions
- Provides reproducible builds across all platforms

**Note:** Tagged releases (v1.0, v1.0.1) use legacy `setup.py` with CMake compatibility issues on modern systems.

## Testing

All tests pass locally on macOS ARM64:

```bash
$ pytest cosmolayer/tests/test_cosmosac_installation.py -v
============================= test session starts ==============================
cosmolayer/tests/test_cosmosac_installation.py::test_cCOSMO_imported PASSED
cosmolayer/tests/test_cosmosac_installation.py::test_cCOSMO_has_basic_structure PASSED
cosmolayer/tests/test_cosmosac_installation.py::test_cCOSMO_key_classes PASSED
============================== 3 passed in 3.03s ===============================
```

Import verification:
```python
>>> import cCOSMO
>>> hasattr(cCOSMO, 'COSMO1')
True
>>> hasattr(cCOSMO, 'VirginiaTechProfileDatabase')
True
```

## Benefits

- **Automated validation**: CI now tests against reference implementation
- **Cross-platform support**: Works on Windows, Linux, and macOS
- **Reproducible builds**: Pinned to specific commit for consistency
- **Early failure detection**: Import checks catch installation issues immediately
- **Better debugging**: Clear error messages and verification steps in CI logs

## Future Work

- Compare `cosmolayer` results with COSMOSAC reference calculations
- Add regression tests using COSMOSAC as ground truth
- Benchmark performance differences between implementations

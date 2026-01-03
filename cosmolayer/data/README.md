# COSMO Data Files

This directory contains COSMO (Conductor-like Screening Model) output files from quantum mechanical calculations. These files are used as test data and examples in the cosmolayer package.

## Files

- **`C=C(N)O.cosmo`**: COSMO output file for formamide (C₂H₃NO) generated using **TurboMole**.
- **`NCCO.cosmo`**: COSMO output file generated using **DMol-3**.

## Usage

These files can be loaded using Python's `importlib.resources`:

```python
from importlib.resources import files
from cosmolayer.sac import Component

# Load a COSMO file
component = Component(files("cosmolayer.data") / "C=C(N)O.cosmo")
```

## File Format

COSMO files contain information about molecular surface segments including:

- Segment coordinates (x, y, z)
- Surface charge densities
- Surface areas
- Atom assignments

The parser in `cosmolayer.parser` can read these files and extract the necessary information for COSMO-SAC activity coefficient calculations.

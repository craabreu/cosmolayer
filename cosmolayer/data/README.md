# COSMO Data Files

This directory contains COSMO (Conductor-like Screening Model) output files and derived sigma profiles used as test data and examples in the cosmolayer package.

## Files

### COSMO output files (`.cosmo`)

Raw COSMO output from quantum mechanical calculations:

- **`C=C(N)O.cosmo`** – 1-Aminoethenol, **TurboMole**
- **`CF.cosmo`** – Fluoromethane, **DMol-3**
- **`NCCO.cosmo`** – 2-Aminoethanol, **DMol-3**
- **`O.cosmo`** – Water, **DMol-3**

### COSMO-SAC 2002 Sigma profiles (`.sigma`)

Precomputed sigma profiles for the **COSMO-SAC 2002** model (see `cosmolayer.sac.CosmoSac2002Mixture` and `create_cosmo_sac_2002_matrix`):

- **`CF.sigma`** – Fluoromethane
- **`NCCO.sigma`** – 2-Aminoethanol
- **`O.sigma`** – Water

### COSMO-SAC 2010 Sigma profiles (`.sigma3`)

Precomputed sigma profiles for the **COSMO-SAC 2010** model (see `cosmolayer.sac.CosmoSac2010Mixture` and `create_cosmo_sac_2010_matrices`):

- **`CF.sigma3`** – Fluoromethane
- **`NCCO.sigma3`** – 2-Aminoethanol
- **`O.sigma3`** – Water

### NIST COSMOSAC reference data

Data formatted for validation against the [NIST COSMOSAC reference implementation](https://github.com/usnistgov/COSMOSAC):

#### `cosmo-sac-2002/` – Virginia Tech 2005 format

Sigma profiles reformatted for compatibility with NIST COSMOSAC's COSMO-SAC 2002 implementation:

- **`Sigma_Profile_Database_Index_v2.txt`** – Profile database index
- **`VT2005-*.txt`** – Reformatted sigma profile files

These files use the Virginia Tech 2005 format and are used to validate CosmoLayer against the NIST reference.

#### `cosmo-sac-2010/` – COSMO-SAC 2010 format

Symbolic links to `.sigma3` files with modified extensions (`.sigma3` → `.sigma`) for compatibility with NIST COSMOSAC's COSMO-SAC 2010 reader. Used in validation tests against the NIST reference implementation.

## Usage

COSMO files can be loaded with `importlib.resources` and used to build components or mixtures:

```python
from importlib.resources import files
from cosmolayer.sac import Component

# Load a COSMO file
component = Component(files("cosmolayer.data") / "C=C(N)O.cosmo")
```

Example with multiple components (e.g. for a Mixture):

```python
from importlib.resources import files
from cosmolayer.sac import CosmoSac2010Mixture

components = {
    "fluoromethane": files("cosmolayer.data") / "CF.cosmo",
    "water": files("cosmolayer.data") / "O.cosmo",
}
mixture = CosmoSac2010Mixture(components)
```

## File formats

**COSMO files** (`.cosmo`) contain segment-level data, including:

- Segment coordinates (x, y, z)
- Surface charge densities
- Surface areas
- Atom assignments

The parsers in `cosmolayer.parser` read these files and provide the data needed for COSMO-SAC activity coefficient calculations. The `.sigma` and `.sigma3` files store precomputed sigma profiles for the COSMO-SAC 2002 and COSMO-SAC 2010 models respectively (see `cosmolayer.sac`); they are used by tests and validation scripts.

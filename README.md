Differentiable COSMO-Type Activity Coefficient Layer
====================================================

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/craabreu/cosmolayer/actions/workflows/Linux.yaml/badge.svg?branch=main&event=push)](https://github.com/craabreu/cosmolayer/actions/workflows/Linux.yaml?query=branch%3Amain+event%3Apush)
[![GitHub Actions Build Status](https://github.com/craabreu/cosmolayer/actions/workflows/MacOS.yaml/badge.svg?branch=main&event=push)](https://github.com/craabreu/cosmolayer/actions/workflows/MacOS.yaml?query=branch%3Amain+event%3Apush)
[![GitHub Actions Build Status](https://github.com/craabreu/cosmolayer/actions/workflows/Windows.yaml/badge.svg?branch=main&event=push)](https://github.com/craabreu/cosmolayer/actions/workflows/Windows.yaml?query=branch%3Amain+event%3Apush)
[![GitHub Actions Build Status](https://github.com/craabreu/cosmolayer/actions/workflows/Linter.yaml/badge.svg?branch=main&event=push)](https://github.com/craabreu/cosmolayer/actions/workflows/Linter.yaml?query=branch%3Amain+event%3Apush)
[![Documentation Status](https://github.com/craabreu/cosmolayer/actions/workflows/Docs.yaml/badge.svg?branch=main&event=push)](https://github.com/craabreu/cosmolayer/actions/workflows/Docs.yaml?query=branch%3Amain+event%3Apush)
[![Coverage Report](https://craabreu.github.io/cosmolayer/development/coverage/coverage.svg)](https://craabreu.github.io/cosmolayer/development/coverage)

[![Conda version](https://img.shields.io/conda/v/mdtools/cosmolayer.svg)](https://anaconda.org/mdtools/cosmolayer)
[![Conda platforms](https://img.shields.io/conda/pn/mdtools/cosmolayer.svg)](https://anaconda.org/mdtools/cosmolayer)
[![Conda downloads](https://img.shields.io/conda/dn/mdtools/cosmolayer.svg)](https://anaconda.org/mdtools/cosmolayer)

[![PyPI version](https://img.shields.io/pypi/v/cosmolayer.svg)](https://pypi.org/project/cosmolayer)
[![PyPI version](https://img.shields.io/pypi/pyversions/cosmolayer.svg)](https://pypi.org/project/cosmolayer)
[![PyPI version](https://img.shields.io/pypi/dm/cosmolayer.svg)](https://pypi.org/project/cosmolayer)

[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![License](https://img.shields.io/badge/License-MIT-yellowgreen.svg?style=flat)](https://github.com/craabreu/cosmolayer/blob/main/LICENSE.md)

### Overview

CosmoLayer is a package implementing differentiable COSMO-type activity coefficient calculation layers for neural network models.

CosmoLayer leverages automatic differentiation and GPU acceleration to enable efficient computation and gradient-based optimization of COSMO model parameters.

### Installation and Usage

CosmoLayer is available on [PyPI] and as a conda package on the [mdtools] channel.

To install from PyPI:

```bash
    pip install cosmolayer
```

To install with conda:

```bash
    conda install -c conda-forge -c mdtools cosmolayer
```

Or:

```bash
    mamba install -c mdtools cosmolayer
```

To use CosmoLayer in your own Python script or Jupyter notebook, simply import it as follows:

```python
    import cosmolayer
```

### Documentation

Documentation for the latest CosmoLayer version is available at [Github Pages].

### Copyright

Copyright (c) 2026 [Charlles Abreu](https://github.com/craabreu)


[Github Pages]: https://craabreu.github.io/cosmolayer/latest
[mdtools]: https://anaconda.org/mdtools/cosmolayer
[PyPI]: https://pypi.org/project/cosmolayer

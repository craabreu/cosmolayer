# Rename the installable package to `cosmotools`

Do this **in this repository**. Do not fork. GitHub will redirect clones, issues, and stars if the repo is renamed; PyPI and conda will not.

Target imports after the rename:

```text
cosmotools/
  cosmolayer/     # today's nested layer implementation
  cosmosac/
  parser/
  tests/
  data/
```

```python
from cosmotools.cosmolayer import CosmoLayer
from cosmotools.cosmosac import Component
from cosmotools.parser import parse_cosmo_file
```

The current top-level shims (`cosmolayer/cosmosolver.py`, `cosmolayer/cosmodata.py`, `cosmolayer/cosmolightning.py`) exist only to keep today's `import cosmolayer` paths working. Delete them as part of this rename; do not carry them into `cosmotools`.

## 1. Source tree

- [ ] Rename the top-level Python package directory `cosmolayer/` → `cosmotools/`
- [ ] Delete the three compatibility shims that would otherwise become `cosmotools/cosmosolver.py`, `cosmotools/cosmodata.py`, and `cosmotools/cosmolightning.py`
- [ ] Update every `import cosmolayer` / `from cosmolayer...` in code, tests, and doctests
- [ ] Update Sphinx `.. module::` headers and `:class:`~cosmolayer...`` / `:mod:` targets
- [ ] Update resource paths such as `files("cosmolayer.data")` in [`cosmolayer/data/README.md`](../../cosmolayer/data/README.md) and tests

## 2. Packaging metadata

- [ ] [`pyproject.toml`](../../pyproject.toml): `project.name`, URLs, `[project.scripts]` (`cosmoview = "cosmotools.cosmosac.visualize:main"`), `[tool.setuptools.package-data]`, `[tool.versioningit.write]`, `[tool.pytest.ini_options]` (`--cov`, `testpaths`), `[tool.mypy.overrides]`
- [ ] [`setup.cfg`](../../setup.cfg): coverage omit path for `_version.py`
- [ ] [`devtools/conda-recipes/anaconda/meta.yaml`](../../devtools/conda-recipes/anaconda/meta.yaml): package name, `test.imports`, home/doc/dev URLs
- [ ] Decide whether `COSMOLAYER_VERSION` in Docs/PyPI/Anaconda workflows and [`docs/conf.py`](../conf.py) becomes `COSMOTOOLS_VERSION`

## 3. CI, docs, and scripts

- [ ] [`.github/workflows/Linter.yaml`](../../.github/workflows/Linter.yaml): `ruff` / `mypy` paths
- [ ] [`.github/workflows/PyPI.yaml`](../../.github/workflows/PyPI.yaml) and [`Anaconda.yaml`](../../.github/workflows/Anaconda.yaml): log strings and version env
- [ ] [`docs/conf.py`](../conf.py), [`docs/getting_started.rst`](../getting_started.rst), [`docs/visualization.rst`](../visualization.rst), [`docs/index.rst`](../index.rst)
- [ ] [`devtools/scripts/format_and_check.sh`](../../devtools/scripts/format_and_check.sh)
- [ ] [`README.md`](../../README.md) badges, install commands, and `import` example
- [ ] [`.github/CONTRIBUTING.md`](../../.github/CONTRIBUTING.md)

## 4. Publish `cosmotools`, then retire `cosmolayer`

A README-only deprecation is not enough: `pip install cosmolayer` and `conda install -c mdtools cosmolayer` would still install the old codebase.

- [ ] Reserve/publish the `cosmotools` project on PyPI and the `mdtools` conda channel
- [ ] Cut a first `cosmotools` release with the new import paths
- [ ] Cut a **last** `cosmolayer` release that depends on `cosmotools` and re-exports it (with a `DeprecationWarning` on import)
- [ ] After a grace period, stop releasing `cosmolayer` (optional: yank non-wrapper versions only if that will not break pinned installs)

## 5. GitHub repo (optional, last)

- [ ] Rename `craabreu/cosmolayer` → `craabreu/cosmotools` on GitHub (old URLs redirect)
- [ ] Confirm GitHub Pages moves to `https://craabreu.github.io/cosmotools/` and update remaining doc URLs
- [ ] Update the conda recipe `home` / `doc_url` / `dev_url` if they still point at the old name
- [ ] Delete this file (`docs/plans/rename-to-cosmotools.md`)

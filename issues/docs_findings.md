# Documentation findings

Open issues in labscript-optimization's code, documentation and packaging, for decision item by item. Each entry says what is wrong and the smallest fix.

1. **The package is not on PyPI.** `README.md:23-24`, `UPGRADING.md:14-15` and `docs/source/getting-started.rst:13-14` install from a clone of the repository, because PyPI has no project of this name. Fix: Ian may publish the package, after which the three places give `pip install "labscript-optimization[lyse]"`.
2. **The `lyse` extra names no versions.** `pyproject.toml:49-50` lists `lyse` and `runmanager` bare, but the routine needs their Development branches (`README.md:29`), which have `lyse.Routine`, `lyse.data(where=...)`, `save_to_h5` and `submit_shots` with `sequence`, so a plain install may get an upstream release that lacks them. Fix: pin minimum versions once the suite releases them.

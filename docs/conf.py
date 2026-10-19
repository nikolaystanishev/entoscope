"""Sphinx configuration: Markdown pages through MyST, the API reference through autodoc.

Build with ``uv run --group docs sphinx-build -W --keep-going docs docs/_build/html``. There is no
intersphinx: a build that fetches inventories over the network fails whenever one of those sites does.
"""

from importlib.metadata import version as package_version

project = "entoscope"
author = "Nikolay Stanishev"
copyright = "2026, Nikolay Stanishev"
release = package_version("entoscope")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["_build"]

# GitHub-style anchors, so ``[...](#dry-runs)`` resolves the same on the site and on GitHub.
myst_heading_anchors = 3
myst_enable_extensions = ["alert", "dollarmath"]

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_preserve_defaults = True

# Docstrings are Google style; the types are in the signature, so the sections carry only descriptions.
# Attributes render as a field list: an entry of their own would make an attribute named ``type`` a
# second target for every ``type[...]`` annotation.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
napoleon_use_ivar = True

html_theme = "furo"
html_title = "entoscope"

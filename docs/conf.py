"""Sphinx configuration: Markdown pages through MyST, the API reference through autodoc.

Build with ``uv run --group docs sphinx-build -W --keep-going docs docs/_build/html``. There is no
intersphinx: a build that fetches inventories over the network fails whenever one of those sites does.
"""

from importlib.metadata import version as package_version

project = "intex"
author = "Nikolay Stanishev"
copyright = "2026, Nikolay Stanishev"
release = package_version("intex")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["_build"]

# GitHub-style anchors, so ``[...](#dry-runs)`` resolves the same on the site and on GitHub.
myst_heading_anchors = 3
myst_enable_extensions = ["dollarmath"]

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_preserve_defaults = True

html_theme = "furo"
html_title = "intex"

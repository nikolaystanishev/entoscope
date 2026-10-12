"""Importing the package registers its built-in lenses, so a config can name one whatever was imported first."""

from entoscope.lens.method.factory import LensMethodFactory

__all__ = ["LensMethodFactory"]

"""Importing the package registers its built-in patching methods, so a config can name one whatever was
imported first."""

from entoscope.patching.method.factory import PatchingMethodFactory

__all__ = ["PatchingMethodFactory"]

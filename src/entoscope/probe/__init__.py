"""Importing the package registers its built-in probes, so a config can name one whatever was imported first."""

from entoscope.probe.method.factory import ProbeMethodFactory

__all__ = ["ProbeMethodFactory"]

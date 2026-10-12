"""Importing the package registers its built-in circuit-discovery methods, so a config can name one whatever
was imported first."""

from entoscope.circuit.method.factory import CircuitMethodFactory

__all__ = ["CircuitMethodFactory"]

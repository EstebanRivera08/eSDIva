"""
transducers
===========

Available transducer types:
  - LinearArrayTransducer(n_elements, element_width_mm, ...)
  - MatrixArrayTransducer(tx_N_elem_x, tx_elem_width_mm, ...)

Usage:
  from transducers import LinearArrayTransducer
  t = LinearArrayTransducer(...)
  print(t)
"""

from .linear import LinearArrayTransducer
from .matrix import MatrixArrayTransducer
from .saved_transducers import Domino, Zeus_Matrix

__all__ = ["LinearArrayTransducer", "MatrixArrayTransducer", "Domino", "Zeus_Matrix"]


def available_transducers():
    """List all available transducer classes."""
    return __all__[:]


# Optional: helper factory


def create_transducer(kind, **kwargs):
    """
    Factory to instantiate a transducer by name:
      kind: 'linear' or 'matrix'
    """
    kind = kind.lower()
    if kind == "linear":
        return LinearArrayTransducer(**kwargs)
    if kind == "matrix":
        return MatrixArrayTransducer(**kwargs)
    raise ValueError(f"Unknown transducer kind: {kind}")

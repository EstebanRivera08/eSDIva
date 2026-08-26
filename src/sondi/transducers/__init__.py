"""Transducer geometry classes for the SonDI acoustic simulator.

Notes
-----
**Array transducers** (multi-element, electronic steering):

- ``LinearArrayTransducer`` -- 1-D row of rectangular elements, optional elevation lens.
- ``MatrixArrayTransducer`` -- 2-D grid of rectangular elements.

**Mono-element transducers** (geometric focusing, no electronic delays):

- ``FlatCircularTransducer`` -- Flat piston disc.
- ``ConcaveCircularTransducer`` -- Spherically curved bowl (HIFU / TUS).
- ``ConvexCircularTransducer`` -- Spherically convex dome; virtual focus behind
  transducer. Models transducers with acoustic refractive lenses.
- ``FocusedCircularTransducer`` -- Circular-disk aperture with single-axis curvature
  (line focus).

**Composite arrays**:

- ``CustomTransducer`` -- Assemble any number of mono-element transducers at
  arbitrary positions and orientations (e.g. TUS helmet).

**Field II import**:

- ``FieldIITransducer`` -- Transducer built from ``xdc_get(Th, 'all')`` patch data.
- ``from_fieldii_xdc_data`` -- Factory: parse ``xdc_get`` struct → FieldIITransducer.
- ``from_fieldii_rect_data`` -- Factory: 26-row ``xdc_get(Th, 'rect')`` matrix →
  FieldIITransducer.
- ``from_fieldii_patch_arrays`` -- Factory: explicit patch arrays → FieldIITransducer.

**Pre-defined transducers**:

- ``Domino`` -- 128-element linear array (clinical probe).
- ``Zeus_Matrix`` -- 55x55 matrix array (research probe).

**Utilities**:

- ``available_transducers()`` -- List all exported class names.
- ``create_transducer()`` -- Simple factory for 'linear' and 'matrix' kinds.
"""

from .base import TransducerBase
from .circular import (
    ConcaveCircularTransducer,
    ConvexCircularTransducer,
    FlatCircularTransducer,
    FocusedCircularTransducer,
)
from .custom import CustomTransducer
from .fieldii_compat import (
    FieldIITransducer,
    from_fieldii_patch_arrays,
    from_fieldii_rect_data,
    from_fieldii_xdc_data,
)
from .linear import LinearArrayTransducer, ConvexArrayTransducer
from .matrix import MatrixArrayTransducer
from .saved_transducers import Domino, Zeus_Matrix

__all__ = [
    # Base
    "TransducerBase",
    # Array transducers
    "LinearArrayTransducer",
    "ConvexArrayTransducer",
    "MatrixArrayTransducer",
    # Mono-element transducers
    "FlatCircularTransducer",
    "ConcaveCircularTransducer",
    "ConvexCircularTransducer",
    "FocusedCircularTransducer",
    # Composite
    "CustomTransducer",
    # Field II import
    "FieldIITransducer",
    "from_fieldii_xdc_data",
    "from_fieldii_rect_data",
    "from_fieldii_patch_arrays",
    # Pre-defined
    "Domino",
    "Zeus_Matrix",
]


def available_transducers() -> list:
    """Return the names of the concrete transducer classes.

    Excludes the abstract base and the Field II factory functions — only the
    instantiable transducer types.

    Returns
    -------
    list of str
        Names of all concrete transducer classes.
    """
    return [
        name
        for name in __all__
        if isinstance(obj := globals()[name], type)
        and issubclass(obj, TransducerBase)
        and obj is not TransducerBase
    ]


def create_transducer(kind: str, **kwargs):
    """
    Instantiate a transducer by kind name.

    Parameters
    ----------
    kind : str
        One of ``'linear'``, ``'matrix'``, ``'flat_circular'``,
        ``'concave_circular'``, ``'focused_circular'``.
    **kwargs
        Passed directly to the transducer constructor.

    Returns
    -------
    TransducerBase
        A transducer instance of the requested kind.
    """
    mapping = {
        "linear": LinearArrayTransducer,
        "convex": ConvexArrayTransducer,
        "matrix": MatrixArrayTransducer,
        "flat_circular": FlatCircularTransducer,
        "concave_circular": ConcaveCircularTransducer,
        "convex_circular": ConvexCircularTransducer,
        "focused_circular": FocusedCircularTransducer,
    }
    key = kind.lower()
    if key not in mapping:
        raise ValueError(
            f"Unknown transducer kind '{kind}'. Available: {list(mapping.keys())}"
        )
    return mapping[key](**kwargs)

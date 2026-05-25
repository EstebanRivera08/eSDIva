"""PyField: deprecated alias for Emission.

Use `pyfield.psimulation.Emission` instead.
"""

import warnings

import numpy as np

from .emission import Emission


class PyField(Emission):
    """Deprecated: use `Emission` instead.

    Backward-compatible wrapper around `Emission`.  Defaults to
    ``monochromatic=True`` to preserve the old `PyField` behavior
    (``Emission`` defaults to ``False``).

    Parameters
    ----------
    transducer : TransducerBase
        Transducer instance.
    monochromatic : bool, default True
        CW mode.  Set to False for pulsed/transient output.
    **kwargs
        Forwarded to `Emission.__init__`.
    """

    def __init__(self, transducer, *, monochromatic=True, **kwargs):
        warnings.warn(
            "PyField is deprecated, use pyfield.psimulation.Emission instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(transducer, monochromatic=monochromatic, **kwargs)

    def __call__(
        self,
        field_points_mm,
        *,
        method="auto",
        excitation=None,
        normalize=False,
        **kwargs,
    ):
        """Backward-compatible call accepting legacy kwargs.

        Parameters
        ----------
        field_points_mm : dict or ndarray
        method : str
        excitation : ndarray or None, optional
            Excitation pulse passed at call time (legacy API).
        normalize : bool, optional
            If True, normalize pressure to its maximum value. Default False.
        **kwargs
            Ignored (absorbs old ``monochromatic``, etc.).
        """
        if excitation is not None:
            saved = self.excitation
            saved_mono = self.monochromatic
            self.excitation = np.asarray(excitation, dtype=np.float32)
            self.monochromatic = False
            try:
                pressure, coords = super().__call__(field_points_mm, method=method)
            finally:
                self.excitation = saved
                self.monochromatic = saved_mono
        else:
            pressure, coords = super().__call__(field_points_mm, method=method)

        if normalize:
            m = float(np.max(np.abs(pressure)))
            if m > 0:
                pressure = pressure / m
        return pressure, coords

    def set_field(self, attribute_name, value):
        """Set an attribute by name (legacy API; prefer `set()`).

        Parameters
        ----------
        attribute_name : str
            Attribute to update.
        value : object
            New value.
        """
        if not hasattr(self, attribute_name):
            raise AttributeError(
                f"{attribute_name} is not a valid attribute of PyField."
            )
        setattr(self, attribute_name, value)

    def __repr__(self) -> str:
        return (
            f"PyField(transducer={self.tx}, c={self.c} m/s, fs={self.fs} Hz, "
            f"fc={self.fc} Hz, alpha0={self.alpha0} dB/(MHz^y cm), "
            f"freq_power={self.freq_power})"
        )

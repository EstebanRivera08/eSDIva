"""Shared base for the acoustic simulators (`Emission`, `Reception`, `ReceptionSDI`).

`SimulationBase` holds the small amount of state and plumbing that emission and
reception genuinely share — the medium parameters (speed of sound, density,
sampling rate, attenuation), the runtime ``set()`` validation, the excitation
resolution, and the per-element patch grouping — so neither side re-implements it.

It deliberately does NOT share any physics: the emitted-pressure chain
(``ρ₀ · dv/dt ⊛ h``) and the pulse-echo chain (``v_pe ⊛ h_tx ⊛ h_rx``, with the
four integrations ``I⁴``) live entirely in the respective subclasses, so each
acoustic derivation can be read in one file.
"""

from typing import TYPE_CHECKING, Any

import numpy as np


class SimulationBase:
    """Medium state + runtime config shared by every simulator.

    Subclasses set ``tx`` (and, for reception, ``rx``), provide a ``_SETTABLE``
    map and a ``_refresh_sub_elem_attributes`` method, and implement their own
    physics core. Everything here is medium bookkeeping and array plumbing.
    """

    if TYPE_CHECKING:
        tx: Any
        excitation: Any
        _SETTABLE: dict

    # ------------------------------------------------------------------
    # Runtime parameter update (shared validation core)
    # ------------------------------------------------------------------

    def _apply_settable(self, name: str, value) -> None:
        """Validate ``value`` against ``_SETTABLE[name]`` and assign it.

        The shared tail of every ``set()``: checks the name is known and the
        type matches, casts an excitation to float32, and stores the attribute.
        Subclasses handle their structural keys (``"transducer"``/``"tx"``/
        ``"rx"``) first, then delegate the plain parameters here.

        Raises
        ------
        ValueError
            If ``name`` is not a recognised parameter.
        TypeError
            If ``value`` has the wrong type for ``name``.
        """
        if name not in self._SETTABLE:
            raise ValueError(
                f"Unknown parameter '{name}'. Valid: {list(self._SETTABLE)}"
            )
        expected = self._SETTABLE[name][0]
        if not isinstance(value, expected):
            raise TypeError(f"'{name}' expects {expected}, got {type(value)}")
        if name == "excitation" and value is not None:
            value = np.asarray(value, dtype=np.float32)
        setattr(self, name, value)

    # ------------------------------------------------------------------
    # Excitation resolution
    # ------------------------------------------------------------------

    def _resolve_excitation(self):
        """Effective excitation pulse: ``self.excitation`` else ``tx.excitation``.

        Returns a float32 array or None. A global pulse is ``(L,)``; a
        per-transmit-element pulse is ``(L, E)`` (one column per element) and is
        preserved as 2-D — only a transducer-stored 1-D pulse is ravelled.
        """
        exc = self.excitation
        if exc is None:
            tx_exc = getattr(self.tx, "excitation", None)
            if tx_exc is not None:
                tx_exc = np.asarray(tx_exc, dtype=np.float32)
                exc = tx_exc.ravel() if tx_exc.ndim == 1 else tx_exc
        return exc

    # ------------------------------------------------------------------
    # Per-element patch grouping
    # ------------------------------------------------------------------

    @staticmethod
    def _group_patches_by_element(n_elements, sub_el_idx, arrays):
        """Split flat per-patch arrays into one tuple per element.

        ``arrays`` is the 7-tuple ``(centers, wx, wy, apod, delays, eu, ev)`` of
        per-patch quantities; this returns a list of ``n_elements`` such tuples,
        each holding only the patches belonging to that element (``sub_el_idx == e``).
        Used to compute one element's SIR at a time (per-element excitation,
        per-element attenuation, raw per-channel RF).
        """
        return [tuple(a[sub_el_idx == e] for a in arrays) for e in range(n_elements)]

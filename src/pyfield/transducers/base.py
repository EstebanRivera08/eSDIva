"""
Base class for all PyField transducer types.

Every transducer is built from rectangular patches that approximate the
physical aperture. Subclasses implement the geometry (element centers and
subdivision patches); this base class provides the shared simulation
interface: delay computation, apodization, visualization, and state
management.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from . import geometry_utils, validators


class TransducerBase(ABC):
    """
    Abstract base class for all transducer types.

    Subclasses must implement:
      - ``_compute_element_centers()`` → element positions in 3-D space
      - ``_build_subdivisions()``      → rectangular patch geometry

    Everything else (delay law, apodization setter/getter, mesh generation,
    visualization, state dict) is provided here and shared by all types.

    Attributes
    ----------
    type : str
        Short type identifier (e.g. ``'linear'``, ``'matrix'``, ``'circular'``).
    name : str
        Human-readable class name.
    n_elements : int
        Number of independently controlled elements (1 for mono-element types).
    elem_width : float
        Characteristic element width in metres (used for patch-size reporting).
    elem_height : float
        Characteristic element height in metres.
    no_sub_x, no_sub_y : int
        Subdivision count along each axis (controls simulation accuracy).
    fc : float
        Centre frequency in Hz (required by the PyField simulator).
    speed_of_sound_mps : float
        Default propagation speed used when no ``c`` argument is supplied.
    """

    def __init__(self) -> None:
        # Filled in by subclasses before __init__ returns
        self.type: str = "unknown"
        self.name: str = "TransducerBase"
        self.n_elements: int = 0
        self.elem_width: float = 0.0
        self.elem_height: float = 0.0
        self.no_sub_x: int = 1
        self.no_sub_y: int = 1
        self.fc: float = 1e6
        self.speed_of_sound_mps: float = 1540.0

        # State
        self._apodization: Optional[np.ndarray] = None
        self._delays: Optional[np.ndarray] = None
        self.apodization_type: Optional[str] = None
        self.FoverD: Optional[float] = None

        # Geometry cache (populated lazily)
        self._element_centers: Optional[np.ndarray] = None
        self._sub_quad_verts: Optional[List[np.ndarray]] = None
        self._sub_area: Optional[float] = None
        self._sub_el_idx: Optional[List[int]] = None
        self._sub_patch_frames: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Lazy-loaded geometry properties
    # ------------------------------------------------------------------

    @property
    def element_centers(self) -> np.ndarray:
        """3-D element centre positions, shape ``(n_elements, 3)`` in metres."""
        if self._element_centers is None:
            self._element_centers = self._compute_element_centers()
        return self._element_centers

    @property
    def sub_quad_verts(self) -> List[np.ndarray]:
        """List of quad-vertex arrays ``(4, 3)`` for every patch, in metres."""
        if self._sub_quad_verts is None:
            self._sub_quad_verts, self._sub_area, self._sub_el_idx = (
                self._build_subdivisions()
            )
        return self._sub_quad_verts

    @property
    def sub_area(self) -> float:
        """Patch area in m² (same for all patches in a uniform grid)."""
        if self._sub_area is None:
            _ = self.sub_quad_verts  # trigger lazy load
        assert self._sub_area is not None
        return self._sub_area

    @property
    def sub_el_idx(self) -> List[int]:
        """Element index for each patch; maps patch → parent element."""
        if self._sub_el_idx is None:
            _ = self.sub_quad_verts
        assert self._sub_el_idx is not None
        return self._sub_el_idx

    @property
    def n_sub_patches(self) -> int:
        """Total number of rectangular sub-patches across all elements."""
        return len(self.sub_quad_verts)

    @property
    def sub_patch_frames(self) -> Dict:
        """
        Per-patch rigid-body frames used by the SIR kernel.

        Returns a dict with keys ``centers``, ``normals``, ``tangents_u``,
        ``tangents_v``, ``wu``, ``wv`` — all ndarrays indexed by patch.

        For flat transducers the default implementation computes frames from
        the vertex edge vectors (``v[1]-v[0]`` and ``v[3]-v[0]``), which is
        exact for any flat, arbitrarily-oriented patch.  Curved transducers
        override ``_build_patch_frames`` to return surface-accurate frames
        derived from the parametric surface equations.
        """
        if self._sub_patch_frames is None:
            self._sub_patch_frames = self._build_patch_frames()
        return self._sub_patch_frames

    def _build_patch_frames(self) -> Dict:
        """
        Default patch-frame builder for **flat** transducers.

        Computes each patch's local frame directly from its corner vertices:

        * ``tangent_u`` = normalised ``v[1] − v[0]``  (first edge)
        * ``tangent_v`` = ``v[3] − v[0]`` orthogonalised against ``tangent_u``
        * ``normal``    = ``tangent_u × tangent_v``
        * ``wu``        = ``‖v[1] − v[0]‖``
        * ``wv``        = ``‖v[3] − v[0]‖``

        This is exact for truly flat patches and gives a good approximation
        for very gently curved surfaces.  Subclasses whose geometry is
        significantly curved should override this method (or set
        ``_sub_patch_frames`` as a side-effect inside ``_build_subdivisions``).
        """
        verts = self.sub_quad_verts
        centers = np.array([v.mean(axis=0) for v in verts], dtype=np.float64)

        e_u = np.array([v[1] - v[0] for v in verts], dtype=np.float64)
        e_v = np.array([v[3] - v[0] for v in verts], dtype=np.float64)

        wu = np.linalg.norm(e_u, axis=1).astype(np.float32)
        wv = np.linalg.norm(e_v, axis=1).astype(np.float32)

        # Normalise u-tangents
        tu = e_u / np.where(wu[:, None] > 1e-30, wu[:, None], 1.0)

        # Gram-Schmidt: orthogonalise v-tangents against u-tangents
        tv_raw = e_v / np.where(wv[:, None] > 1e-30, wv[:, None], 1.0)
        tv = tv_raw - np.einsum("ij,ij->i", tv_raw, tu)[:, None] * tu
        tv_len = np.linalg.norm(tv, axis=1, keepdims=True)
        tv /= np.where(tv_len > 1e-30, tv_len, 1.0)

        normals = np.cross(tu, tv)

        return {
            "centers": centers,
            "normals": normals,
            "tangents_u": tu,
            "tangents_v": tv,
            "wu": wu,
            "wv": wv,
        }

    # ------------------------------------------------------------------
    # State properties: apodization and delays
    # ------------------------------------------------------------------

    @property
    def apodization(self) -> np.ndarray:
        """Per-element apodization weights, shape ``(n_elements,)``."""
        if self._apodization is None:
            self._apodization = np.ones(self.n_elements, dtype=float)
        return self._apodization

    @apodization.setter
    def apodization(self, weights: np.ndarray) -> None:
        self._apodization = validators.validate_apodization_weights(
            weights, self.n_elements
        )

    @property
    def delays(self) -> np.ndarray:
        """Per-element delays in seconds, shape ``(n_elements,)``."""
        if self._delays is None:
            self._delays = np.zeros(self.n_elements, dtype=float)
        return self._delays

    @delays.setter
    def delays(self, delay_values: np.ndarray) -> None:
        self._delays = validators.validate_delays(delay_values, self.n_elements)

    @property
    def tx_N_active(self) -> int:
        """Number of elements with non-zero apodization."""
        return int(np.sum(self.apodization > 0))

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by every subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_element_centers(self) -> np.ndarray:
        """Return element centre positions, shape ``(n_elements, 3)`` in metres."""

    @abstractmethod
    def _build_subdivisions(
        self,
    ) -> Tuple[List[np.ndarray], float, List[int]]:
        """
        Build rectangular sub-patches for the entire aperture.

        Returns
        -------
        sub_quad_verts : list of ndarray (4, 3)
        sub_area : float
        sub_el_idx : list of int
        """

    # ------------------------------------------------------------------
    # Shared methods — provided by the base class
    # ------------------------------------------------------------------

    def compute_delays(
        self,
        focus_mm,
        *,
        c: Optional[float] = None,
        inline: bool = True,
        plot: bool = False,
    ) -> np.ndarray:
        """
        Compute per-element time delays for electronic focusing.

        The delay law is distance-based: each element fires at a time that
        makes the wave front arrive at ``focus_mm`` simultaneously.

        Parameters
        ----------
        focus_mm : array-like, shape (2,) or (3,)
            Focal point in mm. If 2-D ``[x, z]``, y=0 is assumed.
        c : float, optional
            Speed of sound in m/s. Defaults to ``speed_of_sound_mps`` (1540).
        inline : bool
            If True (default) store result in ``self.delays``.
        plot : bool
            If True, display a delay plot after computation.

        Returns
        -------
        delays : ndarray, shape (n_elements,)
            Delays in seconds (minimum delay is always 0).
        """
        if self.n_elements == 1:
            warnings.warn(
                f"{self.name} is a mono-element transducer. "
                "compute_delays() is ignored — the entire aperture surface "
                "acts simultaneously, so element-level delays do not apply.",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros(1)

        c = (
            validators.validate_speed_of_sound(c)
            if c is not None
            else self.speed_of_sound_mps
        )
        focus_m = validators.validate_focus_coordinates(focus_mm)

        dist = np.linalg.norm(self.element_centers - focus_m, axis=1)
        if focus_m[2] <= 0:
            # Diverging wave: earliest element fires first
            delays = dist - dist.min()
        else:
            # Focusing: farthest element fires first
            delays = dist.max() - dist

        delays /= c

        if inline:
            self.delays = delays
        if plot:
            self.plot_delays(delays)
        return delays

    def compute_apodization(
        self,
        focus_mm=None,
        *,
        FoverD: Optional[float] = None,
        apodization_type: Optional[str] = None,
        plot: bool = False,
        inline: bool = True,
    ) -> np.ndarray:
        """
        Return uniform full-aperture apodization (all ones).

        Mono-element transducers use the full aperture by definition.
        Multi-element subclasses (linear, matrix) override this method with
        window-based aperture selection.

        For mono-element transducers, patch-wise apodization can still be
        set directly via ``set_apodization()``.

        Parameters
        ----------
        focus_mm : ignored
            Accepted for API consistency with multi-element subclasses.
        FoverD, apodization_type, plot : ignored
            Accepted for API consistency.
        inline : bool
            If True (default), store result in ``self.apodization``.

        Returns
        -------
        apodization : ndarray, shape (n_elements,)
        """
        if self.n_elements == 1:
            warnings.warn(
                f"{self.name} is a mono-element transducer. "
                "compute_apodization() returns uniform weights. "
                "Use set_apodization() for custom patch-wise weighting.",
                UserWarning,
                stacklevel=2,
            )

        apod = np.ones(self.n_elements, dtype=float)
        if inline:
            self.apodization = apod
            self.apodization_type = "full"
        if plot:
            self.plot_apodization()
        return apod

    def set_apodization(self, weights: np.ndarray) -> None:
        """Set per-element apodization weights directly."""
        self.apodization = np.asarray(weights, dtype=float)

    def set_delays(self, delays: np.ndarray) -> None:
        """Set per-element delays directly (normalised so minimum = 0)."""
        d = np.atleast_1d(delays).astype(float)
        self.delays = d - d.min()

    def get_mesh(self) -> pv.PolyData:
        """
        Build a PyVista surface mesh of the transducer.

        Returns
        -------
        pyvista.PolyData
            Mesh with ``'Apodization'`` and ``'Delays'`` as cell arrays.
        """
        return geometry_utils.create_mesh_from_quads(
            self.sub_quad_verts,
            self.sub_el_idx,
            self.apodization,
            self.delays,
            scale_to_mm=True,
        )

    def show(
        self,
        *,
        window_size: Tuple[int, int] = (800, 600),
        scalars: str = "Apodization",
        notebook: bool = False,
        jupyter_backend: Optional[str] = None,
        colorbar_title: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Interactive 3-D visualisation of the transducer surface.

        Parameters
        ----------
        window_size : (int, int)
            Pixel dimensions of the render window.
        scalars : {'Apodization', 'Delays'}
            Which cell array to colour by.
        notebook : bool
            Enable Jupyter notebook rendering.
        jupyter_backend : str, optional
            Backend string passed to PyVista (``'static'``, ``'trame'`` …).
        colorbar_title : str, optional
            Override the default colour-bar label.
        **kwargs
            Forwarded to ``plotter.add_mesh()``.
        """
        if scalars not in ("Apodization", "Delays"):
            raise ValueError(
                f"scalars must be 'Apodization' or 'Delays', got '{scalars}'"
            )

        mesh = self.get_mesh()
        plotter = pv.Plotter(window_size=window_size, notebook=notebook)

        if scalars == "Apodization":
            cmap, title, clim = "cool", "Apodization", [0, 1]
        else:
            cmap, title, clim = "rainbow", "Delays (s)", None

        if colorbar_title is not None:
            title = colorbar_title

        defaults = {
            "scalars": scalars,
            "cmap": cmap,
            "clim": clim,
            "show_scalar_bar": True,
            "scalar_bar_args": {
                "title": title,
                "vertical": True,
                "position_x": 0.8,
                "position_y": 0.1,
            },
            "opacity": 1.0,
            "show_edges": True,
            "ambient": 1,
        }
        for key, val in defaults.items():
            kwargs.setdefault(key, val)

        plotter.add_mesh(mesh, **kwargs)
        plotter.add_axes()
        plotter.show_grid(
            font_size=10,
            xtitle="X (mm)",
            ytitle="Y (mm)",
            ztitle="Z (mm)",
            n_xlabels=3,
            n_ylabels=3,
            n_zlabels=3,
        )

        if jupyter_backend is not None:
            plotter.show(jupyter_backend=jupyter_backend)
        else:
            plotter.show()

    def plot_apodization(
        self,
        apodization: Optional[np.ndarray] = None,
        *,
        figsize: Tuple[int, int] = (6, 5),
        ax=None,
    ):
        """
        Plot apodization weights as a line/stem chart.

        For 2-D matrix transducers, override this method to produce an image.
        """
        standalone = ax is None
        if apodization is None:
            apodization = self.apodization
        if standalone:
            _, ax = plt.subplots(figsize=figsize)

        assert ax is not None
        ax.plot(
            np.arange(self.n_elements),
            apodization,
            "k-",
            marker="o",
            markerfacecolor="r",
        )
        ax.set_title(f"Apodization: {self.apodization_type}")
        ax.set_xlabel("Element #")
        ax.set_ylabel("Weight")
        ax.grid(True)

        if standalone:
            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            return ax

    def plot_delays(
        self,
        delays: Optional[np.ndarray] = None,
        *,
        figsize: Tuple[int, int] = (6, 5),
        ax=None,
    ):
        """Plot per-element delays in microseconds."""
        standalone = ax is None
        if delays is None:
            delays = self.delays
        if standalone:
            _, ax = plt.subplots(figsize=figsize)

        assert ax is not None
        ax.plot(
            np.arange(self.n_elements),
            delays * 1e6,
            "k-",
            marker="o",
            markerfacecolor="r",
        )
        ax.set_title("Delays")
        ax.set_xlabel("Element #")
        ax.set_ylabel("Delay (µs)")
        ax.grid(True)

        if standalone:
            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            return ax

    def plot_delays_apodization(self, figsize: Tuple[int, int] = (10, 4)) -> None:
        """Side-by-side delay and apodization plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        self.plot_delays(ax=ax1)
        self.plot_apodization(ax=ax2)
        plt.tight_layout()
        plt.show()
        plt.close()

    def clean(self) -> None:
        """Release cached geometry arrays to free memory."""
        self._element_centers = None
        self._sub_quad_verts = None
        self._sub_area = None
        self._sub_el_idx = None
        self._sub_patch_frames = None
        print(f"{self.name} cleaned up.")

    def get_state_dict(self) -> Dict[str, Any]:
        """Return a snapshot of the current apodization / delay state."""
        return {
            "apodization": self.apodization.copy(),
            "delays": self.delays.copy(),
            "apodization_type": self.apodization_type,
            "FoverD": self.FoverD,
        }

    def set_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore apodization / delay state from a dictionary."""
        if "apodization" in state:
            self.apodization = state["apodization"]
        if "delays" in state:
            self.delays = state["delays"]
        if "apodization_type" in state:
            self.apodization_type = state["apodization_type"]
        if "FoverD" in state:
            self.FoverD = state["FoverD"]

    def __repr__(self) -> str:
        return (
            f"{self.name}(n_elements={self.n_elements}, "
            f"type='{self.type}', fc={self.fc / 1e6:.2f} MHz)"
        )

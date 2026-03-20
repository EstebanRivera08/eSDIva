"""
Abstract base class for all transducers.

This module defines the TransducerBase abstract class which provides common
functionality for all transducer types (linear, matrix, concave, annular, etc.).
It handles geometry initialization, state management, computation methods,
and visualization.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List
import warnings
import numpy as np
import pyvista as pv
from time import time as TIME

from . import validators, geometry_utils


class TransducerBase(ABC):
    """
    Abstract base class for all transducer types.

    Defines the common interface for transducers, including geometry management,
    delay and apodization computation, visualization, and state persistence.

    Attributes
    ----------
    type : str
        Transducer type identifier (e.g., 'linear', 'matrix').
    name : str
        Human-readable name of the transducer.
    n_elements : int
        Total number of active elements.
    elem_width : float
        Width of elements in meters.
    elem_height : float
        Height of elements in meters.
    no_sub_x, no_sub_y : int
        Number of subdivisions per element.
    frequency_Hz : float
        Operating frequency in Hz (optional).
    """

    def __init__(self) -> None:
        """
        Initialize base transducer (called by subclasses).

        Subclasses should set the following attributes:
        - type : str
        - name : str
        - n_elements : int
        - elem_width : float (in meters)
        - elem_height : float (in meters)
        - no_sub_x : int
        - no_sub_y : int
        - frequency_Hz : Optional[float]
        """
        # These will be set by subclasses
        self.type: str = "unknown"
        self.name: str = "TransducerBase"
        self.n_elements: int = 0
        self.elem_width: float = 0.0
        self.elem_height: float = 0.0
        self.no_sub_x: int = 1
        self.no_sub_y: int = 1
        self.frequency_Hz: Optional[float] = None

        # State attributes
        self._apodization: Optional[np.ndarray] = None
        self._delays: Optional[np.ndarray] = None
        self.apodization_type: Optional[str] = None
        self.FoverD: Optional[float] = None
        self.speed_of_sound_mps: float = 1540.0  # Default for soft tissue

        # Geometry attributes (lazy-loaded via properties)
        self._element_centers: Optional[np.ndarray] = None
        self._sub_quad_verts: Optional[List[np.ndarray]] = None
        self._sub_area: Optional[float] = None
        self._sub_el_idx: Optional[List[int]] = None

    # =========================================================================
    # Properties: Lazy-loaded geometry
    # =========================================================================

    @property
    def element_centers(self) -> np.ndarray:
        """
        Element center positions in 3D space (meters).

        Returns
        -------
        ndarray, shape (n_elements, 3)
            X, Y, Z coordinates of element centers.
        """
        if self._element_centers is None:
            self._element_centers = self._compute_element_centers()
        return self._element_centers

    @property
    def sub_quad_verts(self) -> List[np.ndarray]:
        """
        Subdivision quad vertices for all patches.

        Returns
        -------
        list of ndarray
            Each element is shape (4, 3) with quad vertices.
        """
        if self._sub_quad_verts is None:
            self._sub_quad_verts, self._sub_area, self._sub_el_idx = (
                self._build_subdivisions()
            )
        return self._sub_quad_verts

    @property
    def sub_area(self) -> float:
        """
        Area of each subdivision patch (square meters).

        Returns
        -------
        float
            Patch area (same for all patches in a simple rectangular grid).
        """
        if self._sub_area is None:
            self.sub_quad_verts  # Trigger lazy load
        return self._sub_area

    @property
    def sub_el_idx(self) -> List[int]:
        """
        Element index for each subdivision patch.

        Returns
        -------
        list of int
            Maps each patch to its parent element index.
        """
        if self._sub_el_idx is None:
            self.sub_quad_verts  # Trigger lazy load
        return self._sub_el_idx

    @property
    def n_sub_patches(self) -> int:
        """
        Total number of subdivision patches.

        Returns
        -------
        int
            Number of patches across all elements.
        """
        return len(self.sub_quad_verts)

    # =========================================================================
    # State Properties: Apodization and Delays
    # =========================================================================

    @property
    def apodization(self) -> np.ndarray:
        """
        Per-element apodization weights.

        Returns
        -------
        ndarray, shape (n_elements,)
            Apodization weights [0, 1].
        """
        if self._apodization is None:
            self._apodization = np.ones(self.n_elements, dtype=float)
        return self._apodization

    @apodization.setter
    def apodization(self, weights: np.ndarray) -> None:
        """
        Set apodization weights.

        Parameters
        ----------
        weights : ndarray, shape (n_elements,)
            Apodization weights.
        """
        weights = validators.validate_apodization_weights(
            weights, self.n_elements
        )
        self._apodization = weights

    @property
    def delays(self) -> np.ndarray:
        """
        Per-element delays (seconds).

        Returns
        -------
        ndarray, shape (n_elements,)
            Delay values.
        """
        if self._delays is None:
            self._delays = np.zeros(self.n_elements, dtype=float)
        return self._delays

    @delays.setter
    def delays(self, delay_values: np.ndarray) -> None:
        """
        Set delays.

        Parameters
        ----------
        delay_values : ndarray, shape (n_elements,)
            Delay values in seconds.
        """
        delay_values = validators.validate_delays(
            delay_values, self.n_elements
        )
        self._delays = delay_values

    @property
    def tx_N_active(self) -> int:
        """
        Number of active (non-zero apodization) elements.

        Returns
        -------
        int
            Count of elements with apodization > 0.
        """
        return int(np.sum(self.apodization > 0))

    # =========================================================================
    # Abstract Methods: Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _compute_element_centers(self) -> np.ndarray:
        """
        Compute element center positions.

        Returns
        -------
        ndarray, shape (n_elements, 3)
            Element center coordinates in meters.
        """
        pass

    @abstractmethod
    def _build_subdivisions(
        self,
    ) -> Tuple[List[np.ndarray], float, List[int]]:
        """
        Build subdivision patches for all elements.

        Returns
        -------
        sub_quad_verts : list of ndarray
            Quad vertices for each patch.
        sub_area : float
            Patch area.
        sub_el_idx : list of int
            Element index for each patch.
        """
        pass

    @abstractmethod
    def compute_delays(
        self,
        focus_mm: np.ndarray,
        *,
        c: Optional[float] = None,
        inline: bool = True,
        plot: bool = False,
    ) -> np.ndarray:
        """
        Compute delays for focusing at a given focal point.

        Parameters
        ----------
        focus_mm : sequence of float
            Focal point coordinates in millimeters [x, z] or [x, y, z].
        c : float, optional
            Speed of sound in m/s. Default is 1540 (soft tissue).
        inline : bool, optional
            If True, store result in self.delays. Default is True.
        plot : bool, optional
            If True, plot the delays. Default is False.

        Returns
        -------
        ndarray, shape (n_elements,)
            Computed delays in seconds.
        """
        pass

    @abstractmethod
    def compute_apodization(
        self,
        focus_mm: np.ndarray,
        *,
        FoverD: Optional[float] = None,
        apodization_type: Optional[str] = None,
        plot: bool = False,
    ) -> np.ndarray:
        """
        Compute apodization weights for focusing at a given focal point.

        Parameters
        ----------
        focus_mm : sequence of float
            Focal point coordinates in millimeters [x, z] or [x, y, z].
        FoverD : float, optional
            F/D ratio for determining aperture size.
        apodization_type : str, optional
            Window type: 'none', 'rect', 'hanning', 'hamming', etc.
        plot : bool, optional
            If True, plot the apodization. Default is False.

        Returns
        -------
        ndarray, shape (n_elements,)
            Apodization weights [0, 1].
        """
        pass

    # =========================================================================
    # Concrete Methods: Provided by base class
    # =========================================================================

    def set_apodization(self, weights: np.ndarray) -> None:
        """
        Set per-element apodization weights.

        Parameters
        ----------
        weights : ndarray, shape (n_elements,)
            Apodization weights [0, 1].

        Raises
        ------
        ValueError
            If weights size doesn't match n_elements.
        """
        self.apodization = weights

    def set_delays(self, delays: np.ndarray) -> None:
        """
        Set per-element delays.

        Parameters
        ----------
        delays : ndarray, shape (n_elements,)
            Delay values in seconds.

        Raises
        ------
        ValueError
            If delays size doesn't match n_elements.
        """
        # Normalize to zero minimum
        delays_array = np.atleast_1d(delays).astype(float)
        delays_array = delays_array - np.min(delays_array)
        self.delays = delays_array

    def get_mesh(self) -> pv.PolyData:
        """
        Generate a PyVista mesh of the transducer surface.

        Returns
        -------
        pyvista.PolyData
            Mesh with 'Apodization' and 'Delays' as cell data.
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
        Visualize the transducer using PyVista.

        Parameters
        ----------
        window_size : tuple of int, optional
            Window size (width, height). Default is (800, 600).
        scalars : str, optional
            Scalar field to display: 'Apodization' or 'Delays'. Default is 'Apodization'.
        notebook : bool, optional
            If True, use Jupyter notebook rendering. Default is False.
        jupyter_backend : str, optional
            Jupyter backend ('static', 'trame', etc.). Only used if notebook=True.
        colorbar_title : str, optional
            Custom colorbar title.
        **kwargs
            Additional arguments passed to plotter.add_mesh().

        Raises
        ------
        ValueError
            If scalars is not 'Apodization' or 'Delays'.
        """
        if scalars not in ("Apodization", "Delays"):
            raise ValueError(
                f"scalars must be 'Apodization' or 'Delays', got '{scalars}'"
            )

        mesh = self.get_mesh()
        plotter = pv.Plotter(window_size=window_size, notebook=notebook)

        # Determine colormap and colorbar title
        if scalars == "Apodization":
            cmap = "cool"
            title = "Apodization" if colorbar_title is None else colorbar_title
            clim = [0, 1]
        else:  # Delays
            cmap = "rainbow"
            title = "Delays (s)" if colorbar_title is None else colorbar_title
            clim = None

        # Merge defaults with user kwargs
        default_kwargs = {
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
        }
        for key, value in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        plotter.add_mesh(mesh, **kwargs)
        plotter.add_axes()
        plotter.show_grid(
            font_size=10,
            xtitle="X (mm)",
            ytitle="Y (mm)",
            ztitle="Z (mm)",
            show_zlabels=False,
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
        Plot apodization weights (matplotlib).

        Parameters
        ----------
        apodization : ndarray, optional
            Custom apodization to plot. If None, uses self.apodization.
        figsize : tuple of int, optional
            Figure size (width, height). Default is (6, 5).
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates new figure.

        Returns
        -------
        ax or None
            If ax was provided, returns it. Otherwise returns None.
        """
        import matplotlib.pyplot as plt

        flag = False
        if apodization is None:
            apodization = self.apodization

        if ax is None:
            flag = True
            _, ax = plt.subplots(figsize=figsize)

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

        if flag:
            plt.tight_layout()
            plt.show()
            plt.close()
            return None
        else:
            return ax

    def plot_delays(
        self,
        delays: Optional[np.ndarray] = None,
        *,
        figsize: Tuple[int, int] = (6, 5),
        ax=None,
    ):
        """
        Plot delays (matplotlib).

        Parameters
        ----------
        delays : ndarray, optional
            Custom delays to plot. If None, uses self.delays.
        figsize : tuple of int, optional
            Figure size (width, height). Default is (6, 5).
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates new figure.

        Returns
        -------
        ax or None
            If ax was provided, returns it. Otherwise returns None.
        """
        import matplotlib.pyplot as plt

        flag = False
        if delays is None:
            delays = self.delays

        if ax is None:
            flag = True
            _, ax = plt.subplots(figsize=figsize)

        ax.plot(
            np.arange(self.n_elements),
            delays * 1e6,  # Convert to microseconds for display
            "k-",
            marker="o",
            markerfacecolor="r",
        )
        ax.set_title("Delays")
        ax.set_xlabel("Element #")
        ax.set_ylabel("Delay (µs)")
        ax.grid(True)

        if flag:
            plt.tight_layout()
            plt.show()
            plt.close()
            return None
        else:
            return ax

    def plot_delays_apodization(self, figsize: Tuple[int, int] = (10, 4)) -> None:
        """
        Plot delays and apodization side by side.

        Parameters
        ----------
        figsize : tuple of int, optional
            Figure size (width, height). Default is (10, 4).
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        self.plot_delays(ax=ax1)
        self.plot_apodization(ax=ax2)
        plt.tight_layout()
        plt.show()
        plt.close()

    def clean(self) -> None:
        """
        Clean up large arrays to free memory.

        This clears cached geometry arrays. They can be recomputed if needed.
        """
        self._element_centers = None
        self._sub_quad_verts = None
        self._sub_area = None
        self._sub_el_idx = None
        print(f"{self.name} cleaned up.")

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get a dictionary representation of the transducer state.

        Returns
        -------
        dict
            Dictionary with 'apodization', 'delays', 'apodization_type', 'FoverD'.
        """
        return {
            "apodization": self.apodization.copy(),
            "delays": self.delays.copy(),
            "apodization_type": self.apodization_type,
            "FoverD": self.FoverD,
        }

    def set_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Restore transducer state from a dictionary.

        Parameters
        ----------
        state : dict
            Dictionary with keys matching those from get_state_dict().
        """
        if "apodization" in state:
            self.apodization = state["apodization"]
        if "delays" in state:
            self.delays = state["delays"]
        if "apodization_type" in state:
            self.apodization_type = state["apodization_type"]
        if "FoverD" in state:
            self.FoverD = state["FoverD"]

    def __repr__(self) -> str:
        """Return a string representation of the transducer."""
        return f"{self.name}(n_elements={self.n_elements}, type='{self.type}')"

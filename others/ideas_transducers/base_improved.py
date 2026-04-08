"""
Improved abstract base class for all transducers using parameter-driven design.

This base class accepts geometric parameters and handles all patch generation,
curvature, rotation, and subdivision logic. Any transducer geometry can be
created by just providing parameters - no geometry computation needed in subclasses.

Parameters (all per-element unless specified):
  - n_elements: number of elements
  - element_centers_mm: (n_elements, 3) positions of element centers
  - element_width_mm: width in X (can be per-element)
  - element_height_mm: height in Y (can be per-element)
  - normal_vectors: (n_elements, 3) surface normal direction for each element
  - rotation_angles_deg: (n_elements,) rotation of patches in element plane
  - no_sub_x: subdivisions in X (can be per-element)
  - no_sub_y: subdivisions in Y (can be per-element)
  - frequency_Hz: operating frequency
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List, Union
import warnings
import numpy as np
import pyvista as pv
from time import time as TIME

from . import validators, geometry_utils


class TransducerBase(ABC):
    """
    Parameter-driven base class for all transducer types.

    This class accepts geometric parameters and automatically generates
    patch-based representations for any transducer geometry. Subclasses
    only need to compute parameters and call super().__init__().

    The key insight: ANY transducer geometry (linear, matrix, concave, annular,
    phased, custom) can be represented by:
      1. Element center positions (3D)
      2. Element surface orientations (normal vectors)
      3. Element dimensions (width, height)
      4. Element patch rotations (in-plane rotation)
      5. Subdivision counts (patches per dimension)

    By providing these parameters, the base class handles all geometry generation.
    """

    def __init__(
        self,
        *,
        n_elements: int,
        element_centers_mm: np.ndarray,
        element_width_mm: Union[float, np.ndarray],
        element_height_mm: Union[float, np.ndarray],
        normal_vectors: np.ndarray,
        rotation_angles_deg: Optional[Union[float, np.ndarray]] = None,
        no_sub_x: Union[int, np.ndarray] = 1,
        no_sub_y: Union[int, np.ndarray] = 1,
        frequency_Hz: Optional[float] = None,
    ) -> None:
        """
        Initialize transducer from parameters.

        Parameters
        ----------
        n_elements : int
            Total number of elements.
        element_centers_mm : ndarray, shape (n_elements, 3)
            3D center position of each element in millimeters.
        element_width_mm : float or ndarray
            Width of elements in X direction. Either scalar (all same) or
            shape (n_elements,) for per-element variation.
        element_height_mm : float or ndarray
            Height of elements in Y direction. Either scalar (all same) or
            shape (n_elements,) for per-element variation.
        normal_vectors : ndarray, shape (n_elements, 3)
            Unit normal vector for each element surface. Defines surface
            orientation (flat, curved, tilted, etc.).
        rotation_angles_deg : float or ndarray, optional
            Rotation angle of patches in element plane (degrees). Either
            scalar or shape (n_elements,). Default is 0.
        no_sub_x : int or ndarray, optional
            Subdivisions in X per element. Either scalar or shape (n_elements,).
            Default is 1.
        no_sub_y : int or ndarray, optional
            Subdivisions in Y per element. Either scalar or shape (n_elements,).
            Default is 1.
        frequency_Hz : float, optional
            Operating frequency in Hz.

        Raises
        ------
        ValueError
            If input shapes or values are invalid.
        """
        start_time = TIME()
        
        # Subclasses should set these before calling super().__init__()
        self.type: str = "unknown"
        self.name: str = "TransducerBase"

        # Store and validate basic parameters
        self.n_elements = validators.validate_integer(n_elements, "n_elements", 1)
        self.frequency_Hz = validators.validate_speed_of_sound(frequency_Hz) if frequency_Hz else None

        # Convert and validate element centers
        self.element_centers_mm = np.atleast_2d(element_centers_mm).astype(float)
        if self.element_centers_mm.shape != (n_elements, 3):
            raise ValueError(
                f"element_centers_mm must have shape ({n_elements}, 3), "
                f"got {self.element_centers_mm.shape}"
            )

        # Normalize and validate normal vectors
        self.normal_vectors = np.atleast_2d(normal_vectors).astype(float)
        if self.normal_vectors.shape != (n_elements, 3):
            raise ValueError(
                f"normal_vectors must have shape ({n_elements}, 3), "
                f"got {self.normal_vectors.shape}"
            )
        # Normalize to unit vectors
        norms = np.linalg.norm(self.normal_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.normal_vectors = self.normal_vectors / norms

        # Handle per-element or global dimensions
        self.element_width_mm = self._normalize_per_element_param(
            element_width_mm, n_elements, "element_width_mm"
        )
        self.element_height_mm = self._normalize_per_element_param(
            element_height_mm, n_elements, "element_height_mm"
        )

        # Validate dimensions are positive
        if np.any(self.element_width_mm <= 0) or np.any(self.element_height_mm <= 0):
            raise ValueError("Element dimensions must be positive")

        # Handle rotation angles
        if rotation_angles_deg is None:
            rotation_angles_deg = np.zeros(n_elements)
        self.rotation_angles_deg = self._normalize_per_element_param(
            rotation_angles_deg, n_elements, "rotation_angles_deg"
        )

        # Handle subdivision counts
        self.no_sub_x = self._normalize_per_element_param(
            no_sub_x, n_elements, "no_sub_x", dtype=int
        )
        self.no_sub_y = self._normalize_per_element_param(
            no_sub_y, n_elements, "no_sub_y", dtype=int
        )

        # Validate subdivisions are positive integers
        if np.any(self.no_sub_x < 1) or np.any(self.no_sub_y < 1):
            raise ValueError("Subdivisions must be >= 1")

        # State attributes
        self._apodization_per_patch: Optional[np.ndarray] = None
        self._delays: Optional[np.ndarray] = None
        self.apodization_type: Optional[str] = None
        self.FoverD: Optional[float] = None
        self.speed_of_sound_mps: float = 1540.0

        # Geometry attributes (lazy-loaded)
        self._sub_quad_verts: Optional[List[np.ndarray]] = None
        self._sub_area: Optional[float] = None
        self._sub_el_idx: Optional[List[int]] = None
        self._patch_to_element_map: Optional[List[int]] = None

        end_time = TIME()
        print(
            f"\n{self.name} initialized in {end_time - start_time:.4f} seconds "
            f"({self.n_elements} elements, "
            f"~{np.sum(self.no_sub_x * self.no_sub_y)} patches)"
        )

    @staticmethod
    def _normalize_per_element_param(
        param: Union[float, int, np.ndarray],
        n_elements: int,
        name: str,
        dtype: type = float,
    ) -> np.ndarray:
        """
        Convert parameter to per-element array.

        Parameters
        ----------
        param : scalar or ndarray
            Input parameter (scalar or array).
        n_elements : int
            Number of elements.
        name : str
            Parameter name for error messages.
        dtype : type
            Desired data type.

        Returns
        -------
        ndarray, shape (n_elements,)
            Parameter as array (repeated if scalar).
        """
        param_array = np.atleast_1d(param).astype(dtype)
        
        if param_array.shape[0] == 1:
            return np.tile(param_array, n_elements)
        elif param_array.shape[0] == n_elements:
            return param_array
        else:
            raise ValueError(
                f"{name} must be scalar or shape ({n_elements},), "
                f"got shape {param_array.shape}"
            )

    # =========================================================================
    # Properties: Lazy-loaded geometry
    # =========================================================================

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
            self._sub_quad_verts, self._sub_area, self._patch_to_element_map = (
                self._build_subdivisions()
            )
        return self._sub_quad_verts

    @property
    def sub_area(self) -> float:
        """
        Area of each subdivision patch (square millimeters).

        Note: This is average area. Actual area can vary per element
        if element_width_mm or element_height_mm vary.

        Returns
        -------
        float
            Average patch area (mm^2).
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
        if self._patch_to_element_map is None:
            self.sub_quad_verts  # Trigger lazy load
        return self._patch_to_element_map

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
    def apodization_per_patch(self) -> np.ndarray:
        """
        Apodization weights per subdivision patch (not per element!).

        Returns
        -------
        ndarray, shape (n_patches,)
            Apodization weights [0, 1] for each patch.
        """
        if self._apodization_per_patch is None:
            self._apodization_per_patch = np.ones(self.n_sub_patches, dtype=float)
        return self._apodization_per_patch

    @apodization_per_patch.setter
    def apodization_per_patch(self, weights: np.ndarray) -> None:
        """
        Set apodization weights per patch.

        Parameters
        ----------
        weights : ndarray, shape (n_patches,)
            Apodization weights per patch.
        """
        weights = validators.validate_apodization_weights(
            weights, self.n_sub_patches, "apodization_per_patch"
        )
        self._apodization_per_patch = weights

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
        Set delays per element.

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
    def apodization_per_element(self) -> np.ndarray:
        """
        Average apodization weight per element (computed from patches).

        Returns
        -------
        ndarray, shape (n_elements,)
            Average apodization per element.
        """
        apod_per_patch = self.apodization_per_patch
        apod_per_elem = np.zeros(self.n_elements)
        
        for patch_idx, elem_idx in enumerate(self.sub_el_idx):
            apod_per_elem[elem_idx] += apod_per_patch[patch_idx]
        
        # Divide by number of patches per element
        for elem_idx in range(self.n_elements):
            n_patches_in_elem = np.sum(np.array(self.sub_el_idx) == elem_idx)
            if n_patches_in_elem > 0:
                apod_per_elem[elem_idx] /= n_patches_in_elem
        
        return apod_per_elem

    @property
    def tx_N_active(self) -> int:
        """
        Number of active elements (average apodization > 0).

        Returns
        -------
        int
            Count of elements with non-zero average apodization.
        """
        return int(np.sum(self.apodization_per_element > 0))

    # =========================================================================
    # Core Geometry Building (Implemented in Base Class)
    # =========================================================================

    def _build_subdivisions(
        self,
    ) -> Tuple[List[np.ndarray], float, List[int]]:
        """
        Build rectangular subdivision patches for all elements.

        This method uses normal vectors and rotation angles to create
        3D quad patches for any element orientation.

        Returns
        -------
        sub_quad_verts : list of ndarray, shape (4, 3) each
            Quad vertices for each patch in 3D.
        sub_area : float
            Average patch area.
        patch_to_element : list of int
            Element index for each patch.
        """
        all_quads = []
        all_elem_idx = []
        
        for elem_idx in range(self.n_elements):
            center = self.element_centers_mm[elem_idx]
            width = self.element_width_mm[elem_idx]
            height = self.element_height_mm[elem_idx]
            normal = self.normal_vectors[elem_idx]
            rotation_deg = self.rotation_angles_deg[elem_idx]
            n_sub_x = self.no_sub_x[elem_idx]
            n_sub_y = self.no_sub_y[elem_idx]
            
            quads = self._build_element_patches(
                center, width, height, normal, rotation_deg, n_sub_x, n_sub_y
            )
            
            all_quads.extend(quads)
            all_elem_idx.extend([elem_idx] * len(quads))
        
        # Calculate average area
        if len(all_quads) > 0:
            areas = [np.linalg.norm(
                np.cross(quad[1] - quad[0], quad[3] - quad[0])
            ) / 2 for quad in all_quads]
            avg_area = np.mean(areas)
        else:
            avg_area = 0.0
        
        return all_quads, avg_area, all_elem_idx

    def _build_element_patches(
        self,
        center_mm: np.ndarray,
        width_mm: float,
        height_mm: float,
        normal: np.ndarray,
        rotation_deg: float,
        no_sub_x: int,
        no_sub_y: int,
    ) -> List[np.ndarray]:
        """
        Build patch quads for a single element with given orientation.

        This handles:
          1. Creating local rectangular grid
          2. Rotating patches by rotation_deg in local frame
          3. Transforming to global frame using normal vector
          4. Applying any curvature from element position offset

        Parameters
        ----------
        center_mm : ndarray, shape (3,)
            Element center position.
        width_mm : float
            Element width.
        height_mm : float
            Element height.
        normal : ndarray, shape (3,)
            Unit normal vector of element surface.
        rotation_deg : float
            Rotation angle in degrees (in element plane).
        no_sub_x : int
            Subdivisions in X.
        no_sub_y : int
            Subdivisions in Y.

        Returns
        -------
        list of ndarray
            Quad vertices for patches.
        """
        # Create local coordinate system for element
        local_x, local_y = self._create_local_frame(normal)
        
        rotation_rad = np.radians(rotation_deg)
        cos_r = np.cos(rotation_rad)
        sin_r = np.sin(rotation_rad)
        
        local_x_rot = cos_r * local_x - sin_r * local_y
        local_y_rot = sin_r * local_x + cos_r * local_y
        
        # Subdivision grid
        xs = np.linspace(-width_mm / 2, width_mm / 2, no_sub_x + 1)
        ys = np.linspace(-height_mm / 2, height_mm / 2, no_sub_y + 1)
        
        quads = []
        for i in range(no_sub_x):
            for j in range(no_sub_y):
                # Four corners in local coordinates
                corners_local = [
                    np.array([xs[i], ys[j]]),
                    np.array([xs[i + 1], ys[j]]),
                    np.array([xs[i + 1], ys[j + 1]]),
                    np.array([xs[i], ys[j + 1]]),
                ]
                
                # Transform to global coordinates
                corners_global = []
                for corner_local in corners_local:
                    corner_global = (
                        center_mm +
                        corner_local[0] * local_x_rot +
                        corner_local[1] * local_y_rot
                    )
                    corners_global.append(corner_global)
                
                quads.append(np.array(corners_global))
        
        return quads

    @staticmethod
    def _create_local_frame(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create orthonormal basis for element plane from normal vector.

        Parameters
        ----------
        normal : ndarray, shape (3,)
            Unit normal vector.

        Returns
        -------
        local_x : ndarray, shape (3,)
            X-axis of element plane.
        local_y : ndarray, shape (3,)
            Y-axis of element plane.
        """
        normal = normal / np.linalg.norm(normal)
        
        # Find a vector not parallel to normal
        if abs(normal[2]) < 0.9:
            up = np.array([0, 0, 1])
        else:
            up = np.array([1, 0, 0])
        
        # Create orthonormal basis
        local_x = np.cross(up, normal)
        local_x = local_x / np.linalg.norm(local_x)
        local_y = np.cross(normal, local_x)
        
        return local_x, local_y

    # =========================================================================
    # Abstract Methods: Subclasses Implement Computation
    # =========================================================================

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
        focus_mm : ndarray, shape (3,)
            Focal point coordinates in millimeters [x, y, z].
        c : float, optional
            Speed of sound in m/s. Default is 1540.
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
        Compute apodization weights for focusing.

        Parameters
        ----------
        focus_mm : ndarray, shape (3,)
            Focal point coordinates in millimeters [x, y, z].
        FoverD : float, optional
            F/D ratio for determining aperture size.
        apodization_type : str, optional
            Window type: 'none', 'rect', 'hanning', 'hamming', etc.
        plot : bool, optional
            If True, plot the apodization. Default is False.

        Returns
        -------
        ndarray, shape (n_elements,)
            Apodization weights per element [0, 1].
        """
        pass

    # =========================================================================
    # Concrete Methods: Utilities Provided by Base Class
    # =========================================================================

    def set_apodization_per_patch(self, weights: np.ndarray) -> None:
        """
        Set apodization weights per patch.

        Parameters
        ----------
        weights : ndarray, shape (n_patches,)
            Apodization weights.
        """
        self.apodization_per_patch = weights

    def set_delays(self, delays: np.ndarray) -> None:
        """
        Set per-element delays.

        Parameters
        ----------
        delays : ndarray, shape (n_elements,)
            Delay values in seconds.
        """
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
        apod_per_elem = self.apodization_per_element
        
        return geometry_utils.create_mesh_from_quads(
            self.sub_quad_verts,
            self.sub_el_idx,
            apod_per_elem,
            self.delays,
            scale_to_mm=False,  # Already in mm
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
            Window size (width, height).
        scalars : str, optional
            Scalar field: 'Apodization' or 'Delays'.
        notebook : bool, optional
            Use Jupyter notebook rendering.
        jupyter_backend : str, optional
            Jupyter backend ('static', 'trame', etc.).
        colorbar_title : str, optional
            Custom colorbar title.
        """
        if scalars not in ("Apodization", "Delays"):
            raise ValueError(
                f"scalars must be 'Apodization' or 'Delays', got '{scalars}'"
            )

        mesh = self.get_mesh()
        plotter = pv.Plotter(window_size=window_size, notebook=notebook)

        if scalars == "Apodization":
            cmap = "cool"
            title = "Apodization" if colorbar_title is None else colorbar_title
            clim = [0, 1]
        else:
            cmap = "rainbow"
            title = "Delays (s)" if colorbar_title is None else colorbar_title
            clim = None

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
        Plot apodization weights per element.

        Parameters
        ----------
        apodization : ndarray, optional
            Custom apodization array. If None, uses computed average per element.
        figsize : tuple of int, optional
            Figure size.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        ax or None
            Axes if provided, else None.
        """
        import matplotlib.pyplot as plt

        flag = False
        if apodization is None:
            apodization = self.apodization_per_element

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
        Plot delays per element.

        Parameters
        ----------
        delays : ndarray, optional
            Custom delays array. If None, uses self.delays.
        figsize : tuple of int, optional
            Figure size.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        ax or None
            Axes if provided, else None.
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
            delays * 1e6,
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
            Figure size.
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
        """
        self._sub_quad_verts = None
        self._sub_area = None
        self._patch_to_element_map = None
        print(f"{self.name} cleaned up.")

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get dictionary representation of transducer state.

        Returns
        -------
        dict
            State including apodization, delays, etc.
        """
        return {
            "apodization_per_patch": self.apodization_per_patch.copy(),
            "delays": self.delays.copy(),
            "apodization_type": self.apodization_type,
            "FoverD": self.FoverD,
        }

    def set_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Restore transducer state from dictionary.

        Parameters
        ----------
        state : dict
            State dictionary from get_state_dict().
        """
        if "apodization_per_patch" in state:
            self.apodization_per_patch = state["apodization_per_patch"]
        if "delays" in state:
            self.delays = state["delays"]
        if "apodization_type" in state:
            self.apodization_type = state["apodization_type"]
        if "FoverD" in state:
            self.FoverD = state["FoverD"]

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.name}(n_elements={self.n_elements}, type='{self.type}')"

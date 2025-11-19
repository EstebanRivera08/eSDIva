import warnings
from time import time as TIME

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# LinearArrayTransducer class


class LinearArrayTransducer:
    def __init__(
        self,
        *,
        n_elements,
        element_width_mm,
        element_height_mm,
        kerf_mm,
        no_sub_x,
        no_sub_y,
        elevation_focus_mm=None,
        frequency_Hz=None,
    ):
        """
        Defines a linear array transducer geometry with optional elevation focusing.

        Parameters
        ----------
        n_elements : int
            Number of elements in the array.
        element_width : float
            Width of each element (m).
        element_height : float
            Height of each element (m).
        kerf : float
            Gap between elements (m).
        elevation_focus : float or None
            Elevation focus distance (m). If None, elements are flat.
        no_sub_x, no_sub_y : int
            Number of subdivisions (patches) in x (element width) and y (element height).

        """
        start_time = TIME()
        self.type = "linear"
        self.name = "LinearArrayTransducer"

        if kerf_mm < 0:
            raise ValueError("Kerf must be non-negative.")
        if no_sub_x <= 0 or no_sub_y <= 0:
            raise ValueError("Number of subdivisions must be positive.")
        # no_sub must be positive integers
        if not isinstance(no_sub_x, int) or not isinstance(no_sub_y, int):
            raise ValueError("Number of subdivisions must be positive integers.")
        if element_height_mm <= 0 or element_width_mm <= 0:
            raise ValueError("Element dimensions must be positive.")

        if elevation_focus_mm is None:
            elevation_focus_mm = 0

        if elevation_focus_mm < 0:
            raise ValueError("Elevation focus must be non-negative or None.")

        element_height, element_width = (
            element_height_mm * 1e-3,
            element_width_mm * 1e-3,
        )

        kerf, elevation_focus = kerf_mm * 1e-3, elevation_focus_mm * 1e-3
        self.n_elements = n_elements
        self.elem_width = element_width  # m
        self.elem_height = element_height  # m
        self.kerf = kerf  # m
        self.pitch = element_width + kerf  # m
        self.elev_focus = elevation_focus  # m
        self.no_sub_x = no_sub_x
        self.no_sub_y = no_sub_y

        if elevation_focus is not None and elevation_focus > 0 and no_sub_y < 2:
            raise ValueError(
                "Elevation focus requires at least 2 subdivisions in y-dir to model elevation focusing."
            )

        if frequency_Hz is not None:
            self.fc = frequency_Hz
        else:
            self.fs = 1e6
            print("Warning: No central frequency provided. Defaulting to 1 MHz.")

        # Per-element apodization weights and delay placeholders

        self.apodization = np.ones(n_elements, dtype=float)
        self.delays = np.zeros(n_elements, dtype=float)
        self.tx_N_active = int(np.sum(self.apodization > 0))
        self.apodization_type = None
        self.FoverD = None

        # Compute element centers along x-axis
        total_width = n_elements * element_width + (n_elements - 1) * kerf
        start_x = -total_width / 2 + element_width / 2
        self.element_centers = np.array(
            [
                [start_x + i * (element_width + kerf), 0.0, 0.0]
                for i in range(n_elements)
            ]
        )

        # Build subdivisions boundaries and areas, apply elevation curvature
        self.sub_quad_verts, self.sub_area, self.sub_el_idx = self._build_subdivisions()

        end_time = TIME()
        print(
            f"\nLinearArrayTransducer initialized in {end_time - start_time:.4f} seconds."
        )

    def _build_subdivisions(self):
        """
        Generate vertices for each subdivision quad of each element, applying elevation focus curvature if set.
        Returns
        -------
        sub_quad_verts : list of arrays (4x3) for each patch quad vertices
        sub_area : float, area of each patch
        sub_el_idx : list mapping each patch to its element index
        """
        # Local grid edges in element coordinates
        xs = np.linspace(-self.elem_width / 2, self.elem_width / 2, self.no_sub_x + 1)
        ys = np.linspace(-self.elem_height / 2, self.elem_height / 2, self.no_sub_y + 1)

        patch_area = (self.elem_width / self.no_sub_x) * (
            self.elem_height / self.no_sub_y
        )

        quads = []
        el_indices = []
        for idx, center in enumerate(self.element_centers):
            for i in range(self.no_sub_x):
                for j in range(self.no_sub_y):
                    # four corners of the patch in local coords
                    corners_local = np.array(
                        [
                            [xs[i], ys[j], 0.0],
                            [xs[i + 1], ys[j], 0.0],
                            [xs[i + 1], ys[j + 1], 0.0],
                            [xs[i], ys[j + 1], 0.0],
                        ]
                    )
                    # translate to global x,y
                    corners = corners_local.copy()
                    corners[:, 0] += center[0]
                    corners[:, 1] += center[1]
                    # apply elevation curvature in z
                    if self.elev_focus is not None and self.elev_focus > 0:
                        y_vals = corners[:, 1]
                        z_offset = self.elev_focus - np.sqrt(
                            np.clip(self.elev_focus**2 - y_vals**2, 0, None)
                        )
                        corners[:, 2] += z_offset
                    else:
                        corners[:, 2] += center[2]
                    quads.append(corners)
                    el_indices.append(idx)
        return quads, patch_area, el_indices

    def compute_apodization(
        self,
        focus_mm,
        *,
        FoverD=None,
        apodization_type=None,
        plot=False,
        equiv_energy=False,
    ):
        """
        Compute per‑element apodization for focusing at a given spot.

        Parameters
        ----------
        focus_mm : sequence of three floats (x, y, z)
            Lateral (x) and axial (z) coordinates of the focus, in millimeters
            relative to the array center.
        apodization_type : {'none', 'rect', 'hanning', 'hamming'}
            Type of window to apply.
        plot : bool
            If True, show a quick plot of the resulting apodization.

        Returns
        -------
        apod : ndarray, shape (N_elements,)
            Normalized apodization weights.
        """
        defined_types = {None, "none", "rect", "hanning", "hamming"}
        if apodization_type not in defined_types:
            raise ValueError(
                f"Unknown apodization_type '{apodization_type}' \n \
                             Must be one of {defined_types}"
            )

        # Unpack and convert to meters
        if isinstance(focus_mm, (tuple, list)):
            if len(focus_mm) not in (2, 3):
                raise ValueError(
                    "Focus must be a sequence of 2 [x,z] or 3 [x,y,z] values."
                )
            else:
                focus = np.array(focus_mm) * 1e-3

        if isinstance(focus_mm, np.ndarray):
            focus_mm = np.squeeze(focus_mm)
            if focus_mm.ndim != 1 or focus_mm.shape[0] not in (2, 3):
                raise ValueError(
                    "Focus must be a 1D array of 2 [x,z] or 3 [x,y,z] values."
                )
            else:
                focus = focus_mm * 1e-3

        if focus.shape == (3,):
            x_foc, y_foc, z_foc = focus[0], focus[1], focus[2]
            # print(f"Focus: {focus_mm[0]:.3f} mm, {focus_mm[1]:.3f} mm, {focus_mm[2]:.3f} mm")
        elif focus.shape == (2,):
            x_foc, z_foc = focus[0], focus[1]
            y_foc = 0
            # print(f"Focus: {focus_mm[0]:.3f} mm, 0.000 mm, {focus_mm[1]:.3f} mm")

        if z_foc <= 0:
            raise ValueError("Wrong focus: z_foc must be positive")

        N = self.n_elements
        pitch = self.elem_width + self.kerf  # element pitch in meters
        total_ap = N * pitch  # total array aperture (m)

        if apodization_type is None:
            print("Warning: No apodization type provided. Using 'rect'.")
            apodization_type = "rect"

        if apodization_type == "none":
            apod = np.ones(N, dtype=float)

        else:
            # require ratio_F_over_D property
            if FoverD is not None:
                self.FoverD = FoverD

            if self.FoverD is None:
                print("Warning: F/D ratio not set. Using default value of 1.0.")
                self.FoverD = 1.0

            # physical extent (in meters) of active aperture for given F/D
            D = z_foc / self.FoverD
            # how many elements that corresponds to (must be even)
            if self.n_elements % 2 == 1:
                N_virt = int(round((D / total_ap) * N / 2) * 2 + 1)
            else:
                N_virt = int(round((D / total_ap) * N / 2) * 2)

            # window factor for equivalent energy in Hanning/Hamming
            if equiv_energy:
                # If we want to keep the same energy as a rectangular window
                # we need to scale the Hanning/Hamming window by a factor to use more elements
                factor = {"rect": 1.0, "hanning": 0.5, "hamming": 0.54}[
                    apodization_type
                ]
            else:
                factor = 1

            N_ext = int(np.round(N_virt / factor))

            # clamp and warn if outside
            if N_ext > N:
                warnings.warn("Focus outside imaging window: using full aperture")
                N_ext = N

            # build window
            if apodization_type == "rect":
                wins = np.ones(N_ext)
            elif apodization_type == "hanning":
                wins = np.hanning(N_ext)
            elif apodization_type == "hamming":
                wins = np.hamming(N_ext)
            else:
                raise ValueError(f"Unknown apodization_type '{apodization_type}'")

            # now slide this window so its center aligns with x_foc
            # compute how many elements to shift
            shift_elems = int(np.round(x_foc / pitch)) - 1
            # the shift must be between 0 and N-1
            if shift_elems < -(N - 1) // 2:
                shift_elems = -(N - 1) // 2
            if shift_elems > (N - 1) // 2:
                shift_elems = (N - 1) // 2 + 1

            center = (N_ext - 1) // 2 - shift_elems
            idxs = np.arange(N_ext) - center + N // 2

            # only keep those inside the real array
            valid = (idxs >= 0) & (idxs < N)
            apod = np.zeros(N)
            apod[idxs[valid]] = wins[valid]

        # optionally plot
        if plot:
            self.plot_apodization()

        # save into object for later reference
        self.apodization = apod
        self.apodization_type = apodization_type
        self.tx_N_active = int(np.sum(apod > 0))
        return apod

    def plot_apodization(self, apodization=None, *, figsize=(6, 5), ax=None):
        """
        Plot the current apodization weights.
        """
        flag = False
        if apodization is None:
            apodization = self.apodization

        if ax is None:
            flag = True
            fig, ax = plt.subplots(figsize=figsize)

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
            plt.show()
            plt.close()
        else:
            return ax

    def compute_delays(self, focus_mm, *, c=None, inline=True, plot=False):
        """
        Compute per-element delays for focusing at a given spot.

        Parameters
        ----------
        focus_mm : sequence of two floats (x, z)
            Lateral (x) and axial (z) coordinates of the focus, in millimeters
            relative to the array center.

        Returns
        -------
        delays : ndarray, shape (N_elements,)
            Delays in seconds.
        """

        if c is None:
            c = 1540.0
            print("Warning: No speed of sound provided. Defaulting to 1540 m/s.")

        # Unpack and convert to meters
        focus = np.array(focus_mm) * 1e-3

        # Compute distances from each element to the focus point
        delays = np.linalg.norm(self.element_centers - focus, axis=1) / c

        # Compute delays based on the speed of sound in soft tissue
        delays = -delays + delays.max()  # time delays for focusing
        # delays = -delays + delays.min()  # time delays for focusing

        # optionally plot
        if inline:
            self.delays = delays
        if plot:
            self.plot_delays()

        return delays

    def plot_delays(self, delays=None, *, figsize=(6, 5), ax=None):
        """
        Plot the current delays.
        """
        flag = False
        if delays is None:
            delays = self.delays

        if ax is None:
            flag = True
            fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            np.arange(self.n_elements),
            delays * 1e6,
            "k-",
            marker="o",
            markerfacecolor="r",
        )
        ax.set_title("Delays")
        ax.set_xlabel("Element #")
        ax.set_ylabel("Delay (us)")
        ax.grid(True)

        if flag:
            plt.show()
            plt.close()
        else:
            return ax

    def plot_delays_apodization(self):
        """
        Plot the current delays and apodization side by side.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.plot_delays(ax=ax1)
        self.plot_apodization(ax=ax2)
        plt.tight_layout()
        plt.show()
        plt.close()

    def set_apodization(self, weights):
        """Set per-element apodization weights (length = n_elements)."""
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != self.n_elements:
            raise ValueError(
                f"Apodization array must match number of elements. Input size: {weights.size}, expected: {self.n_elements}"
            )
        self.apodization = weights

    def set_delays(self, delays):
        """Set per-element delays in seconds (length = n_elements)."""
        delays = np.asarray(delays, dtype=float)
        if delays.shape[0] != self.n_elements:
            raise ValueError(
                f"Delay array must match number of elements. Input size: {delays.size}, expected: {self.n_elements}"
            )
        self.delays = delays

    def get_mesh(self):
        """
        Returns a PyVista PolyData mesh of all subdivided quads, with per-cell apodization scalars.
        """
        # Aggregate all points and faces
        verts = []
        faces = []
        scalars = []
        scalars2 = []  # delays
        pt_index = 0
        for quad, el_idx in zip(self.sub_quad_verts, self.sub_el_idx):
            # quad is 4x3 array, create face [4, p0, p1, p2, p3]
            verts.extend(quad.tolist())
            face = [4, pt_index, pt_index + 1, pt_index + 2, pt_index + 3]
            faces.append(face)
            scalars.append(self.apodization[el_idx])
            scalars2.append(self.delays[el_idx])
            pt_index += 4
        # Flatten verts and faces
        verts = np.array(verts) * 1e3  # Convert to mm for visualization
        faces_flat = np.hstack(faces)
        mesh = pv.PolyData(verts, faces_flat)
        mesh.cell_data["Apodization"] = np.array(scalars)
        mesh.cell_data["Delays"] = np.array(scalars2)
        return mesh

    def show(
        self,
        *,
        window_size=[800, 600],
        scalars="Apodization",
        notebook=False,
        jupyter_backend=None,
        colorbar_title=None,
        **kwargs,
    ):
        """
        Visualize the transducer surface mesh and apodization with PyVista.
        """
        mesh = self.get_mesh()
        plotter = pv.Plotter(window_size=window_size, notebook=notebook)

        if scalars == "Apodization":
            title_name = "Apodization"
            cmap = "cool"
        elif scalars == "Delays":
            title_name = "Delays (s)"
            cmap = "rainbow"
        else:
            raise ValueError("Scalars must be 'Apodization' or 'Delays'")

        if colorbar_title is not None:
            title_name = colorbar_title

        default_kwargs = {
            "scalars": scalars,
            "cmap": cmap,
            "clim": [0, 1] if scalars == "Apodization" else None,
            "show_scalar_bar": True,
            "scalar_bar_args": {
                "title": title_name,
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

        plotter.add_mesh(
            mesh,
            **kwargs,
        )

        plotter.add_axes()
        plotter.show_grid(
            font_size=10,
            xtitle="X (mm)",
            ytitle="Y (mm)",
            ztitle="Z (mm)",
            show_zlabels=False,
        )
        plotter.camera_position = [
            (12.520367408261166, 13.689471886505752, 13.940982550648721),
            (1.4163759408294876, 0.20198691702220328, -0.9914130664803784),
            (-0.5077013315077692, -0.41679734969120574, 0.7540022064129689),
        ]
        if jupyter_backend is not None:
            plotter.show(jupyter_backend=jupyter_backend)
        else:
            plotter.show()
        # plotter.close()

    def clean(self):
        """
        Clean up the transducer object by removing large arrays.
        """
        self.sub_quad_verts = None
        self.sub_area = None
        self.sub_el_idx = None
        self.element_centers = None
        self.apodization = None
        self.delays = None
        print("Transducer cleaned up.")

    def __repr__(self):
        params = {
            "n_elements": self.n_elements,
            "elem_width_mm": self.elem_width * 1e3,
            "elem_height_mm": self.elem_height * 1e3,
            "kerf_mm": self.kerf * 1e3,
            "elev_focus_mm": self.elev_focus * 1e3 if self.elev_focus else None,
            "no_sub_x": self.no_sub_x,
            "no_sub_y": self.no_sub_y,
            "fc_Hz": self.fc,
            "Apod type": self.apodization_type,
            "tx_N_active": self.tx_N_active,
            "FoverD": self.FoverD,
        }
        parts = [f"{k}={v}" for k, v in params.items()]
        return f"{self.__class__.__name__}({', '.join(parts)})"

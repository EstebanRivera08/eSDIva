import warnings
from time import time as TIME

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.signal import windows

warnings.filterwarnings("ignore", category=UserWarning)


def create_ellipse_mask(nx, ny):
    """Generate a 2D elliptical mask (similar to MATLAB's createEllipseMask)."""
    y, x = np.ogrid[-1 : 1 : complex(0, ny), -1 : 1 : complex(0, nx)]
    mask = x**2 + y**2 <= 1

    return mask.astype(float)


class MatrixArrayTransducer:
    """
    Defines a 2D matrix (multi-row) array transducer similar to FieldII's xdc_linear_multirow.
    """

    def __init__(
        self,
        *,
        N_elem_x,
        N_elem_y,
        elem_width_mm,
        elem_height_mm,
        kerf_x_mm,
        kerf_y_mm,
        no_sub_x,
        no_sub_y,
        frequency_Hz=None,
        dir_angle_deg=30,
    ):
        start_time = TIME()
        # Convert mm to meters

        self.type = "matrix"
        self.name = "MatrixArrayTransducer"
        self.n_elem_x = N_elem_x
        self.n_elem_y = N_elem_y
        self.n_elements = N_elem_x * N_elem_y

        if kerf_x_mm < 0 or kerf_y_mm < 0:
            raise ValueError("Kerf must be non-negative.")
        if no_sub_x <= 0 or no_sub_y <= 0:
            raise ValueError("Number of subdivisions must be positive.")
        # no_sub must be positive integers
        if not isinstance(no_sub_x, int) or not isinstance(no_sub_y, int):
            raise ValueError("Number of subdivisions must be positive integers.")
        if elem_height_mm <= 0 or elem_width_mm <= 0:
            raise ValueError("Element dimensions must be positive.")

        self.elem_width = elem_width_mm * 1e-3
        self.elem_height = elem_height_mm * 1e-3

        self.kerf_x = kerf_x_mm * 1e-3
        self.kerf_y = kerf_y_mm * 1e-3
        self.pitch_x = self.elem_width + self.kerf_x
        self.pitch_y = self.elem_width + self.kerf_x
        self.no_sub_x = no_sub_x
        self.no_sub_y = no_sub_y
        self.fc = frequency_Hz or 1.0
        self.dir_angle_deg = dir_angle_deg

        # Per-element apodization weights and delay placeholders
        self.apodization_type = None
        self.FoverD = None
        self.apodization = np.ones(self.n_elements, dtype=float)
        self.delays = np.zeros(self.n_elements, dtype=float)

        # compute element centers in x and y
        total_w = self.n_elem_x * self.elem_width + (self.n_elem_x - 1) * self.kerf_x
        total_h = self.n_elem_y * self.elem_height + (self.n_elem_y - 1) * self.kerf_y
        start_x = -total_w / 2 + self.elem_width / 2
        start_y = -total_h / 2 + self.elem_height / 2
        centers = []

        for iy in range(self.n_elem_y):
            y = start_y + iy * (self.elem_width + self.kerf_y)
            for ix in range(self.n_elem_x):
                x = start_x + ix * (self.elem_width + self.kerf_x)
                z = 0.0
                centers.append([x, y, z])

        self.element_centers = np.array(centers)
        # subdivisions
        self.sub_quad_verts = []
        self.sub_area = []
        self.sub_el_idx = []
        for idx, center in enumerate(self.element_centers):
            xs = np.linspace(
                -self.elem_width / 2, self.elem_width / 2, self.no_sub_x + 1
            )
            ys = np.linspace(
                -self.elem_height / 2, self.elem_height / 2, self.no_sub_y + 1
            )
            for i in range(self.no_sub_x):
                for j in range(self.no_sub_y):
                    corners_local = np.array(
                        [
                            [xs[i], ys[j], 0.0],
                            [xs[i + 1], ys[j], 0.0],
                            [xs[i + 1], ys[j + 1], 0.0],
                            [xs[i], ys[j + 1], 0.0],
                        ]
                    )
                    corners = corners_local + center
                    self.sub_quad_verts.append(corners)
                    self.sub_area.append(
                        (self.elem_width / self.no_sub_x)
                        * (self.elem_height / self.no_sub_y)
                    )
                    self.sub_el_idx.append(idx)

        print(
            f"MatrixArrayTransducer initialized with {self.n_elements} elements and {len(self.sub_quad_verts)} patches."
        )
        end_time = TIME()
        print(f"\nTransducer initialized in {end_time - start_time:.4f} seconds.")

    def compute_apodization(
        self,
        focus_mm,
        *,
        FoverD=None,
        apodization_type="circular",
        plot=False,
        inline=True,
    ):
        """
        Compute per‑element apodization for focusing at a given spot.

        Parameters
        ----------
        focus_mm : sequence of three floats (x, y, z)
            Lateral (x) and axial (z) coordinates of the focus, in millimeters
            relative to the array center.
        apodization_type : {'none', 'rect', 'circular', 'hanning', 'hamming'}
            Type of window to apply.
        plot : bool
            If True, show a quick plot of the resulting apodization.

        Returns
        -------
        apod : ndarray, shape (N_elements,)
            Normalized apodization weights.
        """
        # Unpack and convert to meters
        focus = np.array(focus_mm) * 1e-3
        N_x = self.n_elem_x
        N_y = self.n_elem_y

        if focus.shape == (3,):
            x_foc, y_foc, z_foc = focus[0], focus[1], focus[2]
            # print(f"Focus: {focus_mm[0]:.3f} mm, {focus_mm[1]:.3f} mm, {focus_mm[2]:.3f} mm")
        else:
            raise ValueError("Focus must be a 3D coordinate (x, y, z)")

        if z_foc <= 0:
            print("z_foc is negative. Setting Diverging waves...")

        if apodization_type is None:
            pass
        elif apodization_type == "none":
            apod = np.ones((N_x, N_y))
        else:
            # require ratio_F_over_D property
            if FoverD is not None:
                self.FoverD = FoverD

            if self.FoverD is None:
                print("Warning: F/D ratio not set. Using default value of 1.0.")
                self.FoverD = 1.0

            d_tx = 2 * abs(z_foc) * np.tan(np.radians(self.dir_angle_deg)) / self.FoverD
            Nvx = round(d_tx / (self.pitch_x))
            Nvy = round(d_tx / (self.pitch_y))
            if Nvx % 2 == 0:
                Nvx += 1
            if Nvy % 2 == 0:
                Nvy += 1

            if apodization_type == "rect":
                profile = np.ones((Nvx, Nvy))
            elif apodization_type == "circular":
                # ellipse mask
                Y, X = np.ogrid[:Nvx, :Nvy]
                cx, cy = (Nvx - 1) / 2, (Nvy - 1) / 2
                a, b = Nvx / 2, Nvy / 2
                mask = ((X - cx) ** 2 / a**2 + (Y - cy) ** 2 / b**2) <= 1
                profile = mask
            elif apodization_type == "hanning":
                profile = np.outer(windows.hann(Nvx), windows.hann(Nvy))
            elif apodization_type == "hamming":
                profile = np.outer(windows.hamming(Nvx), windows.hamming(Nvy))
            else:
                raise ValueError(
                    "Invalid apodization type. Must be one of: 'none', 'rect', 'circular', 'hanning', 'hamming'."
                )

            # compute shifts
            sx = int(np.round(x_foc / self.elem_width))
            sy = int(np.round(y_foc / self.elem_height))
            ix = np.arange(Nvx) - (Nvx - 1) // 2 + (N_x // 2) + sx
            iy = np.arange(Nvy) - (Nvy - 1) // 2 + (N_y // 2) + sy
            valid_x = (ix >= 0) & (ix < N_x)
            valid_y = (iy >= 0) & (iy < N_y)
            apod = np.zeros((N_x, N_y))
            apod[np.ix_(ix[valid_x], iy[valid_y])] = profile[np.ix_(valid_x, valid_y)]

        if plot:
            self.plot_apodization(apod.T.flatten())
        if inline:
            self.apodization = apod.T.flatten()
            self.apodization_type = apodization_type
        return apod.T.flatten()

    def compute_delays(self, *, focus_mm, c=None, plot=False, inline=True):
        """
        Compute per-element delays for focusing at a given spot.

        Parameters
        ----------
        focus_mm : sequence of two floats (x,y, z)
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

        if focus.shape != (3,):
            raise ValueError("Focus must be a 3D coordinate (x, y, z)")

        # Compute distances from each element to the focus point
        delays = np.linalg.norm(self.element_centers - focus, axis=1) / c

        # Compute delays based on the speed of sound in soft tissue
        if focus[2] <= 0:
            delays = delays - np.min(delays)  # time delays for diverging waves
        else:
            delays = delays.max() - delays  # time delays for focusing (in microseconds)

        if inline:
            self.delays = delays
        # optionally plot
        if plot:
            self.plot_delays(delays)

        return delays

    def plot_apodization(self, apodization=None, *, figsize=(6, 5), ax=None, **kwargs):
        flag = False
        if apodization is None:
            apodization = self.apodization

        if ax is None:
            flag = True
            fig, ax = plt.subplots(figsize=figsize)

        ax.imshow(
            apodization.reshape((self.n_elem_x, self.n_elem_y)),
            cmap="cool",
            vmin=0,
            vmax=1,
            **kwargs,
        )
        ax.set_title("Apodization")
        ax.set_xlabel("Element x #")
        ax.set_ylabel("Element y #")
        ax.grid(True)

        if flag:
            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            return ax

    def plot_delays(self, delays=None, *, figsize=(6, 5), ax=None, **kwargs):
        flag = False

        if delays is None:
            delays = self.delays

        if ax is None:
            flag = True
            fig, ax = plt.subplots(figsize=figsize)

        ax.imshow(
            delays.reshape((self.n_elem_x, self.n_elem_y)) * 1e6, cmap="jet", **kwargs
        )
        ax.set_title("Delays")
        ax.set_xlabel("Element #")
        ax.set_ylabel("Delay (us)")
        ax.grid(True)

        if flag:
            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            return ax

    def plot_delays_apodization(self, figsize=(8, 4)):
        """
        Plot the current delays and apodization side by side.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        self.plot_delays(ax=ax1)
        self.plot_apodization(ax=ax2)
        plt.tight_layout()
        plt.show()
        plt.close()

    def set_apodization(self, weights):
        weights = np.asarray(weights, dtype=float)
        if weights.size != self.n_elem_x * self.n_elem_y:
            raise ValueError(
                f"Apodization must match total elements. Input size: {weights.size}, expected: {self.n_elem_x * self.n_elem_y}"
            )
        self.apodization = weights

    def set_delays(self, delays):
        delays = np.asarray(delays, dtype=float)
        if delays.size != self.n_elem_x * self.n_elem_y:
            raise ValueError(
                f"Delays must match total elements. Input size: {delays.size}, expected: {self.n_elem_x * self.n_elem_y}"
            )
        self.delays = delays - np.min(delays)  # Normalize to min delay

    def get_mesh(self):
        verts = []
        faces = []
        scalars = []
        scalars2 = []  # delays
        pt_index = 0
        for quad, el_idx in zip(self.sub_quad_verts, self.sub_el_idx):
            verts.extend(quad.tolist())
            faces.append([4, pt_index, pt_index + 1, pt_index + 2, pt_index + 3])
            scalars.append(self.apodization[el_idx])
            scalars2.append(self.delays[el_idx])
            pt_index += 4
        verts = np.array(verts) * 1e3
        mesh = pv.PolyData(verts, np.hstack(faces))
        mesh.cell_data["Apodization"] = np.array(scalars)
        mesh.cell_data["Delays"] = np.array(scalars2) * 1e-6  # in seconds
        return mesh

    def show(
        self,
        *,
        window_size=[800, 600],
        scalars="Apodization",
        notebook=False,
        jupyter_backend=None,
        **kwargs,
    ):
        """
        Visualize the transducer surface mesh and apodization with PyVista.
        """
        mesh = self.get_mesh()
        plotter = pv.Plotter(window_size=window_size, notebook=notebook)

        if scalars == "Apodization":
            title = "Apodization"
            cmap = "cool"
        elif scalars == "Delays":
            title = "Delays (s)"
            cmap = "rainbow"
        else:
            raise ValueError("Scalars must be 'Apodization' or 'Delays'")

        default_kwargs = {
            "scalars": scalars,
            "cmap": cmap,
            "clim": [0, 1] if scalars == "Apodization" else None,
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

        mesh = self.get_mesh()
        plotter = pv.Plotter(notebook=notebook)
        plotter.add_mesh(
            mesh,  # Convert to mm for visualization
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
            (16.72465241530815, 20.611591228182785, 26.54115950699113),
            (1.31674789370372, 1.5498150789167457, -1.4859004666360698),
            (-0.568581023901881, -0.5004125236360828, 0.6529187740039761),
        ]
        plotter.show(jupyter_backend=jupyter_backend)
        plotter.close()

    def clean(self):
        """
        Clean up the transducer object by removing large arrays.
        """
        self.apodization = None
        self.delays = None
        self.element_centers = None
        self.sub_quad_verts = None
        self.sub_area = None
        self.sub_el_idx = None
        print("Transducer cleaned up.")

    def __repr__(self):
        params = {
            "n_elem_x": self.n_elem_x,
            "n_elem_y": self.n_elem_y,
            "elem_width_mm": self.elem_width * 1e3,
            "kerf_x_mm": self.kerf_x * 1e3,
            "kerf_y_mm": self.kerf_y * 1e3,
            "no_sub_x": self.no_sub_x,
            "no_sub_y": self.no_sub_y,
            "fc_Hz": self.fc,
        }
        parts = [f"{k}={v}" for k, v in params.items()]
        return f"{self.__class__.__name__}({', '.join(parts)})"

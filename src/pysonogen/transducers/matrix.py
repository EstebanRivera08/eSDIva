import warnings
from time import time as TIME

import numpy as np
import pyvista as pv
from scipy.signal import windows

from .base import BaseTransducer

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
        self.n_elem_x = N_elem_x
        self.n_elem_y = N_elem_y
        self.n_elements = N_elem_x * N_elem_y
        self.el_w = elem_width_mm * 1e-3
        self.el_h = elem_height_mm * 1e-3
        self.kerf_x = kerf_x_mm * 1e-3
        self.kerf_y = kerf_y_mm * 1e-3
        self.pitch_x = self.el_w + self.kerf_x
        self.pitch_y = self.el_w + self.kerf_x
        self.no_sub_x = no_sub_x
        self.no_sub_y = no_sub_y
        self.fc = frequency_Hz or 1.0
        self.dir_angle_deg = dir_angle_deg

        # Per-element apodization weights and delay placeholders
        self.apodization_type = None
        self.F_over_D = None
        self.apodization = np.ones(self.n_elements, dtype=float)
        self.delays = np.zeros(self.n_elements, dtype=float)

        # compute element centers in x and y
        total_w = self.n_elem_x * self.el_w + (self.n_elem_x - 1) * self.kerf_x
        total_h = self.n_elem_y * self.el_h + (self.n_elem_y - 1) * self.kerf_y
        start_x = -total_w / 2 + self.el_w / 2
        start_y = -total_h / 2 + self.el_h / 2
        centers = []

        for iy in range(self.n_elem_y):
            y = start_y + iy * (self.el_w + self.kerf_y)
            for ix in range(self.n_elem_x):
                x = start_x + ix * (self.el_w + self.kerf_x)
                z = 0.0
                centers.append([x, y, z])

        self.element_centers = np.array(centers)
        # subdivisions
        self.sub_quad_verts = []
        self.sub_area = []
        self.sub_el_idx = []
        for idx, center in enumerate(self.element_centers):
            row = idx // self.n_elem_x
            xs = np.linspace(-self.el_w / 2, self.el_w / 2, self.no_sub_x + 1)
            ys = np.linspace(-self.el_h / 2, self.el_h / 2, self.no_sub_y + 1)
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
                        (self.el_w / self.no_sub_x) * (self.el_h / self.no_sub_y)
                    )
                    self.sub_el_idx.append(idx)

        print(
            f"MatrixArrayTransducer initialized with {self.n_elements} elements and {len(self.sub_quad_verts)} patches."
        )
        end_time = TIME()
        print(f"Transducer initialized in {end_time - start_time:.4f} seconds.")

    def compute_apodization(
        self,
        focus_mm,
        *,
        F_over_D=None,
        apodization_type="rect",
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
            raise ValueError("Wrong focus: z_foc must be positive")

        if apodization_type is None:
            pass
        elif apodization_type == "none":
            apod = np.ones((N_x, N_y))
        else:
            # require ratio_F_over_D property
            if F_over_D is not None:
                self.F_over_D = F_over_D

            if self.F_over_D is None:
                print("Warning: F/D ratio not set. Using default value of 1.0.")
                self.F_over_D = 1.0

            d_tx = 2 * z_foc * np.tan(np.radians(self.dir_angle_deg)) / self.F_over_D
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
            sx = int(np.round(x_foc / self.el_w))
            sy = int(np.round(y_foc / self.el_h))
            ix = np.arange(Nvx) - (Nvx - 1) // 2 + (N_x // 2) + sx
            iy = np.arange(Nvy) - (Nvy - 1) // 2 + (N_y // 2) + sy
            valid_x = (ix >= 0) & (ix < N_x)
            valid_y = (iy >= 0) & (iy < N_y)
            apod = np.zeros((N_x, N_y))
            apod[np.ix_(ix[valid_x], iy[valid_y])] = profile[np.ix_(valid_x, valid_y)]

        if plot:
            import matplotlib.pyplot as plt

            plt.imshow(apod, cmap="cool", vmin=0, vmax=1)
            plt.colorbar()
            plt.xlabel("Element X")
            plt.ylabel("Element Y")
            plt.show()
            plt.close()

        if inline:
            self.apodization = apod.T.flatten()
            self.apodization_type = apodization_type
        return apod.T

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

        if focus.shape == (3,):
            x_foc, y_foc, z_foc = focus[0], focus[1], focus[2]
            # print(f"Focus: {focus_mm[0]:.3f} mm, {focus_mm[1]:.3f} mm, {focus_mm[2]:.3f} mm")
        else:
            raise ValueError("Focus must be a 3D coordinate (x, y, z)")

        # Compute distances from each element to the focus point
        delays = np.linalg.norm(self.element_centers - focus, axis=1) / c

        # Compute delays based on the speed of sound in soft tissue
        delays = delays.min() - delays  # time delays for focusing

        # optionally plot
        if plot:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(
                np.arange(self.n_elements),
                delays * 1e6,
                "k-",
                marker="o",
                markerfacecolor="r",
                ms=3,
            )
            plt.title(
                f"Focusing at: [{x_foc * 1e3:.3f} mm, {y_foc * 1e3:.3f} mm, {z_foc * 1e3:.3f} mm]"
            )
            plt.xlabel("Element #")
            plt.ylabel("Time delay (us)")
            plt.grid(True)
            plt.show()
            plt.close()

        if inline:
            self.delays = delays
        return delays

    def set_apodization(self, weights):
        weights = np.asarray(weights, dtype=float)
        if weights.size != self.n_elem_x * self.n_elem_y:
            raise ValueError("Apodization must match total elements")
        self.apodization = weights

    def set_delays(self, delays):
        delays = np.asarray(delays, dtype=float)
        if delays.size != self.n_elem_x * self.n_elem_y:
            raise ValueError("Delays must match total elements")
        self.delays = delays

    def get_mesh(self):
        verts, faces, scalars = [], [], []
        pt_index = 0
        for quad, el_idx in zip(self.sub_quad_verts, self.sub_el_idx):
            verts.extend(quad.tolist())
            faces.append([4, pt_index, pt_index + 1, pt_index + 2, pt_index + 3])
            scalars.append(self.apodization[el_idx])
            pt_index += 4
        verts = np.array(verts) * 1e3
        mesh = pv.PolyData(verts, np.hstack(faces))
        mesh.cell_data["Apodization"] = np.array(scalars)
        return mesh

    def show(self, *, notebook=True, show_edges=False):
        """
        Visualize the transducer surface mesh and apodization with PyVista.
        """
        mesh = self.get_mesh()
        plotter = pv.Plotter(notebook=notebook)
        plotter.add_mesh(
            mesh,  # Convert to mm for visualization
            scalars="Apodization",
            cmap="cool",
            clim=[0, 1],
            show_scalar_bar=True,
            scalar_bar_args={"title": "Apodization", "vertical": True},
            opacity=1.0,
            show_edges=show_edges,
        )
        plotter.add_axes()
        plotter.show_grid(
            font_size=10,
            xtitle="X (mm)",
            ytitle="Y (mm)",
            ztitle="Z (mm)",
            show_zlabels=False,
        )
        plotter.show()
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
            "el_w_mm": self.el_w * 1e3,
            "kerf_x_mm": self.kerf_x * 1e3,
            "kerf_y_mm": self.kerf_y * 1e3,
            "no_sub_x": self.no_sub_x,
            "no_sub_y": self.no_sub_y,
            "fc_Hz": self.fc,
        }
        parts = [f"{k}={v}" for k, v in params.items()]
        return f"{self.__class__.__name__}({', '.join(parts)})"

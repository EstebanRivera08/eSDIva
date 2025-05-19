
from .base import BaseTransducer
import numpy as np
import pyvista as pv
from time import time as TIME

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
        frequency_Hz=None
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
        self.no_sub_x = no_sub_x
        self.no_sub_y = no_sub_y
        self.fc = frequency_Hz or 1.0
        # initialize apodization and delays
        total_elements = self.n_elem_x * self.n_elem_y
        self.apodization = np.ones(total_elements, dtype=float)
        self.delays = np.zeros(total_elements, dtype=float)
        # compute element centers in x and y
        total_w = self.n_elem_x * self.el_w + (self.n_elem_x - 1) * self.kerf_x
        total_h = self.n_elem_y * self.el_h + (self.n_elem_y - 1) * self.kerf_y
        start_x = -total_w / 2 + self.el_w / 2
        start_y = -total_h / 2 + self.el_h / 2
        centers = []
        
        for iy in range(self.n_elem_y):
            y = start_y + iy * (self.el_w + self.kerf_y )
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
            xs = np.linspace(-self.el_w/2, self.el_w/2, self.no_sub_x + 1)
            ys = np.linspace(-self.el_h/2, self.el_h/2, self.no_sub_y + 1)
            for i in range(self.no_sub_x):
                for j in range(self.no_sub_y):
                    corners_local = np.array([
                        [xs[i], ys[j], 0.0], [xs[i+1], ys[j], 0.0],
                        [xs[i+1], ys[j+1], 0.0], [xs[i], ys[j+1], 0.0]
                    ])
                    corners = corners_local + center
                    self.sub_quad_verts.append(corners)
                    self.sub_area.append((self.el_w/self.no_sub_x)*(self.el_h/self.no_sub_y))
                    self.sub_el_idx.append(idx)

        print(f"MatrixArrayTransducer initialized with {total_elements} elements and {len(self.sub_quad_verts)} patches.")
        end_time = TIME()
        print(f"Transducer initialized in {end_time - start_time:.4f} seconds.")


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
            faces.append([4, pt_index, pt_index+1, pt_index+2, pt_index+3])
            scalars.append(self.apodization[el_idx])
            pt_index += 4
        verts = np.array(verts)*1e3
        mesh = pv.PolyData(verts, np.hstack(faces))
        mesh.cell_data['Apodization'] = np.array(scalars)
        return mesh
    
    def show(self, *, notebook = True):
        """
        Visualize the transducer surface mesh and apodization with PyVista.
        """
        mesh = self.get_mesh()
        plotter = pv.Plotter(notebook=notebook)
        plotter.add_mesh(
            mesh,  # Convert to mm for visualization
            scalars='Apodization',
            cmap='cool',
            clim=[0, 1],
            show_scalar_bar=True,
            scalar_bar_args={'title':'Apodization', 'vertical': True},
            opacity=1.0,
            show_edges=True,
        )
        plotter.add_axes()
        plotter.show_grid(font_size = 10, xtitle = "X (mm)", ytitle = "Y (mm)", ztitle = "Z (mm)", show_zlabels=False)
        plotter.show()
    
    def __repr__(self):
        params = {
            'n_elem_x': self.n_elem_x,
            'n_elem_y': self.n_elem_y,
            'el_w_mm': self.el_w*1e3,
            'kerf_x_mm': self.kerf_x*1e3,
            'kerf_y_mm': self.kerf_y*1e3,
            'no_sub_x': self.no_sub_x,
            'no_sub_y': self.no_sub_y,
            'focus_mm': self.focus*1e3 if self.focus else None,
            'fc_Hz': self.fc
        }
        parts = [f"{k}={v}" for k, v in params.items()]
        return f"{self.__class__.__name__}({', '.join(parts)})"

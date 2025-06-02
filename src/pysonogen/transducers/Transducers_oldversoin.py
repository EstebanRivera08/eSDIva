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
    
# LinearArrayTransducer class

class LinearArrayTransducer:
    def __init__(
        self,
        *,
        n_elements,
        element_width_mm,
        element_height_mm,
        kerf_mm,
        elevation_focus_mm,
        no_sub_x,
        no_sub_y,
        frequency_Hz = None
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
        element_height, element_width = element_height_mm *1e-3, element_width_mm *1e-3
        kerf, elevation_focus = kerf_mm *1e-3, elevation_focus_mm *1e-3
        self.n_elements = n_elements
        self.el_w = element_width # m
        self.el_h = element_height  # m
        self.kerf = kerf  # m
        self.elev_focus = elevation_focus  # m
        self.no_sub_x = no_sub_x
        self.no_sub_y = no_sub_y
        
        if elevation_focus is not None and elevation_focus > 0 and no_sub_y < 2:
            raise ValueError("Elevation focus requires at least 2 subdivisions in y to model elevation focusing.")

        if frequency_Hz is not None:
            self.fc = frequency_Hz
        else:
            self.fs = 1
            print("Warning: No central frequency provided. Defaulting to 1 Hz.")

        # Per-element apodization weights and delay placeholders
        self.apodization = np.ones(n_elements, dtype=float)
        self.delays = np.zeros(n_elements, dtype=float)

        # Compute element centers along x-axis
        total_width = n_elements * element_width + (n_elements - 1) * kerf
        start_x = -total_width / 2 + element_width / 2
        self.element_centers = np.array([
            [start_x + i * (element_width + kerf), 0.0, 0.0]
            for i in range(n_elements)
        ])

        # Build subdivisions boundaries and areas, apply elevation curvature
        self.sub_quad_verts, self.sub_area, self.sub_el_idx = self._build_subdivisions()
        end_time = TIME()
        print(f"Transducer initialized in {end_time - start_time:.4f} seconds.")

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
        xs = np.linspace(-self.el_w/2, self.el_w/2, self.no_sub_x + 1)
        ys = np.linspace(-self.el_h/2, self.el_h/2, self.no_sub_y + 1)

        patch_area = (self.el_w / self.no_sub_x) * (self.el_h / self.no_sub_y)

        quads = []
        el_indices = []
        for idx, center in enumerate(self.element_centers):
            for i in range(self.no_sub_x):
                for j in range(self.no_sub_y):
                    # four corners of the patch in local coords
                    corners_local = np.array([
                        [xs[i],   ys[j],   0.0],
                        [xs[i+1], ys[j],   0.0],
                        [xs[i+1], ys[j+1], 0.0],
                        [xs[i],   ys[j+1], 0.0]
                    ])
                    # translate to global x,y
                    corners = corners_local.copy()
                    corners[:,0] += center[0]
                    corners[:,1] += center[1]
                    # apply elevation curvature in z
                    if self.elev_focus is not None and self.elev_focus > 0:
                        y_vals = corners[:,1]
                        z_offset = self.elev_focus - np.sqrt(np.clip(self.elev_focus**2 - y_vals**2, 0, None))
                        corners[:,2] += z_offset
                    else:
                        corners[:,2] += center[2]
                    quads.append(corners)
                    el_indices.append(idx)
        return quads, patch_area, el_indices

    def set_apodization(self, weights):
        """Set per-element apodization weights (length = n_elements)."""
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != self.n_elements:
            raise ValueError("Apodization array must match number of elements.")
        self.apodization = weights

    def set_delays(self, delays):
        """Set per-element delays in seconds (length = n_elements)."""
        delays = np.asarray(delays, dtype=float)
        if delays.shape[0] != self.n_elements:
            raise ValueError("Delay array must match number of elements.")
        self.delays = delays

    def get_mesh(self):
        """
        Returns a PyVista PolyData mesh of all subdivided quads, with per-cell apodization scalars.
        """
        # Aggregate all points and faces
        verts = []
        faces = []
        scalars = []
        pt_index = 0
        for quad, el_idx in zip(self.sub_quad_verts, self.sub_el_idx):
            # quad is 4x3 array, create face [4, p0, p1, p2, p3]
            verts.extend(quad.tolist())
            face = [4, pt_index, pt_index+1, pt_index+2, pt_index+3]
            faces.append(face)
            scalars.append(self.apodization[el_idx])
            pt_index += 4
        # Flatten verts and faces
        verts = np.array(verts)*1e3  # Convert to mm for visualization
        faces_flat = np.hstack(faces)
        mesh = pv.PolyData(verts, faces_flat)
        mesh.cell_data['Apodization'] = np.array(scalars)
        return mesh
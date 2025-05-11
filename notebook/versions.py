import numpy as np
from numba import njit, prange

@njit(parallel=True)
def _compute_element_sir(
    x_p, y_p,
    corners,
    t,
    c,
    w_el,
    delay_samples,
    r_avg
):
    N = t.shape[0]
    h_el = np.zeros(N)
    for n in prange(N):
        R = c * t[n]
        if R <= 0:
            continue
        # compute intersections for 4 edges
        Q = np.empty(8)
        qcnt = 0
        for j in range(4):
            x1, y1 = corners[j]
            x2, y2 = corners[(j+1)%4]
            dx = x2 - x1
            dy = y2 - y1
            A = dx*dx + dy*dy
            if A == 0:
                continue
            B = 2*(dx*(x1-x_p) + dy*(y1-y_p))
            C = (x1-x_p)**2 + (y1-y_p)**2 - R*R
            D = B*B - 4*A*C
            if D < 0:
                continue
            sqrtD = np.sqrt(D)
            u1 = (-B + sqrtD) / (2*A)
            u2 = (-B - sqrtD) / (2*A)
            if 0 <= u1 <= 1:
                xi = x1 + u1*dx
                yi = y1 + u1*dy
                Q[qcnt] = np.arctan2(yi - y_p, xi - x_p)
                qcnt += 1
            if 0 <= u2 <= 1:
                xi = x1 + u2*dx
                yi = y1 + u2*dy
                Q[qcnt] = np.arctan2(yi - y_p, xi - x_p)
                qcnt += 1
        if qcnt < 2:
            continue
        # sort and sum arcs
        # copy valid entries
        Qs = np.empty(qcnt)
        for k in range(qcnt):
            Qs[k] = Q[k]
        # unwrap and sort
        Qs = np.sort(np.unwrap(Qs))
        deltaQ = 0.0
        for k in range(0, qcnt-1, 2):
            deltaQ += Qs[k+1] - Qs[k]
        h_val = (c/(2*np.pi)) * deltaQ * w_el / r_avg
        # apply delay shift
        idx = n + delay_samples
        if idx < N:
            h_el[idx] += h_val
    return h_el


def compute_sir(
    transducer,
    points,
    c=1540.0,
    fs=100e6
):
    """
    Optimized SIR computation using Jensen's boundary-intersection algorithm accelerated with Numba.
    """
    pts = np.atleast_2d(points)
    P = pts.shape[0]

    # Pre-calculate element corners and per-element params
    num_els = transducer.n_elements
    elem_corners = np.zeros((num_els, 4, 2))
    delays_samp = np.zeros(num_els, dtype=np.int32)
    r_avgs = np.zeros(num_els)
    weights = transducer.apod
    hw, hh = transducer.el_w/2, transducer.el_h/2
    for i in range(num_els):
        x0, y0, _ = transducer.element_centers[i]
        elem_corners[i, 0] = (x0 - hw, y0 - hh)
        elem_corners[i, 1] = (x0 + hw, y0 - hh)
        elem_corners[i, 2] = (x0 + hw, y0 + hh)
        elem_corners[i, 3] = (x0 - hw, y0 + hh)
        delays_samp[i] = int(round(transducer.delays[i] * fs))
        # r_avg per element varies by point, will pass later

    # Estimate t_max over all points and corners
    max_delay = 0.0
    for p in range(P):
        for i in range(num_els):
            for corner in elem_corners[i]:
                r = np.linalg.norm([pts[p,0]-corner[0], pts[p,1]-corner[1], pts[p,2]])
                d = r/c + transducer.delays[i]
                if d > max_delay:
                    max_delay = d
    t_max = max_delay + 1e-6
    N = int(np.ceil(t_max * fs)) + 1
    t = np.arange(N) / fs

    # Compute h
    h = np.zeros((P, N), dtype=np.float64)
    for p in range(P):
        x_p, y_p, z_p = pts[p]
        for i in range(num_els):
            # r_avg based on center
            center = transducer.element_centers[i]
            r_avgs[i] = np.linalg.norm(pts[p] - center)
        # accumulate element contributions
        for i in range(num_els):
            h_el = _compute_element_sir(
                x_p, y_p,
                elem_corners[i],
                t,
                c,
                weights[i],
                delays_samp[i],
                r_avgs[i]
            )
            h[p] += h_el
    return t, h

class PyField:
    def __init__(self, transducer):
        """
        Computational framework for ultrasound fields. Wraps a transducer object.
        """
        self.tx = transducer
        self.c = 1540.0  # Speed of sound in soft tissue (m/s)
        self.fs = 100e6 # Sampling frequency (Hz)
        self.t_max = 50e-6  # Max time for field computation

        self.use_attenuation = False  # Flag for attenuation computation
        self.att_f0 = transducer.fc  # Central frequency for attenuation
        self.f_wv = transducer.fc  # Frequency for field computation
        self.lambda_wv = self.c / self.f_wv # Wavelength (um)

        # Frequency independent attenuation at the frequency f0 [dB/m]
        self.att = 0.5  # Attenuation coefficient (dB/m)

    
    def set_field(self, name_struct_str, value_float):
        """
        Dynamically modifies a class property if it exists.

        Parameters
        ----------
        name_struct_str : str
            The name of the property to modify.
        value_float : float
            The new value to assign to the property.

        Raises
        ------
        AttributeError
            If the property does not exist in the class.
        TypeError
            If the value is not a float.
        """
        if not isinstance(value_float, (float, int)):  # Allow integers as well
            raise TypeError(f"The value must be a float or int, got {type(value_float).__name__}.")
        
        if hasattr(self, name_struct_str):
            setattr(self, name_struct_str, value_float)
            print(f"Property '{name_struct_str}' updated to {value_float}.")
        else:
            raise AttributeError(f"Property '{name_struct_str}' does not exist in the class.")

    def show_transducer(self, kwargs= None):
        """
        Visualize the transducer surface mesh and apodization with PyVista.
        """
        mesh = self.tx.get_mesh()
        plotter = pv.Plotter()
        plotter.add_mesh(
            mesh,  # Convert to mm for visualization
            scalars='Apodization',
            cmap='viridis',
            show_scalar_bar=True,
            scalar_bar_args={'title':'Apodization', 'vertical': True},
            opacity=1.0,
            show_edges=True,
        )
        plotter.add_axes()
        plotter.show_grid(font_size = 10, xtitle = "X (mm)", ytitle = "Y (mm)", ztitle = "Z (mm)", show_zlabels=False)
        plotter.show()


    def spatial_impulse_response(self, field_points):
        """
        Compute the Spatial Impulse Response (SIR) of a linear array transducer at given field points,
        incorporating per-element delays.

        Parameters
        ----------
        transducer : LinearArrayTransducer
            Instance with subdivisions built and optional element delays (s).
        points : ndarray, shape (P, 3)
            Field points in meters where SIR is evaluated.
        c : float
            Speed of sound in m/s (default: 1540 m/s).
        fs : float
            Sampling frequency in Hz for the impulse response (default: 100 MHz).
        t_max : float or None
            Maximum time in seconds. If None, set to (max distance + max delay)/c + buffer.

        Returns
        -------
        t : ndarray, shape (N,)
            Time vector for the impulse response.
        h : ndarray, shape (P, N)
            SIR waveform at each of the P points.

        Notes
        -----
        - Each subdivision patch is approximated as a point source located at its centroid.
        - Per-element electronic delays are added to each patch’s propagation delay.
        - Amplitude term: (weight * area) / (2*pi*r*c).
        """
        
        sub_h = self.tx.el_h / self.tx.no_sub_y
        sub_w = self.tx.el_w / self.tx.no_sub_x
        largest_dim = max(sub_h, sub_w)

        if field_points.ndim > 2:
            raise ValueError("Field points must be 3D array (P, 3) or (3,), thus np.array.ndim 2 or 1. np.arrayrray.ndim > 2 is not supported.")
        elif field_points.ndim == 1:
            field_points = field_points.reshape((1, 3))
        
        if field_points.shape[1] != 3:
            raise ValueError("Field points must be 3D array (P, 3) or (3,), thus np.array.ndim 2 or 1. np.array.shape[1] != 3 is not supported.")
        
        pts = np.atleast_2d(field_points)

        closest_distance = np.linalg.norm(pts, axis=1).min()
        criteria = np.sqrt(closest_distance*self.lambda_wv*4)
        if largest_dim > 10*criteria:
            print(f"Warning: The largest dimension of the transducer ({largest_dim:.2f} m) is greater than 10 times the criteria ({criteria:.2f} m).")
            print("This may lead to inaccurate results.")

        P = pts.shape[0]

        # Pre-calculate element corners and bounding lines for each element
        # Each element: centered at transducer.element_centers[i], width w, height h
        elem_w = self.tx.el_w
        elem_h = self.tx.el_h
        corners = []  # per element, 4x2 (x,y) in element plane
        for center in self.tx.element_centers:
            x0, y0, _ = center
            hw, hh = elem_w/2, elem_h/2
            # rectangle corners in global x-y plane
            corners.append(np.array([
                [x0 - hw, y0 - hh],
                [x0 + hw, y0 - hh],
                [x0 + hw, y0 + hh],
                [x0 - hw, y0 + hh]
            ]))

        # Determine global max time
        # For each element corner and each point, compute delay + element delay
        delays = []
        for p in range(P):
            for i, corner in enumerate(corners):
                for x_c, y_c in corner:
                    for z_corner in [0]:  # flat element
                        r = np.linalg.norm([pts[p,0]-x_c, pts[p,1]-y_c, pts[p,2]-z_corner])
                        delays.append(r/self.c + self.tx.delays[i])
        t_max = max(delays) + 1e-6
        N = int(np.ceil(t_max * self.fs)) + 1
        t = np.arange(N) / self.fs
        h = np.zeros((P, N), dtype=float)

        # Loop over points and elements, compute analytic SIR per element
        for p in range(P):
            x_p, y_p, z_p = pts[p]
            for i, center in enumerate(self.tx.element_centers):
                x0, y0, _ = center
                # Project field point onto element plane: at z=0
                # For each time sample, determine intersection of circle radius R=c*t_c with rectangle boundary
                # Use Jensen's Eq. (6): h_el(t) = (c/(2*pi)) * sum_over_arcs [theta2 - theta1]
                # plus apodization and delay shifting
                # Here we discretize: for element i, compute its SIR vector h_el
                # 1. Compute circle radius at each t: r_circle = c*t
                # 2. For each t, find intersections with rectangle edges (4 lines): compute angles Q
                # 3. Sort Q, sum arcs inside rectangle to get arc angle deltaQ
                # 4. h_el[t_idx] = (c/(2*pi)) * deltaQ * apod[i]
                # 5. Shift h_el by delay = transducer.delays[i] (sample index shift)
                # 6. Normalize amplitude by 1/r_avg
                # ----- implementation details -----
                # For performance, one would precompute discontinuity times and only update Q around those.
                # Here, for clarity, we perform brute-force per sample.
                h_el = np.zeros(N)
                # Apodization weight
                w_el = self.tx.apod[i]
                # Average distance for amplitude normalization
                r_avg = np.linalg.norm(pts[p] - center)
                for n in range(N):
                    R = self.c * t[n]
                    if R <= 0:
                        continue
                    # intersection angles list
                    Q = []
                    # edges defined by pairs of corner points
                    elem_corners = corners[i]
                    for j in range(4):
                        x1, y1 = elem_corners[j]
                        x2, y2 = elem_corners[(j+1)%4]
                        # line segment intersection with circle (x-x_p)^2+(y-y_p)^2=R^2
                        # parametric line L(u) = (x1,y1) + u*(dx,dy), u in [0,1]
                        dx, dy = x2-x1, y2-y1
                        # solve (x1 + u*dx - x_p)^2 + (y1 + u*dy - y_p)^2 = R^2
                        A = dx*dx + dy*dy
                        B = 2*(dx*(x1-x_p) + dy*(y1-y_p))
                        C = (x1-x_p)**2 + (y1-y_p)**2 - R*R
                        D = B*B - 4*A*C
                        if D < 0 or A == 0:
                            continue
                        sqrtD = np.sqrt(D)
                        for sign in (-1, 1):
                            u = (-B + sign*sqrtD) / (2*A)
                            if 0 <= u <= 1:
                                xi = x1 + u*dx
                                yi = y1 + u*dy
                                # angle relative to projected point
                                Q.append(np.arctan2(yi - y_p, xi - x_p))
                    if len(Q) < 2:
                        continue
                    Q = np.sort(np.unwrap(Q))
                    # sum alternating arcs (inside polygon)
                    deltaQ = 0.0
                    for k in range(0, len(Q)-1, 2):
                        deltaQ += Q[k+1] - Q[k]
                    # impulse response value
                    h_el[n] = (self.c/(2*np.pi)) * deltaQ * w_el / r_avg
                # shift by electronic delay
                delay_samples = int(round(self.tx.delays[i] * self.fs))
                if delay_samples > 0:
                    h_el = np.concatenate((np.zeros(delay_samples), h_el[:-delay_samples]))
                # accumulate
                h[p] += h_el
        return t, h
    
    import numpy as np
import time
from numba import njit, prange
import sys

@njit(parallel=True)
def _compute_element_sir(
    x_p, y_p,
    corners,
    t,
    c,
    w_el,
    delay_samples,
    r_avg
):
    N = t.shape[0]
    h_el = np.zeros(N)
    for n in prange(N):
        R = c * t[n]
        if R <= 0:
            continue
        Q = np.empty(8)
        qcnt = 0
        for j in range(4):
            x1, y1 = corners[j]
            x2, y2 = corners[(j+1)%4]
            dx = x2 - x1
            dy = y2 - y1
            A = dx*dx + dy*dy
            if A == 0:
                continue
            B = 2*(dx*(x1-x_p) + dy*(y1-y_p))
            C = (x1-x_p)**2 + (y1-y_p)**2 - R*R
            D = B*B - 4*A*C
            if D < 0:
                continue
            sqrtD = np.sqrt(D)
            u1 = (-B + sqrtD) / (2*A)
            u2 = (-B - sqrtD) / (2*A)
            if 0 <= u1 <= 1:
                xi = x1 + u1*dx
                yi = y1 + u1*dy
                Q[qcnt] = np.arctan2(yi - y_p, xi - x_p)
                qcnt += 1
            if 0 <= u2 <= 1:
                xi = x1 + u2*dx
                yi = y1 + u2*dy
                Q[qcnt] = np.arctan2(yi - y_p, xi - x_p)
                qcnt += 1
        if qcnt < 2:
            continue
        Qs = np.empty(qcnt)
        for k in range(qcnt):
            Qs[k] = Q[k]
        Qs = np.sort(Qs)
        deltaQ = 0.0
        for k in range(0, qcnt-1, 2):
            deltaQ += Qs[k+1] - Qs[k]
        h_val = (c/(2*np.pi)) * deltaQ * w_el / r_avg
        idx = n + delay_samples
        if idx < N:
            h_el[idx] += h_val
    return h_el


class PyField:
    def __init__(self, transducer):
        self.tx = transducer
        self.c = 1540.0
        self.fs = 100e6
        self.t_max = 50e-6

        self.use_attenuation = False
        self.att_f0 = transducer.fc
        self.f_wv = transducer.fc
        self.lambda_wv = self.c / self.f_wv
        self.att = 0.5

    def set_field(self, name_struct_str, value_float):
        if not isinstance(value_float, (float, int)):
            raise TypeError(f"The value must be a float or int, got {type(value_float).__name__}.")
        if hasattr(self, name_struct_str):
            setattr(self, name_struct_str, value_float)
            print(f"Property '{name_struct_str}' updated to {value_float}.")
        else:
            raise AttributeError(f"Property '{name_struct_str}' does not exist in the class.")

    def show_transducer(self, kwargs=None):
        mesh = self.tx.get_mesh()
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalars='Apodization', cmap='viridis', show_scalar_bar=True,
                         scalar_bar_args={'title': 'Apodization', 'vertical': True}, opacity=1.0, show_edges=True)
        plotter.add_axes()
        plotter.show_grid(font_size=10, xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)", show_zlabels=False)
        plotter.show()

    def spatial_impulse_response(self, field_points):
        sub_h = self.tx.el_h / self.tx.no_sub_y
        sub_w = self.tx.el_w / self.tx.no_sub_x
        largest_dim = max(sub_h, sub_w)

        if field_points.ndim > 2:
            raise ValueError("Field points must be 3D array (P, 3) or (3,).")
        elif field_points.ndim == 1:
            field_points = field_points.reshape((1, 3))
        if field_points.shape[1] != 3:
            raise ValueError("Field points must be of shape (P, 3) or (3,).")

        pts = np.atleast_2d(field_points)
        P = pts.shape[0]

        closest_distance = np.linalg.norm(pts, axis=1).min()
        criteria = np.sqrt(closest_distance * self.lambda_wv * 4)
        if largest_dim > 10 * criteria:
            print(f"Warning: The largest dimension ({largest_dim:.2e} m) > 10x criteria ({criteria:.2e} m). Results may be inaccurate.")

        num_els = self.tx.n_elements
        elem_corners = np.zeros((num_els, 4, 2))
        delays_samp = np.zeros(num_els, dtype=np.int32)
        r_avgs = np.zeros(num_els)
        weights = self.tx.apod
        hw, hh = self.tx.el_w / 2, self.tx.el_h / 2
        for i in range(num_els):
            x0, y0, _ = self.tx.element_centers[i]
            elem_corners[i, 0] = (x0 - hw, y0 - hh)
            elem_corners[i, 1] = (x0 + hw, y0 - hh)
            elem_corners[i, 2] = (x0 + hw, y0 + hh)
            elem_corners[i, 3] = (x0 - hw, y0 + hh)
            delays_samp[i] = int(round(self.tx.delays[i] * self.fs))

        max_delay = 0.0
        for p in range(P):
            for i in range(num_els):
                for corner in elem_corners[i]:
                    r = np.linalg.norm([pts[p, 0] - corner[0], pts[p, 1] - corner[1], pts[p, 2]])
                    d = r / self.c + self.tx.delays[i]
                    if d > max_delay:
                        max_delay = d
        t_max = max_delay + 1e-6
        N = int(np.ceil(t_max * self.fs)) + 1
        t = np.arange(N) / self.fs

        h = np.zeros((P, N), dtype=np.float64)

        # Estimate runtime
        print("Estimating total runtime...")
        start_single = time.time()
        x_p, y_p, z_p = pts[0]
        for i in range(num_els):
            r_avgs[i] = np.linalg.norm(pts[0] - self.tx.element_centers[i])
            _compute_element_sir(x_p, y_p, elem_corners[i], t, self.c, weights[i], delays_samp[i], r_avgs[i])
        est_duration = (time.time() - start_single) * P
        print(f"Estimated total time: {est_duration:.1f} s for {P} points.")

        # Main computation with progress bar
        print("Computing SIR...")
        step = max(1, P // 20)
        start_all = time.time()
        for p in range(P):
            x_p, y_p, z_p = pts[p]
            for i in range(num_els):
                r_avgs[i] = np.linalg.norm(pts[p] - self.tx.element_centers[i])
                h_el = _compute_element_sir(x_p, y_p, elem_corners[i], t, self.c, weights[i], delays_samp[i], r_avgs[i])
                h[p] += h_el
            if p % step == 0:
                progress = 100 * p // P
                elapsed = time.time() - start_all
                remaining = est_duration - elapsed
                sys.stdout.write(f"\rProgress: {progress}% — Estimated time remaining: {remaining:.1f} s")
                sys.stdout.flush()

        print("\nSIR computation complete.")
        return t, h
    

    




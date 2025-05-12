from scipy.interpolate import interp1d

def _compute_rectangle_SIR(wx,wy,xp,yp,l,c0, apodization, delay) :
    """
    Compute the SIR (Spatial impulse Response) for a rectangle defined by its center and corners.
    The SIR is computed with the far-field approximation.
    """
    # print(f"wx: {wx}, wy: {wy}, xp: {xp}, yp: {yp}, l: {l}, c0: {c0}, apodization: {apodization}, delay: {delay}")
    # Compute the normal vector of the rectangle
    Dt1 = np.abs([wx*xp/c0, wy*yp/c0]).min() # Convert to microseconds
    Dt2 = np.abs([wx*xp/c0, wy*yp/c0]).max()

    # print(f"Dt1: {1/Dt1:.2e} us, Dt2: {1/Dt2:.2e} us")
    area = wx*wy/(2*np.pi*l) # m^2

    t1 = l/c0 - (Dt1 + Dt2)/ 2 # s
    t2 = t1 + Dt1 # s
    t3 = t1 + Dt2 # s
    t4 = t1 +  Dt1 + Dt2 # s

    # Area = h_max * [ ((t2-t1)+(t4-t3)) / 2 + (t3-t2)]
    time_denomidator = ((t2-t1)+(t4-t3)) / 2 + (t3-t2)
    h_max = area/time_denomidator* apodization
    if t1 > t2 or t2 > t3 or t3 > t4:
        raise ValueError("Invalid time values: t1, t2, t3, t4 must be in increasing order.")
    t = np.array([t1,t2,t3,t4]) + delay
    h = np.array([0, h_max, h_max, 0])
    
    return t, h

class PyField:
    def __init__(self, transducer):
        """
        Computational framework for ultrasound fields. Wraps a transducer object.
        """
        self.tx = transducer
        self.c = 1540.0  # Speed of sound in soft tissue (m/s)
        self.fs = 100e6 # Sampling frequency (Hz)

        self.use_attenuation = False  # Flag for attenuation computation
        self.att_f0 = transducer.fc*1e6  # Central frequency for attenuation (Hz)
        self.f_wv = transducer.fc*1e6  # Frequency for field computation (Hz)
        self.lambda_wv = self.c / self.f_wv # Wavelength (m)

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

    def spatial_impulse_response(self, field_points, return_all = False):
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
        if field_points.ndim > 2:
            raise ValueError("Field points must be 3D array (P, 3) or (3,).")
        
        pts = np.atleast_2d(field_points)
        P = pts.shape[0]

        sub_elem_h = self.tx.el_h / self.tx.no_sub_y
        sub_elem_w = self.tx.el_w / self.tx.no_sub_x
        largest_dim = max(sub_elem_h, sub_elem_w)
        
        closest_distance = np.linalg.norm(pts, axis=1).min()
        criteria = np.sqrt(closest_distance * self.lambda_wv * 4)
        if largest_dim > 10 * criteria:
            print(f"Warning: The largest dimension ({largest_dim:.2e} m) > 10x criteria ({criteria:.2e} m). Results may be inaccurate.")

        nb_elements = self.tx.n_elements
        nb_sub_div_per_elem = self.tx.no_sub_x * self.tx.no_sub_y
        t_tx = np.zeros((P, nb_elements*nb_sub_div_per_elem, 4))
        h_tx = np.zeros((P, nb_elements*nb_sub_div_per_elem, 4))

        for p in range(P):
            rect_count = 0
            for elem in range(nb_elements):
                for sub_div_per_elem in range(nb_sub_div_per_elem) :
                    # Get the patch center
                    xp, yp, zp = pts[p,:] # m
                    patch_center = self.tx.sub_quad_verts[elem*nb_sub_div_per_elem+sub_div_per_elem].mean(axis=0) # m
                    point2patch = [xp-patch_center[0], yp-patch_center[1], zp-patch_center[2]]
                    distance_point2patch =  np.linalg.norm(point2patch) # m
                    point2patch = np.array(point2patch) / distance_point2patch  # Normalize the vector, m
                    
                    ti, hi =_compute_rectangle_SIR( sub_elem_w, sub_elem_h, point2patch[0], point2patch[1],
                                            distance_point2patch, self.c,
                                            self.tx.apodization[elem], self.tx.delays[elem])
                    t_tx[p, rect_count, :] = ti
                    h_tx[p, rect_count, :] = hi
                    rect_count += 1
                

        
        # Compute the time vector
        times_all = t_tx.flatten()  # save all the time values
        times_all = np.unique(times_all) # remove duplicates
        times_all = times_all[np.argsort(times_all)] # sort the time vector

        # Downsample to the desired sampling frequency
        start_time, end_time = times_all.min(), times_all.max()
        
        # Compute the time vector with a size that is a power of 2
        duration = end_time - start_time
        num_samples = int(np.ceil(duration * self.fs))  # Calculate the number of samples
        next_power_of_2 = 2 ** int(np.ceil(np.log2(num_samples))+1)  # Find the next power of 2
        if next_power_of_2 < 2**5:
            next_power_of_2 = 2**5
        duration_adjusted = next_power_of_2 / self.fs  # Adjust the duration to match the new number of samples
        time = np.linspace(start_time, start_time + duration_adjusted, next_power_of_2, endpoint=False)  # Create the time vector

        # Initialize the SIR array

        h_subdiv = np.zeros((P, time.shape[0]))

        for p in range(P):
            h_all = np.zeros((times_all.shape[0]))  # Initialize the SIR array
            for i in range(h_tx.shape[1]): # Loop over all patches
                h_all += np.interp(times_all, t_tx[p, i, :], h_tx[p, i, :])    
            
            h_subdiv[p,:] = np.interp(time, times_all, h_all) # Interpolate to the desired time vector

        
        if return_all :
            return time, h_subdiv.T, t_tx, h_tx, 
        else :
            return start_time, h_subdiv.T



# ---------------------



    def spatial_impulse_response(self, field_points, return_all=False):
        pts = np.atleast_2d(field_points)
        P = pts.shape[0]
        M = self._centers.shape[0]

        # collect all patch times & amplitudes
        t_tx = np.zeros((P, M, 4))
        h_tx = np.zeros((P, M, 4))

        # progress bar
        for p in tqdm(range(P), desc="Computing SIR", unit="pt"):
            xp, yp, zp = pts[p]
            diffs = self._centers - pts[p]
            dists = np.linalg.norm(diffs, axis=1)
            # normalized directions
            dirs = diffs / dists[:, None]
            for i in range(M):
                t_patch, h_patch = _compute_rectangle_SIR_numba(
                    self._wx, self._wy,
                    dirs[i,0], dirs[i,1],
                    dists[i], self.c,
                    self._apods[i], self._delays[i]
                )
                t_tx[p, i, :] = t_patch
                h_tx[p, i, :] = h_patch

        # flatten times and build global time vector
        all_times = np.unique(t_tx.reshape(-1))
        start, end = all_times.min(), all_times.max()
        duration = end - start
        num_samples = int(np.ceil(duration * self.fs))
        n2 = 2 ** max(int(np.ceil(np.log2(num_samples))) + 1, 5)
        global_time = np.linspace(start, start + n2/self.fs, n2, endpoint=False)

        # aggregate
        h_subdiv = np.zeros((P, n2))
        for p in tqdm(range(P), desc="Interpolating", unit="pt"):
            # sum contributions
            h_all = np.zeros(all_times.shape)
            for i in range(M):
                h_all += np.interp(all_times, t_tx[p,i], h_tx[p,i])
            h_subdiv[p] = np.interp(global_time, all_times, h_all)

        if return_all:
            return global_time, h_subdiv.T, t_tx, h_tx
        return start, h_subdiv.T
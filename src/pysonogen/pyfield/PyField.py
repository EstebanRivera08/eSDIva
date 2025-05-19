import numpy as np
from tqdm import tqdm
from numba import njit, prange
from time import time as TIME
import pyvista as pv


def create_simulation_grid(simulation_struct):
    """
    Create a simulation mesh for the ultrasound field.
    
    Parameters
    ----------
    simulation_grid_dict : dict
        Dictionary containing the simulation parameters:
        - x_extent : list
            The extent of the simulation in the x direction (in mm).
        - y_extent : list
            The extent of the simulation in the y direction (in mm).
        - z_extent : list
            The extent of the simulation in the z direction (in mm).
        - dx : float
            The grid spacing in the x direction (in mm).
        - dy : float
            The grid spacing in the y direction (in mm).
        - dz : float
            The grid spacing in the z direction (in mm).
    
    Returns
    -------
    grid_points : ndarray
        Array of points in the simulation space.
    """
    # Create a grid of points in the simulation space
    [x0, xf], [y0, yf], [z0, zf] = simulation_struct["x_extent"], simulation_struct["y_extent"], simulation_struct["z_extent"]
    Nx = int((xf-x0) / simulation_struct["dx"])
    Ny = int((yf-y0) / simulation_struct["dy"])
    Nz = int((zf-z0) / simulation_struct["dz"])
    if Nx % 2 == 0 :
        Nx += 1 
    if Ny % 2 == 0 :
        Ny += 1
    if Nz % 2 == 0 :
        Nz += 1

    x = np.linspace(x0, xf, Nx)
    y = np.linspace(y0, yf, Ny)
    z = np.linspace(z0, zf, Nz)
    # Create a meshgrid of points
    grid_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    return x, y, z, grid_points*1e-3

@njit
def compute_patch_sir(wx, wy, xp, yp, l, c0, apod, delay, sampling_rate_Hz, lambda_mm):
    # Common sampling rate is 100 MHz
    # Then minimum time step is 0.01 us, 
    epsilon = 1/(2*sampling_rate_Hz) # 1 ns
    Dt1 = min(abs(wx * xp / c0), abs(wy * yp / c0))
    Dt2 = max(abs(wx * xp / c0), abs(wy * yp / c0))
    area = wx * wy / (2 * np.pi * l)

    t1 = l / c0 - 0.5 * (Dt1 + Dt2)
    t2 = t1 + Dt1
    t3 = t1 + Dt2
    t4 = t1 + Dt1 + Dt2

    # peak amplitude
    if not (t1 <= t2 <= t3 <= t4):
        raise ValueError("Invalid time sequence in patch SIR")
    
    # avoid division by zero
    width_trapezoid = ((t2 - t1) + (t4 - t3)) * 0.5 + (t3 - t2) if (t4-t1) > epsilon else epsilon

    # trapezoid area
    h_max = area * apod / (width_trapezoid)
    return t1 + delay, t2 + delay, t3 + delay, t4 + delay, h_max

@njit(parallel=True)
def compute_all_events(P, M, pts, centers, wx, wy, c, apods, delays, events, sampling_rate_Hz, lambda_mm):
    for p in prange(P):
        for i in range(M):
            dx = pts[p,0] - centers[i,0]
            dy = pts[p,1] - centers[i,1]
            dz = pts[p,2] - centers[i,2]
            dist = (dx*dx + dy*dy + dz*dz)**0.5
                
            xp, yp = dx/(dist), dy/(dist)
            t1, t2, t3, t4, h_max = compute_patch_sir(wx, wy, xp, yp, dist, c,
                                                    apods[i], delays[i], sampling_rate_Hz, lambda_mm)
            events[p,i,0] = t1
            events[p,i,1] = t2
            events[p,i,2] = t3
            events[p,i,3] = t4
            events[p,i,4] = h_max

@njit(parallel=True)
def accumulate_from_events(P, M, events, fs, t0, h_out):
    """
    Parallel accumulation of trapezoidal SIR contributions for all patches.
    events shape: (P, M, 5) storing t1, t2, t3, t4, h_max.
    h_out: (P, n2) output array, t0: start time, fs: sampling rate.
    """
    dt = 1.0 / fs
    n2 = h_out.shape[1]
    for p in prange(P):
        for i in range(M):
            t1, t2, t3, t4, h_max = (
                events[p,i,0], events[p,i,1],
                events[p,i,2], events[p,i,3],
                events[p,i,4]
            )            
            k1 = max(0, int((t1 - t0) * fs)+1)
            k2 = min(n2, int((t2 - t0) * fs)+1)
            k3 = min(n2, int((t3 - t0) * fs)+1)
            k4 = min(n2, int((t4 - t0) * fs)+1)
            
            # rising
            for k in range(k1, k2):
                t = t0 + k * dt
                h_out[p,k] += h_max * ((t - t1) / (t2 - t1))
            # plateau
            for k in range(k2, k3):
                h_out[p,k] += h_max
            # falling
            for k in range(k3, k4):
                t = t0 + k * dt
                h_out[p,k] += h_max * (1 - (t - t3) / (t4 - t3))

class PyField:
    def __init__(self, transducer):
        self.tx = transducer
        self.c = 1540.0 
        self.fs = 100e6 # Hz
        self.fc = transducer.fc # Hz
        self.lambda_mm = self.c / self.fc
        # compute patch centers/apods/delays once
        el_h = self.tx.el_h / self.tx.no_sub_y
        el_w = self.tx.el_w / self.tx.no_sub_x
        centers, apods, delays = [], [], []
        for elem in range(self.tx.n_elements):
            for sub_elem in range(self.tx.no_sub_x * self.tx.no_sub_y):
                verts = self.tx.sub_quad_verts[elem*(self.tx.no_sub_x*self.tx.no_sub_y)+sub_elem]
                centers.append(verts.mean(axis=0))
                apods.append(self.tx.apodization[elem])
                delays.append(self.tx.delays[elem])
        self.centers = np.array(centers, dtype=np.float32)
        self.apods = np.array(apods, dtype=np.float32)
        self.delays = np.array(delays, dtype=np.float32)
        self.wx = el_w
        self.wy = el_h

        print(f"Successfully initialized PyField with \n {transducer}")
        
    def spatial_impulse_response(self, field_points, return_all=False):

        start_comput_time = TIME()
        pts = np.atleast_2d(field_points).astype(np.float32)
        P, M = pts.shape[0], self.centers.shape[0]
        
        # allocate events
        events = np.zeros((P, M, 5), dtype=np.float32)
        tqdm.write("Computing all patch events...")
        compute_all_events(P, M, pts, self.centers, self.wx, self.wy,
                           self.c, self.apods, self.delays, events, self.fs, self.lambda_mm)
        # build global time vector from real event times
        all_times = np.unique(events[:,:,0:4].ravel())
        all_times.sort()
        t0, tN = all_times[0], all_times[-1]
        # create sampling grid
        dt = 1.0/self.fs
        num_samples = int(np.ceil((tN - t0)*self.fs))
        # next power of two
        n2 = 2**max(int(np.ceil(np.log2(num_samples)))+1, 5)
        t_global = t0 + np.arange(n2, dtype=np.float32)*dt
        h_out = np.zeros((P, n2), dtype=np.float32)
        tqdm.write("Accumulating SIR from events...")
        accumulate_from_events(P, M, events, self.fs, t0, h_out)

        print(f"Total computation time: {TIME() - start_comput_time:.4f} seconds.")
        if return_all:
            return t_global, h_out.T, events
        return t0, h_out.T
    
    def compute_pr_from_sir(self, h_sir, x, y, z):
        """
        Compute the pressure field from the Spatial Impulse Response (SIR).
        
        Parameters
        ----------
        field_points : ndarray
            Array of points in the simulation space.
    
        Returns
        -------
        pressure : ndarray
            The computed pressure field.
        """
        # Reshape the SIR to match the grid dimensions
        print(f"Original h shape: {h_sir.shape}")
        spatial_impulse_response_field = h_sir.reshape( -1, z.shape[0], x.shape[0], y.shape[0]).transpose(0, 2, 3, 1)
        print(f"Reshaped h shape: {spatial_impulse_response_field.shape}")

        # Perform FFT along the first axis
        print("Performing FFT...")
        spatial_impulse_response_field_FT = np.fft.fft(spatial_impulse_response_field, axis=0)
        # Generate the frequency vector
        freq_vect = np.linspace(0, self.fs, spatial_impulse_response_field_FT.shape[0])

        # Find the index of the desired frequency
        idx_freq = np.argmin(np.abs(freq_vect -  self.fc))
        print(f"Frequency searched: {self.fc} Hz. Found: {freq_vect[idx_freq]} Hz")

        # Amplitude for the given frequency
        amp_response_tx_freq = np.abs(spatial_impulse_response_field_FT[idx_freq, :, :, :])
        
        return amp_response_tx_freq
    
    def compute_pressure_field(self,field_info, *, normalize=True):
        """
        Compute the pressure field from the Spatial Impulse Response (SIR).
        
        Parameters
        ----------
        field_info : dict
        - x_extent : list
            The extent in mm of the simulation in the x direction.
        - y_extent : list
            The extent in mm of the simulation in the y direction.
        - z_extent : list
            The extent in mm of the simulation in the z direction.
        - dx : float
            The grid spacing in mm along the x direction.
        - dy : float
            The grid spacing in mm along the y direction.
        - dz : float
            The grid spacing in mm along the z direction.
    
        Returns
        -------
        pressure : ndarray
            The computed pressure field.
        """
        print("Creating simulation grid...")
        x, y, z, grid_points = create_simulation_grid(field_info)
        print("Computing spatial impulse response...")
        start_time, h_sir = self.spatial_impulse_response(grid_points)
        print("Computing pressure field...")
        pressure_field = self.compute_pr_from_sir(h_sir, x, y, z)

        if normalize:
            pressure_field = pressure_field/np.max(pressure_field)

        return pressure_field, x, y, z
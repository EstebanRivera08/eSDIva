import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


# ------------------- reading functions -------------------
# These functions read the scan and BPS files, extracting the data and metadata.
def read_scan(path_scan, verbose=False):
    """
    Read a 3D scan from a given path and return the data and metadata.
    Args:
        path_scan (str): Path to the .scan file.
        verbose (bool): If True, print additional information about the scan.
    Returns:
        tuple: A tuple containing the data and metadata from the scan.
    """

    file_scan = h5py.File(path_scan, "r")
    data = file_scan["Data"][()]
    metadata = file_scan["acqMetaData"]
    # metadata = {key: raw_metadata[key][()] for key in raw_metadata.keys()}

    if verbose:
        print("\n------scan keys------")
        print(file_scan.keys())
        for key in file_scan.keys():
            print(f"{key}: {file_scan[key]} ")
        print("\ndata shape:", data.shape)
        print("\n-----aqui. metadata keys------")
        for key in metadata.keys():
            print(f"{key}  : {metadata[key]}")

    return data, metadata


def read_bps(path_bps, verbose=False):
    """
    Read a BPS file and return BrainToLab affine matrix.
    Args:
        path_bps (str): Path to the .bps file.
        verbose (bool): If True, print additional information about the BPS file.
    Returns:
        tuple: A tuple containing the BrainToLab affine matrix.
    """

    metadata_bps = h5py.File(path_bps, "r")
    # metadata_bps = {key: raw_metadata_bps[key][()] for key in raw_metadata_bps.keys()}

    if verbose:
        print("------bps keys------")
        print(metadata_bps.keys())
        for key in metadata_bps.keys():
            print(f"{key}: {metadata_bps[key].shape} {metadata_bps[key].dtype}")
    return metadata_bps


# -------------------------- processing functions --------------------------
# These functions process the scan data to create 3D or 2D arrays suitable for visualization.


def volume3D_from_scan_stack(
    scan_stack_data, *, probe_type="Linear", mean_frames=True, verbose=False
):
    """
    Process a 3D scan data array and return a 3D data array suitable for visualization.
    Args:
        scan3D_data (np.ndarray): 5D array with shape (N_scan, frames, Z_vox, Y_vox, X_vox).
        probe_type (str): Type of probe used for the scan ('Linear' or other).
        verbose (bool): If True, print additional information about the processing.
    Returns:
        np.ndarray: A 3D array with shape (X_vox, Y_vox, Z_vox) suitable for visualization.
    """

    if scan_stack_data.ndim != 5:
        raise ValueError(
            "Input scan3D_data must be a 5D array with shape (N_scan, frames, Z_vox, Y_vox, X_vox)"
        )

    n_scan, nt, nz, ny, nx = scan_stack_data.shape

    if mean_frames and nt > 1:
        # If mean_frames is True, average over the time dimension
        print(scan_stack_data)
        scan_stack_data = np.mean(scan_stack_data, axis=1, keepdims=True)
        print(scan_stack_data)
    else:
        scan_stack_data = scan_stack_data[:, 0, :, :, :]

    if probe_type == "Linear":
        if ny != 1:
            raise ValueError("For linear probes, the Y dimension should be 1.")
        # For linear probes, we assume N_scan corresponds to different scan lines, and we want to
        # create a 3D volume where each scan line is a slice in the Y dimension.
        # Thus, we assume dy (distance between adjacent scan lines) is small
        # enough to consider it as a voxel in the Y dimension.

        stack_processed = scan_stack_data.squeeze().transpose([2, 0, 1])
    else:
        print(f"Probe type {probe_type} not recognized. Returning original data.")
        stack_processed = scan_stack_data

    if verbose:
        print("Processing scan stack data...")
        print("Original scan_stack_data shape:", scan_stack_data.shape)
        print("Processed scan_stack_data shape:", stack_processed.shape)

    return stack_processed


def image2D_from_scan(
    scan_data, *, probe_type="Linear", mean_frames=True, verbose=False
):
    """
    Process a 2D scan data array and return a 2D data array suitable for visualization.
    Args:
        scan_data (np.ndarray): 4D array with shape (frames, Z_vox, Y_vox, X_vox).
        probe_type (str): Type of probe used for the scan ('Linear' or other).
        mean_frames (bool): If True, average over the time dimension.
        verbose (bool): If True, print additional information about the processing.
    Returns:
        np.ndarray: A 2D array with shape (X_vox, Y_vox=1, Z_vox) suitable for visualization.
    """

    if scan_data.ndim != 4:
        raise ValueError(
            "Input scan_data must be a 4D array with shape (frames, Z_vox, Y_vox, X_vox)"
        )

    nt, nz, ny, nx = scan_data.shape

    if mean_frames and nt > 1:
        # If mean_frames is True, average over the time dimension
        scan_data = np.mean(scan_data, axis=0, keepdims=False)
    else:
        scan_data = scan_data[0, :, :, :]

    if probe_type == "Linear":
        if ny != 1:
            raise ValueError("For linear probes, the Y dimension should be 1.")

        scan_processed = scan_data.transpose([2, 1, 0])
    else:
        print(f"Probe type {probe_type} not recognized. Returning original data.")
        scan_processed = scan_data

    if verbose:
        print("Processing scan data...")
        print("Original scan_data shape:", scan_data.shape)
        print("Processed scan_data shape:", scan_processed.shape)
    return scan_processed


# --------------------- mesh creation functions ---------------------
# These functions create 3D or 2D meshes from the processed scan data for visualization using PyVista.


def create_3D_volume_mesh(data):
    """
    Create a 3D mesh from the processed scan data for Doppler visualization.
    Args:
        processed_scan_3D (np.ndarray): The processed 3D scan data.
        clip (tuple, optional): A tuple of two values (minclip_value, maxclip_value) to clip the Doppler data.
    Returns:
        pv.Plotter: The PyVista plotter with the Doppler mesh added.
    """

    # Create the 3D UniformGrid
    doppler3D_voxels = pv.ImageData(dimensions=data.shape)

    # Attach pressure data to the grid
    doppler3D_voxels.point_data["doppler"] = data.ravel(
        order="F"
    )  # VERY important: Fortran order

    return doppler3D_voxels


def create_2D_image_mesh(data):
    """
    Compute a 2D image mesh from a B-mode ultrasound image and a transformation matrix.
    Args:
        data (np.ndarray): 2D array of B-mode ultrasound data in dB.
    Returns:
        pv.StructuredGrid: A structured grid representing the 2D image in world coordinates.
    """
    H, W = data.shape

    # Create grid indices
    i_grid, j_grid = np.meshgrid(np.arange(W), np.arange(H))

    # Construct homogeneous voxel coordinates [j, 0, i, 1]
    points_voxel = np.stack(
        [j_grid, np.zeros_like(j_grid), i_grid, np.ones_like(j_grid)], axis=-1
    )
    points_voxel_flat = points_voxel.reshape(-1, 4).T  # Shape (4, H*W)

    points_voxel_flat  # Shape (4, H*W)

    # Create structured grid
    xx = points_voxel_flat[0].reshape(H, W)
    yy = points_voxel_flat[1].reshape(H, W)
    zz = points_voxel_flat[2].reshape(H, W)
    grid = pv.StructuredGrid(xx, yy, zz)
    grid["doppler"] = data.ravel(order="F").astype(
        np.float32
    )  # Add scalar data # FORTRAN ORDER IMPORTANT
    return grid


# --------------------- DopplerScan class ---------------------
# This class represents a Doppler scan, allowing for loading, processing, and visualization of the scan data.


class DopplerScan:
    def __init__(
        self,
        scan_PATH=None,
        bps_PATH=None,
        *,
        probe_type="Linear",
        mean_frames=True,
        verbose=False,
    ):
        self.scan_PATH = scan_PATH
        self.bps_PATH = bps_PATH
        self.probe_type = probe_type
        self.mean_frames = mean_frames
        self.verbose = verbose
        self.load_scan(scan_PATH, bps_PATH=bps_PATH)
        self.reset_mesh()  # Initialize the PyVista mesh attribute

    def load_scan(self, scan_PATH, bps_PATH=None):
        """
        Load the scan data from the specified path and pre-process it.
        Args:
            scan_PATH (str): The path to the scan file.
            bps_PATH (str, optional): The path to the BPS file. If None, no BPS metadata will be loaded.
        """

        try:
            # Load the scan data from the file
            scan_data, scan_metadata = read_scan(scan_PATH)
        except Exception as e:
            raise ValueError(f"Error loading scan data from {scan_PATH}: {e}")

        print(f"Scan data loaded from {scan_PATH}. Shape: {scan_data.shape}")
        self.scan_PATH = scan_PATH

        # Pre-process it
        if scan_data.ndim == 5:
            # If the scan data is 5D, we assume it is a stack of 2D scans
            # [N_scan, frames, Z_vox, Y_vox, X_vox] (2D, then Y_vox is 1)
            self.data = volume3D_from_scan_stack(
                scan_data,
                probe_type=self.probe_type,
                mean_frames=self.mean_frames,
                verbose=self.verbose,
            )
            self.scan_type = "3D"
        elif scan_data.ndim == 4:
            # If the scan data is 4D, we assume it is a single 2D scan
            # [frames, Z_vox, Y_vox, X_vox]  (2D, then Y_vox is 1)
            self.scan_type = "2D"
            self.data = image2D_from_scan(
                scan_data,
                probe_type=self.probe_type,
                mean_frames=self.mean_frames,
                verbose=self.verbose,
            )
        else:
            raise ValueError(
                "Input scan_data must be a 4D or 5D array with shape (frames, Z_vox, Y_vox, X_vox) or (N_scan, frames, Z_vox, Y_vox, X_vox)"
            )

        self.probeToLab = scan_metadata["probeToLab"][()]
        print(
            f"Probe to Lab transformation matrix loaded. Shape: {self.probeToLab.shape}"
        )
        if self.probeToLab.ndim == 3:
            # If probeToLab is a 3D array, we assume the first slice corresponds to the probe to lab transformation
            print("Probe to Lab transformation matrix is 3D. Taking the first slice.")
            self.probeToLab = self.probeToLab[0].squeeze()
        elif self.probeToLab.ndim > 3 or self.probeToLab.shape != (4, 4):
            raise ValueError(
                "probeToLab must be a 3D array with shape (4, 4) or (N_scan, 4, 4)"
            )

        self.voxelsToProbe = scan_metadata["voxelsToProbe"][()]

        # Attributes to hold the scan data and metadata
        self.set_qform(self.get_qform())
        self.set_sform(self.get_sform(bps_PATH=bps_PATH))
        del scan_data, scan_metadata  # Free memory
        print(f"DopplerScan {self.scan_type} object created successfully.")

    def set_qform(self, qform):
        """
        Set the qform matrix for the scan.
        Args:
            qform (np.ndarray): The qform matrix to set.
        """
        self.qform = qform

    def set_sform(self, sform):
        """
        Set the sform matrix for the scan.
        Args:
            sform (np.ndarray): The sform matrix to set.
        """
        self.sform = sform

    def reset_mesh(self):
        """
        Set the PyVista mesh for the scan.
        Args:
            pv_mesh (pv.Plotter): The PyVista mesh to set.
        """
        self.pv_mesh = self.get_mesh(self.qform)
        print("PyVista mesh reset on Lab coordinate system.")

    def get_qform(self):
        invert_z = np.diag([1, 1, -1, 1])  # Invert z-axis
        # We must invert the z-axis to match the probe coordinate system
        return (
            invert_z @ self.probeToLab @ self.voxelsToProbe
        )  # voxel -> scabber (Lab) coordinates transformation

    def get_sform(self, bps_PATH):
        if bps_PATH is not None:
            print("BPS path provided. Using it to compute the sform matrix.")
            self.bps_PATH = bps_PATH  # Reset the BPS path on the object

        elif self.bps_PATH is not None:
            print("No BPS path provided, but BPS path was found on the scan object.")
        else:
            print(
                "No BPS path provided. No way to get to the Brain space. Returning s_form as identity."
            )
            return np.eye(4)

        bps_metadata = read_bps(self.bps_PATH, verbose=False)
        self.BrainToLab = bps_metadata["BrainToLab"][()]

        del bps_metadata
        invert_z = np.diag([1, 1, -1, 1])  # Invert z-axis

        return (
            invert_z @ np.linalg.inv(self.BrainToLab) @ invert_z @ self.qform
        )  # voxel -> anatomical (Brain) coordinates transformation

    def get_mesh(self, T_matrix=None, *, dB=True):
        if dB:
            data = self.dB(inplace=False)
        else:
            data = self.data

        if self.scan_type == "3D":
            # Create a 3D mesh from the scan data
            mesh = create_3D_volume_mesh(data)
        elif self.scan_type == "2D":
            # Create a 2D mesh from the scan data
            mesh = create_2D_image_mesh(data.squeeze())
        else:
            raise ValueError(
                "Something went wrong when loading the scan data. \n"
                "The scan type should be either '3D' or '2D', but it is not recognized."
            )

        if T_matrix is not None:
            # If a transformation matrix is provided, apply it to the mesh
            mesh.transform(T_matrix, inplace=True)

        return mesh

    def transform(self, pv_mesh=None, T_matrix=None, *, inplace=False):
        """
        Transform the scan data using a transformation matrix.
        Args:
            T_matrix (np.ndarray): The transformation matrix to apply.
            inplace (bool): If True, modify the scan data in place.
        Returns:
            np.ndarray: The transformed scan data.
        """
        if self.pv_mesh is None and pv_mesh is None:
            raise ValueError("No mesh provided. pv_mesh is NONE.")

        if pv_mesh is None and not inplace:
            pv_mesh = self.pv_mesh.copy()
        elif pv_mesh is None and inplace:  # if inplace, transform the existing mesh
            pv_mesh = self.pv_mesh

        if T_matrix is not None:
            pv_mesh.transform(T_matrix, inplace=True)
        else:
            print(
                "No transformation matrix provided. No transformation applied to the mesh."
            )
        return pv_mesh

    def dB(self, data=None, *, inplace=False):
        """
        Convert the scan data to dB scale.
        Args:
            data (np.ndarray): The scan data to convert.
        Returns:
            np.ndarray: The scan data in dB scale.
        """
        if data is None:
            data = self.data

        if np.any(data == 0):
            print("Data contains zero values. No convertion to dB is done.")
            return data

        if inplace:
            self.data = 20 * np.log10(np.abs(data) / np.max(np.abs(data)))
            return self.data
        else:
            return 20 * np.log10(np.abs(data) / np.max(np.abs(data)))

    def show(self, *, dB=True, **kwargs):
        nx, ny, nz = self.data.shape
        print(f"Plotting y-planes of scan with shape ({nx}, {ny}, {nz})")

        if self.scan_type == "3D":
            ncols = 9
            nrows = ny // ncols + (ny % ncols > 0)
        else:
            ncols = 1
            nrows = 1

        fig, ax = plt.subplots(
            nrows, ncols, figsize=(ncols * 5, nrows * 5), facecolor="black"
        )

        ax = ax.flatten() if nrows > 1 else [ax]
        if dB:
            for i in range(ny):
                ax[i].imshow(
                    self.dB(self.data[:, i, :].squeeze().T), cmap="gray", **kwargs
                )
                ax[i].set_title(f"Y-plane {i + 1}", color="white")
                ax[i].axis("off")
        else:
            for i in range(ny):
                ax[i].imshow(self.data[:, i, :].squeeze().T, cmap="gray", **kwargs)
                ax[i].set_title(f"Y-plane {i + 1}", color="white")
                ax[i].axis("off")

        for i in range(ny, nrows * ncols):
            ax[i].axis("off")

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def summary(self):
        """
        Print a summary of the DopplerScan object.
        """
        print("----------DopplerScan Summary:----------")
        for key, value in self.__dict__.items():
            if key == "data":
                if isinstance(value, np.ndarray):
                    print(f"{key}: array with shape {value.shape}")
                else:
                    print(f"{key}: {value}")
            else:
                print(f"{key}: {value}")

    def clean(self):
        """
        Clean the DopplerScan object by deleting the scan data and metadata.
        """
        self.scan_PATH = None
        self.bps_PATH = None
        self.data = None
        self.scan_type = None
        self.probeToLab = None
        self.voxelsToProbe = None
        self.qform = None
        self.sform = None
        self.pv_mesh = None
        print("DopplerScan object cleaned.")

    def __repr__(self):
        return f"DopplerScan(scan_PATH={self.scan_PATH}, bps_PATH={self.bps_PATH}, probe_type={self.probe_type}, mean_frames={self.mean_frames}, scan_type={self.scan_type})"

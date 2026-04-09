import brainglobe_space as bg_space
import numpy as np
import pyvista as pv
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi import show_atlases as bg_show_atlases

# ---------------------------------------------------------------------------
# Known atlas landmark data
# ---------------------------------------------------------------------------
# Each entry maps an atlas name (as returned by bg_atlas.metadata["name"])
# to a dict with either a ``whs_voxels`` key (origin/bregma/lambda in voxels)
# or a ``manual_fit`` key (origin voxel + bregma-lambda distance in µm).
#
# Add new atlases here to register them without touching the class logic.
_ATLAS_LANDMARKS: dict = {
    "whs_sd_rat": {
        "whs_voxels": {
            "origin": np.array([244, 623, 248]),
            "bregma": np.array([246, 653, 440]),
            "lambda": np.array([244, 442, 434]),
        }
    },
    "allen_mouse": {
        "manual_fit": {
            "origin": np.array([228, 330, 118]),
            "bregma_lambda_um": 2300,  # distance between bregma and lambda in µm
        }
    },
}


class BG_Atlas:
    def __init__(
        self,
        atlas_name=None,
        region_names=None,
        *,
        whs_voxels=None,
        manual_fit=None,
        verbose=False,
    ):
        if atlas_name is None:
            print("No atlas name provided. Showing available BrainGlobe atlases:")
            self.show_atlases()

        else:
            self.set_bgatlas(
                atlas_name,
                region_names=region_names,
                whs_voxels=whs_voxels,
                manual_fit=manual_fit,
                verbose=verbose,
            )

    def set_bgatlas(
        self,
        atlas_name,
        region_names=None,
        *,
        whs_voxels=None,
        manual_fit=None,
        verbose=False,
    ):
        """
        Set a new BrainGlobe Atlas and update the attributes.
        Args:
            atlas_name (str): The name of the BrainGlobe Atlas to load.
            region_names (str or list/tuple/set): The name(s) of the structure(s) to retrieve.
            whs_voxels (dict, optional): A dictionary containing the WHS origin, bregma, and lambda voxels.
            manual_fit (dict, optional): A dictionary containing manual fit parameters.
            verbose (bool, optional): Whether to print verbose output.
        """
        self.atlas_name = atlas_name

        try:
            self.bg_atlas = BrainGlobeAtlas(atlas_name)
        except Exception as e:
            print(f"Error loading atlas '{atlas_name}': {e}")
            self.show_atlases()
            return

        if region_names is None:
            print("No region names provided. Defaulting to 'root'.")
            region_names = "root"

        self.region_names = region_names
        self.whs_voxels = whs_voxels
        self.manual_fit = manual_fit
        self.verbose = verbose

        self.bgatlasToBrain = self.get_bgatlasToBrain(
            self.bg_atlas,
            whs_voxels=self.whs_voxels,
            manual_fit=self.manual_fit,
            verbose=self.verbose,
        )  # Get the transformation matrix from BrainGlobe Atlas to Brain-space (BPS Atlas)

        self.reset_mesh()
        print(f"BrainGlobe Atlas '{self.atlas_name}' loaded successfully.")

    def show_atlases(self):
        """
        Show available BrainGlobe atlases.
        """
        bg_show_atlases()

    def get_bgatlasToBrain(
        self, bg_atlas, *, whs_voxels=None, manual_fit=None, verbose=False
    ):
        """
        Compute the 4×4 affine matrix from BrainGlobe Atlas voxel space to
        normalised Brain-space (BPS).

        The function first checks the module-level ``_ATLAS_LANDMARKS`` registry
        for pre-calibrated landmark data.  User-supplied ``whs_voxels`` or
        ``manual_fit`` take precedence over the registry (allowing per-subject
        overrides).  If neither is available the atlas is returned in its raw
        voxel space with a warning.

        Parameters
        ----------
        bg_atlas : BrainGlobeAtlas
            The loaded BrainGlobe atlas object.
        whs_voxels : dict, optional
            Landmark voxel coordinates with keys ``"origin"``, ``"bregma"``,
            ``"lambda"`` (each a length-3 ndarray).
        manual_fit : dict, optional
            Coarser alignment via keys ``"origin"`` (voxel, length-3 ndarray)
            and ``"bregma_lambda_um"`` (scalar distance in µm).
        verbose : bool, optional
            Print intermediate matrices for debugging.  Default False.

        Returns
        -------
        bgatlasToBrain : ndarray, shape (4, 4)
            Homogeneous transform from atlas voxel indices to normalised
            brain coordinates.
        """
        name = bg_atlas.metadata["name"]
        resolution = np.array(bg_atlas.metadata["resolution"])

        # Check registry for pre-calibrated landmarks (user args take priority)
        if whs_voxels is None and manual_fit is None:
            known = _ATLAS_LANDMARKS.get(name)
            if known is not None:
                whs_voxels = known.get("whs_voxels")
                manual_fit = known.get("manual_fit")
                method = "whs_voxels" if whs_voxels is not None else "manual_fit"
                print(f"Using pre-calibrated landmarks for '{name}' ({method}).")
            else:
                print(
                    f"WARNING: No landmark data for atlas '{name}'.\n"
                    "Provide whs_voxels or manual_fit, or add the atlas to "
                    "_ATLAS_LANDMARKS.\nNo normalisation will be applied — "
                    "coordinates returned in raw voxel space."
                )
        elif whs_voxels is not None:
            print("Using provided WHS voxels for alignment.")
        else:
            print("Using provided manual fit for alignment.")

        if whs_voxels is not None:
            bregma2lambda = np.linalg.norm(
                (whs_voxels["lambda"] - whs_voxels["bregma"])
            )
            normalize_bregma2sigma = np.diag(
                np.concatenate(([1 / bregma2lambda] * 3, [1]))
            )  # Add a 1 for homogeneous coordinates
            set_whs_origin = np.eye(4)
            set_whs_origin[:3, 3] = -whs_voxels["origin"]

        elif manual_fit is not None:
            # If bregma_lambda_mm is provided, we use it scale the atlas
            bregma2lambda = np.linalg.norm(manual_fit["bregma_lambda_um"] / resolution)
            normalize_bregma2sigma = np.diag(
                np.concatenate(([1 / bregma2lambda] * 3, [1]))
            )
            set_whs_origin = np.eye(4)
            set_whs_origin[:3, 3] = -manual_fit["origin"]

        else:
            normalize_bregma2sigma = np.eye(4)
            set_whs_origin = np.eye(4)

        # Take to whs orientation RAS+
        target_space = bg_space.AnatomicalSpace(
            "lpi"
        )  # New resolution to change the units from um to mm
        set_whs_orientation = bg_atlas.space.transformation_matrix_to(target_space)

        # Create the transformation matrix from BrainGlobe Atlas to whs Atlas
        resolution2Voxel = np.diag(
            np.concatenate((1 / resolution, [1]))
        )  # Add a 1 for homogeneous coordinates

        # Create the transformation matrix from BrainGlobe Atlas to BPS Atlas
        bgatlasToBrain = (
            normalize_bregma2sigma
            @ set_whs_origin
            @ set_whs_orientation
            @ resolution2Voxel
        )

        if verbose:
            print("resolutionToVoxel:\n", resolution2Voxel)
            print("set_whs_orientation:\n", set_whs_orientation)
            print("set_whs_origin:\n", set_whs_origin)
            print("normalize_bregma2sigma:\n", normalize_bregma2sigma)
            print("bgatlasToBrain:\n", bgatlasToBrain)

        return bgatlasToBrain

    def get_pv_mesh_from_atlas(self, bg_atlas, region_names):
        """
        Get a PyVista mesh from the BrainGlobe Atlas for a given structure name.
        Args:
            bg_atlas (BrainGlobeAtlas): The BrainGlobe Atlas object.
            structure_name (str or list/tuple/set): The name of the structure(s) to retrieve.
        Returns:
            pv_mesh (dict): A dictionary containing the PyVista mesh(es) for the specified structure(s).
        """
        pv_mesh = {}
        if not isinstance(region_names, str):
            try:
                n_reg = len(region_names)
                for i, region_name in enumerate(region_names):
                    print(f"Processing region {i + 1}/{n_reg}: {region_name}")
                    print(
                        f"Found structure: {bg_atlas.structures[region_name]['name']}"
                    )
                    pv_mesh[region_name] = pv.read(
                        bg_atlas.structures[region_name]["mesh_filename"]
                    )
            except KeyError as e:
                raise ValueError(
                    f"Error: {e}. Please check the structure names provided."
                )

            return pv_mesh
        else:
            print(f"Structure name: {bg_atlas.structures[region_names]['name']}")
            pv_mesh[region_names] = pv.read(
                bg_atlas.structures[region_names]["mesh_filename"]
            )
            return pv_mesh

    def transform(self, T_matrix=None, pv_mesh=None, *, inplace=False):
        """
        Transform the PyVista mesh using a transformation matrix.
        Args:
            transformation_matrix (np.ndarray): The transformation matrix to apply to the mesh.
        Returns:
            pv_mesh_transformed (dict): A dictionary containing the transformed PyVista mesh(es).
        """
        if pv_mesh is None and not inplace:
            if self.pv_mesh is None:
                raise ValueError("No mesh available. Load mesh before transforming.")
            pv_mesh = {}
            for region_name, mesh in self.pv_mesh.items():
                pv_mesh[region_name] = mesh.copy()
        elif pv_mesh is None and inplace:  # if inplace, transform the existing mesh
            # If no mesh is provided, and inplace is True use the atlas mesh
            if self.pv_mesh is None:
                raise ValueError("No mesh available. Load mesh before transforming.")
            pv_mesh = self.pv_mesh

        if T_matrix is not None:
            trans_pv_mesh = {}
            assert pv_mesh is not None
            for region_name, mesh in pv_mesh.items():
                trans_pv_mesh[region_name] = mesh.transform(T_matrix, inplace=True)
        else:
            print(
                "No transformation matrix provided. No transformation applied to the mesh."
            )
            trans_pv_mesh = pv_mesh
        if not inplace:
            return trans_pv_mesh

    def reset_mesh(self):
        """
        Reset the PyVista mesh to the original mesh.
        Returns:
            pv_mesh (dict): A dictionary containing the original PyVista mesh(es).
        """
        self.pv_mesh = self.get_pv_mesh_from_atlas(self.bg_atlas, self.region_names)
        self.transform(self.bgatlasToBrain, inplace=True)
        print("PyVista mesh atlas reset.")
        return self.pv_mesh

    def summary(self):
        """
        Print a summary of the BG_Atlas object.
        """
        print("----------BG_Atlas Summary:----------")
        for key, value in self.__dict__.items():
            if key == "pv_mesh":
                if isinstance(value, dict):
                    print(f"{key}: dictionary with {len(value)} regions")
                else:
                    print(f"{key}: {value}")
            else:
                print(f"{key}: {value}")

    def clean(self):
        """
        Clean the BG_Atlas object by removing the PyVista mesh and other attributes.
        """
        self.pv_mesh = None
        self.bg_atlas = None
        self.bgatlasToBrain = None
        self.region_names = None
        self.whs_voxels = None
        self.manual_fit = None
        print("BG_Atlas object cleaned.")

    def __repr__(self):
        return f"BG_Atlas(atlas_name={self.atlas_name}, region_names={self.region_names}, whs_voxels={self.whs_voxels}, manual_fit={self.manual_fit})"

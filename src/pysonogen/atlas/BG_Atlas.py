
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi import show_atlases as bg_show_atlases
import brainglobe_space as bg_space
import numpy as np
import pyvista as pv


class BG_Atlas:
    def __init__(self, atlas_name = 'whs_sd_rat_39um', region_names='root', *,
                 show_atlases = False,
                 whs_voxels = None, manual_fit = None, verbose = False):
        
        if show_atlases:
            bg_show_atlases()
        self.atlas_name = atlas_name
        self.bg_atlas = BrainGlobeAtlas(atlas_name)
        
        self.region_names = region_names
        self.whs_voxels = whs_voxels
        self.manual_fit = manual_fit
        self.verbose = verbose

        self.bgatlasToBrain = self.get_bgatlasToBrain(self.bg_atlas,
                     whs_voxels = self.whs_voxels, manual_fit = self.manual_fit, verbose = self.verbose)  # Get the transformation matrix from BrainGlobe Atlas to Brain-space (BPS Atlas)

        self.reset_mesh()
        print(f"BrainGlobe Atlas '{self.atlas_name}' loaded successfully.")

    def get_bgatlasToBrain(self, bg_atlas, *, whs_voxels = None, manual_fit = None, verbose = False):
        """
        Compute the transformation matrix from BrainGlobe Atlas to Brain-space (BPS Atlas).
        Args:
            bg_atlas (BrainGlobeAtlas): The BrainGlobe Atlas object.
            brain_mesh_dict (dict, optional): A dictionary of PyVista meshes for different brain regions.
            whs_voxels (dict, optional): A dictionary containing the WHS origin, bregma, and lambda voxels.
        Returns:
            np.ndarray: The transformation matrix from BrainGlobe Atlas to Brain-space (BPS Atlas).
            transformed_brain_mesh_dict (dict, optional): A dictionary of transformed PyVista meshes for different brain regions, if provided.
        """
        name = bg_atlas.metadata['name']
        # Get the information to compute transformation matrix
        resolution = np.array(bg_atlas.metadata['resolution']) 

        if  name == "whs_sd_rat":
            # We need to apply a specific transformation to align the BrainGlobe atlas
            # with the WHS atlas space.
            print(f"WHS origin, bregma, and lambda coordinates available for `{name}`.")
            whs_voxels = {
            'origin' : np.array([244, 623, 248]),
            'bregma' : np.array([246, 653, 440]),
            'lambda' : np.array([244, 442, 434]),
            }
        elif name == "allen_mouse":
            # We need to apply a specific transformation to align the BrainGlobe atlas
            # with the WHS atlas space.
            print(f"WHS origin, bregma, and lambda coordinates available for `{name}`.")
            manual_fit = {
            'origin' : np.array([228, 330, 118]),
            'bregma_lambda_um' : 2300,  # Distance between bregma and lambda in micrometers (4mm)
            }
        elif whs_voxels is not None:
            # If whs_voxels is provided, we use it to align the atlas
            print("Using provided WHS voxels for alignment.")
            pass
        elif manual_fit is not None:
            # If manual_fit is provided, we use it to align the atlas
            print("Using provided manual fit for alignment.")
            pass
        else:
            print("WARNING: No available information for this atlas. \n" \
            "To compute BrainGlobe to Brain-space, the WHS origin, bregma, and lambda voxels must be provided, \n" \
            "or a manual fit could be done by setting the `manual_fit` dict with bregma_sigma_um and origin_voxel keys.\n" \
            "No subject normalization will be applied, and the atlas will be returned in its voxel space.")

        if whs_voxels is not None:
            bregma2lambda = np.linalg.norm((whs_voxels['lambda'] - whs_voxels['bregma']))
            normalize_bregma2sigma =  np.diag(np.concatenate(([1/bregma2lambda]*3, [1])))  # Add a 1 for homogeneous coordinates
            set_whs_origin = np.eye(4)
            set_whs_origin[:3, 3] = -whs_voxels['origin']
            
        elif manual_fit is not None:
            # If bregma_lambda_mm is provided, we use it scale the atlas
            bregma2lambda = np.linalg.norm(manual_fit['bregma_lambda_um']/resolution)
            normalize_bregma2sigma =  np.diag(np.concatenate(([1/bregma2lambda]*3, [1])))        
            set_whs_origin = np.eye(4)
            set_whs_origin[:3, 3] = -manual_fit['origin']

        else:
            normalize_bregma2sigma = np.eye(4)
            set_whs_origin = np.eye(4)

        # Take to whs orientation RAS+
        target_space = bg_space.AnatomicalSpace("lpi") # New resolution to change the units from um to mm
        set_whs_orientation = bg_atlas.space.transformation_matrix_to(target_space)

        # Create the transformation matrix from BrainGlobe Atlas to whs Atlas
        resolution2Voxel =  np.diag(np.concatenate((1/resolution, [1])))  # Add a 1 for homogeneous coordinates

        # Create the transformation matrix from BrainGlobe Atlas to BPS Atlas
        bgatlasToBrain =  normalize_bregma2sigma @ set_whs_origin @ set_whs_orientation @ resolution2Voxel 

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
                    print(f"Processing region {i+1}/{n_reg}: {region_name}")
                    print(f"Found structure: {bg_atlas.structures[region_name]['name']}")
                    pv_mesh[region_name] = pv.read(bg_atlas.structures[region_name]['mesh_filename'])
            except KeyError as e:
                raise ValueError(f"Error: {e}. Please check the structure names provided.")
            
            return pv_mesh
        else:
            print(f"Structure name: {bg_atlas.structures[region_names]['name']}")
            pv_mesh[region_names] = pv.read(bg_atlas.structures[region_names]['mesh_filename'])
            return pv_mesh
    
    def transform(self, T_matrix = None, pv_mesh = None, *, inplace = False):
        """
        Transform the PyVista mesh using a transformation matrix.
        Args:
            transformation_matrix (np.ndarray): The transformation matrix to apply to the mesh.
        Returns:
            pv_mesh_transformed (dict): A dictionary containing the transformed PyVista mesh(es).
        """
        if pv_mesh is None and not inplace:
            pv_mesh = {}
            for region_name, mesh in self.pv_mesh.items():
                pv_mesh[region_name] = mesh.copy()
        elif pv_mesh is None and inplace:  # if inplace, transform the existing mesh
            # If no mesh is provided, and inplace is True use the atlas mesh
            pv_mesh = self.pv_mesh

        if T_matrix is not None:
            trans_pv_mesh = {}
            for region_name, mesh in pv_mesh.items():
                trans_pv_mesh[region_name] = mesh.transform(T_matrix, inplace = True)
        else:
            print("No transformation matrix provided. No transformation applied to the mesh.")
            trans_pv_mesh = pv_mesh

        return pv_mesh     

    
    def reset_mesh(self):
        """
        Reset the PyVista mesh to the original mesh.
        Returns:
            pv_mesh (dict): A dictionary containing the original PyVista mesh(es).
        """
        self.pv_mesh = self.get_pv_mesh_from_atlas(self.bg_atlas, self.region_names)
        self.transform(self.bgatlasToBrain, inplace = True)
        print("PyVista mesh atlas reset.")
        return self.pv_mesh

    
    def summary(self):
        """
        Print a summary of the BG_Atlas object.
        """
        print("----------BG_Atlas Summary:----------")
        for key, value in self.__dict__.items():
            if key == 'pv_mesh':
                if isinstance(value, dict):
                    print(f"{key}: dictionary with {len(value)} regions")
                else:
                    print(f"{key}: {value}")
            else:
                print(f"{key}: {value}")
                
    def __repr__(self):
        return f"BG_Atlas(atlas_name={self.atlas_name}, region_names={self.region_names}, whs_voxels={self.whs_voxels}, manual_fit={self.manual_fit})"
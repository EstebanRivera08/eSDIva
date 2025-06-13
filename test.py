

MAIN_FOLDER_PATH = r"..\src\pysonogen\datatype\Silvia"

bps_PATH = MAIN_FOLDER_PATH + r"\3Dscan_angio3D.source.bps"
file_scan_3D_PATH = MAIN_FOLDER_PATH + r"\3Dscan_angio3D.source.scan"
file_scan_2D_PATH = MAIN_FOLDER_PATH + r"\2Dscan.source.scan"

# Let's take the 008 scan. It shouwld be placed at one sagittal plane on the left side of the brain

Doppler3D = DopplerScan(scan_PATH=file_scan_3D_PATH, bps_PATH = bps_PATH)
Doppler3D.summary()
Doppler3D.show()

Doppler2D = DopplerScan(scan_PATH=file_scan_2D_PATH)
Doppler2D.summary()
Doppler2D.show(interpolation = 'bilinear')

atlas_name = "allen_mouse_25um"

if atlas_name == "allen_mouse_25um":
    region_names = ("root")
elif atlas_name == "whs_sd_rat_39um":
    region_names = ("root", "M1", "S1-hl")

Brain_Atlas = BG_Atlas(atlas_name, region_names = region_names)
Brain_Atlas.summary()

invertz = np.diag([1, 1, -1, 1])  # Invert z-axis for the transducer mesh
Brain_Atlas.reset_mesh()  # Reset the Brain Atlas mesh to the original mesh
Brain_Atlas.transform(T_matrix = invertz @ Doppler3D.BrainToLab  , inplace=True) 



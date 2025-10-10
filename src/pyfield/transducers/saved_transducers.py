from .linear import LinearArrayTransducer
from .matrix import MatrixArrayTransducer

# -------------- Define the domino transducer --------------


def Domino():
    """
    Create a Domino transducer with specific parameters.

    Returns:
        LinearArrayTransducer: An instance of the Domino transducer.
    """

    # Create transducer with elevation focus
    n_elements = 128
    pitch_mm = 0.11
    element_width_mm = 0.108
    element_height_mm = 1.5
    elevation_focus_mm = 8  # mm
    tx_freq = 12.5  # MHz

    return LinearArrayTransducer(
        n_elements=n_elements,
        element_width_mm=element_width_mm,
        element_height_mm=element_height_mm,
        elevation_focus_mm=elevation_focus_mm,
        kerf_mm=pitch_mm - element_width_mm,
        no_sub_x=1,
        no_sub_y=10,
        frequency_Hz=tx_freq * 1e6,  # MHz),
    )


# -------------- Define the ZeUS transducer --------------


def Zeus_Matrix():
    """
    Create a Zeus Matrix transducer with specific parameters.
    Returns:
        MatrixArrayTransducer: An instance of the Zeus Matrix transducer.
    """

    # Create transducer with elevation focus
    tx_N_elem_x = 55
    tx_N_elem_y = 55
    tx_elem_width_mm = 0.29
    tx_elem_height_mm = 0.29
    tx_pitch_mm = 0.3
    tx_kerf_x_mm = tx_pitch_mm - tx_elem_width_mm
    tx_kerf_y_mm = tx_pitch_mm - tx_elem_height_mm
    tx_freq = 10  # MHz
    directivity = 30

    return MatrixArrayTransducer(
        N_elem_x=tx_N_elem_x,
        N_elem_y=tx_N_elem_y,
        elem_width_mm=tx_elem_width_mm,
        elem_height_mm=tx_elem_height_mm,
        kerf_x_mm=tx_kerf_x_mm,
        kerf_y_mm=tx_kerf_y_mm,
        no_sub_x=2,
        no_sub_y=2,
        frequency_Hz=tx_freq * 1e6,
        dir_angle_deg=directivity,
    )

"""Pre-defined transducer configurations."""

from .linear import LinearArrayTransducer
from .matrix import MatrixArrayTransducer

# -------------- Define the domino transducer --------------


def Domino():
    """Create a Domino transducer with specific parameters.

    Returns
    -------
    LinearArrayTransducer
        An instance of the Domino transducer.
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


def Zeus_Matrix(no_sub_x=2, no_sub_y=2):
    """Create a Zeus Matrix transducer with specific parameters.

    Parameters
    ----------
    no_sub_x : int, default: 2
        Patch subdivisions per element along x (lateral). Each 0.29 mm element is
        split into ``no_sub_x`` x ``no_sub_y`` rectangular patches for the SIR; 1
        keeps the whole element as a single patch (useful for subdivision-convergence
        or benchmark studies).
    no_sub_y : int, default: 2
        Patch subdivisions per element along y (elevation).

    Returns
    -------
    MatrixArrayTransducer
        An instance of the Zeus Matrix transducer.
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
        n_elements_x=tx_N_elem_x,
        n_elements_y=tx_N_elem_y,
        element_width_mm=tx_elem_width_mm,
        element_height_mm=tx_elem_height_mm,
        kerf_x_mm=tx_kerf_x_mm,
        kerf_y_mm=tx_kerf_y_mm,
        no_sub_x=no_sub_x,
        no_sub_y=no_sub_y,
        frequency_Hz=tx_freq * 1e6,
        dir_angle_deg=directivity,
    )

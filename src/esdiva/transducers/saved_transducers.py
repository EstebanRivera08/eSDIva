"""Pre-defined transducer configurations."""

from .linear import LinearArrayTransducer
from .matrix import MatrixArrayTransducer


def LinearArrayExample():
    """Ready-made 128-element linear array at 12.5 MHz.

    Pitch 0.11 mm (element 0.108 mm wide, 1.5 mm tall) with a cylindrical
    elevation lens of radius 8 mm. Each element is split into 1 x 10 patches so
    the curved lens is resolved in elevation. Use it for demos and quick checks.

    Returns
    -------
    LinearArrayTransducer
        The configured linear array.
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


def MatrixArrayExample(no_sub_x=2, no_sub_y=2):
    """Ready-made 55 x 55 matrix array at 10 MHz for 3-D imaging.

    Pitch 0.3 mm (element 0.29 x 0.29 mm), element directivity limited to 30 deg.

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
        The configured matrix array.
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

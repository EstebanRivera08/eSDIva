"""Runtime projection for a planned acquisition or field sweep.

Wall-clock numbers are not asserted — they are a property of whatever machine
the suite runs on, which is the whole reason the helper exists. What is checked
is that the projection is self-consistent and that the inputs a real call uses
(a moving cloud, an amplitude array, an emission grid dict) are accepted.
"""

import numpy as np
import pytest

from esdiva.emission import Emission
from esdiva.reception import Reception
from esdiva.utilities import estimate_sequence_runtime


def _probe():
    """Unfocused 4-element array: RX must carry no delays or the RF is weighted."""
    from esdiva.transducers import LinearArrayTransducer

    return LinearArrayTransducer(
        n_elements=4,
        element_width_mm=0.25,
        element_height_mm=5.0,
        kerf_mm=0.05,
        no_sub_x=1,
        no_sub_y=2,
        frequency_Hz=5e6,
    )


@pytest.fixture
def sim():
    return Reception(_probe(), _probe(), verbose=False)


@pytest.fixture
def cloud():
    rng = np.random.default_rng(3)
    return np.column_stack(
        [
            rng.uniform(-2, 2, 200),
            rng.uniform(-1, 1, 200),
            rng.uniform(15, 25, 200),
        ]
    ).astype(np.float32)


class TestEstimateSequenceRuntime:
    def test_projection_is_self_consistent(self, sim, cloud):
        """Total must be per-emission x emissions, over the real scatterer count."""
        est = estimate_sequence_runtime(sim, cloud, n_emissions=50, verbose=False)

        assert est["n_points"] == cloud.shape[0]
        assert est["n_emissions"] == 50
        assert est["seconds_per_emission"] > 0
        assert est["total_seconds"] == pytest.approx(
            est["seconds_per_emission"] * 50, rel=1e-9
        )
        assert est["total_hours"] == pytest.approx(
            est["total_seconds"] / 3600, rel=1e-9
        )
        assert len(est["probes"]) == 2
        assert est["machine"]["cpu_count"] >= 1

    def test_scales_with_emission_count(self, sim, cloud):
        """Doubling the planned emissions doubles the projected total."""
        a = estimate_sequence_runtime(sim, cloud, n_emissions=10, verbose=False)
        b = estimate_sequence_runtime(sim, cloud, n_emissions=20, verbose=False)
        assert b["total_seconds"] / b["seconds_per_emission"] == pytest.approx(
            2 * a["total_seconds"] / a["seconds_per_emission"], rel=1e-9
        )

    def test_accepts_moving_cloud(self, sim, cloud):
        """A (N_events, N_scat, 3) phantom is probed at its first emission."""
        moving = np.stack([cloud] * 4)
        est = estimate_sequence_runtime(sim, moving, n_emissions=4, verbose=False)
        assert est["n_points"] == cloud.shape[0]

    def test_tiny_cloud_does_not_divide_by_zero(self, sim):
        """A cloud smaller than the probe sizes collapses to one probe point."""
        tiny = np.array([[0.0, 0.0, 20.0], [1.0, 0.0, 21.0]], dtype=np.float32)
        est = estimate_sequence_runtime(sim, tiny, n_emissions=3, verbose=False)
        assert np.isfinite(est["seconds_per_emission"])
        assert est["seconds_per_emission"] > 0

    @pytest.mark.parametrize("monochromatic", [True, False])
    def test_emission_field_sweep(self, monochromatic):
        """Emission is probed on its field points; a grid dict counts every point."""
        tx = _probe()
        exc = np.sin(2 * np.pi * 5e6 * np.arange(40) / 100e6).astype(np.float32)
        sim = Emission(tx, excitation=exc, monochromatic=monochromatic, verbose=False)
        grid = {"x_extent": [-3, 3], "y_extent": [0, 0], "z_extent": [10, 20],
                "dx": 0.25, "dy": 1.0, "dz": 0.5}  # fmt: skip
        est = estimate_sequence_runtime(sim, grid, n_emissions=7, verbose=False)
        assert est["n_points"] == 25 * 21
        assert est["total_seconds"] == pytest.approx(
            7 * est["seconds_per_emission"], rel=1e-9
        )

<h1 align="center">🌊 eSDIva</h1>
<p align="center"><b>Efficient Sparse Delta Integration for Vectorized Acoustics.</b></p>
<p align="center">
A friendly acoustic field simulator — your probe into ultrasound fields: <b>fast and exact</b>.
</p>

[![PyPI version](https://img.shields.io/pypi/v/esdiva)](https://pypi.org/project/esdiva/)
[![Python versions](https://img.shields.io/pypi/pyversions/esdiva)](https://pypi.org/project/esdiva/)
[![DOI]()]()
[![codecov](https://codecov.io/gh/EstebanRivera08/eSDIva/graph/badge.svg)](https://codecov.io/gh/EstebanRivera08/eSDIva)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://estebanrivera08.github.io/eSDIva/)

📖 **Documentation:** <https://estebanrivera08.github.io/eSDIva/>

> [!WARNING]
> eSDIva is currently under development. The API is subject to change, and some features may be incomplete or unstable.

eSDIva is an open‑source Spatial Impulse Response (SIR) and pressure‑field simulation library that supports arbitrary transducer geometries composed of small rectangular patches with apodization and delays.
eSDIva implements both the Fully Sampled Trapezoid (FST) and the Sparse Delta Integration (SDI) methods for computing SIRs following the Tupholme–Stepanishen formulation.
FST reproduces the classic Field II approach, while SDI is a new, algorithmically and mathematically improved method that computes the same SIRs — under identical assumptions — but substantially faster; an automatic mode picks the best method for each simulation.

> [!NOTE]
> eSDIva is designed as complementary material to the work presented in [https://arxiv.org/abs/2608.26891]. Its goal is to provide fundamental building blocks that researchers can inspect, reuse, contribute to, or adapt. It also leaves room for community‑driven extensions that integrate naturally with the broader scientific Python ecosystem.
> Utilities such as the integration with the BrainGlobe atlas may still evolve to improve robustness.

### Main Features

- **Transducer objects** — Tools to create and assemble common transducer types: linear arrays, convex arrays, matrix arrays, flat/concave/focused circular transducers, and arbitrary custom arrays. These utilities compute geometric focal laws, generate apodization windows for specified F/D ratios, and more.

- **SIR simulation** — The `H_sir` module computes discrete spatial impulse responses \( h(r, t) \) produced by apertures discretized into rectangular patches. It includes naïve, SDI, and automatic methods implemented with Numba‑accelerated kernels for field‑point‑parallel execution.

- **Emission simulation** — Converts time‑domain SIRs into acoustic pressure fields via the `Emission` class. Supports monochromatic fields (spatial‑only, CW amplitude at `fc`) and broadband transient simulations with defined excitation pulses (spatio‑temporal pressure matrices), with global or per‑element excitation.

- **Pulse‑echo reception** — The `Reception` class simulates pulse‑echo RF from scatterers using a fast closed‑form PE‑SDI spectral kernel (plus conventional Tupholme–Stepanishen and pedagogic reference backends). Generates PSFs, focused B‑mode lines, plane‑wave / diverging‑wave event sequences, and full‑matrix / synthetic‑aperture (FMC) acquisitions, with crash‑safe checkpointing of long runs.

- **Beamforming** — Numba‑accelerated 3‑D delay‑and‑sum (`das_volume`, `das_rca_volume`) and focused scanline (`DAS_focused_scanline`) reconstructors for plane‑wave, diverging‑wave, focused, and row‑column sequences, with optional coherence weighting and envelope/log‑compression helpers.

- **Attenuation** — Causal power‑law (frequency‑dependent) attenuation transfer functions applicable per patch in both emission and reception.

- **Phantoms & I/O** — Random‑scatterer phantom generation with echogenicity maps, and a checkpointed on‑disk RF store (`RFDataset`, `.npz`) with HDF5 export (UFF‑compatible fields) for MATLAB/USTB interchange.

- **Brain Atlas Integration** — Maps pressure simulations onto standard brain atlases (via BrainGlobe) for neuro‑ultrasound research.

- **Visualization** — Rich plotting utilities using Matplotlib and PyVista for visualizing transducers, pressure fields, pulse‑echo setups, and brain atlases.

## Gallery

<table>
<tr>
<td width="50%"><img src="docs/examples/assets/ex03_matrix_array_field.png" width="100%" alt="Focused pressure field"><br><sub><b>Focused CW field</b> — matrix array</sub></td>
<td width="50%"><img src="docs/examples/assets/ex05_matrix_pw_3d.gif" width="100%" alt="Steered plane-wave transient"><br><sub><b>Steered plane wave</b> — 3-D transient</sub></td>
</tr>
<tr>
<td width="50%"><img src="docs/examples/assets/ex13_rat_brain_zones.png" width="100%" alt="Transcranial targeting"><br><sub><b>Transcranial targeting</b> — rat-brain atlas</sub></td>
<td width="50%"><img src="docs/examples/assets/ex21_zeus10_volume_3d.png" width="100%" alt="3-D B-mode volume"><br><sub><b>3-D B-mode volume</b> — Zeus matrix, fast RF + DAS</sub></td>
</tr>
</table>

---

## Installation

### 1. Set up a virtual environment

We recommend installing eSDIva in a virtual environment to avoid dependency conflicts with other Python packages. Using [uv](https://docs.astral.sh/uv/guides/install-python/), you can create a new project folder with a virtual environment as follows:

```bash
uv init new_project
```

If you already have a project folder, create a virtual environment with:

```bash
uv venv
```

### 2. Install eSDIva

To install the latest development version from GitHub:

```bash
uv add git+https://github.com/EstebanRivera08/eSDIva.git
```

eSDIva will soon be available on PyPI.

### 3. Check installation

Check that eSDIva is correctly installed by opening a Python interpreter and
importing the package:

```python
import esdiva
```

If no error is raised, you have installed eSDIva correctly.

---

## Quick Start

```python
import esdiva as diva

# Define transducer (mm units; no_sub_x/no_sub_y are keyword-only)
tx = diva.transducers.LinearArrayTransducer(
    n_elements=64,
    element_width_mm=0.25,
    element_height_mm=12.0,
    kerf_mm=0.05,
    no_sub_x=2,
    no_sub_y=4,
    frequency_Hz=5e6,
)
tx.compute_delays(focus_mm=[0, 0, 30])
tx.compute_apodization(focus_mm=[0, 0, 30], FoverD=2.0)

# Define field grid
field_points = {
    "x_extent": [-5, 5],
    "y_extent": [-0.5, 0.5],
    "z_extent": [5, 55],
    "dx": 0.1,
    "dy": 1.0,
    "dz": 0.2,
}

# Run a monochromatic (CW) simulation → pressure amplitude at fc
sim = diva.Emission(tx, monochromatic=True)
p, coords = sim(field_points, method="auto")

# Visualize
diva.plot2D_pressure_slices(p, coords=coords, db_scale=True, vmin=-40)
```

From the project folder you can also run the bundled examples directly:

```bash
uv run examples/example03_multielements_monochromatic_CW.py
uv run examples/example04_lineararray_excitation_DW.py
uv run examples/example01_transducer_gallery.py
```

---


## Citing eSDIva

If you use eSDIva in your research, please cite it using the following reference:

```bibtex
@misc{rivera2026sparsedeltaintegrationmethod,
      title={Sparse Delta Integration method for the calculation of spatiotemporal pressure fields of arbitrary ultrasound transducer geometries}, 
      author={Deyver E. Rivera and Charlie Demene and Mickael Tanter},
      year={2026},
      eprint={2608.26891},
      archivePrefix={arXiv},
      primaryClass={physics.med-ph},
      url={https://arxiv.org/abs/2608.26891}, 
}
```


## References

These works underpin the theory and helped as inspiration for the methods implemented in eSDIva (far-field trapezoid SIR for rectangular apertures, pulse‑echo modelling, power‑law attenuation, and related simulators).

1. B. T. Cox, S. Kara, S. R. Arridge, and P. C. Beard, "k‑space propagation models for acoustically heterogeneous media: Application to biomedical photoacoustics," *The Journal of the Acoustical Society of America*, vol. 121, no. 6, pp. 3453–3464, Jun. 2007. [Online]. Available: <https://pubs.aip.org/jasa/article/121/6/3453/537252/>
2. G. Pinton, J. Dahl, S. Rosenzweig, and G. Trahey, "A heterogeneous nonlinear attenuating full‑wave model of ultrasound," *IEEE Trans. Ultrason., Ferroelect., Freq. Contr.*, vol. 56, no. 3, pp. 474–488, Mar. 2009. [Online]. Available: <http://ieeexplore.ieee.org/document/4816057/>
3. E. Bossy, M. Talmant, and P. Laugier, "Three‑dimensional simulations of ultrasonic axial transmission velocity measurement on cortical bone models," *The Journal of the Acoustical Society of America*, vol. 115, no. 5, pp. 2314–2324, May 2004. [Online]. Available: <https://pubs.aip.org/jasa/article/115/5/2314/546299/>
4. B. E. Treeby and B. T. Cox, "k‑Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields," *J. Biomed. Opt.*, vol. 15, no. 2, p. 021314, 2010. [Online]. Available: <http://biomedicaloptics.spiedigitallibrary.org/article.aspx?doi=10.1117/1.3360308>
5. J. A. Jensen, "FIELD: A program for simulating ultrasound systems," *Medical & Biological Engineering & Computing*, vol. 34, no. Supplement 1, Part 1, pp. 351–352, Jan. 1996.
6. G. E. Tupholme, "Generation of acoustic pulses by baffled plane pistons," *Mathematika*, vol. 16, no. 2, pp. 209–224, Dec. 1969. [Online]. Available: <https://onlinelibrary.wiley.com/doi/abs/10.1112/S0025579300008184>
7. P. R. Stepanishen, "Transient Radiation from Pistons in an Infinite Planar Baffle," *Journal of the Acoustical Society of America*, vol. 49, pp. 1629–1638, Mar. 1971. [Online]. Available: <https://doi.org/10.1121/1.1912541>
8. P. R. Stepanishen, "The Time‑Dependent Force and Radiation Impedance on a Piston in a Rigid Infinite Planar Baffle," *Journal of the Acoustical Society of America*, vol. 49, pp. 841–849, Mar. 1971. [Online]. Available: <https://doi.org/10.1121/1.1912424>
9. J. Jensen and N. Svendsen, "Calculation of pressure fields from arbitrarily shaped, apodized, and excited ultrasound transducers," *IEEE Trans. Ultrason., Ferroelect., Freq. Contr.*, vol. 39, no. 2, pp. 262–267, Mar. 1992. [Online]. Available: <http://ieeexplore.ieee.org/document/139123/>
10. D. Garcia, "SIMUS: An open‑source simulator for medical ultrasound imaging. Part I: Theory & examples," *Computer Methods and Programs in Biomedicine*, vol. 218, p. 106726, May 2022. [Online]. Available: <https://linkinghub.elsevier.com/retrieve/pii/S0169260722001122>
11. A. Cigier, F. Varray, and D. Garcia, "SIMUS: An open‑source simulator for medical ultrasound imaging. Part II: Comparison with four simulators," *Computer Methods and Programs in Biomedicine*, vol. 220, p. 106774, 2022. [Online]. Available: <https://www.sciencedirect.com/science/article/pii/S0169260722001602>
12. G. S. Kino, *Acoustic waves: devices, imaging, and analog signal processing*, ser. Prentice‑Hall signal processing series. Englewood Cliffs: Prentice‑Hall, 1987.
13. J. Jensen, D. Gandhi, and W. O'Brien, Jr., "Ultrasound fields in an attenuating medium," in *1993 Proceedings IEEE Ultrasonics Symposium*, Baltimore, MD, USA: IEEE, 1993, pp. 943–946 vol.2. [Online]. Available: <https://ieeexplore.ieee.org/document/5727212/>
14. J. A. Jensen, "A model for the propagation and scattering of ultrasound in tissue," *The Journal of the Acoustical Society of America*, vol. 89, no. 1, pp. 182–190, Jan. 1991. [Online]. Available: <https://pubs.aip.org/jasa/article/89/1/182/678841/>
15. B. A. J. Angelsen, "A Theoretical Study of the Scattering of Ultrasound from Blood," *IEEE Transactions on Biomedical Engineering*, vol. BME‑27, no. 2, pp. 61–67, Feb. 1980.
16. J. Jensen and I. Nikolov, "Fast simulation of ultrasound images," in *2000 IEEE Ultrasonics Symposium. Proceedings*, vol. 2, San Juan, Puerto Rico: IEEE, 2000, pp. 1721–1724. [Online]. Available: <http://ieeexplore.ieee.org/document/921654/>

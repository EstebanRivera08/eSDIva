<h1 align="center">🌊 eSDIva</h1>
<p align="center"><b>Efficient Sparse Delta Integration for Vectorized Acoustics.</b></p>
<p align="center">
Ultrasound pressure-field simulation for arbitrary transducer geometries — <b>fast and exact</b>.
</p>

[![PyPI version](https://img.shields.io/pypi/v/esdiva)](https://pypi.org/project/esdiva/)
[![Python versions](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://pypi.org/project/esdiva/)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2608.26891-b31b1b)](https://doi.org/10.48550/arXiv.2608.26891)
[![codecov](https://codecov.io/gh/EstebanRivera08/eSDIva/graph/badge.svg)](https://codecov.io/gh/EstebanRivera08/eSDIva)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://estebanrivera08.github.io/eSDIva/)

📖 **Documentation:** <https://estebanrivera08.github.io/eSDIva/>

> [!WARNING]
> eSDIva is currently under development. The API is subject to change, and some features may be incomplete or unstable.

eSDIva is an open‑source Spatial Impulse Response (SIR) and pressure‑field simulation library that supports arbitrary transducer geometries composed of small rectangular patches with apodization and delays.
eSDIva implements both the Fully Sampled Trapezoid (FST) and the Sparse Delta Integration (SDI) methods for computing SIRs following the Tupholme–Stepanishen formulation.
FST reproduces the classic Field II approach, while SDI is a new, algorithmically and mathematically improved method that computes the same SIRs — under identical assumptions — but substantially faster.
The SIR is available through **two kernels that feed emission and reception alike**: a *temporal* kernel (the sampled `h(r, t)`, FST/SDI) and a *spectral* kernel (the closed-form `H(r, ω)`, exact at any frequency). Excitation, impulse responses, attenuation and baffle obliquity are then applied in one shared frequency-domain chain, so emission and reception mirror each other.

> [!NOTE]
> eSDIva is designed as complementary material to the work presented in [https://arxiv.org/abs/2608.26891]. Its goal is to provide fundamental building blocks that researchers can inspect, reuse, contribute to, or adapt. It also leaves room for community‑driven extensions that integrate naturally with the broader scientific Python ecosystem.
> Utilities such as the integration with the BrainGlobe atlas may still evolve to improve robustness.

### Main Features

- **Transducer objects** — Tools to create and assemble common transducer types: linear arrays, convex arrays, matrix arrays, flat/concave/focused circular transducers, and arbitrary custom arrays. These utilities compute geometric focal laws, generate apodization windows for specified F/D ratios, and more.

- **SIR simulation** — The `hsir` module computes the spatial impulse response of apertures discretized into rectangular patches, with Numba‑accelerated, field‑point‑parallel kernels: `compute_h_sir` samples \( h(r, t) \) (FST, SDI or automatic), and `compute_h_sir_spectrum` evaluates its closed‑form spectrum \( H(r, \omega) \) (per patch: area × two sincs × a delay phasor) — exact, no time sampling.

- **Emission simulation** — The `Emission` class turns the SIR into acoustic pressure fields: monochromatic fields (pressure amplitude at exactly `fc`) and broadband transient fields (signed pressure in pascals), with global or per‑element excitation, attenuation, a soft or rigid baffle and user transfer functions.

- **Pulse‑echo reception** — The `Reception` class simulates pulse‑echo RF from scatterers with the same two kernels (spectral by default, temporal on request) and the same options as emission — per‑element excitation, attenuation, baffle. Generates PSFs, focused B‑mode lines, plane‑wave / diverging‑wave event sequences, moving scatterers for flow and Doppler, and full‑matrix / synthetic‑aperture (FMC) acquisitions, with crash‑safe checkpointing of long runs.

- **Attenuation** — Causal power‑law (frequency‑dependent) attenuation with Kramers–Kronig dispersion referenced at the centre frequency, applied in both emission and reception along each propagation path, with either kernel. See the [attenuation guide](https://estebanrivera08.github.io/eSDIva/user-guide/attenuation/).

- **Phantoms & I/O** — Random‑scatterer phantom generation with echogenicity maps, and a checkpointed on‑disk RF store (`RFDataset`, `.npz`) with HDF5 export (UFF‑compatible fields) for MATLAB/USTB interchange.

- **Brain Atlas Integration** — Maps pressure simulations onto standard brain atlases (via BrainGlobe) for neuro‑ultrasound research.

- **AI assistant skills** — Portable Agent Skills (`skills/`) that teach a coding assistant how to build transducers, run emission and pulse-echo simulations, feed the RF to your own beamformer, and contribute back. Work as-is in Claude Code (`/plugin marketplace add EstebanRivera08/eSDIva`), OpenAI Codex and OpenCode (copy into `~/.agents/skills/`), or any assistant that reads Markdown. See the [AI assistant guide](https://estebanrivera08.github.io/eSDIva/user-guide/ai-assistant/).

- **Visualization** — Rich plotting utilities using Matplotlib and PyVista for visualizing transducers, pressure fields, pulse‑echo setups, and brain atlases.

## Gallery

<table>
<tr>
<td width="50%"><img src="docs/examples/assets/ex03_matrix_array_field.png" width="100%" alt="Focused pressure field"><br><sub><b>Focused monochromatic field</b> — matrix array</sub></td>
<td width="50%"><img src="docs/examples/assets/ex05_matrix_pw_3d.gif" width="100%" alt="Steered plane-wave transient"><br><sub><b>Steered plane wave</b> — 3-D transient</sub></td>
</tr>
<tr>
<td width="50%"><img src="docs/examples/assets/ex13_rat_brain_zones.png" width="100%" alt="Atlas-registered targeting"><br><sub><b>Atlas-registered targeting</b> — rat brain, free-field beam (no skull in the acoustic model)</sub></td>
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

From PyPI, with everything switched on:

```bash
uv add "esdiva[all]"          # or:  pip install "esdiva[all]"
```

If you installed with `uv`, you can also run the bundled demo for fun:

```bash
uv run esdiva                 # simulates and renders a focused 3-D field
```

For the latest development version from GitHub:

```bash
uv add "esdiva[all] @ git+https://github.com/EstebanRivera08/eSDIva.git"
```

#### Slimming the install

`[all]` is the recommended install and pulls every feature. If you don't need
some of them — on a headless cluster, in CI, or inside a container — install
plain `esdiva` and add back only the extras you use:

| Install | What you get |
| --- | --- |
| `esdiva` | The simulator core: fields, transducers, reception, beamforming, 2-D and 3-D plotting. Enough for everything in the Quick Start. |
| `esdiva[atlas]` | `utilities.BG_Atlas`, which maps computed fields onto brain anatomy through [BrainGlobe](https://brainglobe.info/). Adds pandas, pyarrow and tifffile. |
| `esdiva[video]` | Saving animations as `.mp4`/`.avi`/`.gif` from the plotting utilities. Adds imageio and a bundled ffmpeg. Without it, static plots and screenshots still work. |
| `esdiva[jupyter]` | Interactive 3-D PyVista scenes rendered *inside* a notebook. Adds the trame and ipywidgets stack. Without it, 3-D still opens in a desktop window. |
| `esdiva[all]` | All of the above. |

Extras combine: `pip install "esdiva[atlas,video]"`. Using a feature whose extra
is missing raises an error naming the extra to install, so nothing fails
silently.

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

# Run a monochromatic simulation → pressure amplitude at fc
sim = diva.Emission(tx, monochromatic=True)
p, coords = sim(field_points)

# Visualize
diva.plot2D_pressure_slices(p, coords=coords, db_scale=True, vmin=-40)
```

From the project folder you can also run the bundled examples directly:

```bash
uv run examples/example03_multielements_monochromatic_CW.py
uv run examples/example04_lineararray_excitation_DW.py
uv run examples/example01_transducer_gallery.py
```

### Two SIR kernels: temporal and spectral

Both simulators take `method=`. The physics is identical — the far‑field trapezoidal SIR of each patch — and the two agree to about 1 %; only the route differs:

| `method` | SIR | Best for |
| --- | --- | --- |
| `"spectral"` | closed‑form `H(r, ω)`, evaluated only at the frequencies the pulse occupies | monochromatic fields, per‑element drives or attenuation, and **all reception** |
| `"temporal"` (= `"sdi"`), `"fst"`, `"auto"` | sampled `h(r, t)`, then an FFT | single‑group transient emission fields on dense grids |

```python
p, coords = diva.Emission(tx, excitation=pulse)(field_points)          # method=None: fastest for the mode
rf, coords = diva.Reception(tx, rx, excitation=pulse).pulse_echo_rf(pos_mm, amp)  # spectral by default
p_t, _ = diva.Emission(tx, excitation=pulse, method="temporal")(field_points)     # pin a kernel
```

`Emission(method=None)` picks the kernel measured fastest for the mode; `Reception` defaults to `"spectral"`, which was the faster in every measured case.

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

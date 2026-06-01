# Graph Report - src  (2026-05-31)

## Corpus Check
- Corpus is ~46,253 words - fits in a single context window. You may not need a graph.

## Summary
- 613 nodes · 1054 edges · 37 communities (34 shown, 3 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 64 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Circular Transducer Geometry|Circular Transducer Geometry]]
- [[_COMMUNITY_Brain Atlas Integration|Brain Atlas Integration]]
- [[_COMMUNITY_SDI Reception Engine|SDI Reception Engine]]
- [[_COMMUNITY_Conventional Reception|Conventional Reception]]
- [[_COMMUNITY_Per-Element SIR and PE SDI|Per-Element SIR and PE SDI]]
- [[_COMMUNITY_Transducer Input Validation|Transducer Input Validation]]
- [[_COMMUNITY_Geometry Utilities|Geometry Utilities]]
- [[_COMMUNITY_Linear and Convex Array Transducers|Linear and Convex Array Transducers]]
- [[_COMMUNITY_3D PyVista Visualization|3D PyVista Visualization]]
- [[_COMMUNITY_Circular Transducer Classes|Circular Transducer Classes]]
- [[_COMMUNITY_Custom Transducer|Custom Transducer]]
- [[_COMMUNITY_Matrix Array Transducer|Matrix Array Transducer]]
- [[_COMMUNITY_PyVista Plot Helpers|PyVista Plot Helpers]]
- [[_COMMUNITY_Emission Batch Methods|Emission Batch Methods]]
- [[_COMMUNITY_2D Matplotlib Plotting|2D Matplotlib Plotting]]
- [[_COMMUNITY_TransducerBase Core Class|TransducerBase Core Class]]
- [[_COMMUNITY_TransducerBase Properties|TransducerBase Properties]]
- [[_COMMUNITY_Attenuation Computation|Attenuation Computation]]
- [[_COMMUNITY_Plot Export Utilities|Plot Export Utilities]]
- [[_COMMUNITY_Transducer Registry|Transducer Registry]]
- [[_COMMUNITY_Delay-and-Sum Beamforming|Delay-and-Sum Beamforming]]
- [[_COMMUNITY_Causal Attenuation Transfer Function|Causal Attenuation Transfer Function]]
- [[_COMMUNITY_PyField Legacy Alias|PyField Legacy Alias]]
- [[_COMMUNITY_Plane Parsing Utilities|Plane Parsing Utilities]]
- [[_COMMUNITY_TransducerBase Beamforming State|TransducerBase Beamforming State]]
- [[_COMMUNITY_Emission Dispatch and SIR Call|Emission Dispatch and SIR Call]]
- [[_COMMUNITY_SIR Core Kernel Numba|SIR Core Kernel Numba]]
- [[_COMMUNITY_Emission Class Init and Config|Emission Class Init and Config]]
- [[_COMMUNITY_TransducerBase Copy and State|TransducerBase Copy and State]]
- [[_COMMUNITY_MATLAB File Utilities|MATLAB File Utilities]]
- [[_COMMUNITY_TransducerBase Subdivision Geometry|TransducerBase Subdivision Geometry]]
- [[_COMMUNITY_TransducerBase Active Patch Indexing|TransducerBase Active Patch Indexing]]
- [[_COMMUNITY_Transient Plane Plotting|Transient Plane Plotting]]
- [[_COMMUNITY_Package Top-Level Init|Package Top-Level Init]]
- [[_COMMUNITY_TransducerBase PyVista Mesh|TransducerBase PyVista Mesh]]
- [[_COMMUNITY_TransducerBase Apodization|TransducerBase Apodization]]
- [[_COMMUNITY_TransducerBase Impulse Response|TransducerBase Impulse Response]]

## God Nodes (most connected - your core abstractions)
1. `TransducerBase` - 68 edges
2. `Emission` - 22 edges
3. `causal_attenuation_tf()` - 17 edges
4. `ReceptionSDI` - 17 edges
5. `Reception` - 16 edges
6. `ndarray` - 14 edges
7. `MatrixArrayTransducer` - 14 edges
8. `CustomTransducer` - 13 edges
9. `BG_Atlas` - 13 edges
10. `LinearArrayTransducer` - 12 edges

## Surprising Connections (you probably didn't know these)
- `plot_pressure_planes()` --calls--> `plot2D_pressure_slices()`  [INFERRED]
  __init__.py → plotting/plotting2D.py
- `envelope_db()` --calls--> `to_dB()`  [INFERRED]
  beamforming/das.py → utilities/helper_functions.py
- `plot2D_pressure_slices()` --calls--> `to_dB()`  [INFERRED]
  plotting/plotting2D.py → utilities/helper_functions.py
- `str` --uses--> `Emission`  [INFERRED]
  emission/PyField.py → emission/emission.py
- `from_sir_to_monochromatic_pressure()` --calls--> `reshape_to_mapped_points()`  [INFERRED]
  emission/sir_to_pressure.py → utilities/helper_functions.py

## Import Cycles
- 3-file cycle: `transducers/__init__.py -> transducers/saved_transducers.py -> transducers/matrix.py -> transducers/__init__.py`
- 3-file cycle: `transducers/__init__.py -> transducers/saved_transducers.py -> transducers/linear.py -> transducers/__init__.py`

## Communities (37 total, 3 thin omitted)

### Community 0 - "Circular Transducer Geometry"
Cohesion: 0.08
Nodes (23): bool, float, int, ndarray, str, Single element centred at the origin., Single element; centre is placed at the bowl's deepest point (origin)., Subdivide the spherical cap using the chosen method. (+15 more)

### Community 1 - "Brain Atlas Integration"
Cohesion: 0.07
Nodes (25): BG_Atlas, BrainGlobe atlas wrapper for mapping acoustic fields to anatomy., Show available BrainGlobe atlases., Compute the 4x4 affine from BrainGlobe voxel space to brain-space (BPS)., Load PyVista mesh(es) from a BrainGlobe atlas for the given structures., Apply an affine transformation to the stored or supplied PyVista mesh., Reload the PyVista mesh from the atlas and re-apply the BPS transform., Print a summary of the BG_Atlas object. (+17 more)

### Community 2 - "SDI Reception Engine"
Cohesion: 0.08
Nodes (20): Pulse-echo RF simulation engine., _next_pow2(), str, ReceptionSDI: pulse-echo RF via the combined PE SDI kernel.  Redistributes all, Extract patch arrays from both TX and RX transducers., Update a simulation parameter at runtime.          Parameters         -------, Return effective excitation: self.excitation or tx.excitation., Pre-extract per-RX-element patch arrays. (+12 more)

### Community 3 - "Conventional Reception"
Cohesion: 0.08
Nodes (21): compute_reception_distances(), Round-trip distances for per-element Reception attenuation.      Two-path mode, _method_to_flag(), _next_pow2(), str, Reception: conventional FieldII-style pulse-echo RF simulation.  Implements th, Extract patch arrays from both TX and RX transducers., Update a simulation parameter at runtime.          Parameters         ------- (+13 more)

### Community 4 - "Per-Element SIR and PE SDI"
Cohesion: 0.09
Nodes (28): compute_h_sir_per_element(), compute_pe_sdi_per_element(), Per-element pulse-echo SIR computation.  Mirrors `transducer_sir_pe.py` but re, Compute Dh_pe independently for each RX element.      Parameters     --------, Per-element SIR computation.  Mirrors `transducer_sir.py` but groups patches b, Compute h_sir independently for each transducer element.      Parameters, compute_h_sir(), compute_parallelized_sir_optimized() (+20 more)

### Community 5 - "Transducer Input Validation"
Cohesion: 0.13
Nodes (26): bool, float, int, ndarray, str, Validation utilities for transducer geometries and parameters.  This module pr, Validate kerf (gap between elements).      Parameters     ----------     ker, Validate that a value is positive (or non-negative).      Parameters     ---- (+18 more)

### Community 6 - "Geometry Utilities"
Cohesion: 0.13
Nodes (24): build_all_subdivisions(), build_rectangular_subdivisions(), compute_1d_element_centers(), compute_2d_element_centers(), compute_distances_to_point(), create_mesh_from_quads(), normalize_delays(), bool (+16 more)

### Community 7 - "Linear and Convex Array Transducers"
Cohesion: 0.14
Nodes (15): ConvexArrayTransducer, LinearArrayTransducer, bool, float, int, ndarray, str, Evenly spaced element centres along x at z=0. (+7 more)

### Community 8 - "3D PyVista Visualization"
Cohesion: 0.15
Nodes (20): plot3D_pressure_slices(), plot3D_pressure_vol(), plot3D_transient_slices(), 3D pressure field visualization using PyVista., Plot orthogonal XZ/XY/YZ slices of a 3D pressure field using PyVista.      Par, Plot transient pressure slices with PyVista time slider or video.      Accepts, Plot a 3D pressure field as a PyVista volume with bounding box and axes., add_2D_image() (+12 more)

### Community 9 - "Circular Transducer Classes"
Cohesion: 0.12
Nodes (15): ConcaveCircularTransducer, ConvexCircularTransducer, FlatCircularTransducer, FocusedCircularTransducer, Mono-element circular transducer types.  All four classes model single-element, Flat circular piston transducer (mono-element).      The aperture is approxima, Spherically focused single-element transducer (bowl / concave disc).      The, Frames already built inside _build_subdivisions; just return them. (+7 more)

### Community 10 - "Custom Transducer"
Cohesion: 0.13
Nodes (12): CustomTransducer, bool, int, ndarray, str, Element centres are the user-supplied positions (in metres).          These ar, Assemble patches from all elements, each rigidly transformed.          Each el, Set per-element apodization weights.          For a ``CustomTransducer`` the a (+4 more)

### Community 11 - "Matrix Array Transducer"
Cohesion: 0.16
Nodes (12): MatrixArrayTransducer, bool, float, int, ndarray, str, Rectangular grid of element centres in the z=0 plane.          Handles non-uni, Flat rectangular patches tiling every element.          Each element uses its (+4 more)

### Community 12 - "PyVista Plot Helpers"
Cohesion: 0.16
Nodes (16): Plotting module for 2D and 3D pressure field visualization., add_3D_vol(), add_markers(), add_pressure_vol(), add_regions_mesh(), add_stl_mesh(), add_transducer_mesh(), PyVista 3-D visualisation utilities for pressure fields and transducers. (+8 more)

### Community 13 - "Emission Batch Methods"
Cohesion: 0.15
Nodes (10): _next_pow2(), Pre-extract per-element patch arrays (outside E-loop for efficiency)., Compute h_sir for a batch, returns (cols, T) float32.          Parameters, Batch size for P-loop: 400 MB budget (float32 h_pad + 2× complex64 arrays)., Wrap with tqdm if importable, else return plain iterable., Evaluate causal attenuation TF at fc only. Returns (P,) complex64., Global path for pulsed and global-excitation modes.          Parameters, Monochromatic, per-element: dot(h_e, exp(-j2πfc·t)) × H_att_e, accumulate. (+2 more)

### Community 14 - "2D Matplotlib Plotting"
Cohesion: 0.19
Nodes (14): plot2D_planes(), plot2D_pressure_plane(), plot2D_pressure_slices(), Matplotlib plotting helpers for 2D pressure field visualization., Plot three orthogonal 2D planes (XZ, XY, YZ) side-by-side with a shared colorbar, Plot orthogonal 2D slices of a pressure field.      Handles both monochromatic, Plot a 2D pressure plane using ``matplotlib.pyplot.imshow``.      Parameters, check_coords() (+6 more)

### Community 15 - "TransducerBase Core Class"
Cohesion: 0.16
Nodes (9): ABC, Base class for all PyField transducer types.  Every transducer is built from r, Per-patch rigid-body frames used by the SIR kernel.          Returns a dict wi, Default patch-frame builder for **flat** transducers.          Computes each p, Abstract base class for all transducer types.      Subclasses must implement:, Release cached geometry arrays to free memory., TransducerBase, float (+1 more)

### Community 16 - "TransducerBase Properties"
Cohesion: 0.14
Nodes (9): ndarray, Per-element delays in seconds, shape ``(n_elements,)``.          Returns, Set per-element delays.          Parameters         ----------         delay, Excitation pulse for this transducer.          1-D float32 array sampled at th, Set excitation pulse; converts to 1-D float32 or stores None.          Paramet, Return element centre positions, shape ``(n_elements, 3)`` in metres., Set per-element apodization weights directly.          Parameters         ---, Set per-element delays directly (normalised so minimum = 0).          Paramete (+1 more)

### Community 17 - "Attenuation Computation"
Cohesion: 0.20
Nodes (12): compute_attenuation_distances(), convert_alpha0_to_nepers(), float, int, ndarray, str, Causal power-law attenuation transfer functions and distance helpers.  Standal, Propagation distance for attenuation.      Parameters     ----------     fie (+4 more)

### Community 18 - "Plot Export Utilities"
Cohesion: 0.25
Nodes (13): Path, bool, int, str, Shared save/export helpers for plotting functions.  ``save_path`` is always a, Record a PyVista animation by iterating *frame_indices*.      *update_fn(idx)*, Join *save_path* and *file_name*, creating the directory if needed., Save a :class:`~matplotlib.animation.FuncAnimation` to disk.      For extensio (+5 more)

### Community 19 - "Transducer Registry"
Cohesion: 0.19
Nodes (10): available_transducers(), Transducer geometry classes for the PyField acoustic simulator.  Notes -----, Return a list of all available transducer class names.      Returns     -----, Linear and convex array transducers.  LinearArrayTransducer     N rectangular, 2-D matrix array transducer.  A matrix array is a rectangular grid of N_x × N_, Domino(), Pre-defined transducer configurations., Create a Domino transducer with specific parameters.      Returns     ------- (+2 more)

### Community 20 - "Delay-and-Sum Beamforming"
Cohesion: 0.22
Nodes (11): das(), envelope_db(), float, NDArray, Delay-and-sum (DAS) beamforming for pulse-echo RF data., Delay-and-sum beamformer for a single focused scanline.      Applies per-chann, Compute log-compressed Hilbert envelope.      Parameters     ----------, Post-processing beamforming functions for PyField RF data. (+3 more)

### Community 21 - "Causal Attenuation Transfer Function"
Cohesion: 0.24
Nodes (11): causal_attenuation_tf(), Causal power-law attenuation transfer function H_att(f, d).      Absorption an, Emission: compute emitted acoustic pressure fields., from_sir_to_monochromatic_pressure(), from_sir_to_pressure(), _next_pow2(), Convert a spatial impulse response (SIR) to a pressure field., Compute the transient pressure field from the SIR and an excitation pulse. (+3 more)

### Community 22 - "PyField Legacy Alias"
Cohesion: 0.17
Nodes (7): Acoustic pressure field emission engine., str, PyField, PyField: deprecated alias for Emission.  Use `pyfield.emission.Emission` inste, Deprecated: use `Emission` instead.      Backward-compatible wrapper around `E, Backward-compatible call accepting legacy kwargs.          Parameters, Set an attribute by name (legacy API; prefer `set()`).          Parameters

### Community 23 - "Plane Parsing Utilities"
Cohesion: 0.21
Nodes (12): infer_coords(), _infer_coords_from_planes(), parse_planes(), _PlaneMeta, PlaneSpec, Unified plane handling utilities for 2D and 3D pressure field plotting.  Singl, Infer missing coordinate arrays from plane shapes.      For each axis, finds a, Infer missing coordinate arrays from plane data shapes. (+4 more)

### Community 24 - "TransducerBase Beamforming State"
Cohesion: 0.21
Nodes (7): bool, Compute per-element time delays for electronic focusing or plane-wave steering., Return uniform full-aperture apodization (all ones).          Mono-element tra, Interactive 3-D visualisation of the transducer surface.          Parameters, Plot apodization weights as a line/stem chart.          For 2-D matrix transdu, Plot per-element delays in microseconds.          Parameters         --------, Side-by-side delay and apodization plot.          Parameters         --------

### Community 25 - "Emission Dispatch and SIR Call"
Cohesion: 0.21
Nodes (8): _method_to_flag(), ndarray, Convolve excitation with impulse response (if not None).          Returns floa, Compute h_sir summed over all patches, returns (T, P) float32.          Parame, Monochromatic, global path: full h_sir → monochromatic pressure., Compute the pressure field at given field points.          Behavior is determi, compute_time_grid(), Compute the time axis needed to capture the full SIR response.      The earlie

### Community 26 - "SIR Core Kernel Numba"
Cohesion: 0.23
Nodes (11): compute_h_sir(), _compute_rectangle_SIR_params(), _compute_sir_parallel_points(), _fully_sampled_trapezoid(), _place_sir_sdi_deltas(), Far-field rectangular patch SIR computation kernels., Compute trapezoidal SIR parameters for a rectangular patch.      Parameters, Compute SIR in parallel over field points.      Parameters     ---------- (+3 more)

### Community 27 - "Emission Class Init and Config"
Cohesion: 0.27
Nodes (5): Emission, str, Update a simulation parameter at runtime.          Parameters         -------, Print mode summary before heavy computation., Compute emitted acoustic pressure fields.      Parameters     ----------

### Community 28 - "TransducerBase Copy and State"
Cohesion: 0.25
Nodes (5): Any, str, Return a deep copy of this transducer, including all state and cached geometry., Return a snapshot of the current apodization / delay state.          Returns, Restore apodization / delay state from a dictionary.          Parameters

### Community 29 - "MATLAB File Utilities"
Cohesion: 0.29
Nodes (7): explore_mat(), mat_struct_fields(), mat_struct_to_dict(), Utilities for reading and exploring MATLAB .mat file structures., Convert a MATLAB struct recursively to a Python dict.      Parameters     ---, Print the hierarchical structure of a MATLAB .mat object.      Parameters, Return field names from a MATLAB struct-like object.      Parameters     ----

### Community 30 - "TransducerBase Subdivision Geometry"
Cohesion: 0.29
Nodes (4): float, List of quad-vertex arrays ``(4, 3)`` for every patch, in metres.          Ret, Patch area in m² (same for all patches in a uniform grid).          Returns, Build rectangular sub-patches for the entire aperture.          Returns

### Community 31 - "TransducerBase Active Patch Indexing"
Cohesion: 0.29
Nodes (4): int, Element index for each patch; maps patch to parent element.          Returns, Total number of rectangular sub-patches across all elements.          Returns, Number of elements with non-zero apodization.          Returns         ------

### Community 32 - "Transient Plane Plotting"
Cohesion: 0.33
Nodes (6): compute_plane_extents(), Compute imshow extents for each PlaneSpec from coordinate arrays.      Sets ``, plot2D_transient_slices(), Animate orthogonal pressure slices of transient data with Matplotlib.      Acc, Convert a matrix to decibel (dB) scale.      Parameters     ----------     m, to_dB()

### Community 33 - "Package Top-Level Init"
Cohesion: 0.33
Nodes (5): main(), plot_pressure_planes(), PyField: acoustic field simulator based on the spatial impulse response method., Plot pressure planes (deprecated, use `plot2D_pressure_slices`).      Paramete, Print a greeting message from the pyfield package.

## Knowledge Gaps
- **9 isolated node(s):** `int`, `float32`, `float64`, `ndarray`, `bool` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TransducerBase` connect `TransducerBase Core Class` to `Circular Transducer Geometry`, `TransducerBase PyVista Mesh`, `TransducerBase Apodization`, `TransducerBase Impulse Response`, `Linear and Convex Array Transducers`, `Circular Transducer Classes`, `Custom Transducer`, `Matrix Array Transducer`, `TransducerBase Properties`, `Transducer Registry`, `TransducerBase Beamforming State`, `TransducerBase Copy and State`, `TransducerBase Subdivision Geometry`, `TransducerBase Active Patch Indexing`?**
  _High betweenness centrality (0.324) - this node is a cross-community bridge._
- **Why does `to_dB()` connect `Transient Plane Plotting` to `Brain Atlas Integration`, `Delay-and-Sum Beamforming`, `2D Matplotlib Plotting`?**
  _High betweenness centrality (0.255) - this node is a cross-community bridge._
- **Why does `subdivide_parametric_surface()` connect `Circular Transducer Geometry` to `Brain Atlas Integration`?**
  _High betweenness centrality (0.228) - this node is a cross-community bridge._
- **Are the 29 inferred relationships involving `TransducerBase` (e.g. with `ConcaveCircularTransducer` and `ConvexCircularTransducer`) actually correct?**
  _`TransducerBase` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Emission` (e.g. with `str` and `PyField`) actually correct?**
  _`Emission` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `PyField: acoustic field simulator based on the spatial impulse response method.`, `Plot pressure planes (deprecated, use `plot2D_pressure_slices`).      Paramete`, `Print a greeting message from the pyfield package.` to the rest of the system?**
  _267 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Circular Transducer Geometry` be split into smaller, more focused modules?**
  _Cohesion score 0.08333333333333333 - nodes in this community are weakly interconnected._
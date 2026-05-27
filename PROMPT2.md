# PyField: Emission / Reception / Attenuation Architecture

Read `.claude/rules/physics-context.md` (§1-10) and `.claude/rules/attenuation.md`
for physics behind every design decision.

---

## Scope

Restructure `psimulation/` into three independent modules and create new SIR derivative
kernels to support them:

1. **`h_sir/sir_derivatives.py`** — new Numba kernels exposing `d2h`, `dh`, `h` at
   summed-all and per-element grouping levels.
2. **`psimulation/attenuation.py`** — causal power-law frequency-domain transfer function.
3. **`psimulation/emission.py`** — `Emission` class replacing `PyField` class, with
   pulsed-by-default behavior and excitation shape dispatch.
4. **`psimulation/reception.py`** — `Reception` class for RF/pulse-echo simulation with
   separate TX/RX transducers.

Old `PyField` class becomes a **deprecated alias** for `Emission` during transition, then
removed.

### Field II correspondence

Design informed by Field II API (Jensen 1996). Key mapping:

| Field II | PyField | Notes |
|----------|---------|-------|
| `xdc_impulse(Th, ir)` | `transducer.impulse_response = ir` | Electromechanical IR, per transducer |
| `xdc_excitation(Th, exc)` | `transducer.excitation = ex / Emission.excitation` / `Reception.excitation` | can be input to the TX or to the class (in Emission/Reception class is to be taken from or set to TX only) |
| `xdc_focus(Th, ...)` | `transducer.compute_delays(focus_mm=...)` | Already exists |
| `xdc_apodization(Th, ...)` | `transducer.compute_apodization(...)` | Already exists |
| `calc_hp(Th, pts)` | `Emission(tx)(field_points)` | Emitted pressure |
| `calc_scat_multi(tx, rx, pos, amp)` | `Reception(tx, rx)(pos, amp)` | Per-element RF data |
| `calc_scat_all(tx, rx, pos, amp)` | `Reception.compute_all(...)` | Full matrix capture |
| `set_field('att', ...)` | `Emission/Reception(alpha0=..., freq_power=...)` | Attenuation |
| `set_field('use_att', 1)` | `alpha0` not None → auto-enabled | Simpler toggle |
| `xdc_baffle(Th, soft)` | Future extension | Not in this iteration |
| `xdc_dynamic_focus(...)` | Future extension | Requires timeline system |
| `ele_waveform(Th, el, wav)` | Future extension | Per-element IR override |
| `calc_scat(tx, rx, pos, amp)` | **Not implemented** | Beamforming is user's job |

**Improvements over Field II**:
- Causal power-law attenuation with K-K dispersion (Field II uses non-causal minimum-phase)
- Python/NumPy ecosystem (no MATLAB dependency)
- Explicit per-element excitation support via shape dispatch

### Complete signal chain

```
Emission:
  p(r,t) = excitation(t) * ir_tx(t) * dh_sir(r,t)/dt

Reception (pulse-echo):
  rf_e(t) = [excitation * ir_tx * dh_tx/dt] *_t [d2h_rx_e/dt2 * ir_rx]
           * f_m(r)  [summed over scatterers]
           * rho / (2*c^2)
           * H_att(f, d)  [if attenuation enabled]
```

---

## 1. New SIR Derivative Kernels (`src/pyfield/h_sir/`)

### 1.1 Existing files (DO NOT modify)

| File | Output | Notes |
|------|--------|-------|
| `farfield_rect_patch.py` | `h (P, T)` summed over M patches | Hybrid naive/SDI. Proven. Reference. |
| `hsir_SDI.py` | `d2h (P, T)` summed over M patches | Pure SDI, no integration. |
| `h_sir.py` | Wrapper class, returns `(t0, h_sir.T)` | Calls `compute_h_sir`. |

Existing kernel (lines 297-311 of `farfield_rect_patch.py`) computes `d2h` via SDI then
double-integrates. Intermediates `d2h` and `dh` are discarded. No per-element output.

### 1.2 New file: `src/pyfield/h_sir/sir_derivatives.py`

All Numba kernels: `@njit(parallel=True, fastmath=True)`, `prange` over P.
Reuse `compute_rectangle_SIR_params` from `farfield_rect_patch.py`.
SDI-only (no naive fallback — these feed freq-domain pipeline where SDI always wins).

| Function | Integration | Grouping | Output | Use case |
|----------|------------|----------|--------|----------|
| `compute_d2h` | 0 cumsums | sum all M | `(P, T)` | Building block, testing |
| `compute_dh` | 1 cumsum | sum all M | `(P, T)` | Emission with excitation (§8: `p = rho * v *_t dh/dt`) |
| `compute_d2h_per_element` | 0 cumsums | per element | `(P, E, T)` | Reception: `d2h_rx` per RX element (§9) |
| `compute_dh_per_element` | 1 cumsum | per element | `(P, E, T)` | Reception: `dh_tx` per TX element (§9) |

Per-element variants need extra args: `sub_el_idx: int32[M]`, `n_elements: int`.
Accumulate into `out[p, sub_el_idx[m], k]`.

**Memory**: `(P, E, T) * 4 bytes` can be huge. Python wrappers accept `batch_size_points`
to chunk P. Numba kernel processes full P; batching at Python level.

**Reference kernel** (testing only):

| Function | Description | Output |
|----------|------------|--------|
| `compute_h_sir_patch_parallel` | Like `farfield_rect_patch.py` but `prange` over M. | `(P, T)` |

### 1.6 Parallelism modes

All kernels in `sir_derivatives.py` support two parallelism axes:

| `parallel_axis` | `prange` over | When to use | Notes |
|-----------------|--------------|-------------|-------|
| `"points"` (default) | P (field points) | P >> n_threads | No race conditions. Current approach. |
| `"patches"` | M (patches) | P < n_threads (e.g. single scatterer in Reception) | Thread-local reduction needed. |

**Patches-parallel implementation** (avoids race conditions):
1. Pre-allocate thread-local buffer: `thread_out (n_threads, P, T)` or `(n_threads, P, E, T)`.
2. Each thread accumulates its patch subset into `thread_out[thread_id]`.
3. Final serial reduce: `out = thread_out.sum(axis=0)`.

```python
# Numba pattern for patches-parallel (thread-local reduction):
@njit(parallel=True, fastmath=True)
def compute_d2h_mpar(pts, centers, wx, wy, apod, delays, dt, c, n_threads):
    P, T = pts.shape[0], <T>
    thread_out = np.zeros((n_threads, P, T), dtype=np.float32)
    for m in prange(len(centers)):
        tid = numba.get_thread_id()
        # accumulate patch m into thread_out[tid, p, k]
        ...
    # reduce
    out = np.zeros((P, T), dtype=np.float32)
    for t in range(n_threads):
        out += thread_out[t]
    return out
```

Python wrapper signature:
```python
def compute_d2h(pts, transducer_data, ..., parallel_axis="points"):
    if parallel_axis == "points":
        return _compute_d2h_ppar(...)   # prange over P
    else:
        return _compute_d2h_mpar(...)   # prange over M, thread-local reduce
```

Applies to: `compute_d2h`, `compute_dh`, `compute_d2h_per_element`,
`compute_dh_per_element`, `compute_h_sir_patch_parallel`.
Default always `"points"` — patches mode opt-in only.

### 1.3 Integration helpers (`sir_derivatives.py` or separate `sir_integration.py`)

```python
@njit
def integrate_d2h_to_dh(d2h, dt):
    """Single cumsum along last axis."""

@njit
def integrate_dh_to_h(dh, dt):
    """Single cumsum along last axis, scale by dt."""
```

Work on any shape `(..., T)`.

### 1.4 `h_sir.py` wrapper update

Add method:

```python
def compute_derivative(self, field_points_mm, *, derivative="h", per_element=False):
    """Return SIR or derivative, optionally per-element.

    Parameters
    ----------
    derivative : {"h", "dh", "d2h"}
    per_element : bool

    Returns
    -------
    t0 : float
    result : ndarray, shape (T, P) or (T, P, E)
    """
```

### 1.5 `compute_sub_elem_attributes` update

Current return: `(centers, apod, delays, M, range_k, wx_arr, wy_arr)`.
**Add** `sub_el_idx_arr: int32[M]` to return tuple. Already iterates over `sub_el_idx`
but doesn't return it as array. Update all call sites: `h_sir.py`, `PyField.py`
(→ `emission.py`).

---

## 2. Attenuation Module (`src/pyfield/psimulation/attenuation.py`)

Standalone module. No dependency on Emission or Reception — both import from it.

### 2.1 Core functions

```python
def causal_attenuation_tf(
    freqs_hz: ndarray,       # (N_freq,)
    distances_m: ndarray,    # (N_points,) or scalar
    alpha0_dB: float,        # dB/(MHz^y cm) — user-facing
    y: float,                # power-law exponent (1.0-1.3 for tissue)
    f0_hz: float,            # reference frequency (transducer fc)
) -> ndarray:
    """Causal power-law attenuation transfer function H_att(f, d).

    General case (y != 1):
        H = exp(-a|w|^y d) * exp(-j a|w|^y tan(y*pi/2) d)

    Special case (y == 1):
        H = exp(-a|w|d) * exp(-j (2a/pi) w ln(|w|/w0) d)

    Returns complex array (N_points, N_freq) or (N_freq,).
    Always causal (includes Kramers-Kronig dispersion).
    """

def convert_alpha0_to_nepers(alpha0_dB: float, y: float) -> float:
    """dB/(MHz^y cm) → Np/(Hz^y m).

    alpha0_neper = alpha0_dB * 100 / (20 * log10(e) * 1e6^y)
    """

def compute_attenuation_distances(
    field_points_m: ndarray,      # (P, 3)
    transducer_center_m: ndarray, # (3,)
    patch_centers_m: ndarray = None, # (M, 3)
    mode: str = "per_point",      # "per_point" or "per_patch"
) -> ndarray:
    """Propagation distance for attenuation.

    per_point: d = |r - r_tx_center|, shape (P,). Fast, approximate.
    per_patch: d = |r - r_patch_m|, shape (P, M). Accurate near-field.
    """

def reduce_patch_distances_to_element(
    distances_pm: ndarray,  # (P, M) from compute_attenuation_distances per_patch
    sub_el_idx: ndarray,    # int32 (M,) — patch-to-element map from compute_sub_elem_attributes
    n_elements: int,
    reduce: str = "mean",   # "mean" | "min" | "max"
) -> ndarray:
    """Reduce per-patch distances to per-element representative distances.

    Returns (P, E) — one distance per field-point/element pair.
    Used to build H_att (P, E, N_freq) for per-element attenuation in Reception.

    reduce="mean": average distances of all patches in each element.
                   Good when patches cluster tightly (near-field, uniform patch size).
    reduce="min":  minimum distance per element. Conservative (least attenuation).
    reduce="max":  maximum distance per element. Opposite of conservative (most attenuation).
    """
```

**H_att shapes and where each applies:**

| Context | Distance shape | H_att shape | Notes |
|---------|---------------|-------------|-------|
| Emission, `per_point` | `(P,)` — `\|r_p - r_tx_center\|` | `(P, N_freq)` | Applied to summed h_sir `(P, T)` |
| Emission, `per_patch` | average all M patches → `(P,)` | `(P, N_freq)` | Reduces to same shape as per_point |
| Emission, per-element excitation | `(P, E_tx)` — per-element TX distances | `(P, E_tx, N_freq)` | Applied to `dh_per_element (P, E_tx, T)` before element sum |
| Reception, `per_point` | `(P,)` — total scalar round-trip | `(P, 1, N_freq)` | Broadcast over E_rx. Ignores element separation — inaccurate for large apertures |
| Reception, `per_element` (default) | `(P, E_rx)` — `d_tx_s + d_rx_se` | `(P, E_rx, N_freq)` | Correct model: TX center-to-scatterer + scatterer-to-each-RX-element |
| Reception, `per_patch` | `reduce_patch_distances_to_element` → `(P, E_rx)` | `(P, E_rx, N_freq)` | More accurate RX distances via patch averaging |

**Reception distance calculation** — two-path model:

```
d_total(s, e) = d_tx(s) + d_rx(s, e)
             = |r_s - r_tx_center| + |r_s - r_rx_element_e|
```

Add dedicated helper:

```python
def compute_reception_distances(
    scatterer_positions_m: ndarray,   # (P, 3)
    tx_center_m: ndarray,             # (3,) — TX transducer center
    rx_element_centers_m: ndarray,    # (E_rx, 3) — one center per RX element
) -> ndarray:
    """Round-trip distances for per-element Reception attenuation.

    Returns (P, E_rx): d_total[p, e] = |r_s_p - r_tx| + |r_s_p - r_rx_e|

    TX path is isotropic (same for all elements). RX path is per-element
    (each element receives from a different direction/distance).
    Feed result directly into causal_attenuation_tf to get H_att (P, E_rx, N_freq).
    """
```

`causal_attenuation_tf` accepts any leading shape — pass `(P,)`, `(P, E_rx)`, etc.,
returns `(..., N_freq)` complex array. Broadcasting handles the rest.

### 2.2 Rules (from `attenuation.md`)

- **Never** modify SIR kernels for attenuation.
- Always use causal model (K-K dispersion). Cost = zero. Non-causal produces precursors.
- User-facing units: `alpha0` in dB/(MHz^y cm). Convert internally.
- `alpha0 = None` or `alpha0 = 0` → no attenuation applied.

---

## 3. Emission Module (`src/pyfield/psimulation/emission.py`)

### 3.1 Design: `Emission` class replaces `PyField`

`PyField` as a class name conflicts with package name. `Emission` is semantically correct
for what this class does: compute emitted acoustic pressure fields.

### 3.2 `Emission` class

```python
class Emission:
    """Compute emitted acoustic pressure fields.

    Parameters
    ----------
    transducer : TransducerBase
        Transducer with geometry, delays, apodization.
    c : float, default 1540.0
        Speed of sound (m/s).
    rho : float, default 1.0
        Medium density (kg/m^3).
    fs : float, default 200e6
        Sampling frequency (Hz).
    alpha0 : float or None, default None
        Attenuation coefficient in dB/(MHz^y cm). None = no attenuation.
    freq_power : float, default 1.0
        Power-law exponent y for attenuation.
    excitation : ndarray or None, default None
        Excitation pulse. Shape determines behavior:
        - None → pulsed (raw SIR, default)
        - (L,) → same excitation for all elements
        - (L, E) → per-element excitation
    monochromatic : bool, default False
        If True, return CW amplitude at fc instead of transient field.
    verbose : bool, default True
    """

    def __init__(
        self, transducer, *,
        c=1540.0, rho=1.0, fs=200e6,
        alpha0=None, freq_power=1.0,
        excitation=None,
        transfer_function=None,   # callable TF(freq) -> array, applied in freq domain
        monochromatic=False,
        fast_attenuation=False,   # True = TX-center distance, no E-loop (fast approx)
        verbose=True,
    ):
        ...

    def set(self, name: str, value):
        """Update a simulation parameter at runtime.

        Validates the parameter name and value type. Recomputes derived
        attributes when needed (e.g., changing transducer refreshes
        sub-element arrays).

        Parameters
        ----------
        name : str
            One of: "c", "rho", "fs", "alpha0", "freq_power", "excitation",
            "monochromatic", "verbose", "transducer".
        value : object
            New value for the parameter.

        Examples
        --------
        >>> sim = Emission(tx)
        >>> sim.set("alpha0", 0.5)          # enable attenuation
        >>> sim.set("monochromatic", True)   # switch to CW mode
        >>> sim.set("excitation", pulse)     # set global excitation
        >>> sim.set("excitation", per_elem)  # (L, E) → per-element
        """

    def __call__(self, field_points_mm, *, method="auto") -> tuple[ndarray, dict]:
        """Compute pressure field at given points.

        Behavior determined by instance state:

        1. monochromatic=True → CW amplitude at fc.
           Output: p (Nx, Ny, Nz), coords {x, y, z}

        2. excitation=None → pulsed transient (raw SIR).
           Output: p (Nt, Nx, Ny, Nz), coords {x, y, z, t0, dt}

        3. excitation=(L,) → transient with global excitation.
           Internally: h_sir (T,P) → FFT conv with dv/dt.
           Output: p (Nt, Nx, Ny, Nz), coords {x, y, z, t0, dt}

        4. excitation=(L, E) → transient with per-element excitation.
           Internally: dh_per_element (P, E, T) → FFT conv per element → sum.
           Output: p (Nt, Nx, Ny, Nz), coords {x, y, z, t0, dt}

        In all transient cases: if alpha0 is not None, attenuation TF applied
        in frequency domain before IFFT.

        For monochromatic with attenuation: frequency-independent shortcut
        p_att = p * exp(-alpha_neper * d) at center frequency.

        Parameters
        ----------
        field_points_mm : dict or ndarray
            Grid dict or (N, 3) array in mm.
        method : str
            SIR computation method: "auto", "naive", "sdi".

        Returns
        -------
        pressure : ndarray
        coords : dict
        """
```

### 3.3 Excitation shape dispatch logic

**Per-element loop trigger:**
```python
per_elem_exc = exc is not None and exc.ndim == 2
use_per_element = (self.alpha0 is not None and not self.fast_attenuation) or per_elem_exc
```

Modes 2 (pulsed) and 3 (global exc) also enter the per-element path when
`alpha0 is not None` and `fast_attenuation=False` (default), so element-center
distances are used for attenuation (accurate near-field model).

```python
# Inside __call__:
exc = self.excitation
if self.monochromatic:
    if use_per_element:
        pressure_flat = self._mono_per_element(points_m, T, dt, time_grid, method_flag)
    else:
        pressure_flat = self._mono_global(points_m, distances_m, method)
    # reshape via reshape_to_mapped_points(x, y, z, flat)[0] — NOT flat.reshape(Nx,Ny,Nz)
    # (meshgrid is z-outer, x-middle, y-inner, so direct reshape is wrong)

elif use_per_element:
    # Per-element loop: pulsed (exc=None), global (L,), or per-element (L, E).
    Pressure_flat = self._transient_per_element(
        points_m, t0, T, dt, time_grid, method_flag, exc
    )

elif exc is None and self.alpha0 is None:
    # Pure pulsed: legacy path via _compute_sir + from_sir_to_pressure
    h, t0 = self._compute_sir(points_m, method=method)
    h[idx_e_h:, :] = 0.0
    Pressure_flat = from_sir_to_pressure(h, ...)

else:
    # Global excitation (L,) or pulsed + fast_attenuation with alpha0 → global FFT path
    Pressure_flat = self._transient_global(
        points_m, t0, T, dt, time_grid, distances_m, method, exc
    )
```

**Per-element path** (`_transient_per_element`): P-outer, E-inner.
Uses `compute_h_sir` (not `compute_dh_per_element`) with element-filtered patches.
Pre-allocates one `h_pad_buf = zeros((batch_P, nfft), float32)` outside all loops.
All E elements accumulate into `acc_H (cols, N_freq)` complex64.
One `irfft + abs` per P-batch preserves inter-element interference.

```python
# Core inner structure (E-inner, P-outer):
h_pad_buf = np.zeros((batch_P, nfft), dtype=np.float32)  # pre-allocated once
for p_start in range(0, P, batch_P):
    h_pad = h_pad_buf[:cols]   # view, no copy; tail stays zero
    acc_H = np.zeros((cols, N_freq), dtype=np.complex64)
    for e in range(n_elements):
        h_e_b = _compute_h_sir_batch(..., patch_slices[e])  # (cols, T) float32
        h_pad[:, :T] = h_e_b                                 # write; tail stays 0
        H_e = rfft(h_pad, axis=1, workers=-1)                # (cols, N_freq) complex64
        H_e *= fft_exc_list[e]    # derivative of excitation
        H_e *= TF                 # optional transfer function
        H_e *= H_att_e            # optional per-element attenuation
        acc_H += H_e
    Pressure_flat[:, p_start:p_end] = abs(irfft(acc_H, ...))[:, :T].T
```

Peak memory per P-batch: `O(batch_P × nfft)` — E-independent.

### 3.4 SIR computation methods (private)

```python
def _compute_sir(self, points, *, method="auto"):
    """Compute h_sir summed over all patches. Returns (h, t0).
    Shape: h (T, P).
    Uses existing compute_h_sir from farfield_rect_patch.py.
    """

def _compute_sir_derivative(self, points, *, derivative="dh", per_element=False, method="sdi"):
    """Compute SIR derivative. Returns (result, t0).
    Shape: (T, P) or (T, P, E) depending on per_element.
    Uses new kernels from sir_derivatives.py.
    """
```

### 3.5 Backward compatibility

Keep `PyField.py` as thin wrapper:

```python
# src/pyfield/psimulation/PyField.py
import warnings
from .emission import Emission

class PyField(Emission):
    """Deprecated: use Emission instead.

    Backward-compatible wrapper. Default monochromatic=True to match
    old PyField behavior (Emission defaults to False).
    """

    def __init__(self, transducer, *, monochromatic=True, **kwargs):
        warnings.warn(
            "PyField is deprecated, use pyfield.psimulation.Emission instead.",
            DeprecationWarning, stacklevel=2,
        )
        super().__init__(transducer, monochromatic=monochromatic, **kwargs)
```

Keep exporting from `psimulation/__init__.py` and `pyfield/__init__.py` so existing
`from pyfield import PyField` works.

---

## 4. Reception Module (`src/pyfield/psimulation/reception.py`)

### 4.1 Design rationale and Field II mapping

Field II provides three scattering functions at different output granularity:

| Field II | Returns | PyField equivalent |
|----------|---------|-------------------|
| `calc_scat(tx, rx, pos, amp)` | Beamformed A-line (1D) | **Not implemented** — beamforming is user's responsibility |
| `calc_scat_multi(tx, rx, pos, amp)` | Per-element RF `(Nt, E_rx)` | `Reception.__call__` — primary use case |
| `calc_scat_all(tx, rx, pos, amp)` | Full matrix `(E_tx, Nt, E_rx)` | `Reception.compute_all` |

PyField returns raw channel data. User applies beamforming externally (delay-and-sum,
MVDR, etc.). This is the research-oriented approach.

Every Reception option is defaulted with downsampling: int |None = None, which means that the
return is with all Nt. Since memory use can be high, a downsampling option is good to
reduce the Nt length. A downsampling = 10, means the result is going to be returned with
temporal size of Nt/10.

Key Field II design choices preserved:
- **TX and RX are separate transducer instances** — can differ in type, geometry, IR, focus
- **Excitation is TX-only** — RX has no excitation
- **Impulse response is per-transducer** — set on TX and/or RX independently
- **Every result includes `t0`** — start time for alignment across lines

### 4.2 Transducer attribute addition: `impulse_response`, and `excitation`

Add to `TransducerBase`:

```python
class TransducerBase:
    def __init__(self, ...):
        ...
        self._impulse_response: Optional[ndarray] = None  # (L_ir,) 1D array
        self._excitation : Optional[ndarray] = None # (L_exc,) a 1D array

    @property
    def impulse_response(self) -> Optional[ndarray]:
        """Electromechanical impulse response of the transducer element.

        1D array sampled at the simulation sampling frequency.
        Represents the electrical-to-acoustic (TX) or acoustic-to-electrical (RX)
        transfer function of each element. Applied via convolution in frequency
        domain.

        None means ideal (delta function) — no filtering.
        """
        return self._impulse_response

    @impulse_response.setter
    def impulse_response(self, value: Optional[ndarray]):
        if value is not None:
            value = np.asarray(value, dtype=np.float32).ravel()
        self._impulse_response = value

    # Same as above for excitation
```

This matches Field II's `xdc_impulse`, `xdc_excitation`. Each transducer (TX or RX) can have a different
impulse response.

### 4.3 `Reception` class

```python
class Reception:
    """Compute received RF signals via pulse-echo simulation.

    Models the full transmit-scatter-receive chain using SIR derivatives
    and frequency-domain convolutions.

    Physics (from physics-context.md §9):

        p_r(t) = (rho/2c^2) * f_m(r) *_r [v'_pe(t) *_t Dh_pe(r, t)]

    Where Dh_pe = dh^e *_t d2h^r computed via PE SDI kernel (compute_pe_sdi),
    and v'_pe = E_m * v (no derivative on v — derivatives in Dh_pe).

    Parameters
    ----------
    tx : TransducerBase
        Transmit transducer (with delays, apodization, optional impulse_response).
    rx : TransducerBase
        Receive transducer (with apodization, optional impulse_response).
        Can be same object as tx for pulse-echo.
    c : float, default 1540.0
        Speed of sound (m/s).
    rho : float, default 1.0
        Medium density (kg/m^3).
    fs : float, default 200e6
        Sampling frequency (Hz).
    alpha0 : float or None, default None
        Attenuation in dB/(MHz^y cm). None = no attenuation.
    freq_power : float, default 1.0
        Attenuation power-law exponent.
    excitation : ndarray or None, default None
        TX excitation pulse (L,). If None, uses delta (impulse).
    verbose : bool, default True

    Examples
    --------
    >>> # Pulse-echo with same transducer
    >>> tx = LinearArrayTransducer(...)
    >>> tx.compute_delays(focus_mm=[0, 0, 30])
    >>> tx.impulse_response = ir_pulse  # electromechanical IR
    >>> tx.excitation = excitation_pulse
    >>>
    >>> rx = LinearArrayTransducer(...)  # or same as tx
    >>> rx.impulse_response = ir_pulse
    >>>
    >>> sim = Reception(tx, rx, fs=200e6, c=1540)
    >>> # or sim.set("excitation", excitation_pulse)
    >>>
    >>> # Single-focus RF
    >>> rf, coords = sim(scatterer_positions_mm, scattering_amplitudes)
    >>> # rf shape: (Nt, E_rx)
    """

    def __init__(
        self, tx, rx, *,
        c=1540.0, rho=1.0, fs=200e6,
        alpha0=None, freq_power=1.0,
        excitation=None,
        verbose=True,
    ):
        self.tx = tx
        self.rx = rx
        self.c = c
        self.rho = rho
        self.fs = fs
        self.alpha0 = alpha0
        self.freq_power = freq_power
        self.excitation = excitation
        self.verbose = verbose

    def set(self, name: str, value):
        """Update simulation parameter. Same pattern as Emission.set()."""

    def __call__(
        self,
        scatterer_positions_mm: ndarray,  # (N_scat, 3)
        scattering_amplitudes: ndarray,   # (N_scat,)
        *,
        method="sdi",
    ) -> tuple[ndarray, dict]:
        """Compute RF signal for all receive elements.

        Pipeline per RX element:
        1. Extract RX element patches (mask by sub_el_idx)
        2. Call compute_pe_sdi(pts, tx_all_patches, rx_elem_patches, ...)
           → Dh_pe (P, T) for this element
        3. FFT(Dh_pe) → multiply by FFT(v'_pe) × FFT(ir_tx) × FFT(ir_rx) × H_att
        4. IFFT → weight by f_m(r), sum over scatterers
        5. Scale by rho / (2 * c^2)

        Parameters
        ----------
        scatterer_positions_mm : (N_scat, 3) ndarray
            Scatterer positions in mm.
        scattering_amplitudes : (N_scat,) ndarray
            Scattering coefficient f_m at each position.
        method : str
            SIR kernel method (default "sdi" — derivatives always use SDI).

        Returns
        -------
        rf : (Nt, E_rx) ndarray
            RF signal per receive element.
        coords : dict
            {"t0": float, "dt": float}
        """

    def compute_sequence(
        self,
        scatterer_positions_mm,
        scattering_amplitudes,
        tx_events,  # list of dicts: [{"delays": [...], "apodization": [...]}]
        *,
        method="sdi",
    ) -> tuple[ndarray, dict]:
        """Compute RF for multiple TX events (e.g., scan line sweep).

        For each TX event:
        1. Set TX delays/apodization per event dict
        2. Compute dh_tx at scatterer positions
        3. Compute RF per RX element
        Returns (N_lines, Nt, E_rx), coords.

        Analogous to looping calc_scat_multi over scan lines in Field II.
        TX transducer state is restored after all events.
        """

    def compute_all(
        self,
        scatterer_positions_mm,
        scattering_amplitudes,
        *,
        method="sdi",
    ) -> tuple[ndarray, dict]:
        """Full matrix capture: each element transmits, all receive.

        Element e_tx transmits with delta excitation (or self.excitation),
        all E_rx elements receive. Repeats for each TX element.
        Returns (E_tx, Nt, E_rx), coords.

        Analogous to Field II's calc_scat_all. Used for synthetic aperture
        imaging and full matrix capture research.
        """
```

### 4.4 Data flow diagram

```
TX transducer                         RX transducer
    │                                     │
    ├─ delays, apodization                ├─ apodization
    ├─ impulse_response (optional)        ├─ impulse_response (optional)
    │                                     │
    └──────────────┬──────────────────────┘
                   ▼
    Per RX element e_rx:
    ┌─────────────────────────────────────────────────────┐
    │ compute_pe_sdi(pts, tx_all_patches, rx_elem_patches)│
    │   → Dh_pe (P, T) — combined PE SIR derivative      │
    │                                                     │
    │ FFT(Dh_pe)                                          │
    │   × FFT(v)         ← excitation (no derivative)     │
    │   × FFT(ir_tx)     ← TX impulse response            │
    │   × FFT(ir_rx)     ← RX impulse response            │
    │   × H_att(f, d)    ← causal attenuation TF          │
    │                                                     │
    │ IFFT → weight by f_m(r) → sum over scatterers       │
    │ → rf[:, e_rx]                                       │
    └─────────────────────────────────────────────────────┘
                   ▼
    Scale by rho / (2 * c^2)
                   ▼
         rf(t, e_rx)  ← output shape (Nt, E_rx)
```

**Frequency domain multiplications** are the key efficiency win: all convolutions become
element-wise multiplies. IR_tx, IR_rx, V, and H_att are each precomputed once. Only
Dh_pe varies per RX element. When IR is None, corresponding FFT term = 1
(identity).

### 4.5 Memory strategy for reception

**Element-loop (same pattern as Emission, decided during Batch 3 implementation).**

Monolithic pre-allocation of per-element SIR arrays causes OOM for matrix arrays (E_rx=3025).
Instead, loop over RX elements using PE SDI kernel:

```python
rf = zeros((T, E_rx), float32)

for e_rx in range(n_rx_elements):
    mask_e = rx_sub_el_idx_arr == e_rx
    # Compute Dh_pe via PE SDI kernel — 16 deltas per (m_e, m_r) pair + 1 cumsum.
    Dh_pe = compute_pe_sdi(
        scatterer_positions, tx_all_patches, rx_patches[mask_e], ...
    )  # (P, T)
    H_pe = rfft(Dh_pe, n=nfft, axis=1)  # (P, N_freq)
    H_pe *= fft_v             # excitation (no derivative)
    H_pe *= fft_ir_tx * fft_ir_rx  # impulse responses
    if do_attenuation:
        H_pe *= H_att_e       # per-element attenuation
    rf_pe = irfft(H_pe, n=nfft, axis=1)[:, :T]
    rf[:, e_rx] = (rf_pe * amps[:, None]).sum(axis=0) * scale
```

Peak memory per e_rx iteration: `O(P × nfft)` float32 — E_rx-independent.
Print memory estimate before main loop: `P × E_rx × T × 4` bytes (for user awareness even though we don't pre-allocate).

---

## 5. Module Organization

### Final structure

```
src/pyfield/
├── h_sir/
│   ├── __init__.py                    # UPDATE: export new kernels + compute_derivative
│   ├── farfield_rect_patch.py         # NO CHANGE (proven core)
│   ├── hsir_SDI.py                    # NO CHANGE
│   ├── h_sir.py                       # UPDATE: add compute_derivative method
│   └── sir_derivatives.py             # NEW: d2h, dh, per-element kernels, integration helpers
├── psimulation/
│   ├── __init__.py                    # UPDATE: export Emission, Reception, PyField (compat)
│   ├── PyField.py                     # REWRITE: thin deprecated subclass of Emission
│   ├── emission.py                    # NEW: Emission class (replaces PyField)
│   ├── sir_to_pressure.py             # UPDATE: add attenuation param to existing functions
│   ├── attenuation.py                 # NEW: causal TF, distance helpers, unit conversion
│   └── reception.py                   # NEW: Reception class, RF computation
├── transducers/
│   └── base.py                        # UPDATE: add impulse_response property
└── utilities/
    └── helper_functions.py            # UPDATE: return sub_el_idx_arr from compute_sub_elem_attributes
```

### Files NOT to touch

- `farfield_rect_patch.py` — proven core engine
- `hsir_SDI.py` — existing reference implementation
- `plotting/` — no changes
- `cache/` — independent, secret
- `scans/` — independent, personal

### Exports

```python
# psimulation/__init__.py
from .emission import Emission
from .reception import Reception
from .PyField import PyField  # deprecated alias
from .attenuation import (
    causal_attenuation_tf,
    compute_reception_distances,
    reduce_patch_distances_to_element,
)

__all__ = [
    "Emission", "Reception", "PyField",
    "causal_attenuation_tf",
    "compute_reception_distances",
    "reduce_patch_distances_to_element",
]

# pyfield/__init__.py — add Emission, Reception to top-level
```

---

## 6. `.set()` Pattern

Both `Emission` and `Reception` share same `.set()` design:

```python
# Allowed parameter names and their types/validators
_SETTABLE = {
    "c": (float, "Speed of sound (m/s)"),
    "rho": (float, "Density (kg/m^3)"),
    "fs": (float, "Sampling frequency (Hz)"),
    "alpha0": ((float, type(None)), "Attenuation dB/(MHz^y cm) or None"),
    "freq_power": (float, "Attenuation exponent"),
    "excitation": ((np.ndarray, type(None)), "Excitation pulse or None"),
    "monochromatic": (bool, "CW mode flag"),  # Emission only
    "verbose": (bool, "Print diagnostics"),
}

def set(self, name: str, value):
    if name not in self._SETTABLE:
        raise ValueError(f"Unknown parameter '{name}'. Valid: {list(self._SETTABLE)}")
    # type check
    expected = self._SETTABLE[name][0]
    if not isinstance(value, expected):
        raise TypeError(f"'{name}' expects {expected}, got {type(value)}")
    setattr(self, name, value)
    # trigger recomputation if needed
    if name == "transducer":
        self._refresh_sub_elem_attributes()
    if name == "fs":
        # fs change may invalidate cached excitation FFTs
        self._invalidate_caches()
```

Usage:
```python
sim = Emission(tx)
sim.set("alpha0", 0.5)           # enable attenuation
sim.set("excitation", pulse)     # (L,) global excitation
sim.set("excitation", per_elem)  # (L, E) per-element
sim.set("monochromatic", True)   # switch to CW
sim.set("excitation", None)      # back to pulsed

# Also direct attribute access works (no validation):
sim.alpha0 = 0.5
```

---

## 7. Transducer `impulse_response` Property

### What it represents

The electromechanical impulse response models the transduction between electrical and
acoustic domains. In Field II this is set separately for TX and RX via `xdc_impulse`.

- **TX**: electrical signal → acoustic pressure (how element vibrates given voltage)
- **RX**: incoming pressure → electrical signal (how element converts pressure to voltage)
- **Default None**: ideal transducer = delta function = no filtering

### Where it's used

- `Emission`: if `tx.impulse_response` is not None, convolve with excitation in freq
  domain: `V_eff(f) = FFT(excitation) * FFT(ir_tx)`
- `Reception`: both `ir_tx` and `ir_rx` enter the frequency-domain product chain.

### Implementation

Add to `TransducerBase.__init__`:
```python
self._impulse_response: Optional[np.ndarray] = None
```

Add property with getter/setter (shown in §4.2 above).

No changes to existing transducer subclasses needed — inherited from base.

---

## 7.2 `compute_delays` — Plane-Wave Steering Extension

### Motivation

Current `compute_delays(focus_mm)` only handles focused beams. Two common PW cases
need steering:

1. **Linear array** (elements along x, z≈0): single xz-plane angle θ_x.
2. **Matrix array or 3D custom array** (elements span x-y, may have different z):
   two angles θ_x (xz-plane) and θ_y (yz-plane). Single angle would only steer in
   one direction and ignore the other lateral dimension.
3. **Depth-staggered custom array at θ=0** (like `05_bigelementstx`): straight-ahead
   PW — element z-positions already provide the needed depth correction. Same formula,
   angles both zero.

### New signature

```python
def compute_delays(
    self,
    focus_mm=None,
    *,
    angle_steering_deg=None,  # float or (float, float) — see below
    c: Optional[float] = None,
    inline: bool = True,
    plot: bool = False,
) -> np.ndarray:
```

**Mutual exclusivity**: exactly one of `focus_mm` / `angle_steering_deg` must be non-None.
Both None or both set → `ValueError`.

**`angle_steering_deg` input forms**:
- `float` → linear array, steer in xz-plane only: `(θ_x, θ_y=0)`
- `(float, float)` → 2D steer `(θ_x, θ_y)` for matrix or 3D arrays

### Steering delay formula

```python
# Parse angles
if isinstance(angle_steering_deg, (int, float)):
    theta_x_deg, theta_y_deg = float(angle_steering_deg), 0.0
else:
    theta_x_deg, theta_y_deg = float(angle_steering_deg[0]), float(angle_steering_deg[1])

theta_x = np.deg2rad(theta_x_deg)
theta_y = np.deg2rad(theta_y_deg)
sin_x, sin_y = np.sin(theta_x), np.sin(theta_y)
nz_sq = 1.0 - sin_x**2 - sin_y**2
if nz_sq < 0:
    raise ValueError(
        f"Steering angles ({theta_x_deg:.1f}°, {theta_y_deg:.1f}°) exceed physical limit "
        f"(sin²θ_x + sin²θ_y must be ≤ 1)."
    )
n = np.array([sin_x, sin_y, np.sqrt(nz_sq)])  # unit steering direction

# Project each element center onto the steering direction
d = self.element_centers @ n  # (E,) signed projections

# Delay: element with maximum projection fires first (zero delay)
delays = (d.max() - d) / c   # (E,) non-negative, minimum always 0
```

**Physical interpretation**: element e fires at delay `(d_max - d_e) / c` so that all
emitted wavefronts reach the same steering-direction phase plane simultaneously, producing
a plane wave traveling in direction `n`.

### Case table

| Array type | `angle_steering_deg` | θ_y | `n` | `d_e` reduces to | Result |
|-----------|---------------------|-----|-----|-------------------|--------|
| Linear flat (z=0) | `0` | 0 | `[0, 0, 1]` | `z_e = 0` | All fire simultaneously |
| Linear flat (z=0) | `30` | 0 | `[0.5, 0, 0.866]` | `0.5 * x_e` | Rightmost fires first |
| Custom 3D (varying z) | `0` | 0 | `[0, 0, 1]` | `z_e` | Deeper elements fire later |
| Matrix (x-y grid, z=0) | `(20, 10)` | 10° | `[sin20°, sin10°, nz]` | `x_e sin20° + y_e sin10°` | 2D steered PW |

The custom-3D case with `angle_steering_deg=0` matches the manual delay computation in
`others/05_bigelementstx/visualize_transient.py` — the formula reproduces it exactly
via the z-projection term.

### Mono-element behavior unchanged

Mono-element transducers still warn and return `np.zeros(1)` regardless of `focus_mm`
or `angle_steering_deg` — entire aperture fires simultaneously.

### Files to update

- `src/pyfield/transducers/base.py`: modify `compute_delays` signature and body.
- `src/pyfield/transducers/validators.py`: add `validate_steering_angles` if needed.
- Tests: focused unchanged. Add: `angle_steering_deg=0` on flat array → all-zero delays.
  `angle_steering_deg=30` on flat linear → monotone delays proportional to x-position.
  `angle_steering_deg=0` on depth-staggered custom array → delays proportional to z.

---

## 8. Implementation Order

### Phase 1: Foundation (no API changes visible to user)
1. `compute_sub_elem_attributes` → return `sub_el_idx_arr`. Update call sites.
2. `compute_delays` → add `angle_steering_deg` param (§7.2). Focus path unchanged.
3. `sir_derivatives.py`: implement `compute_d2h`.
4. Test: `compute_d2h` + 2 cumsums ≈ existing `compute_h_sir` (float32 tolerance).
   Test: `angle_steering_deg=0` flat array → zero delays. `angle_steering_deg=30` → monotone x. Depth-staggered → proportional to z.

### Phase 2: Remaining derivative kernels
4. `compute_dh`, `compute_d2h_per_element`, `compute_dh_per_element`.
5. Integration helpers.
6. `compute_h_sir_patch_parallel` reference kernel.
7. Test: sum over E of per-element == summed-all version.
8. `h_sir.py`: add `compute_derivative` method.

### Phase 3: Attenuation module
9. `attenuation.py`: `causal_attenuation_tf`, `convert_alpha0_to_nepers`,
   `compute_attenuation_distances`.
10. Integrate into `sir_to_pressure.py` (add `alpha0`, `freq_power`, `f0`, `distances`
    params to both functions — default None = no attenuation).
11. Test: `alpha0=None` identical output. `alpha0>0` reduces amplitude with distance.
    `y=1` special case vs `y=1.001`.

### Phase 4: Emission class
12. Create `emission.py` with `Emission` class.
13. Move logic from `PyField.__call__` into `Emission.__call__`, adapting for new defaults
    (pulsed by default, excitation shape dispatch).
14. Add per-element excitation path: `from_sir_to_pressure_per_element`.
15. Rewrite `PyField.py` as deprecated subclass.
16. Update `__init__.py` exports.
17. Test: old `PyField` usage still works (deprecation warning). `Emission` with
    same params gives same results. Uniform per-element == global excitation.

### Phase 5: Transducer impulse response
18. Add `impulse_response` property to `TransducerBase`.
19. Wire into `Emission` freq-domain pipeline.
20. Test: `ir=None` gives same result. `ir=delta` gives same result.

### Phase 6: Reception class
21. Create `reception.py` with `Reception` class.
22. Implement `__call__` for RF.
23. Test: single point scatterer gives symmetric PSF. Pulse-echo with identical TX/RX.
24. Implement `compute_sequence` and `compute_all`.
25. Test: multi-line with single focus == single call.

---

## 9. Performance Constraints

- **Numba kernels**: `@njit(parallel=True, fastmath=True)`, `prange` over P.
- **Memory budgets**: print estimate before big allocations. Batch P when
  `P * E * T * 4 bytes > threshold` (configurable, default 2 GB).
- **Float32 throughout FFT**: all kernel outputs AND freq-domain ops stay float32 →
  complex64. No upcast to float64 (half memory vs original spec). FFT batch size
  `batch_P = 400 MB / (nfft×4 + 2×N_freq×8)` bytes.
- **Freq-domain**: `scipy.fft.rfft/irfft(workers=-1)` — multithreaded. **Not** numpy.fft.
  scipy with `workers=-1` uses all cores automatically; provides ~2–3× speedup over
  single-threaded numpy for large arrays.
- **Pre-allocated h_pad**: single `zeros((batch_P, nfft), float32)` outside all loops.
  scipy.fft on already-nfft-length input creates no internal zero-padding buffer per call.
  Without pre-allocation: `E × n_batches × ~140 MB` = ~17 GB zero-allocations → OS swap.
  With pre-allocation: buffer created once, reused every iteration.
- **Loop structure for per-element**: P-outer, E-inner. Accumulate freq-domain across E,
  then one irfft per P-batch. Total irfft = n_batches (not `E × n_batches`).
  Rationale: `irfft(sum_e H_e) = sum_e irfft(H_e)` (linearity), and one irfft
  preserves inter-element interference (abs only after full E sum).
- **ETA print**: after first P-batch, print `~total_min = t_first × n_batches`.
- **Attenuation TF**: compute per-element inside element loop, `(P_batch, N_rfft)` per iteration.
- **Emission**: element-loop over E, P-batched inside. Peak `O(P_batch × nfft)` per element.
- **Reception**: element-loop over E_rx, P-batched inside. Same peak bound.

### Measured benchmark (E=128, P=60501, nfft=8192)

| Mode | Time |
|------|------|
| Monochromatic global | ~29 s |
| Pulsed global | ~11 s |
| Global excitation | ~24 s |
| Per-element (E=128) | ~17 min |

Per-element bottleneck is irreducible on CPU: `E × n_batches × t_rfft`.
For E=128, 15 P-batches, `t_rfft ≈ 0.53 s` → `128 × 15 × 0.53 ≈ 1020 s`.
Options to reduce: coarser grid (fewer P → fewer batches), GPU, or use global
path when all elements share the same pulse.

---

## 10. API Summary

### Transducer delays (focused and plane-wave)

```python
tx = LinearArrayTransducer(...)

# Focused beam (existing)
tx.compute_delays(focus_mm=[0, 0, 30])

# Steered PW — linear array, xz-plane only
tx.compute_delays(angle_steering_deg=0)     # straight-ahead PW
tx.compute_delays(angle_steering_deg=20)    # steer 20° in xz-plane
tx.compute_delays(angle_steering_deg=-15)   # steer -15°

# Steered PW — matrix or 3D custom array, two angles
tx.compute_delays(angle_steering_deg=(20, 10))   # θ_x=20°, θ_y=10°
tx.compute_delays(angle_steering_deg=(0, 0))     # straight-ahead on depth-staggered array
```

### Emission (replaces PyField)

```python
from pyfield.psimulation import Emission

# Pulsed (default — raw SIR)
sim = Emission(tx)
p, coords = sim(field_points)

# Monochromatic CW
sim = Emission(tx, monochromatic=True)
p, coords = sim(field_points)

# Transient with excitation
sim = Emission(tx, excitation=pulse, fs=200e6)
p, coords = sim(field_points)

# With attenuation
sim = Emission(tx, excitation=pulse, alpha0=0.5, freq_power=1.1)
p, coords = sim(field_points)

# Per-element excitation
exc_per_elem = np.stack([pulse * w for w in weights])  # (E, L)
sim = Emission(tx, excitation=exc_per_elem)
p, coords = sim(field_points)

# Runtime changes
sim.set("alpha0", 0.3)
sim.set("excitation", None)  # back to pulsed
p2, coords2 = sim(field_points)
```

### Reception

```python
from pyfield.psimulation import Reception

tx = LinearArrayTransducer(...)
tx.compute_delays(focus_mm=[0, 0, 30])
tx.impulse_response = ir_tx  # optional electromechanical IR

rx = LinearArrayTransducer(...)  # can be same or different
rx.impulse_response = ir_rx

sim = Reception(tx, rx, excitation=pulse, fs=200e6, alpha0=0.5)

# Single-focus RF
scatterer_pos = np.array([[0, 0, 30], [1, 0, 35]])  # mm
scatterer_amp = np.array([1.0, 0.5])
rf, coords = sim(scatterer_pos, scatterer_amp)
# rf shape: (Nt, E_rx)

# Multi-line (sweep TX focus)
focuses = [[0, 0, 30], [1, 0, 30], [2, 0, 30]]
rf_multi, coords = sim.compute_sequence(scatterer_pos, scatterer_amp, focuses)
# rf_multi shape: (N_lines, Nt, E_rx)
```

### Backward compatible

```python
from pyfield import PyField  # still works, emits DeprecationWarning
sim = PyField(tx)  # monochromatic=True by default (old behavior)
p, coords = sim(field_points)
```

---

## 11. Testing Strategy

| Test | Validates |
|------|-----------|
| `compute_d2h` + 2 cumsums ≈ `compute_h_sir` | SDI delta-placement in new kernel |
| `angle_steering_deg=0` flat array → `delays == 0` | Straight-ahead PW, flat geometry |
| `angle_steering_deg=30` flat linear → delays ∝ x | xz-plane steering formula |
| `angle_steering_deg=0` depth-staggered → delays ∝ z | z-projection term handles depth |
| `angle_steering_deg=(θ_x, θ_y)` flat matrix → delays ∝ x·sin(θ_x)+y·sin(θ_y) | 2D steering |
| `focus_mm` + `angle_steering_deg` both set → `ValueError` | Mutual exclusivity |
| Per-element sum over E == summed-all | Grouping logic |
| `Emission` pulsed == old `PyField(monochromatic=False)` no excitation | Migration correctness |
| `Emission` mono == old `PyField(monochromatic=True)` | Migration correctness |
| `Emission` + excitation == old `PyField` + excitation | Migration correctness |
| `alpha0=None` identical to no-attenuation path | Attenuation doesn't break default |
| Uniform per-element excitation == global excitation | Per-element reduces to global |
| `alpha0>0` amplitude decreases with distance | Physics sanity |
| `y=1` special case ≈ `y=1.001` general case (continuous) | Log branch correctness |
| `impulse_response=None` == `impulse_response=delta` | IR default behavior |
| `Reception` single scatterer on axis → symmetric RF | Pulse-echo PSF |
| `Reception` TX==RX same transducer → valid pulse-echo | Self-echo correctness |
| `compute_sequence` with 1 focus == single `__call__` | Multi-line consistency |
| `PyField` deprecated wrapper gives same results + warning | Backward compatibility |

Use `numpy.testing.assert_allclose` with `rtol=1e-4` for float32.

---

## 12. Future Extensions (NOT in this iteration)

Documented here for awareness — do not implement now, but design should not block these:

| Feature | Field II equivalent | Notes |
|---------|-------------------|-------|
| **Dynamic receive focusing** | `xdc_dynamic_focus`, timeline-based `xdc_focus` | Focus tracks with depth (time). Requires timeline system for apodization/delays. Design `Reception` so delays can be swapped between calls without breaking. |
| **Baffle conditions** | `xdc_baffle(Th, soft_flag)` | Soft baffle multiplies SIR by `z_p/(c*t)` directivity factor. Would need modification in SIR kernel or post-processing. |
| **Per-element impulse response** | `ele_waveform(Th, el, wav)` | Different IR per element (models manufacturing variation). Would extend `impulse_response` from `(L,)` to `(L, E)` or dict. |
| **Per-sub-element apodization** | `ele_apodization(Th, el, apo)` | Apodization at mathematical element (patch) level, independent of physical element apodization. |
| **Beamformed output** | `calc_scat` | Delay-and-sum on receive. Keep as user's responsibility — provide utility function if needed. |
| **Speed of sound per scatterer** | Field II `speeds` parameter | For heterogeneous media. Would need per-scatterer `c` in SIR computation. |

---

## 13. Documentation Updates

After implementation, update:

- `CLAUDE.md`: new module structure, `Emission`/`Reception` examples, attenuation usage.
- `docs/user-guide/`: add guides for emission, reception, attenuation.
- `docs/api/`: add reference pages for new modules.
- `CHANGELOG.md`: document breaking change (PyField → Emission).
- `zensical.toml`: add new pages to nav.

# How eSDIva computes a pulse-echo RF — the backends, in plain terms

Every reception backend computes the **same** physical signal. They differ only in *how*
they do the arithmetic, and that choice decides the speed. This page explains each one
with no jargon assumed, so you can pick the right one and understand why it wins.

## The one equation they all solve

A point scatterer at position `r` sends back an echo. The recorded echo on one receive
element is

```
echo(t)  =  amplitude · ( pulse  ⊛  h_tx(r,t)  ⊛  h_rx(r,t) )
```

- `⊛` means **convolution** (sliding-overlap of two signals).
- `h_tx(r,t)` = the **transmit spatial impulse response (SIR)**: the sound the whole
  transmit aperture delivers to `r` over time if you hit it with an instantaneous spark.
  For one small rectangular patch it is a **trapezoid** in time; the aperture SIR is the
  sum of all patch trapezoids, each delayed by its focusing delay.
- `h_rx(r,t)` = the same thing on receive (by reciprocity, identical machinery).
- `pulse` = the electrical drive and the transducer's own ringing, all convolved together
  (`exc ⊛ ir_tx ⊛ ir_rx`). It is **short** — tens of samples.

The full RF is this echo summed over every scatterer. **That sum, and those two
convolutions, are the whole cost.** The backends are just three ways to pay it.

Two facts make the problem interesting:

1. **Each scatterer's echo is SHORT in time** — it only exists in a narrow window around
   its round-trip arrival. A scatterer at 5 mm and one at 50 mm have echoes in completely
   different, small slices of the recording.
2. **The convolutions can be done in the time domain (direct sliding sum) or in the
   frequency domain (FFT → multiply → inverse FFT).** FFT is cheap *per long signal* but
   its cost is set by the length of the signal you transform, not by how much of it is
   non-zero.

Those two facts are the entire story of why one backend beats another.

---

## Backend 1 — `spectral` (the default)

**Idea: skip sampling the trapezoid entirely. Write its Fourier transform in closed form.**

A trapezoid is two rectangles convolved, and the Fourier transform of a rectangle is a
`sinc`. So each patch's SIR spectrum is known exactly, with no time axis at all:

```
H_patch(ω) = area/(2π·dist) · directivity · sinc(ω·Δt₁/2) · sinc(ω·Δt₂/2) · e^{-jω·t_center}
```

Sum over patches → the aperture spectrum `H_tx(ω)`. The two-way spectrum is just
`H_tx(ω)·H_rx(ω)` (convolution becomes a product in frequency). Multiply by the pulse
spectrum, inverse-FFT once, done.

- **No time sampling, no forward FFT of any SIR, and it only touches the frequency bins
  the pulse actually occupies** (band-limited → few bins).
- **Cost grows with: number of patches × number of in-band frequency bins.**
- **Fastest when** the pulse is narrow-band relative to the sampling rate and the scatterer
  echoes are *wide* in time (deep focus, steered/unfocused transmit) — exactly where the
  time-domain methods get expensive.
- **Slower when** there are very many patches and a high centre frequency (many bins),
  because its cost is dominated by evaluating `H(ω)` for every patch at every bin.

## Backend 2 — conventional / `fst` / `sdi` (sampled SIR + FFT convolution)

**Idea: sample both trapezoids onto a time grid, FFT-convolve them.**

- `fst` fills the trapezoid sample by sample; `sdi` places the 4 corner deltas of its
  2nd derivative and integrates (cheaper for wide trapezoids). Both produce the sampled
  `h_tx` and `h_rx`.
- Then it FFTs them, multiplies, inverse-FFTs.

The catch: it builds **one FFT whose length spans the entire depth window**, for *all*
scatterers at once. A scatterer at 5 mm and one at 50 mm share the same giant transform,
even though each echo fills only a sliver of it. **Most of that FFT is transforming zeros.**

- **Cost is dominated by the FFT, and the FFT length is set by the whole acquisition depth,
  not by how short each echo is.** Making the SIR kernel faster barely helps — the SIR is
  a tiny fraction of the time; the FFT is the wall.
- This is why it lands close to open-source Field II: same FFT-convolution structure.
- `depth-binning` (`n_depth_bins`) softens this by grouping scatterers by depth so each
  bin's FFT is shorter — a partial fix, with its own per-bin overhead.

## Backend 3 — temporal placement (`method="temporal"`/`"fst"`/`"sdi"` on the conventional path)

**Idea: exploit fact #1. Each echo is short, so do the convolution in the time domain over
just its non-zero window, and drop it into the output at its own time offset.**

For each receive channel and each scatterer:
1. Take the scatterer's transmit SIR (computed once for all channels, kept with its
   non-zero support).
2. Sample this channel's receive SIR **only over its own short support**, into a small
   reused buffer — no scatterers × time receive array is ever built.
3. Convolve the two **tight** supports directly in time (small × small) and **add** the
   amplitude-weighted result into the channel at the sample where the echo starts.

Then, once per channel, convolve the summed two-way SIR with the two-way pulse
`exc ⊛ ir_tx ⊛ ir_rx`: the pulse is the same for every scatterer, so by linearity it does
not need to be applied scatterer by scatterer. The sums are float64 (float32 drifts ~1e-4
of the peak over 1e5 scatterers).

No FFT at all and no wasted work on zeros. This mirrors the structure Jensen describes for
the multi-threaded Field II (IUS 2014): short per-scatterer signals, pre-allocated buffers
reused from cache, work split over threads. Field II's source is not public, so the exact
order of its convolutions is not known; the equivalence is in structure, not code.

What made the difference (measured, Domino 128 el, 10 000 scatterers, FST): the former
version built a full scatterers × time receive SIR per element (~1.7 s), convolved the
pulse per scatterer (~0.9 s) and allocated a buffer per scatterer (~0.6 s of the
convolution) — 3.3 s total, now 0.39 s.

- **Cost grows with: channels × scatterers × (transmit support × receive support).**
  It does **not** grow with acquisition depth — a scatterer deep in the volume is as cheap
  as a shallow one.
- Wide transmit supports (deep focus, steered/unfocused transmit) cost more multiplies per
  scatterer, but in the measured deep-focus case below it still beats `spectral`.
- Lossless, single global drive only. Attenuation and per-element drives fall back to the
  conventional FFT path.

---

## Which is fastest? Measured, not guessed

Measured on one machine (Intel i7-13700H, 14 cores), correlation ≥ 0.99999 against
`spectral` in every case. Re-measure on your own machine before sizing a long run.

**Domino linear array, 128 elements, 12.5 MHz, focus 8 mm, scatterers at 5–11 mm**
(tight echoes, fine subdivision, high frequency):

| scatterers | spectral | temporal (fst), former | temporal (fst) |
|-----------:|---------:|-----------------------:|---------------:|
| 10 000     | 9.4 s    | 3.3 s                  | **0.39 s**     |
| 100 000    | 89 s     | 26.7 s                 | **3.9 s**      |

**64-element array, 5 MHz, focus 30 mm, 5 000 scatterers over 10–50 mm**
(wide echoes, deep focus):

| spectral | fst | sdi | auto |
|---------:|----:|----:|-----:|
| 1.77 s | 0.88 s | **0.83 s** | 0.88 s |

**Zeus 55 × 55 matrix, 10 MHz, 1 000 scatterers**: temporal (fst) 58.9 s → 5.0 s with the
current kernel.

- **Lossless, global drive** → **temporal** (`fst`/`sdi`/`auto`) was fastest in every
  measured case.
- **Attenuation or per-element drives** → **`spectral`** (temporal falls back to the
  full-window FFT there).
- **Conventional FFT over the full window** is never the fastest; it stays as the exact
  cross-check and as the fallback for those options.

### Why "make the SIR 60× faster" did *not* speed up conventional

Because conventional is **FFT-bound, not SIR-bound**: in a 64-element run the SIR kernel was
~1.6 % of the wall and the full-window FFT was ~98 %. Speeding up 1.6 % of the work can't
move the total (Amdahl's law). Temporal placement is the backend that *removes* the FFT, so
it is the only one where the fast SIR kernel actually decides the runtime.

---

## One-line picker

- Lossless, single global drive → **temporal** (`method="temporal"`/`"fst"`/`"sdi"`/`"auto"`).
- Attenuation, per-element drives, or no measurement for your case → **`spectral`** (the
  default).
- Need a reference that supports every option → **conventional** (`fst`/`sdi` with
  attenuation, or `per_scatterer=True`).

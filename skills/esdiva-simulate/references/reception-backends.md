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

For each scatterer:
1. FST-fill its two short one-way SIRs (fast — the ×60 SIR speedup finally matters here,
   because nothing downstream swamps it).
2. Convolve the two **tight** supports directly in time (small × small).
3. Convolve with the short pulse.
4. **Add** the result into the receive channel at the integer sample where its echo starts.

No FFT at all. No wasted work on zeros. This is exactly how Field II builds its RF
(`calc_scat` places each short response at its `tstart`); Field II **Pro**'s ~20× speedup
over open Field II is this same loop in multithreaded C.

- **Cost grows with: scatterer count × (transmit support width × receive support width).**
  It does **not** grow with acquisition depth — a scatterer deep in the volume is as cheap
  as a shallow one.
- **Fastest when** each echo's support is *tight* (scatterers near the transmit focus,
  shallow geometry, fine subdivision at high frequency where the other methods' FFT/bin
  costs balloon).
- **Slower when** the transmit support is *wide* (deep focus or unfocused/steered
  transmit spreads arrivals over a long window), because the direct `support × support`
  convolution then does a lot of multiplies per scatterer — that is the regime where
  `spectral` wins instead.

---

## Which is fastest? It depends on the echo width — measured, not guessed

Two measured regimes on this machine, both with pulse-echo correlation 1.0000 against the
validated conventional RF (so all three are the same signal):

**Domino linear array, 128 elements, 12.5 MHz, focus 8 mm, scatterers at 5–11 mm**
(tight echoes, fine subdivision, high frequency):

| scatterers | open Field II | conventional (fst) | spectral | temporal splat |
|-----------:|--------------:|-------------------:|---------:|---------------:|
| 100        | 0.088 s       | 0.088 s            | 0.041 s  | **0.030 s**    |
| 1 000      | 0.842 s       | 0.663 s            | 0.277 s  | **0.165 s**    |
| 10 000     | 8.53 s        | 6.30 s             | 2.56 s   | **1.12 s**     |

Here the splat wins: ~2.3× over spectral, ~5.6× over conventional, ~7.6× over open Field II.

**64-element array, 5 MHz, focus 30 mm, 5 000 scatterers over 10–50 mm**
(wide echoes, deep focus):

| spectral | temporal splat |
|---------:|---------------:|
| **0.59 s** | 0.93 s       |

Here spectral wins, because the deep focus makes each transmit support wide and the splat's
`support × support` convolution expensive.

**The discriminator is echo-support width**, set by focus depth and transmit steering:

- **Tight support** (near focus, shallow, unsteered) → **temporal splat** wins.
- **Wide support** (deep focus, steered/unfocused, diverging waves) → **spectral** wins.
- **Conventional (FFT over the full window)** is almost never the fastest — its FFT length
  is tied to depth, not echo width. It stays as the exact cross-check and the backend that
  already supports every option.

### Why "make the SIR 60× faster" did *not* speed up conventional

Because conventional is **FFT-bound, not SIR-bound**: in a 64-element run the SIR kernel was
~1.6 % of the wall and the full-window FFT was ~98 %. Speeding up 1.6 % of the work can't
move the total (Amdahl's law). The temporal splat is the backend that *removes* the FFT, so
it is the only one where the fast SIR kernel actually decides the runtime.

---

## One-line picker

- Default / mixed / deep / steered → **`spectral`**.
- Shallow phantom near focus, high frequency, many scatterers → **temporal**
  (`method="temporal"`/`"fst"`/`"sdi"`).
- Need a byte-for-byte reference or every exotic option → **conventional** (`fst`/`sdi`).

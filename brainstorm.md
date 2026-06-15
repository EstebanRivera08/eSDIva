# SDI Reception — Speedup Brainstorm

Top ideas to make Sparse Delta Integration (SDI) reception significantly faster than
conventional pulse-echo methods. CPU-only now (NumPy / Numba / PyTorch-CPU), but
prioritized for future GPU mapping.

## Tier 1 — biggest win

### 1. Kill the M² wall via separable corner factorization
`Δδ_pe = d2h^e ⊛ d2h^r` = 16 deltas per (mₑ, mᵣ) patch pair → M² cost. But corner
times split additively: `τ(mₑ, mᵣ) = τₑ(mₑ) + τᵣ(mᵣ)`. The two-way delay is the sum of
one-way delays. So instead of an M² pair loop, place the Mₑ TX corner trains once, place
the Mᵣ RX corner trains once, then **outer-sum in delay space** = convolution of two
sparse spike trains. Cost ≈ M + M placement + one sparse conv, not M². This is the real
prize.

### 2. Fourier I⁴ already cumsum-free — push whole chain to one FFT
Already have `p_r = (E_m·v) ⊛ (I⁴ Δδ_pe)` with `I⁴ = ÷(jω)⁴·fs`. Fuse exc·IR·÷(jω)⁴
into one precomputed spectral multiplier `G(ω)`. Per point: scatter 16 deltas → rFFT →
×G(ω) → irFFT. No cumsum, no time-domain conv. High arithmetic intensity, batched over
points.

### 3. Batch all field points as one tensor scatter
Corner times/weights are a pure function of geometry. Stack (P points, M patches, 4
corners) → one big scatter-add into a (P, Nt) buffer. Map/reduce, no per-point Python
loop. NumPy `np.add.at` / Numba `prange` over P / torch `index_add_`. GPU-trivial later.

## Tier 2 — structural

### 4. Low-rank two-way SIR over scatterer grid
Neighboring scatterers share nearly identical corner delays (smooth in space). SVD /
Chebyshev the delay+amplitude field → evaluate few basis points, interpolate the rest.
Turns P evaluations into r ≪ P. Huge for dense PSF grids.

### 5. Reciprocity fold for monostatic / symmetric arrays
TX = RX → `d2h^e = d2h^r`. Two-way kernel = autoconvolution → 16 deltas collapse to 10
distinct (symmetric pairs ×2). ~40% fewer events. Free when same aperture.

### 6. Element-to-element redundancy = shift/translation
Linear array: the patch geometry of element e is element 0 translated by e·pitch. Corner
delays for a far scatterer differ by a smooth phase ramp. Compute the reference element
once, **phase-shift in Fourier** for the rest instead of recomputing the SIR. Converts E
evaluations → 1 + E cheap spectral shifts.

### 7. Delta-train compression — merge coincident corners
Many of the 16 events land in the same/adjacent time bins (interpolation already splits
each to 2). Pre-merge by sorting + segment-reduce → fewer scatter writes, better
locality. Helps SIMD.

## Tier 3 — speculative / future-GPU

### 8. Polynomial-moment (spectral) representation instead of deltas
Trapezoid is piecewise linear → its Fourier transform is closed-form (sum of sinc²·phase
terms). Skip the time-domain scatter entirely: accumulate **complex spectral
contributions** directly per corner. All-algebraic, no irregular indexing, perfect
SIMD/GPU map-reduce. Possibly the cleanest GPU form.

### 9. Fuse emission + reception in one pass
For full PSF/imaging, the h_tx computed in emission overlaps h_tx in reception. Share the
TX corner train between the forward and pulse-echo passes → no recompute.

### 10. Block-sparse pair tiling
If keeping M² (Method 3 complete), tile (mₑ, mᵣ) pairs so each tile reuses cached TX
corners in registers. Raises arithmetic intensity, cache-friendly, maps to GPU warps.

### 11. Analytic w(t) = I⁴ v_pe precompute (complete form)
Move I⁴ onto the velocity once → w is fixed per excitation. Then RF = Σ aᵢaⱼ w(t−τᵢ−τⱼ):
pure gather-add of one precomputed waveform. Exact, but M²; pairs with idea #1 to drop to
M.

## Ranked bets

1. **#1 separable delay outer-sum** — breaks M² → M. Do first.
2. **#2 + #3 fused-spectral batched scatter** — most infra already exists.
3. **#6 element shift-in-Fourier** — kills E redundancy.
4. **#8 spectral-moment** — best long-term GPU shape.

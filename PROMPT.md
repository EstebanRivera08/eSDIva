I need you to clean up and reorganize the SIR core modules. The goal is to keep only code that is physically meaningful, readable, and directly comparable to the equations. Avoid unnecessary abstractions and local helper methods.

**Tasks**

1. **hsir module**
   - `h_sir` was renamed to `hsir` (check all imports).
   - I deleted `h_derivatives`, `h_SDI`, and other bulky scripts.
   - I created `transducer_sir.py` as a clean template (rewritten from `farfield_rect_patch`). Use this style: readable, minimal, and physics‑aligned.  
     Note: `patch_frames` → `patch_nvector`.
   - `transducer_sir_pe.py` contains the pulse‑echo parts. Adapt it so it mirrors the structure of `transducer_sir.py` and includes functions like `compute_Dh_pe()`.
   - I added `element_sir.py` and `element_sir_pe.py`. They should mirror the transducer versions but operate per element.
   - I removed `h_sir.py` entirely; the class was unnecessary.

2. **psimulation module**
   - It no longer exists. It is now split into:
     - `emission`
     - `reception`
     - `attenuation`
   - Some modules need new or updated `__init__.py` files.
   - Several scripts in emission/reception are too long; splitting them into smaller, well‑named modules is required.


Some more especific tasks would include:

1) Emission and Reception validation of dependencies of other modules and correct
deivision of computation per element or global for transducer.

2) Especifically for Reception, create two classes.

These classes computes:
rf = v_pe ⊛_t h_pe ⊛_r  f_m
with ⊛ convolution. So they both needs to be sent to fourier domain.

The main difference is :
Reception: This should admits the 3 methods (sdi, naive, auto) and is the
conventional fieldii implementation with:
v_pe = (ρ₀/2c₀²) × E_m ⊛_t ∂³v/∂t³     ← 3 derivatives on excitation
h_pe = h_tx(r₁→r₅) ⊛_t h_rx(r₅→r₁)     ← no derivatives on SIR

whereas,
ReceptionSDI: Performs the new SDI formulation and just admits SDI for its nature
(redistribute all 3 derivatives onto SIR side) turning the equation to:
rf = v_pe' ⊛_t Dh_pe ⊛_r  f_m

v_pe' = (ρ₀/2c₀²) × E_m × v            ← no derivatives
Dh_pe = dh_tx/dt ⊛_t d²h_rx/dt² = ∫zeta_pe dt           ← 3 derivatives on SIR

With zeta_pe :
∫zeta_pe dt = d²h_tx/dt² ⊛_t d²h_rx/dt²   <- convolution of 4 deltas with 4 deltas
giving 16 deltas (therefore 32 in discrete implementation).
    .


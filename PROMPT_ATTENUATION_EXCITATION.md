
at h_sir create the numba scripts to compute different kinds of SIR
computations:

All are farfield of rectangular patches, and computation is parallelized over points
and admits : 

1)  non-attenuation, non-per-element excitation (current implementation, no need to
change anythin take it as inspiration at ".\src\pyfield\h_sir\farfield_rect_patch.py")
output: h_sir with size [P,T], all patches sirs have ben summed over patches.

2)  per-element-excitation, per-element-attenuations (assumes low curvature)
outpute: h_sir with size [P, T, E], one might need to batch for memory use. And the
patches sirs are summed over element.

3) per-element-excitation, per-patch attenuation.
output: h_sir with size [P, T, M], will need for sure batching processing for memory.

The Idea is that these functions will be use for each computation case, increasing in
memory is and complexity.

In addition to the last core h_sir implementations:

1) One base implementation ( case 1), that instead of parallelized over points, does it
over patches for testing.

2) 4 scripts :
 * d2h_base: just SDI method truncated before integrating twice, output shape [P,T]
 * dh_base : just SDI method truncated after one integration, outpute shape [P,T]
 * d2h_element : The same as d2h_element but returning the sum over elements size [P,T,E]
 * dh_element : The same as d2h_element but returning the sum over elements size [P,T,E]

These last scripts are meant to be use as core methods for RF simulation. 


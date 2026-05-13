Looking at the .claude/rules/physical-context I have one hypothesis:

# Hypothesis: Receive signal (simulation of RF) with the SDI is faster

From the development of the SDI, we see that the implementation computes directly a
sparse distribution of deltas that represents the second derivative of the spatial
impulse response, And the, to retrieve the SIR, we perform two integrations. This is how 
PyField works to compute transmitted pressure fields, for instance.

However, I am now interested in receive pressure fields (or simulation of RF data). By
looking at the alternative form in section 7 of the equation derived by Jensen, we can
compute this received pressure field with using the second and the first time derivative
of the SIR. Interestingly, the SDI method computes the second derivative directly, and
then after one integration it will find the first derivative. So this computation can be
higher speed up using the SDI method.

# Goal: implement the receiving signal computation in PyField using the SDI method

I'm thinking in this organization, if you find a better way suggest it:

1) in pyfield\h_sir\ we can implement the methods for computing the dh/dt and d2h/dt2
using the SDI method. This will be the core of the implementation, and we can test it
with some simple cases. This just needs to replicate numba the far_field_sir.py file but
truncating the integrations. 

2) the RF_simulation is going to be a class located at pyfield\psimulation\ that will be
responsible for simulating the RF data. This class will use the methods implemented in
step 1 to compute the received pressure fields and maybe common helper_functions of
PyFields class.

3) API, we create the instance 


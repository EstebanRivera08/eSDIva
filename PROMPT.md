Read .claude/rules/physical-context to have the physical context.

# Hypothesis: Receive signal (simulation of RF) with the SDI is faster

From the development of the SDI, we see that the implementation computes directly a
sparse distribution of deltas that represents the second derivative of the spatial
impulse response, And then, to retrieve the SIR, we perform two integrations. This is how 
PyField works for emitted pressure fields.

However, I am now interested in receive pressure fields (or simulation of RF data) and to
have the possibilities of adding per-element excitation (until now the hsir core gives 
a spatio-temporal tensor of size [P,T], where hsir have been summed over elements for 
each point). 

looking at physical context rules 7-9 section in  we can
compute this received pressure field, using the second and the first time derivative
of the SIR. Interestingly, the SDI method computes the second derivative directly, so 
the truncation of the integration steps could lead us to the second an first derivative
easily and gain computation time.

# Goal: Achieve the best implementation of memory-management and parallelization with 
# numba for the highest speed performance and friendly API to have a general simulation 
# with attenuation inclusion if needed, excitation per element, and the receiving signal
# computation in PyField using the SDI method

1) Think in the best architecture for this:
- should we create an attenuation module where everything related to attenuation is to
  be added? some day one might add different attenuation types (causal, acausal, with
  dispersion etc) and we can have to types per-patch and per-element (each ones needs an
  specific output from the h_sir module to be then multiplied in the fourier domain
  representing the convolution operations)

- emission comprises the following pressure field :
  monochromatic amplitude, pulsed-transient, with excitation (all elements with the same
          one), and with excitation-per-element (where Hsir can be truncated to one
          temporal integration and have a small gain).

- reception comprises: PSF calculation, Imaging PSF, RF simulation, etc.

- should we do pyfield.psimulation.emission and pyfield.psimulation.reception instead of
  having the PyField class? or is it better to have a module for emission and another
  one for reception?

2) As I stated before, adding attenuation or per-element excitations changes the
expected output shape of the SIR when computing emission. And if working with emission with 
excitation-per-element one temporal integration can be avoided, and for reception the
omission of some temporal integration increases.

Taking into account that, we can increase the functions of the h_sir class as well as
the numba scripts to be used, so each possible output can be executed with the best
practices and performance. So, create the list of files and functions to have the best
implementation of this reducing repetition of helper functions. But, for example:
if it is better to create an script for each sir (even if the method is the same) do it
if it is really better. If the time performance is not changing so much, one base script
    could exist, for example the one that computes the d2h, and then the integration
    steps can be performed with one helper function to increase readeability and
    optimization of scripts and functions. Give me the best option.

Create the best and most confortable API, reutilising ressources, optimizing the modules
organization, the back-compatibility. The biggest concern is time-performance, memory
and ressources management, easy integration and communication between modules, and if
a powerfull feature is to be added back-compatibility can be sacrifice if it merites it.
Do not touch any important and crucial file. Create new ones or use existing ones if possible.

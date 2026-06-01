%  Make differene images for showing the effect of
%  focusing and apodization
%
%  This script assumes that the field_init procedure has been called
%
%  Example by Joergen Arendt Jensen, March 25, 1997.
%  constant F# added by Peter Munk, April 8, 1997
%  Version 1.2, August 13, 2007, JAJ: Printout changed
%
%  Generate the transducer apertures for send and receive

f0=3e6;                  %  Transducer center frequency [Hz]
fs=100e6;                %  Sampling frequency [Hz]
c=1540;                  %  Speed of sound [m/s]
lambda=c/f0;             %  Wavelength [m]
width=lambda;            %  Width of element
element_height=5/1000;   %  Height of element [m]
kerf=0.1/1000;           %  Kerf [m]
focus=[0 0 70]/1000;     %  Fixed focal point [m]
N_elements=128;          %  Number of physical elements
N_active=64;             %  Number of active elements
xmit_N_active=128;       %  Number of active transmit elements for constant F#
rec_N_active=128;	 %  Number of active receive elements for constant F#

%  Set the sampling frequency 

set_sampling(fs);

%  Generate aperture for emission

emit_aperture = xdc_linear_array (N_elements, width, element_height, kerf, 1, 1,focus);

%  Set the impulse response and excitation of the emit aperture

impulse_response=sin(2*pi*f0*(0:1/fs:2/f0));
impulse_response=impulse_response.*hanning(max(size(impulse_response)))';
xdc_impulse (emit_aperture, impulse_response);

excitation=sin(2*pi*f0*(0:1/fs:2/f0));
xdc_excitation (emit_aperture, excitation);

%  Generate aperture for reception

receive_aperture = xdc_linear_array (N_elements, width, element_height, kerf, 1, 1,focus);

%  Set the impulse response for the receive aperture

xdc_impulse (receive_aperture, impulse_response);

%   Load the computer phantom

[phantom_positions, phantom_amplitudes] = pts_pha;

%   Do linear array imaging

no_lines=20;                         %  Number of lines in image
image_width=20/1000;                 %  Size of image sector
d_x=image_width/no_lines;            %  Increment for image

%  Make the different simulations

%  Single focus for emission and reception, no apodization

figure(1)
disp('Making images without apodization (figure 1)')
subplot(161)
disp('Single transmit and receive focus')
sesr
mk_img
title('A')
axis on
ylabel('Axial distance [mm]')
text(40,135,'Lateral distance [mm]')

%  Introduce a number of receive focus zones

subplot(162)
disp('Single transmit and multiple receive foci')
semr
mk_img
title('B')

%  Multiple focusing in both transmit and receive

subplot(163)
disp('Multiple transmit and multiple receive foci')
memr
mk_img
title('C')
xlabel('Lateral distance [mm]')

%  Use 128 elements instead

subplot(164)
disp('Multiple transmit and multiple receive foci (128 elements)')
memr128
mk_img
title('D')

%  Use 128 elements with continous receive focus

subplot(165)
disp('Multiple transmit and continuous receive foci (128 elements)')
mecr128
mk_img
title('E')

%  Use 128 elements instead with no apodization
%  constant F# rec=2 and xmit=4

subplot(166)
disp('Constant F# number focusing')
fnumna
mk_img
title('F')


%  Perform the simulations again using apodization

%  Single focus for emission and reception with apodization

figure(2)
disp('Making images with apodization (figure 2)')
subplot(161)
disp('Single transmit and receive focus')
sesra
mk_img
axis on
title('A')
ylabel('Axial distance [mm]')
text(40,135,'Lateral distance [mm]')

%  Introduce a number of receive focus zones with apodization

subplot(162)
disp('Single transmit and multiple receive foci')
semra
mk_img
title('B')

%  Multiple focusing in both transmit and receive with apodization

subplot(163)
disp('Multiple transmit and multiple receive foci')
memra
mk_img
title('C')
xlabel('Lateral distance [mm]')

%  Use 128 elements instead with apodization

subplot(164)
disp('Multiple transmit and multiple receive foci (128 elements)')
memr128a
mk_img
title('D')

%  Use 128 elements with 2 mm between receive focuses

subplot(165)
disp('Multiple transmit and continuous receive foci (128 elements)')
mecr128a
mk_img
title('E')

%  Use 128 elements instead with apodization
%  constant F# rec=2 and xmit=4

subplot(166)
disp('Constant F# number focusing')
fnumwa
mk_img
title('F')


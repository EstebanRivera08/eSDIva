%  Define a concave piston transducer,
%  set its impulse response and excitation
%  and calculate its point spread function
%
%  Note that the field_init routine must be
%  calculated before the routine is called
%
%  Version 1.0, June 29, 2001 by JAJ

%  Set initial parameters

R=8/1000;             %  Radius of transducer [m]
Rfocus=80/1000;       %  Geometric focus point [m]
ele_size=1/1000;      %  Size of mathematical elements [m]
f0=3e6;               %  Transducer center frequency [Hz]
fs=100e6;             %  Sampling frequency [Hz]

%  Define the transducer

Th = xdc_concave (R, Rfocus, ele_size);

%  Set the impulse response and excitation of the emit aperture

impulse_response=sin(2*pi*f0*(0:1/fs:2/f0));
impulse_response=impulse_response.*hanning(max(size(impulse_response)))';
xdc_impulse (Th, impulse_response);

excitation=sin(2*pi*f0*(0:1/fs:2/f0));
xdc_excitation (Th, excitation);

%  Calculate the pulse echo field and display it

xpoints=(-10:0.2:10);
[RF_data, start_time] = calc_hhp (Th, Th, [xpoints; zeros(1,101); 30*ones(1,101)]'/1000);

%  Make a display of the envelope

figure(1)
env=abs(hilbert(RF_data(1:5:600,:)));
env=20*log10(env/max(max(env)));
[N,M]=size(env);
env=(env+60).*(env>-60) - 60;
mesh(xpoints, ((0:N-1)/fs + start_time)*1e6, env)
ylabel('Time [\mus]')
xlabel('Lateral distance [mm]')
title('Pulse-echo field from 8 mm concave transducer at 30 mm')
axis([-10 10 38.41 39.6 -60 0])
view([-14 80])

%  Create a smaller image for the thumbnail

figure(2)
imagesc(env)
colormap(hot)
axis off

%  Free the aperture after use

xdc_free (Th)

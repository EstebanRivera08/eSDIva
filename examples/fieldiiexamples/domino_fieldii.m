% domino_fieldii.m
% ---------------------------------------------------------------------------
% Build the SonDI "Domino" probe in MATLAB Field II and save its
% mathematical-element geometry, so example 17 can import it and check the
% imported aperture matches sondi.transducers.Domino() patch-for-patch.
%
% Domino = 128-element elevation-focused linear array, 12.5 MHz:
%   pitch 0.11 mm, element 0.108 x 1.5 mm, kerf 0.002 mm,
%   elevation (cylindrical-lens) focus 8 mm, elevation subdivision 10.
%
% Requires the Field II toolbox (field_init / xdc_* on the MATLAB path).
% Run once; it writes domino_fieldii.mat next to this script.
% ---------------------------------------------------------------------------
addpath("C:\Users\INSERM\Documents\Esteban\Simulation\Field II\Field_II_ver_3_30_windows")

field_init(0);

f0     = 12.5e6;     % centre frequency [Hz]
c      = 1540;       % speed of sound [m/s]
fs     = 200e6;      % sampling frequency [Hz]
set_field('c', c);
set_field('fs', fs);

n_elements = 128;
width   = 0.108e-3;          % element width  (x, lateral) [m]
height  = 1.5e-3;            % element height (y, elevation) [m]
pitch   = 0.11e-3;           % [m]
kerf    = pitch - width;     % [m]
Rfocus  = 8e-3;              % elevation lens focal length [m]
no_sub_x = 1;                % lateral  subdivisions per element (SonDI no_sub_x)
no_sub_y = 10;               % elevation subdivisions per element (SonDI no_sub_y)

% Elevation-focused linear array: flat element face at z = 0, cylindrical
% lens curving along y — this is the Field II analogue of a SonDI
% LinearArrayTransducer with elevation_focus_mm set.
Th = xdc_focused_array(n_elements, width, height, kerf, Rfocus, ...
                       no_sub_x, no_sub_y, [0 0 30]/1000);

rect = xdc_get(Th, 'rect');  % 26 x M matrix, one column per mathematical element

here = fileparts(mfilename('fullpath'));
outfile = fullfile(here, 'domino_fieldii.mat');
save(outfile, 'rect', 'f0', 'c', 'fs', 'Rfocus', '-v7');

xdc_free(Th);
field_end;
fprintf('Saved %s (%d mathematical elements)\n', outfile, size(rect, 2));

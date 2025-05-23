%% recreate_setup.m
clear; clc

% Add k-Wave to your path
addpath(genpath('C:\Users\INSERM\Documents\Esteban\Simulation\k-Wave'))

%%-------------------------------------------------
%% 1) Simulation domain & grid
%%-------------------------------------------------

domain_width_mm   = 42;
domain_height_mm  = 26;
dx_mm = 0.07; dy_mm = 0.07;

Nx = round(domain_width_mm/dx_mm) + 1;  % # cols
Ny = round(domain_height_mm/dy_mm) + 1; % # rows

% Coordinates for plotting [mm]
x_mm = linspace(-domain_width_mm/2 , domain_width_mm/2,  Nx);
y_mm = linspace(0 , domain_height_mm,  Ny);

% Create k-Wave grid (inputs in metres)
kgrid = makeGrid(Ny, dy_mm*1e-3, Nx, dx_mm*1e-3);

%%-------------------------------------------------
%% 2) Define medium grid parameters
%%-------------------------------------------------

% Object properties
c_air   = 343;   rho_air   = 100 ; %1.293;
c_water = 1540;  rho_water = 997  ;
c_plast = 2700;  rho_plast = 1200 ;
c_TPX   = 2700;  rho_TPX   = 1200 ;
c_abs   = 1500;  rho_abs   = 1010 ;

% Petri dish parameters (all in mm)
dish_vert_offset = 0 ;
dish_height   = 25;
dish_diameter = 40;
wall_thick    = 0.5;

% Compute petri dish limits
dish_int_rad = dish_diameter/2;
dish_top_y   = dish_vert_offset + dish_height;

% Absorber
Absorber_offset_mm   = wall_thick ;
Absorber_center_mm   = [0, Absorber_offset_mm] ; 
Absorber_height      = 7 ; % mm
Absorber_int_diam    = 15 ; % mm
Absorber_ext_diam    = 25 ; % mm
Microscopic_win_diam = 5 ; % mm

% Cell support
water_height = 20;
TPX_thick = 0.05 ; % mm
TPX_thick = max(TPX_thick, dy_mm) ; % mm

% All the transducer will be define from the sample point
tx_elem_size    = 6    ;  % [mm]
tx_incli_angle  = 45   ;  % degrees
tx_focus_dist   = 12.7 ;  %[mm]

%%-------------------------------------------------
%% 3) Define medium grid 
%%-------------------------------------------------

% Build geometry masks with meshgrid
[XX, YY] = meshgrid(x_mm, y_mm);  % both Ny×Nx


% 2a) Fill the space with air

medium.sound_speed =  c_air   * ones(Ny, Nx);
medium.density     =  rho_air * ones(Ny, Nx);

% 2b) Water inside dish
water_mask = ...
     (abs(XX) <= dish_int_rad) & ... % Use symmetry
     (YY >= dish_vert_offset) & (YY <= dish_vert_offset + water_height);


water_height_mask = ...
     (abs(XX) <= dish_int_rad) & ... % Use symmetry
     (YY >= water_height) & (YY <= TPX_thick + water_height);

medium.sound_speed(water_mask) = c_water;
medium.density    (water_mask) = rho_water;

% 2c) Plastic walls (left, right, bottom)

lateral_wall  = ...
     (abs(XX) >= dish_int_rad) & (abs(XX) <= dish_int_rad + wall_thick) & ...
     (YY >= dish_vert_offset) & (YY <= dish_top_y);

bottom_wall = ...
     (abs(XX) <= dish_int_rad + wall_thick)& ...
     (YY >= dish_vert_offset) & (YY <= dish_vert_offset + wall_thick);

petri_dish_mask = lateral_wall | bottom_wall;

medium.sound_speed(petri_dish_mask) = c_plast;
medium.density    (petri_dish_mask) = rho_plast;

%%-------------------------------------------------
% 2) Cell-suport
%%------------------------------------------------- 

film_bottom = Absorber_center_mm(2)+Absorber_height ;

main_region_mask = ...
     (abs(XX) >= Absorber_int_diam/2 & (abs(XX) <= Absorber_ext_diam/2))  & ...
     (YY >= Absorber_center_mm(2)) & (YY <= film_bottom);

triangle_region_mask = ...
     (abs(XX) >= Microscopic_win_diam/2 & (abs(XX) <= Absorber_int_diam/2))  & ...
     ((YY >= Absorber_center_mm(2)) & (YY <= (abs(XX)-Microscopic_win_diam/2) + Absorber_center_mm(2)));

absorber_mask = main_region_mask | triangle_region_mask ;
 
medium.density(absorber_mask) = rho_abs ;
medium.sound_speed(absorber_mask) = c_abs ;

TPX_film_mask = ...
     (abs(XX) <= Absorber_ext_diam/2)   & ...
     (YY >= film_bottom) & (YY <= film_bottom + TPX_thick);

medium.density(TPX_film_mask) = rho_TPX ;
medium.sound_speed(TPX_film_mask) = c_TPX ;

setup_mask = water_height_mask | TPX_film_mask |...
             absorber_mask     | petri_dish_mask ;
         
%%-------------------------------------------------
%% 3) Define the transducer
%%-------------------------------------------------

% 3a) Define Tranducer

% 1) Define your geometry in mm:
sample_position = [0, film_bottom + TPX_thick];   % where you want the focus
theta           = deg2rad(tx_incli_angle);       % tilt from vertical

% 2) Pick your focus point in mm:
focus_pt_mm     = sample_position;               
%    (we assume you want the beam to focus at that sample point)

% 3) Compute the center of curvature in mm:
curv_center_mm  = focus_pt_mm + tx_focus_dist * [ cos(theta), sin(theta) ];

% 3b) translate mm → grid points
% Translate mm values to grid points
%    col = x→columns; row = y→rows
arc_col = round( (curv_center_mm(1) + domain_width_mm/2) / dx_mm ) + 1;
arc_row = round(  curv_center_mm(2)                      / dy_mm ) + 1;
arc_pos = [arc_row, arc_col];

focus_col = round( (focus_pt_mm(1) + domain_width_mm/2) / dx_mm ) + 1;
focus_row = round(  focus_pt_mm(2)                      / dy_mm ) + 1;
focus_pos = [focus_row, focus_col];

radius_pts   = round(tx_focus_dist / dx_mm);
diameter_pts = round(tx_elem_size   / dx_mm);
if mod(diameter_pts,2)==0
    diameter_pts = diameter_pts + 1;
end

% 5) make the arc mask
source.p_mask = makeArc(...
    [Ny, Nx], ...       % [rows, cols]
    arc_pos, ...        % [row, col] of the center of curvature
    radius_pts, ...     % radius of that curvature
    diameter_pts, ...   % aperture width (odd!)
    focus_pos );        % [row, col] of the focus


%% Define attenuations (absorbtion)

% Build local coords for the entire grid (in mm)
% Shift so that local origin is at the curvature center
Xg = XX - curv_center_mm(1);
Yg = YY - curv_center_mm(2);

% Rotate global → local (so local +Y is “into” the aperture face)
Xloc =  cos(theta-pi/2)*Xg + sin(theta-pi/2)*Yg;
Yloc = -sin(theta-pi/2)*Xg + cos(theta-pi/2)*Yg;

% Define backing layer in local frame
backing_thickness = 1.5;  % mm behind the face
halfW = tx_elem_size/2; % mm

backing_loc = ...
   (Yloc >= 0) & (Yloc <= backing_thickness) & ...  % “behind” the face
   (Xloc >= -halfW) & (Xloc <= +halfW);             % within aperture width
back_tx_water = backing_loc & water_mask ;

% Define attenuations (absorbtion)

% 1) Choose an attenuation law
%   α (dB per MHz^y per cm) — to get “huge” loss, go high (e.g. 50–100)
alpha_coeff_value = 100;  
alpha_power_value = 1.5;      % typical biological tissue: ~1.1–1.5

% 2) Allocate to entire grid (Ny×Nx)
medium.alpha_coeff = zeros(Ny, Nx);
medium.alpha_power = alpha_power_value ;
medium.alpha_mode  = 'no_dispersion';    % enable power‐law absorption

% 3) Paint your absorber region
medium.alpha_coeff(absorber_mask) = alpha_coeff_value;

% 3) Paint your transducer back zone
medium.alpha_coeff(back_tx_water) = alpha_coeff_value;


%% 3c) Define Tranducer Pulse

% time array
sampling_frequency    = 100e6 ; % [Hz]
dt = 1/sampling_frequency ; % [s]
t_end = 20e-6;                 % [s]
% create the time array
kgrid.setTime(round(t_end / dt) + 1, dt);

%  ------------------ Pulse emission params ---------------------
pulse.frequency_Mhz = 10 ; %MHz
pulse.PRF_kHz = 1 ; % kHz
pulse.Stim_Dur_ms = 50 ;  %ms
pulse.Duty_Cycle = 0.15 ;

pulse.burst_period_ms = (1 / pulse.PRF_kHz) ;
pulse.burst_duration_ms = pulse.burst_period_ms*pulse.Duty_Cycle ;
pulse.Nb_burst = pulse.Stim_Dur_ms/pulse.burst_period_ms ;

N_t = length(kgrid.t_array) ;
active_window = zeros(1,N_t) ;
active_window(1:round(N_t*pulse.Duty_Cycle)) = 1 ;

source_mag = 10;         % [Pa]
emitted_burst = source_mag*sin(2*pi*pulse.frequency_Mhz*1e6 * kgrid.t_array);
emitted_burst = emitted_burst.*active_window ;

% define a time varying sinusoidal source
source.p = emitted_burst;

% filter the source to remove any high frequencies not supported by the grid
source.p = filterTimeSeries(kgrid, medium, source.p);

%% 2d) Plot medium
figure;
tiledlayout(2,3, 'TileSpacing','compact','Padding','compact')

k = 1 ;
nexttile(k)
imagesc(x_mm, y_mm, medium.sound_speed);
hold on
plot(curv_center_mm(1), curv_center_mm(2), 'ro', 'DisplayName','Element Center')
plot(focus_pt_mm(1), focus_pt_mm(2), 'bo', 'DisplayName','Focus')
plot([focus_pt_mm(1) curv_center_mm(1)], [focus_pt_mm(2) curv_center_mm(2)],...
    '-k','DisplayName','Focal Distance')
set(gca, 'YDir', 'normal')
axis image tight;
colorbar;
legend show
xlabel('x (mm)');
ylabel('y (mm)');
title('Sound speed (m/s): ');

k = k + 1 ;
nexttile(k)
imagesc(x_mm, y_mm, medium.alpha_coeff);
hold on
plot(curv_center_mm(1), curv_center_mm(2), 'ro', 'DisplayName','Element Center')
plot(focus_pt_mm(1), focus_pt_mm(2), 'bo', 'DisplayName','Focus')
plot([focus_pt_mm(1) curv_center_mm(1)], [focus_pt_mm(2) curv_center_mm(2)],...
    '-k','DisplayName','Focal Distance')
set(gca, 'YDir', 'normal')
axis image tight;
colorbar;
legend show
xlabel('x (mm)');
ylabel('y (mm)');
title('Alpha coefficient (dB/(MHz^y·cm)): ');

k = k + 1 ;
nexttile(k)
imagesc(x_mm, y_mm, source.p_mask*2+setup_mask)
hold on
plot(curv_center_mm(1), curv_center_mm(2), 'ro', 'DisplayName','Element Center')
plot(focus_pt_mm(1), focus_pt_mm(2), 'bo', 'DisplayName','Focus')
plot([focus_pt_mm(1) curv_center_mm(1)], [focus_pt_mm(2) curv_center_mm(2)],'-k')
set(gca, 'YDir', 'normal')
axis image tight;
colorbar;
xlabel('x (mm)');
ylabel('y (mm)');
title('Setup+Source mask');


k = k + 1 ;
nexttile(k, [1,3])  
plot(kgrid.t_array*1e6, emitted_burst, 'k')
xlabel('Time, us')
ylabel('Pressure, Pa')
grid minor

%%

% create a sensor mask covering the entire computational domain using the
% opposing corners of a rectangle
sensor.mask = [1, 1, Ny, Nx].';

% set the record mode to capture the final wave-field and the statistics at
% each sensor point
sensor.record = {'p_final', 'p_max', 'p_rms'};

% create a display mask to display the transducer
display_mask = source.p_mask | setup_mask;

% assign the input options
input_args = {'DisplayMask', display_mask, 'PMLInside', false, 'PlotPML', false,...
    'RecordMovie', true, 'MovieName', 'focused'};

sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, ...
    input_args{:}) ;

%% 2d) Plot medium
figure('color','white');
tiledlayout(3,3, 'TileSpacing','compact','Padding','compact')

k = 1 ;
spancol = 3 ;
nexttile(k, [1,spancol])  
plot(kgrid.t_array*1e6, emitted_burst, 'k')
xlabel('Time, us')
ylabel('Pressure, Pa')
grid minor
k = k+spancol ;

nexttile(k)
imagesc(x_mm, y_mm, medium.sound_speed);
hold on
plot(curv_center_mm(1), curv_center_mm(2), 'ro', 'DisplayName','Element Center')
plot(focus_pt_mm(1), focus_pt_mm(2), 'bo', 'DisplayName','Focus')
plot([focus_pt_mm(1) curv_center_mm(1)], [focus_pt_mm(2) curv_center_mm(2)],...
    '-k','DisplayName','Focal Distance')
set(gca, 'YDir', 'normal')
axis image tight;
colorbar;
legend show
xlabel('x (mm)');
ylabel('y (mm)');
title('Sound speed (m/s): ');

k = k + 1 ;
nexttile(k)
imagesc(x_mm, y_mm, medium.alpha_coeff);
hold on
plot(curv_center_mm(1), curv_center_mm(2), 'ro', 'DisplayName','Element Center')
plot(focus_pt_mm(1), focus_pt_mm(2), 'bo', 'DisplayName','Focus')
plot([focus_pt_mm(1) curv_center_mm(1)], [focus_pt_mm(2) curv_center_mm(2)],...
    '-k','DisplayName','Focal Distance')
set(gca, 'YDir', 'normal')
axis image tight;
colorbar;
legend show
xlabel('x (mm)');
ylabel('y (mm)');
title('Alpha coefficient (dB/(MHz^y·cm)): ');

k = k + 1 ;
nexttile(k)
imagesc(x_mm, y_mm, source.p_mask*2+setup_mask)
hold on
plot(curv_center_mm(1), curv_center_mm(2), 'ro', 'DisplayName','Element Center')
plot(focus_pt_mm(1), focus_pt_mm(2), 'bo', 'DisplayName','Focus')
plot([focus_pt_mm(1) curv_center_mm(1)], [focus_pt_mm(2) curv_center_mm(2)],'-k')
set(gca, 'YDir', 'normal')
axis image tight;
colorbar;
xlabel('x (mm)');
ylabel('y (mm)');
title('Setup+Source mask');

k = k + 1 ;
nexttile(k)
imagesc(x_mm, y_mm, sensor_data.p_final);
set(gca, 'YDir', 'normal')
axis image tight;
colorbar;
xlabel('x (mm)');
ylabel('y (mm)');
title('Final Wave Field');
k = k+1 ;

nexttile(k)
imagesc(x_mm, y_mm, sensor_data.p_max);
set(gca, 'YDir', 'normal')
axis image tight;
colorbar;
xlabel('x (mm)');
ylabel('y (mm)');
title('Maximum pressure');
k = k+1 ;

nexttile(k)
imagesc(x_mm, y_mm, sensor_data.p_rms);
set(gca, 'YDir', 'normal')
axis image tight;
colorbar;
xlabel('x (mm)');
ylabel('y (mm)');
title('RMS pressure');

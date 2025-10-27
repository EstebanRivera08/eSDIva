
clear 

addpath("C:\Users\INSERM\Documents\Esteban\Simulation\Field II\Field_II_ver_3_30_windows")
addpath("C:\Users\INSERM\Documents\Esteban\Simulation\neurosonogene_acousticsimulations\WorkInProgress\Functions")

field_init

%% DEFINE PULSE AND TRANSDUCER PROPERTIES
disp('Defining transducer...')

repetitions = 5 ;
h_calc_time = zeros(1,repetitions) ;    

% ------------------ Saving and plotting options -----------------------
version= '_v1' ;
data_folder = '\' ;
figure_folder = '\' ;

%  -------------- emission ----------------
frequency = 12.5 ; %MHz
data.c = 1540; % Speed of sound [m/s]

% ------- focus and simulation window (input parameters) -------
x_extent = [-5, 5]; % mm
y_extent = [-5, 5]; % mm
z_extent = [1, 16]; % mm
dxyz = 0.5 ; %mm
dx = dxyz ; % mm
dy = dxyz ; % mm
dz = dxyz ; % mm
x_focus = 0 ; % mm
y_focus = 0 ; % mm
z_focus = 8 ; % mm

%  ------------------- transducer characteristics --------------------
data.f0 = frequency*1e6 ; % [Hz]
data.lambda = data.c/data.f0; % Wave length [m]
data.focus_mm = [x_focus y_focus z_focus]; % Fixed focal point [mm]

data.tx_N_elements = 128 ; % Number of elements in the transducer
data.tx_element_height_mm = 1.5 ;% Height of element [mm] (Elevation aperture ?)
data.tx_width_mm = 0.108 ; % Width of element 
data.tx_pitch_mm = 0.11 ; % Kerf [mm]
data.tx_kerf_mm = data.tx_pitch_mm - data.tx_width_mm ; % Kerf [mm]
data.tx_elevationFocus_mm = 8 ; % Elevation focus [m]
data.tx_frequency = data.f0 ; % Transducer center frequency [Hz]
data.no_sub_x = 1  ;% Number of sub-divisions in x-direction of mathematical elements
data.no_sub_y = 10 ;% Number of sub-divisions in y-direction of mathematical elements

% Set the sampling frequency
data.sampling_frequency = 100e6 ;  % Sampling frequency of Veramachine [Hz]
set_sampling(data.sampling_frequency);


% -------- The size of sub-division is correct? -----------
min_distance = z_extent(1)  ; % Min distance from a transducer to the point [m]
max_subdiv_size = max(data.tx_element_height_mm/data.no_sub_y,...
                    data.tx_width_mm/data.no_sub_x) ; %[m]
dist_min_lim = max_subdiv_size^2/(4*data.lambda*1e3) ;
disp(['l_min ~ ',num2str(dist_min_lim)])
disp(['l: ', num2str(min_distance)])

% -------- create mathematical transducer --------- 
transducer = xdc_focused_array( ...
data.tx_N_elements, ...       % Number of elements
data.tx_width_mm/1000, ...            % Width of elements [m]
data.tx_element_height_mm/1000, ...   % Height of elements [m]
data.tx_kerf_mm/1000, ...             % Kerf [m]
data.tx_elevationFocus_mm/1000, ...   % Elevation focus [m]
data.no_sub_x, ...            % Subdivisions in x-direction
data.no_sub_y, ...            % Subdivisions in y-direction
data.focus_mm/1000 );    % Focus


%% DEFINE THE MEASUREMENT GRID OR POSITIONS

start_pressure = tic ;
disp('Defining grid and positions...')

% computationnaly efficient method is to simulate a half Y plane, and then
% mirror its results because it should be symmetrical.

% Calculate the number of points
nx = round(diff(x_extent) / dx);
ny = round(diff(y_extent) / dy);
nz = round(diff(z_extent) / dz);

% Ensure the number of points is greater than 1
if nx < 2
    nx = 2;
end
if ny < 2
    ny = 2;
end
if nz < 2
    nz = 2;
end

if rem(nx, 2) == 0 
    nx = nx + 1 ;
end

if rem(ny, 2) == 0 
    ny = ny + 1 ;
end

if rem(nz, 2) == 0 
    nz = nz + 1 ;
end

% Debugging statements
disp(['Focusing at: ', num2str(data.focus_mm)]);
disp(['x extent: ', num2str(x_extent)]);
disp(['y extent: ', num2str(y_extent)]);
disp(['z extent: ', num2str(z_extent)]);
disp(['nx,ny,nz: ', num2str([nx,ny,nz])]); 
disp(['Number of total points: ', num2str(nz*ny*nx)]);
disp('')


% whole X Z plane measurement

x_vec = linspace(x_extent(1), x_extent(2), nx);
y_vec = linspace(y_extent(1), y_extent(2), ny);
z_vec = linspace(z_extent(1), z_extent(2), nz);

[x_grid, y_grid, z_grid] = meshgrid(x_vec, y_vec, z_vec);
data.points = [x_grid(:) y_grid(:) z_grid(:)];
disp(size(data.points))

%% CALL Field II ROUTINES FOR SPATIAL IMPULSE RESPONSE SIMULATION

disp('Be sure to do no overwrite any data. Starting in 10 seconds...')
% pause(10)

%  (for monochromatic field simulations)disp('Calling Field II routine for spatial impulse response simulation...')

%Purpose: Procedure for calculating the spatial impulse response for an aperture.
%Calling: 

for rep = 1:repetitions
time_start_h = tic ;
[h, start_time] = calc_h(transducer, data.points);

calculation_time(rep) = toc(time_start_h) ;
end

disp(['Elapsed time: ', num2str(calculation_time)])

%% Reshape measurements

data.calculation_time = calculation_time ;

disp('Post-processing of the results...')

spatial_impulse_response_field = reshape(h,[], size(x_grid,1), size(x_grid,2), size(x_grid,3));  % first two dimensions are space, last is time

% get the frequency response
% this is an impulse response, so will contain all the frequencies.
% to obtain the spatial distribution of one frequency, we do a fft and pick
% the wanted frequency.

spatial_impulse_response_field_FT = fft(spatial_impulse_response_field, [], 1);
freq_vect = linspace(0, data.sampling_frequency, size(spatial_impulse_response_field_FT, 1));

%% Pick the selected frequency
freq_searched = data.f0 ;
% freq_searched = 10*1e6 ;
idx_freq = find(freq_vect>=freq_searched,1, 'first');

% Amplitude for a given frequency
amp_response_tx_freq = squeeze(abs(spatial_impulse_response_field_FT(idx_freq,:,:,:)));

scale_factor = 1/max(amp_response_tx_freq, [], 'all');
Pressure_field_monofreq = amp_response_tx_freq ;



%% setup structure for further thermal simulations
% I do not save the max_pressure_field_3D 3D field as it can grow quite
% big, instead I save the scaled_half_field, and when loading the .mat you
% can retrieve the 3D field by executing:
% eval(field_source.code_to_obtain_3Dfield); % note: revolve2D is a kWave toolbox function (be sure to have it on your path)

field_source.frequency            = freq_vect(idx_freq);
field_source.simu_p_scale_factor  = scale_factor;
field_source.simulated_half_field = amp_response_tx_freq;
field_source.scaled_half_field    = Pressure_field_monofreq;
field_source.code_to_obtain_3Dfield= 'field_source.max_pressure_field_3D = permute(field_mirror_yaxis(field_source.scaled_half_field),[2,1,3]);';
field_source.focus_mm = data.focus_mm ;
field_source.window_xy = [dx, dy] ;
field_source.window_z = [dz_up, dz_down] ;
if ~exist('dxyz', 'var')
    field_source.delta_xy = dxy ;
    field_source.delta_z = dz ;
else
    field_source.delta_xyz = dxyz ;
end
field_source.x_vec                = x_vec;
field_source.z_vec                = z_vec;
field_source.y_vec_half           = y_vec;
field_source.y_vec                = y_vec_3D ;

if option.save_data
    disp(['saving   ' data_folder filename '   ...'])
    save([data_folder filename], 'medium', 'data', 'field_source')
end

%% Now we display the results of the simulation
% (With the mirroring part)

disp('Displaying of the results...')


figure('color', 'white','Name',apodization_type, 'WindowState', 'maximize') ;
options_tx.edges_config = 'k' ;
show_xdc_optimized(transducer,options_tx) ;
title('a) Transducer View')
view(-60,30)
xlim([-8,8])
colormap(cool(128))

factor = 1 ;


%% Plot plane

disp(size(Pressure_field_3D))

disp(size(y_grid_3D))

disp(size(x_grid_3D))


figure('color', 'white', 'WindowState', 'maximize') ;
imagesc(x_vec, z_vec, Pressure_field_3D')
colormap jet



%% Plot the slices

figure('color', 'white','Name',apodization_type, 'WindowState', 'maximize') ;
fig1 = tiledlayout(1,3, 'Padding', 'compact', 'TileSpacing', 'compact') ;

% Plot the slices
surf1 = nexttile ;
slice3D = slice(x_grid_3D,y_grid_3D,z_grid_3D, Pressure_field_3D/factor,...
    [], y_focus, []) ;

colormap(surf1,'jet')
set(slice3D, 'EdgeAlpha', 0); % Set alpha to 0.5 (50% transparency)
% cbar = colorbar ;
% Invert the z-axis
set(gca, 'ZDir', 'reverse');
% title(cbar, 'Pressure, MPa ');
axis('image')
xlabel('x, mm')
ylabel('y, mm')
zlabel('z, mm')
grid minor
view(0,0)
title('XZ plane')

surf2 = nexttile ;
slice3D = slice(x_grid_3D,y_grid_3D,z_grid_3D, Pressure_field_3D/factor,...
    [], [], z_focus) ;

colormap(surf2,'jet')
set(slice3D, 'EdgeAlpha', 0); % Set alpha to 0.5 (50% transparency)
% cbar = colorbar ;
% Invert the z-axis
set(gca, 'ZDir', 'reverse');
% title(cbar, 'Pressure, MPa ');
axis('image')
xlabel('x, mm')
ylabel('y, mm')
zlabel('z, mm')
grid minor
view(0,90)
title('b) XY plane')

surf3 = nexttile ;
% Plot the slices
slice3D = slice(x_grid_3D,y_grid_3D,z_grid_3D, Pressure_field_3D/factor,...
    x_focus, [], []) ;

colormap(surf3,'jet')
set(slice3D, 'EdgeAlpha', 0); % Set alpha to 0.5 (50% transparency)
% cbar = colorbar ;
% Invert the z-axis
set(gca, 'ZDir', 'reverse');
% title(cbar, 'Pressure, MPa ');
axis('image')
xlabel('x, mm')
ylabel('y, mm')
zlabel('z, mm')
grid minor
view(90,0)
title('YZ plane')




sgtitle(sprintf('Focalization in %s at (%.1f, %.1f, %.1f) mm, with f_c = %.1f MHz, and F/D = %.1f',chosen_medium, x_focus, y_focus, z_focus,frequency,data.ratio_F_over_D))

if option.save_figure
   disp(['saving   ' figure_folder filename '   ...'])
   savefig([figure_folder filename '.fig'])
end
% end
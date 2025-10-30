
clear 

addpath("C:\Users\INSERM\Documents\Esteban\Simulation\Field II\Field_II_ver_3_30_windows")
addpath("C:\Users\INSERM\Documents\Esteban\Simulation\neurosonogene_acousticsimulations\WorkInProgress\Functions")


%% DEFINE PULSE AND TRANSDUCER PROPERTIES
field_init(0)

disp('Defining transducer...')

% ------------------ Saving and plotting options -----------------------
folder = 'data\Linear\' ;

%  -------------- emission ----------------
frequency = 12.5 ; %MHz
f_sampling = 200 ; %MHz
data.c = 1540; % Speed of sound [m/s]

% ------- focus and simulation window (input parameters) -------
x_extent = [-2, 2]; % mm
y_extent = [-2, 2]; % mm
z_extent = [3, 13]; % mm
dxyz = 0.1 ; %mm
dx = dxyz ; % mm
dy = dxyz*diff(y_extent)/diff(x_extent) ; % mm
dz = dxyz*diff(z_extent)/diff(x_extent) ; % mm
x_focus = 0 ; % mm
y_focus = 0 ; % mm
z_focus = 8 ; % mm

% Set the sampling frequency
data.sampling_frequency = f_sampling*1e6 ; 
set_sampling(data.sampling_frequency);


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
data.no_sub_x = 2  ;% Number of sub-divisions in x-direction of mathematical elements
data.no_sub_y = 20 ;% Number of sub-divisions in y-direction of mathematical elements


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


% whole X Z plane measurement
x_vec = linspace(x_extent(1), x_extent(2), nx);
y_vec = linspace(y_extent(1), y_extent(2), ny);
z_vec = linspace(z_extent(1), z_extent(2), nz);

[x_grid, y_grid, z_grid] = meshgrid(x_vec, y_vec, z_vec);
data.points = [x_grid(:) y_grid(:) z_grid(:)]*1e-3;

data.M = data.tx_N_elements*data.no_sub_x*data.no_sub_y ;
data.P = nx*ny*nz ;
disp(['Number of total points: ', num2str(data.P)]);
disp(['Number of transducers patches: ', num2str(data.M)])

%% CALL Field II ROUTINES FOR SPATIAL IMPULSE RESPONSE SIMULATION

disp('Be sure to do no overwrite any data. Starting in 10 seconds...')
pause(10)

%  (for monochromatic field simulations)disp('Calling Field II routine for spatial impulse response simulation...')

%Purpose: Procedure for calculating the spatial impulse response for an aperture.
%Calling: 

repetitions = 5 ;
h_calc_time = zeros(1,repetitions) ;    

for rep = 1:repetitions
time_start_h = tic ;
[h, start_time] = calc_h(transducer, data.points);

h_calc_time(rep) = toc(time_start_h) ;
end

%% Print some data
data.T = size(h,1) ;
disp(['Number of time grid points: ', num2str(data.T)]);
disp(['Elapsed time: ', num2str(h_calc_time)])
data.h_calc_time = h_calc_time ;
data.mean_time = mean(h_calc_time) ;
data.std_time = std(h_calc_time) ;
disp(['mean time: ', num2str(data.mean_time)])
disp(['std time: ', num2str(data.std_time)])

%% Reshape measurements
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
amp_response_tx_freq = amp_response_tx_freq*scale_factor ;
Pressure_field_monofreq = permute(amp_response_tx_freq,[2,1,3]);

pr_calc_time = toc(start_pressure) ;
disp(['Pre calculation time: ', num2str(pr_calc_time)])
data.x = x_vec ;
data.y = y_vec ;
data.z = z_vec ;
data.pr = Pressure_field_monofreq;

%% Save results

filename = ['Linear', ...
             '_nsubx',num2str(data.no_sub_x),'_nsuby',num2str(data.no_sub_y),...
             '_fs', num2str(f_sampling), '_nxyz', num2str(nx),...
             '_P', num2str(data.P),...
             '_M', num2str(data.M),...
             '_T', num2str(data.T)  ] ;
disp(['File: ', filename])

save([folder, filename], 'data')

%% Plot plane

disp(size(Pressure_field_monofreq))


figure('color', 'white', 'WindowState', 'maximize') ;

fig1 = tiledlayout(1,3, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile
imagesc(x_vec, z_vec, squeeze(Pressure_field_monofreq(:,round(ny/2),:))')
colormap jet
axis image

xlabel('x, mm')
zlabel('z, mm')
title('XZ plane')

nexttile
imagesc(x_vec, y_vec, squeeze(Pressure_field_monofreq(:,:,round(nz/2)))')
colormap jet
axis image
xlabel('x, mm')
ylabel('y, mm')
title('XY plane')

nexttile
imagesc(y_vec, z_vec, squeeze(Pressure_field_monofreq(round(nx/2),:,:))')
colormap jet
axis image
ylabel('y, mm')
zlabel('z, mm')
title('YZ plane')

sgtitle(fig1, 'FIELD II')

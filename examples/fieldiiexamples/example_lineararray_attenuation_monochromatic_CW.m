% Code to compute the 3D pressure field for monochromatic DOMINO probe
% In this code we add the apodization of the elements to maintain
% a ratio F/D constant.
% clc
clear 

addpath("C:\Users\INSERM\Documents\Esteban\Simulation\Field II\Field_II_ver_3_30_windows")
addpath("C:\Users\INSERM\Documents\Esteban\Simulation\neurosonogene_acousticsimulations\WorkInProgress\Functions")

field_init(0)

%% DEFINE PULSE AND TRANSDUCER PROPERTIES
disp('Defining transducer...')

% for i = [1,1.5,2]

% you will see that the transducer is quite large compared to the
% wavelength, therefore the grid is very fine. This explains why a field II
% simulation is preferred to a kwave simulation because the latter would
% take ages on such a 3D grid.

%  -------------- emission ----------------
frequency = 10 ; %MHz
chosen_medium = 'brain';
setup_simu.c = 1540; % Speed of sound [m/s]
setup_simu.f0 = frequency*1e6 ; % [Hz]
setup_simu.lambda = setup_simu.c/setup_simu.f0; % Wave length [m]

% ------- focus and simulation window (input parameters) -------
x_focus = 0 ; % mm
y_focus = 0 ; % mm
z_focus = 5 ; % mm
dx = 8 ; % mm dx window ( we simulate y in [x_focus - dx, x_focus + dx] )
dy = 0 ; % mm dy window ( we simulate y in [y_focus, y_focus + dy] )
dz_up = 4 ; % mm dz window ( we simulate z in [z_focus - dz, z_focus + dz] )
dz_down = 5 ; % mm dz window ( we simulate z in [z_focus - dz, z_focus + dz] )
apodization_type = 'none' ;
setup_simu.ratio_F_over_D = 1 ; % f/D ratio to compute the active aperture
% dxyz = setup_simu.lambda/4; % [m]
dxy = 0.02e-3 ; % [m]
dz = 0.02e-3; % [m]


% ------------------ Saving and plotting options -----------------------
version= '_calib_simu' ;
data_folder = 'Ignacio paper\data\' ;
figure_folder = 'Ignacio paper\figures\' ;
setup_simu.symmetry_yaxis = 0 ; % Whether one want or not to exploit y-symmetry
option.save_data = 0 ;
option.save_figure = 0 ;
setup_simu.calibrated_P_neg_Pa = 1 ; % Pa, measured value in a water tank for 15MHz. (in water!)

%  ------------------- transducer characteristics --------------------
setup_simu.tx_N_elements = 128 ; % Number of elements in the transducer
setup_simu.tx_element_height_mm = 1.5 ;% Height of element [mm] (Elevation aperture ?)
setup_simu.tx_width_mm = 0.108 ; % Width of element 
setup_simu.tx_pitch_mm = 0.11 ; % Kerf [mm]
setup_simu.tx_kerf_mm = setup_simu.tx_pitch_mm - setup_simu.tx_width_mm ; % Kerf [mm]
setup_simu.tx_elevationFocus_mm = 8 ; % Elevation focus [m]
setup_simu.tx_frequency = setup_simu.f0 ; % Transducer center frequency [Hz]

% Total aperture (Note: we suppose space between elements negligable, kerf << width.)
setup_simu.total_aperture_mm = setup_simu.tx_pitch_mm*setup_simu.tx_N_elements ; %mm
max_depth = setup_simu.total_aperture_mm*setup_simu.ratio_F_over_D ; %mm
setup_simu.focus_mm = [x_focus y_focus z_focus]; % Fixed focal point [mm]
% disp(['Note: with F/D = ' num2str(setup_simu.ratio_F_over_D) ', z_max is: ' num2str(max_depth)])

% ---------- mathematical elements ----------
setup_simu.sampling_frequency = 100e6 ;  % Sampling frequency of Veramachine [Hz]
setup_simu.no_sub_x = 1  ;% Number of sub-divisions in x-direction of mathematical elements
setup_simu.no_sub_y = 10 ;% Number of sub-divisions in y-direction of mathematical elements

% -------- The size of sub-division is correct? -----------
min_distance = z_focus - dz_up  ; % Min distance from a transducer to the point [m]
max_subdiv_size = max(setup_simu.tx_element_height_mm/setup_simu.no_sub_y,...
                    setup_simu.tx_width_mm/setup_simu.no_sub_x) ; %[m]
dist_min_lim = max_subdiv_size^2/(4*setup_simu.lambda*1e3) ;

disp(['min distance to the field (I): ', num2str(min_distance),...
    ' mm, and rectangles size is w^2/(5 lambda): ', num2str(dist_min_lim), ' mm.'])
if min_distance > 10*dist_min_lim
    disp('The condition I >> w^2/(5 lambda) is achieved -> good precision.')
else
    disp('The condition I >> w^2/(5 lambda) is not achieved, no guaranteed precision') 
end
% Set the sampling frequency
set_sampling(setup_simu.sampling_frequency);

% -------- create mathematical transducer --------- 
transducer = xdc_focused_array( ...
setup_simu.tx_N_elements, ...       % Number of elements
setup_simu.tx_width_mm/1000, ...            % Width of elements [m]
setup_simu.tx_element_height_mm/1000, ...   % Height of elements [m]
setup_simu.tx_kerf_mm/1000, ...             % Kerf [m]
setup_simu.tx_elevationFocus_mm/1000, ...   % Elevation focus [m]
setup_simu.no_sub_x, ...            % Subdivisions in x-direction
setup_simu.no_sub_y, ...            % Subdivisions in y-direction
[0,0,1000]/1000 );    % Focus

% Set the apodization for the individual mathematical elements

% create the apodization 
options.plot_apodization = 0 ;
options.apodization_type = apodization_type ;
options.equivalent_N_active = 1 ;  
[apodization, setup_simu] = compute_apodization(setup_simu, options) ;
setup_simu.apodization = apodization ;

L = 8*setup_simu.lambda*setup_simu.ratio_F_over_D*...
    setup_simu.tx_N_active/sum(apodization, 'all')*1000 ; % mm
d = 1.02*setup_simu.lambda*(setup_simu.ratio_F_over_D*...
    setup_simu.tx_N_active/sum(apodization, 'all'))^2*1000 ; % mm
disp(['Approx. focal length: ' num2str(L)])
disp(['Approx. focal width: ' num2str(d)])

% Set the apodization into the transducer
element_no = (1:setup_simu.tx_N_elements)';
apo = apodization'.*ones(1,setup_simu.no_sub_x*setup_simu.no_sub_y);
ele_apodization(transducer, element_no, apo)

% Show transducer
setup_simu.transducer_data = xdc_get(transducer,'all');


% Creating the file name
if x_focus < 0; x_sign = 'neg' ; else;  x_sign = 'pos'; end

foc_coord = [x_sign num2str(abs(x_focus)) '_' num2str(y_focus) '_' strrep(num2str(z_focus), '.', '')] ;
str_FoverD = strrep(num2str(setup_simu.ratio_F_over_D), '.', '') ;
str_freq = strrep(num2str(frequency), '.', '-') ;
filename = ['DOMINO_',str_freq, 'MHz_3DPressureField_in_', chosen_medium, '_FoverD_', str_FoverD, '_apod_',...
    apodization_type,'_at_', foc_coord, version] 

%% DEFINE THE MEDIUM PROPERTIES
disp('Defining medium properties...')
disp('')
% here we define the medium properties.
% - do the simulation in water if you want to compare with pressure
% measurements in a water tank, and scale the pressure field to that
% measurement. 
% - do the simulation in brain tissue if you want to know about brain
% attenuation and know what the pressure will be in brain tissue compared
% to water.
% - for thermal simulations, we actually conducted the simulation in water
% (with quasi null attenuation) to be in a 'worst case' scenario and get
% the upper bound of pressure that can be reached in any situation in the
% brain.

medium.nature = chosen_medium;

switch chosen_medium
    
 case 'water'
     % medium velocity
        medium.c = 1540;
        
     % medium attenuation
        medium.alpha_coeff = 2.2e-3;    % [dB/(MHz^y cm)], valeur du Duck à 20°C, table 4.8 (25e-3 Np/m/MHz²)
        medium.alpha_power = 2;         % Duck aussi du coup, vu que c'est en /MHz²
            
    case 'brain'
     % medium velocity
        medium.c = 1546;
              
     %medium attenuation      
        medium.alpha_coeff = 0.5912;    % ITIS: Brain 6.8032 Np/m/MHz^-y with alpha power y = 1.3  , soit 0.5912 dB/cm/MHz^-y   (attenuation coefficient, 1Np = 8.69dB)
        medium.alpha_power = 1.3;         % Duck aussi du coup, vu que c'est en /MHz²
end

% -------- set medium velocity ---------
set_field ('c', medium.c);          % ITIS table (whole brain, Brain (White Matter) is 1552.5), confirmed in guash et al (c=1552.5 rho=1041.0 qf= 302 Brain White Matter, c=1505.0 rho=1044.5 qf=1745 Brain Grey Matter)

% ------ set medium attenuation ---------
set_field ('use_att',1);
set_field ('att_f0',setup_simu.tx_frequency); %  Field II ne permet qu'une approximation linéaire de l'attenuation autour de la fréquence centrale 

medium.attenuation_coeff_at_f0 = medium.alpha_coeff*(setup_simu.tx_frequency/1e6)^medium.alpha_power*100; %[dB/m]
medium.correction_coeff = medium.attenuation_coeff_at_f0/(setup_simu.tx_frequency); %[dB/m/Hz]

set_field ('att',medium.attenuation_coeff_at_f0); % [dB/m]     partie non dépendante de la fréquence (coefficient à setup_simu.tx_frequency);
set_field ('freq_att',medium.correction_coeff); %[dB/m/Hz]  coefficient de correction dépendant de la fréquence

%% DEFINE THE MEASUREMENT GRID OR POSITIONS
disp('Defining grid and positions...')

% computationnaly efficient method is to simulate a half Y plane, and then
% mirror its results because it should be symmetrical.

x_extent = [x_focus-dx, x_focus+dx]*1e-3;  %[m]

if y_focus == 0 && setup_simu.symmetry_yaxis
    y_extent = [y_focus, y_focus+dy]*1e-3;  %[m]
else
    y_extent = [y_focus-dy, y_focus+dy]*1e-3;  %[m]
end

z_extent = [z_focus-dz_up, z_focus+dz_down]*1e-3;  %[m]

if z_extent(1) < 0
    error('z_min must be higher than 0.')
end

% Calculate the number of points
if ~exist('dxyz', 'var')
num_points_x = round(diff(x_extent) / dxy);
num_points_y = round(diff(y_extent) / dxy);
num_points_z = round(diff(z_extent) / dz);
else
num_points_x = round(diff(x_extent) / dxyz);
num_points_y = round(diff(y_extent) / dxyz);
num_points_z = round(diff(z_extent) / dxyz);
end

% Debugging statements
disp(['Focusing at: ', num2str(setup_simu.focus_mm)]);
disp(['x extent: ', num2str(x_extent*1e3)]);
disp(['y extent: ', num2str(y_extent*1e3)]);
disp(['z extent: ', num2str(z_extent*1e3)]);
disp(['Number of total points: ', num2str(num_points_z*num_points_y*num_points_x)]);
disp('')

% Ensure the number of points is greater than 1
if num_points_x < 2
    num_points_x = 2;
end
if num_points_y < 2
    num_points_y = 2;
end
if num_points_z < 2
    num_points_z = 2;
end

% whole X Z plane measurement

if ~exist('dxyz', 'var')
x_vec = linspace(x_extent(1), x_extent(2), round(diff(x_extent)/ dxy)+1);
y_vec = linspace(y_extent(1), y_extent(2), round(diff(y_extent)/ dxy)+1);
z_vec = linspace(z_extent(1), z_extent(2), round(diff(z_extent)/ dz)+1);
else 
x_vec = linspace(x_extent(1), x_extent(2), round(diff(x_extent)/ dxyz));
y_vec = linspace(y_extent(1), y_extent(2), round(diff(y_extent)/ dxyz));
z_vec = linspace(z_extent(1), z_extent(2), round(diff(z_extent)/ dxyz));
end
[x_grid, y_grid, z_grid] = meshgrid(x_vec, y_vec, z_vec);
setup_simu.measurement_points = [x_grid(:) y_grid(:) z_grid(:)];

%% CALL Field II ROUTINES FOR SPATIAL IMPULSE RESPONSE SIMULATION

disp('Be sure to do no overwrite any data. Starting in 10 seconds...')
pause(10)

%  (for monochromatic field simulations)disp('Calling Field II routine for spatial impulse response simulation...')

%Purpose: Procedure for calculating the spatial impulse response for an aperture.
%Calling: 
[h, start_time] = calc_h(transducer, setup_simu.measurement_points);
% 
% point = [1,0,8] ;
% [h, start_time] = calc_h(transducer, point*1e-3) ;%setup_simu.focus_mm*1e-3);
% Input: Th Pointer to the transducer aperture.
% points Field points. Vector with three columns (x,y,z) and one row for each field point.
% Output: h Spatial impulse response in m/s.
% start time The time for the first sample in h.

%%
% 
% figure ;
% 
% time = [start_time:(1/setup_simu.sampling_frequency):(start_time+1/setup_simu.sampling_frequency*(length(h)-1))];
% plot(time, h, 'o-')



%% Reshape measurements

disp('Post-processing of the results...')

spatial_impulse_response_field = reshape(h,[], size(x_grid,1), size(x_grid,2), size(x_grid,3));  % first two dimensions are space, last is time

% get the frequency response
% this is an impulse response, so will contain all the frequencies.
% to obtain the spatial distribution of one frequency, we do a fft and pick
% the wanted frequency.

spatial_impulse_response_field_FT = fft(spatial_impulse_response_field, [], 1);
freq_vect = linspace(0, setup_simu.sampling_frequency, size(spatial_impulse_response_field_FT, 1));

%% Pick the selected frequency
freq_searched = setup_simu.f0 ;
% freq_searched = 10*1e6 ;
idx_freq = find(freq_vect>=freq_searched,1, 'first');

% Amplitude for a given frequency
amp_response_tx_freq = squeeze(abs(spatial_impulse_response_field_FT(idx_freq,:,:,:)));

% scale pressure field 

% Apply a scale factor to the simulation to match the pressure value measurement.
% Method for propagation medium other than water: simulate the pressure
% field in water, calculate and keep the scale factor between simul and
% measured pressure, and apply this scale factor to subsequent simulation
% (e.g. in the brain with larger attenuation).

scale_factor = 1/max(amp_response_tx_freq, [], 'all')*setup_simu.calibrated_P_neg_Pa;
Pressure_field_monofreq = amp_response_tx_freq*scale_factor ;

%% In case of symmetry (we consider just y=0) exploit it.

if y_focus == 0 && setup_simu.symmetry_yaxis

[dim1, dim2, dim3] = size(Pressure_field_monofreq) ;

y_extent_3D = [y_focus-dy, y_focus+dy]*1e-3;  %[m]
y_vec_3D = linspace(y_extent_3D(1), y_extent_3D(2), dim1*2-1);
    
% Mirror along the y-axis
Pressure_field_3D = field_mirror_yaxis(Pressure_field_monofreq); % Pa

[x_grid_3D, y_grid_3D, z_grid_3D] = meshgrid(x_vec*1e3, y_vec_3D*1e3, z_vec*1e3); %mm

else
    
disp('No mirroring in the y-direction has been done...')
y_extent_3D = y_extent ;
y_vec_3D = y_vec ;
Pressure_field_3D = Pressure_field_monofreq ; % MPa
x_grid_3D = x_grid*1e3 ; y_grid_3D = y_grid*1e3 ;z_grid_3D = z_grid*1e3 ;

end

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
field_source.focus_mm = setup_simu.focus_mm ;
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
    save([data_folder filename], 'medium', 'setup_simu', 'field_source')
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




sgtitle(sprintf('Focalization in %s at (%.1f, %.1f, %.1f) mm, with f_c = %.1f MHz, and F/D = %.1f',chosen_medium, x_focus, y_focus, z_focus,frequency,setup_simu.ratio_F_over_D))

if option.save_figure
   disp(['saving   ' figure_folder filename '   ...'])
   savefig([figure_folder filename '.fig'])
end
% end

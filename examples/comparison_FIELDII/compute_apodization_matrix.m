% Compute apodization to adjust the aperture to the focal spot
%
% Calling: compute_apodization(setup_simu, type)
%
% Arguments: setup_simu - structure obtained with the transducer info and
% focus spot
% type - string with the type of window:
% 'none' : No apodization, returns all the elements of the vector equal to 1. (DEFAULT)
% 'rect' : returns the active elements set to 1 amplitude in a rectangular shape, and the rest 0.
% 'circular' : returns the active elements set to 1 amplitude in an elliptical shape, and the rest 0.
% 'hanning' : returns the active elements as a hanning window. 
% 'haming' : returns the active elements as a haming window.
% plot_apodization (optional) - logical variable. Set this variable to 1 
% returns a figure with the plot of the apodization. (DEFAULT = False)
%
% Return: apodization
%
% In this code we add the apodization on the Matrix Probe's elements to maintain
% a ratio F/D constant.

function [apodization, setup_simu] = compute_apodization_matrix(setup_simu, type, plot_apodization)

% Default values
if nargin < 3
    plot_apodization = 0 ;
    if nargin < 2
        type = 'circular' ;
    end
end

if ~isfield(setup_simu,'total_aperture_mm')
    % We use and average number to define the total_aperture_mm
    setup_simu.total_aperture_mm2 = setup_simu.tx_pitch_mm*setup_simu.tx_N_elem_x*...
                                    setup_simu.tx_pitch_mm*setup_simu.tx_N_elem_y ; %mm
end 

% We take the (x,y,z) coordinates of the focal point
x_focus = setup_simu.focus_mm(1) ; 
y_focus = setup_simu.focus_mm(2) ;
z_focus = setup_simu.focus_mm(3) ;

if strcmp(type, 'none')
    apodization = ones(setup_simu.tx_N_elem_x, setup_simu.tx_N_elem_y) ;
    setup_simu.tx_N_active_elem_x = setup_simu.tx_N_elem_x ;
    setup_simu.tx_N_active_elem_y = setup_simu.tx_N_elem_y ;
else
    if isfield(setup_simu, 'dir_angle_deg')
        dir_angle_deg = setup_simu.dir_angle_deg ;
    else
        dir_angle_deg = 30 ;
        fprintf(['Directivity angle for elements set as 30 degrees. \n ',...
           'If you want to change it, define setup_simu.dir_angle_deg. \n'])
    end
    % Active aperture diameter (defined by the ratio F/D)
    d_tx = 2*z_focus*tan(deg2rad(dir_angle_deg))/setup_simu.ratio_F_over_D ; %mm
    
    % we compute the effective total aperture diameter
%     D_eff = sqrt(setup_simu.total_aperture_mm2*4/pi) ;
    
    % Determine the active elements in each dimension
    tx_N_active_x_virt = round(d_tx/ setup_simu.tx_pitch_mm) ; 
    tx_N_active_y_virt = round(d_tx/ setup_simu.tx_pitch_mm) ; 
    % Make them Odd to be well-centered
    if rem(tx_N_active_x_virt, 2) == 0 
        tx_N_active_x_virt = tx_N_active_x_virt + 1;
    end
    if rem(tx_N_active_y_virt, 2) == 0 
        tx_N_active_y_virt = tx_N_active_y_virt + 1;
    end
    
    % We compute the active and center elements on x-direction
    if tx_N_active_x_virt > setup_simu.tx_N_elem_x
        warning('z_focus outside of the imaging window for the chosen ratio F/D.')
        setup_simu.tx_N_x_active = setup_simu.tx_N_elem_x ;
    else
        setup_simu.tx_N_x_active = tx_N_active_x_virt ;
    end
    
    % We compute the active and center elements on y-direction
    if tx_N_active_y_virt > setup_simu.tx_N_elem_y
        warning('z_focus outside of the imaging window for the chosen ratio F/D.')
        setup_simu.tx_N_y_active = setup_simu.tx_N_elem_y ;
    else
        setup_simu.tx_N_y_active = tx_N_active_y_virt ;
    end
    
    %We create our apodization window and its values
    x_center_apo_window = round((tx_N_active_x_virt - setup_simu.tx_N_elem_x)/2) ;
    y_center_apo_window = round((tx_N_active_y_virt - setup_simu.tx_N_elem_y)/2) ;
    apo_x_window = [1:tx_N_active_x_virt] - x_center_apo_window;
    apo_y_window = [1:tx_N_active_y_virt] - y_center_apo_window;
    
    if strcmp(type, 'rect')
            % Total aperture
            apo_2D_profile = ones(tx_N_active_x_virt, tx_N_active_y_virt) ;   
    elseif strcmp(type, 'circular')
    fprintf('Apod. Radii: %d and %d. \n', tx_N_active_x_virt, tx_N_active_y_virt)
            apo_2D_profile = createEllipseMask(tx_N_active_x_virt, tx_N_active_y_virt) ;      
    elseif strcmp(type, 'hanning')
            apo_2D_profile = hanning(tx_N_active_x_virt)*hanning(tx_N_active_y_virt)' ;
    elseif strcmp(type, 'hamming')
            apo_2D_profile = hamming(tx_N_active_x_virt)*hamming(tx_N_active_y_virt)' ;
    else
    error("Active aperture window not in the options. type input must be 'none','circular', 'hanning' or 'haming'.")
    end
       
    % We shift the aperture to the x_focus position. 
    % We compute the shift to the new center of the apodization if there is one.
    shift_x_direction = sign(x_focus) ; 
    shift_x_elements = round(abs(x_focus)/setup_simu.tx_pitch_mm) ; % Pass from mm -> elements
    shift_y_direction = sign(y_focus) ; 
    shift_y_elements = round(abs(y_focus)/setup_simu.tx_pitch_mm) ; % Pass from mm -> elements

    % We check if x_focus exceeds the max shift = N_elements/2 
    if shift_x_elements > setup_simu.tx_N_elem_x/2
        % We make the max center shift equal to the max shift.
        shift_x_elements = setup_simu.tx_N_elem_x/2 ;
    end
    
    % We check if y_focus exceeds the max shift = N_elements/2 
    if shift_y_elements > setup_simu.tx_N_elem_y/2
        % We make the max center shift equal to the max shift.
        shift_y_elements = setup_simu.tx_N_elem_y/2 ;
    end
    
    % We apply the shift to the apodization window
    apo_x_window = apo_x_window + shift_x_direction*shift_x_elements ;
    apo_y_window = apo_y_window + shift_y_direction*shift_y_elements ;
    
    % And we take as the final apodization profile the values that falls
    % within the transducer limits
    apo_within_transducer_x = apo_x_window >= 1 & apo_x_window <= setup_simu.tx_N_elem_x ; 
    apo_within_transducer_y = apo_y_window >= 1 & apo_y_window <= setup_simu.tx_N_elem_y ; 
    
    % We initialize the apodization vector
    apodization = zeros(setup_simu.tx_N_elem_x, setup_simu.tx_N_elem_y) ;
    % And we introduce the values of the elements within the window of the
    % transducer
    apodization(apo_x_window(apo_within_transducer_x),apo_y_window(apo_within_transducer_y)) =...
                apo_2D_profile(apo_within_transducer_x, apo_within_transducer_y) ;
    
    %We compute the limit of the z_focus where we stil have the constant
    %F/D.
    
%     setup_simu.max_z_focus_at_constant_F_D = setup_simu.total_aperture_mm*setup_simu.ratio_F_over_D ;

end 

setup_simu.apodization_type = type ;

if plot_apodization
    % Plot the apodization
    disp('plot_apodization set to True')
    hold on
    
    trans_x_window = [1:setup_simu.tx_N_elem_x] ;
    trans_y_window = [1:setup_simu.tx_N_elem_y] ;
    
    imagesc(apo_y_window,apo_x_window, apo_2D_profile)
    imagesc(trans_y_window, trans_x_window, apodization)
    caxis([0,1])
    grid minor
    axis image
    xlabel('x, element')
    ylabel('y, element')
    xline(setup_simu.tx_N_elem_x/2, 'r')
    yline(setup_simu.tx_N_elem_y/2, 'r')
end


% Compute apodization to adjust the aperture to the focal spot
%
% Calling: compute_apodization(setup_simu, type)
%
% Arguments: setup_simu - structure obtained with the transducer info and
% focus spot
% type - string with the type of window:
% 'none' : No apodization, returns all the elements of the vector equal to 1. (DEFAULT)
% 'rect' : returns the active elements set to 1 amplitude, and the rest 0.
% 'hanning' : returns the active elements as a hanning window. 
% 'haming' : returns the active elements as a haming window.
% plot_apodization (optional) - logical variable. Set this variable to 1 
% returns a figure with the plot of the apodization. (DEFAULT = False)
%
% Return: apodization
%
% Note: In this code we suppose the N of elements even.


function [apodization, setup_simu] = compute_apodization(setup_simu, option)


if nargin < 2
    plot_apodization = 0 ;
    type = 'rect' ;
    equivalence = 0 ; 
else
    if isfield(option, 'plot_apodization')
        plot_apodization = option.plot_apodization ;
    else
        plot_apodization = 0 ;
    end
    if isfield(option, 'apodization_type')
        type = option.apodization_type ;
    else
        type = 'rect' ;
    end
    if isfield(option, 'equivalent_N_active')
        equivalence = option.equivalent_N_active ;
    else
        equivalence = 0 ;
    end
    
end

if ~isfield(setup_simu,'total_aperture_mm')
    setup_simu.total_aperture_mm = setup_simu.tx_pitch_mm*setup_simu.tx_N_elements ; %mm
end 

% We take the (x,z) coordinates of the focal point
x_focus = setup_simu.focus_mm(1) ; 
z_focus = setup_simu.focus_mm(3) ;

if strcmp(type, 'none')
    apodization = ones(1,setup_simu.tx_N_elements) ;
    setup_simu.tx_N_active = setup_simu.tx_N_elements ;
else
    
    % Active aperture (defined by the ratio F/D)
    d_tx = z_focus/setup_simu.ratio_F_over_D ; %mm
    % We compute the number of active elements (must be even)
    tx_N_active_virtual = round(d_tx*setup_simu.tx_N_elements/2/setup_simu.total_aperture_mm)*2 ; 
    
    % change the amount of active element to balance the aperture
    factor = 1 ;
    if strcmp(type, 'hanning')
        if equivalence
            factor = 0.5 ; 
        end
    elseif strcmp(type, 'hamming')
        if equivalence
            factor = 0.54 ; 
        end
    end
    
    
    N_active_extended = tx_N_active_virtual/factor;
    
    % Save the the number of active elements on the setup_simu structure
    if N_active_extended > setup_simu.tx_N_elements
        warning('z_focus outside of the imaging window for the chosen ratio F/D.')
        setup_simu.active_aperture_mm = setup_simu.total_aperture_mm ;
    else
        setup_simu.active_aperture_mm = d_tx ;
    end
    
    %We create our apodization window and its values
    center_apo_window = round((N_active_extended - setup_simu.tx_N_elements)/2) ;
    apo_window = [1:N_active_extended] - center_apo_window;
    if strcmp(type, 'rect')
        apo_profile = ones(1, N_active_extended) ;            
    elseif strcmp(type, 'hanning')
        apo_profile = hanning(N_active_extended) ;
    elseif strcmp(type, 'hamming')
        apo_profile = hanning(N_active_extended) ;
    else
    error("Active aperture window not in the options. type input must be 'none', 'hanning' or 'haming'.")
    end
       
    % We shift the aperture to the x_focus position. 
    % We compute the shift to the new center of the apodization if there is one.
    shift_direction = sign(x_focus) ; 
    shift_elements = round(abs(x_focus)/setup_simu.tx_pitch_mm) ; % Pass from mm -> elements

    % We check if x_focus exceeds the max shift = N_elements/2 
    if shift_elements > setup_simu.tx_N_elements/2
        % We make the max center shift equal to the max shift.
        shift_elements = setup_simu.tx_N_elements/2 ;
    end
    
    % We apply the shift to the apodization window
    apo_window = apo_window + shift_direction*shift_elements ;
    
    % And we take as the final apodization profile the values that falls
    % within the transducer limits
    apo_within_transducer = apo_window >= 1 & apo_window <= setup_simu.tx_N_elements ; 
    
    % We initialize the apodization vector
    apodization = zeros(1, setup_simu.tx_N_elements) ;
    % And we introduce the values of the elements within the window of the
    % transducer
    apodization(apo_window(apo_within_transducer)) = apo_profile(apo_within_transducer) ;

    setup_simu.tx_N_active = sum(apo_within_transducer, 'all') ;
    setup_simu.tx_elem_act_amp = sum(apodization, 'all') ;
    setup_simu.tx_N_equi_active = tx_N_active_virtual ;
end 

setup_simu.apodization_type = type ;

if plot_apodization
    % Plot the apodization
    disp('plot_apodization set to True')
%     figure('color','white')
    xline(setup_simu.tx_N_elements/2, 'r')
    xline(-setup_simu.tx_N_elements/2, 'r')
    transducer_window = [1:setup_simu.tx_N_elements] ;
    shift_to_center = -setup_simu.tx_N_elements/2 - 0.5 ;
    
    hold on
    stem(0,1, 'b')
    stem(shift_direction*shift_elements,1, '--b')
    stem(apo_window + shift_to_center, apo_profile, 'k')
    stem(transducer_window + shift_to_center, apodization, 'color', 	'#EDB120')
    legend('left limit transducer', 'right limit transducer', 'starting center',...
        'x-focus (shifted center)', 'extended aperture apodization', 'real aperture apodization')
    grid minor
    title('apodization per each element of the transducer')
    axis('tight')
end

end


function mask = createEllipseMask(xlength, ylength)
    % Create a grid of coordinates
    [X, Y] = meshgrid(1:xlength, 1:ylength);
    
    % Define the center of the ellipse
    centerX = (xlength+1) / 2;
    centerY = (ylength+1) / 2;
    
    % Define the axes lengths (half the diameters)
    a = (xlength) / 2;  % Semi-major axis
    b = (ylength) / 2;  % Semi-minor axis
    
    % Create the equation of the ellipse (normalized)
    ellipseEq = ((X - centerX).^2 / a^2) + ((Y - centerY).^2 / b^2);
    
    % Create the binary mask where the points inside the ellipse are 1, and outside are 0
    mask = ellipseEq < 1;
end

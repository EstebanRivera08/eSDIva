function res = show_xdc_optimized(Th, options)

    if nargin < 2
        zlims = [-0.5, 0.5] ;
        edges_config = 'none' ; 
        plot_colorbar = 1 ;
        cmap = 'cool' ; 
    else
        if isfield(options, 'edges_config')
            edges_config = options.edges_config ;
        else
            edges_config = 'none' ; 
        end
        if isfield(options, 'zlim')
            zlims = options.zlim ;
        else
            zlims = [-0.5, 0.5] ; 
        end
        if isfield(options, 'colorbar')
            plot_colorbar = options.colorbar ; 
        else
            plot_colorbar = 1 ;
        end
        if isfield(options, 'colormap')
            cmap = options.colormap ;
        else
            cmap = 'cool' ; 
        end
    end
    
    % Get the transducer data
    data = xdc_get(Th, 'all');
    [~, M] = size(data);

    % Preallocate arrays for vertices and colors
    vertices = zeros(4 * M, 3);  % 4 vertices per element
    faces = zeros(M, 4);         % Each element is a quadrilateral
    colors = zeros(M, 1);        % Apodization values for each element

    % Populate the vertices, faces, and colors
    for i = 1:M
        % Vertices for the current element
        v_idx = (i - 1) * 4 + (1:4);  % Index range for current element
        vertices(v_idx, :) = [
            data(11:13, i)';  % Bottom-left corner
            data(14:16, i)';  % Bottom-right corner
            data(17:19, i)';  % Top-right corner
            data(20:22, i)']; % Top-left corner

        % Face definition (quad)
        faces(i, :) = v_idx;

        % Apodization value (color)
        colors(i) = data(5, i);
    end

    % Convert to mm for plotting
    vertices = vertices * 1000;

    % Plot the transducer

    % Use patch for efficient rendering
    
    res = patch('Vertices', vertices, 'Faces', faces, 'FaceVertexCData', colors, ...
          'FaceColor', 'flat', 'EdgeColor', edges_config);
    colormap(cmap)
    % Add labels, colorbar, and adjust view
    caxis([0,1])
    if plot_colorbar
    Hc = colorbar;
    title(Hc, 'Apodization');
    end
    
    xlabel('x [mm]');
    ylabel('y [mm]');
    zlabel('z [mm]');
    axis('image')
    view(3);
    grid on;
    set(gca, 'ZDir', 'reverse');
    zlim(zlims);
end

function euler_plot = generate_euler_plot(ebsd)

    nCol = size(ebsd.orientations.phi1, 2);
    nRow = size(ebsd.orientations.phi1, 1);
    angles = zeros(nRow, nCol, 3);
    angles(:, :, 1) = ebsd.rotations.phi1 / (pi * 2);
    angles(:, :, 2) = ebsd.rotations.Phi / (pi * 2);
    angles(:, :, 3) = ebsd.rotations.phi2 / (pi * 2);
    
    euler_plot = angles;
    
end
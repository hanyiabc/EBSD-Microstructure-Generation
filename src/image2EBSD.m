function ebsd = image2EBSD(img, stepSize, CS)
    [y, x] = find(~isnan(img(:, :, 1)));
    x = x * stepSize(1);
    y = y * stepSize(2);
    img = double(img);
    oneDangles = reshape(img, [], 3);
    phase1 = find(oneDangles(:, 1));
    idx = find(oneDangles(:, 1));
    oneDangles(idx, 1) = oneDangles(idx, 1) / 255 * (pi * 2);
    oneDangles(idx, 2) = oneDangles(idx, 2) / 255 * (pi * 2);
    oneDangles(idx, 3) = oneDangles(idx, 3) / 255 * (pi * 2);

    
    rot = rotation.byEuler(oneDangles(:, 1), oneDangles(:, 2),oneDangles(:, 3), 'Bunge');
    phases = zeros(size(oneDangles, 1), 1, 'int32');

    options = struct();
    
    options.x = x;
    options.y = y;

    unitCell = calcUnitCell([x, y], 'GridType', 'rectangular');
    ebsd = EBSD(rot, phases, CS, options,'unitCell', unitCell);
end
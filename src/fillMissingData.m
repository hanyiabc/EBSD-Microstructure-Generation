function outImg = fillMissingData(img)
    dim = size(img);
    dim = dim(1:2);
    dim3 = size(img);
    nrow = dim(1);
    ncol = dim(2);
    
    zeroIdx = find(all(img == 0, 3));
    [row, col] = ind2sub(dim, zeroIdx);
    
    zeroSubs = [row, col];
    zeroIdxR = sub2ind(dim3, row, col, ones(size(row)) * 1);
    zeroIdxG = sub2ind(dim3, row, col, ones(size(row)) * 2);
    zeroIdxB = sub2ind(dim3, row, col, ones(size(row)) * 3);
    
    combinations = [0 1; 1 0; 0 -1; -1 0];
    randNeighbors = combinations(randsample(size(combinations, 1), size(zeroIdx, 1), true), :);
    neighborSubs = randNeighbors + zeroSubs;
    
    neighborSubs(neighborSubs < 1) = 1;
    neighborSubs(neighborSubs(:, 1) > nrow, 1) = nrow;
    neighborSubs(neighborSubs(:, 2) > ncol, 2) = ncol;
    
    neighborIdxR = sub2ind(dim3, neighborSubs(:, 1), neighborSubs(:, 2), ones(size(row)) * 1);
    neighborIdxG = sub2ind(dim3, neighborSubs(:, 1), neighborSubs(:, 2), ones(size(row)) * 2);
    neighborIdxB = sub2ind(dim3, neighborSubs(:, 1), neighborSubs(:, 2), ones(size(row)) * 3);
    
    outImg = img;
    outImg(zeroIdxR) = img(neighborIdxR);
    outImg(zeroIdxG) = img(neighborIdxG);
    outImg(zeroIdxB) = img(neighborIdxB);
end
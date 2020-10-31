function angles = plot_euler(path, CS, stepSize,targetStepSize ,outName, ctfName)
    [filepath,name,ext] = fileparts(path)
    
    if ext == ".ang"
        ebsd = EBSD.load(path,CS,'interface','ang', 'convertEuler2SpatialReferenceFrame', 'setting 1');
    else
        ebsd = EBSD.load(path,CS,'interface','generic');
    end
    step = targetStepSize;
    
    ebsdProccessed = ebsd;
%     ebsdProccessed = preprocessEBSD(ebsd);
    [grains, ebsdProccessed.grainId] = calcGrains(ebsdProccessed, 'angle', 10 * degree);
    unitCell = [-step -step; -step step; step step; step step];
    ebsdS = ebsdProccessed.gridify('unitCell',unitCell);
  
    F = halfQuadraticFilter;
    F.alpha = 0.5;
    ebsdS = smooth(ebsdS, F, 'fill', grains);

%     figure
%     plot(ebsdS('Copper'),ebsdS('Copper').orientations,'micronbar','off')
    figure
    plot(ebsdS('indexed'),ebsdS('indexed').orientations);
    ebsdS2 = ebsdS('indexed').gridify('unitCell',unitCell);
    angles = generate_euler_plot(ebsdS2);
    angles = uint8(angles * 255);
    angles = processEulerImg(angles);
    imwrite(angles, outName);
    ebsdImg = image2EBSD(angles, [step * 2, step * 2], CS);
    ebsdImg.export(ctfName);
end

function ebsdOut = preprocessEBSD(ebsd)
    [grains, ebsd.grainId, ebsd.mis2mean] = calcGrains(ebsd, 'angle', 10*degree);
    ebsd(grains(grains.grainSize < 10)) = [];
    [grains, ebsd.grainId] = calcGrains(ebsd, 'angle', 10 * degree);
    
    F = halfQuadraticFilter;
    F.alpha = 0.25;
    grains = smooth(grains);
    ebsdOut = smooth(ebsd, F, 'fill', grains);
end


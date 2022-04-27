function angles = plot_euler(path, CS, stepSize,targetStepSize ,outName, ctfName)
    [filepath,name,ext] = fileparts(path);
    
    if ext == ".ang"
        ebsd = EBSD.load(path,CS,'interface','ang','convertEuler2SpatialReferenceFrame', 'setting 1');
    else
        ebsd = EBSD.load(path,CS,'interface','generic');
    end
    
    step = targetStepSize;
    ebsdProccessed = preprocessEBSD(ebsd, step);

    angles = generate_euler_plot(ebsdProccessed);
    angles = uint8(angles * 255);
    angles = processEulerImg(angles);
    imwrite(angles, outName);
    ebsdImg = image2EBSD(angles, [step * 2, step * 2], CS);
    ebsdImg.export(ctfName);

end

function ebsdOut = preprocessEBSD(ebsd, step)

    [grains, ebsd.grainId] = calcGrains(ebsd, 'angle', 10 * degree);
    unitCell = [-step -step; -step step; step step; step step];
    ebsdGrided = ebsd.gridify('unitCell',unitCell);
    F = halfQuadraticFilter;
    F.alpha = 0.5;
    ebsdSmoothed = smooth(ebsdGrided, F, 'fill', grains);
    ebsdGrided2 = ebsdSmoothed('indexed').gridify('unitCell',unitCell);
    ebsdOut = ebsdGrided2;
    
    mtexFig = newMtexFigure('layout',[1,3]);
    plot(ebsd('indexed'),ebsd('indexed').orientations);
    nextAxis
    plot(ebsdGrided('indexed'),ebsdGrided('indexed').orientations);
    nextAxis
    plot(ebsdSmoothed('indexed'),ebsdSmoothed('indexed').orientations);
    saveFigure('../data/figures/preproc.svg')
end



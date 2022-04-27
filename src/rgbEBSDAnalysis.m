function ebsd = rgbEBSDAnalysis(origAngles, genAngles, cleanedAngles, CS, stepSize, savepath, basename)
    %FIXME: Need to convert it back to hex grid
%     plotHistogram(angles);
    UNIFIED_COLOR_RANGE = [0 5];
    UNIFIED_COLOR_RANGE_PF = [0 2];
    
    ebsdOri = image2EBSD(origAngles, stepSize, CS);
    ebsdGen = image2EBSD(genAngles, stepSize, CS);
    ebsdCle = image2EBSD(cleanedAngles, stepSize, CS);
    
    [grainsOri,ebsdOri.grainId,ebsdOri.mis2mean]   = calcGrains(ebsdOri);
    grainsOri = grainsOri(grainsOri.grainSize > 10);
    gBOri = grainsOri.boundary('indexed','indexed');
    
    [grainsCle,ebsdCle.grainId,ebsdCle.mis2mean]   = calcGrains(ebsdCle);
    grainsCle = grainsCle(grainsCle.grainSize > 10);
    gBGen = grainsCle.boundary('indexed','indexed');
    
    figure;
    plotAngleDistribution(gBOri.misorientation)
    saveFigure([savepath, basename, '_ori_mis.eps']) 
    
    figure;
    plotAngleDistribution(gBGen.misorientation)
    saveFigure([savepath, basename, '_gen_mis.eps']) 
    k = 2000;
    oriOrie = ebsdOri('indexed').orientations;
    genOrie = ebsdGen('indexed').orientations;
    idx = randsample(size(oriOrie, 1), k);
    
    figure;
    plotIPDF(oriOrie(idx),xvector)
    saveFigure([savepath, basename, '_ori_ipdf.eps'])
    
    figure;
    plotIPDF(genOrie(idx),xvector)
    saveFigure([savepath, basename, '_gen_ipdf.eps'])
    close all;
%     xlim([0 70])
%     ylim([0 35]) 
% 
%     odf = calcDensity(ebsdS('indexed').orientations);
% %     psi = calcKernel(grains('indexed').meanOrientation);
% %     odfGrain = calcDensity(ebsd('indexed').orientations,'kernel',psi);
%     figure
%     plotSection(odf);
% %     plotSection(odf, 'colorrange', UNIFIED_COLOR_RANGE);
%     mtexColorbar
%     
%     figure
%     h = [Miller(1,0,0,odf.CS),Miller(1,1,0,odf.CS),Miller(1,1,1,odf.CS)];
%     plotPDF(odf,h,'antipodal','silent');
% %     plotPDF(odf,h,'antipodal','silent', 'colorrange', UNIFIED_COLOR_RANGE_PF);
%     mtexColorbar;
end

function plotHistogram(angles)
    edges = linspace(0, 255, 255);
    figure
    subplot(3, 1, 1)
    histogram(angles(:, :, 1), edges);
    ylim([0 10000]) 
    
    subplot(3, 1, 2)
    histogram(angles(:, :, 2), edges);
    ylim([0 10000]) 
    
    subplot(3, 1, 3)
    histogram(angles(:, :, 3), edges);
    ylim([0 10000]) 
end


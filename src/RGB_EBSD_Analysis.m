function ebsd = RGB_EBSD_Analysis(angles, CS, stepSize)
    %FIXME: Need to convert it back to hex grid
    plotHistogram(angles);
    UNIFIED_COLOR_RANGE = [0 5];
    UNIFIED_COLOR_RANGE_PF = [0 2];
    
    image2EBSD(angles, stepSize, CS);
    ebsdS = ebsd;
    
    ipfKey = ipfColorKey(ebsdS('indexed'));
    colors = ipfKey.orientation2color(ebsdS('indexed').orientations);
    figure
    plot(ebsdS('indexed'), colors, 'micronbar', 'off');

    grains  = calcGrains(ebsdS);
%     grains = smooth(grains);
    grains = grains(grains.grainSize > 20);
    gB = grains.boundary('indexed','indexed');
    hold on
    plot(gB,'lineWidth',1)
    hold off
    
    figure
    plotAngleDistribution(gB.misorientation)
    
    xlim([0 70])
    ylim([0 35]) 

    odf = calcDensity(ebsdS('indexed').orientations);
%     psi = calcKernel(grains('indexed').meanOrientation);
%     odfGrain = calcDensity(ebsd('indexed').orientations,'kernel',psi);
    figure
    plotSection(odf, 'colorrange', UNIFIED_COLOR_RANGE);
    mtexColorbar
    
    figure
    h = [Miller(1,0,0,odf.CS),Miller(1,1,0,odf.CS),Miller(1,1,1,odf.CS)];
    plotPDF(odf,h,'antipodal','silent', 'colorrange', UNIFIED_COLOR_RANGE_PF);
    mtexColorbar;
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


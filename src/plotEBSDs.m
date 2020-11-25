pname = '../data/angs';
pname_sim = '../data/sim_data';
pname_al = '../data/angs/Al';
CS_copper = {... 
      'notIndexed',...
      crystalSymmetry('432', [3.6 3.6 3.6], 'mineral', 'Copper', 'color', [0.53 0.81 0.98])};
CS_sim = {... 
      'notIndexed',...
      crystalSymmetry('1', [0.0 0.0 0.0], 'mineral', 'Copper', 'color', [0.0 0.0 0.0])};
CS_al = {... 
  'notIndexed',...
  crystalSymmetry('432', [4 4 4], 'mineral', 'Aluminum', 'color', [0.53 0.81 0.98])};

stepSizeM4 = [0.4 0.346410];
stepSizeM1 = [0.5 0.433013];
stepSizeM2 = [0.6 0.519615];
stepSizeM3 = [0.8 0.692820];
stepSizeSim = [0.05 0.05];
stepSizeAl = [0.800000 0.692820];

targetStepSizeAl = 0.20;
targetStepSize = 0.20;
targetStepSizeSim = 0.025;
tgtStpSz = [targetStepSize targetStepSize];
fname1 = [pname '\300.3.ang'];
fname2 = [pname '\300.9.ang'];
fname3 = [pname '\600.3.ang'];
fname4 = [pname '\600.9.ang'];

fname_sim1 = [pname_sim '\ebsd1.ebsd.ctf'];
fname_sim2 = [pname_sim '\ebsd2.ebsd.ctf'];
fname_sim3 = [pname_sim '\ebsd3.ebsd.ctf'];

fname_al2003 = [pname_al '\200.3 cleaned.ang'];
fname_al6003 = [pname_al '\600.3 cleaned.ang'];

setMTEXpref('xAxisDirection','north');
setMTEXpref('zAxisDirection','outOfPlane');
%%
% generate rgb images from EBSD orientations
angleM4 = plot_euler(fname1, CS_copper, stepSizeM4, targetStepSize, '300.3.png', '300.3.ctf');
angleM1 = plot_euler(fname2, CS_copper, stepSizeM1, targetStepSize, '300.9.png', '300.9.ctf');
angleM2 = plot_euler(fname3, CS_copper, stepSizeM2, targetStepSize, '600.3.png', '600.3.ctf');
angleM3 = plot_euler(fname4, CS_copper, stepSizeM3, targetStepSize, '600.9.png', '600.9.ctf');

angleAl = plot_euler(fname_al2003, CS_al, stepSizeAl, targetStepSizeAl, '200.3.png', '200.3_Al.ctf');
angleAl = plot_euler(fname_al6003, CS_al, stepSizeAl, targetStepSizeAl, '600.3_Al.png', '600.3_Al.ctf');

angleSim1 = plot_euler(fname_sim1, CS_sim, stepSizeSim, targetStepSizeSim, 'sim1.png', 'sim1.ctf');
angleSim2 = plot_euler(fname_sim2, CS_sim, stepSizeSim, targetStepSizeSim, 'sim2.png', 'sim2.ctf');
angleSim3 = plot_euler(fname_sim3, CS_sim, stepSizeSim, targetStepSizeSim, 'sim3.png', 'sim3.ctf');

%%
%Crop generated rgb images randomly (full range)
img = imread("../data/full_range_large/200.3.png");
cropped = randomCrop(img, 448);
imwrite(cropped, '../data/small/200.3_full.png')

img = imread("../data/full_range_large/300.3.png");
cropped = randomCrop(img, 448);
imwrite(cropped, '../data/small/300.3_full.png');

img = imread("../data/full_range_large/300.9.png");
cropped = randomCrop(img, 448);
imwrite(cropped, '../data/small/300.9_full.png');

img = imread("../data/full_range_large/600.3.png");
cropped = randomCrop(img, 448);
imwrite(cropped, '../data/small/600.3_full.png');

img = imread("../data/full_range_large/600.3_Al.png");
cropped = randomCrop(img, 448);
imwrite(cropped, '../data/small/600.3_Al_full.png');

img = imread("../data/full_range_large/600.9.png");
cropped = randomCrop(img, 448);
imwrite(cropped, '../data/small/600.9.png');

%%
%Crop generated rgb images randomly
img = imread("../data/large/200.3.png");
cropped = randomCrop(img, 448);
imwrite(cropped, '../data/small/200.3.png')

img = imread("../data/large/300.3.png");
cropped = randomCrop(img, 448);
imwrite(cropped, '../data/small/300.3.png');

img = imread("../data/large/300.9.png");
cropped = randomCrop(img, 448);
imwrite(cropped, '../data/small/300.9.png');

img = imread("../data/large/600.3.png");
cropped = randomCrop(img, 448);
imwrite(cropped, '../data/small/600.3.png');

img = imread("../data/large/600.3_Al.png");
cropped = randomCrop(img, 448);
imwrite(cropped, '../data/small/600.3_Al.png');

img = imread("../data/large/600.9.png");
cropped = randomCrop(img, 448);
imwrite(cropped, '../data/small/600.9.png');

%%
%convert small images to ctf files
img = imread('small/200.3.png');
ebsd = image2EBSD(img, [targetStepSize * 2, targetStepSize * 2], CS_al);
ebsdImg.export('200.3_filled.ctf');

img = imread('small/300.3_filled_small.png');
ebsd = image2EBSD(img, [targetStepSize * 2, targetStepSize * 2], CS_copper);
ebsdImg.export('300.3_filled_small.ctf');

img = imread('small/600.3_AlFilled.png');
ebsd = image2EBSD(img, [targetStepSize * 2, targetStepSize * 2], CS_al);
ebsdImg.export('600.3_AlFilled.ctf');


img = imread('small/300.9.png');
ebsd = image2EBSD(img, [targetStepSize * 2, targetStepSize * 2], CS_al);
ebsdImg.export('300.9.ctf');

img = imread('small/600.9.png');
ebsd = image2EBSD(img, [targetStepSize * 2, targetStepSize * 2], CS_al);
ebsdImg.export('600.9.ctf');

img = imread('small/cu600.3.png');
ebsd = image2EBSD(img, [targetStepSize * 2, targetStepSize * 2], CS_al);
ebsdImg.export('cu600.3.ctf');

%%
img = imread('../data/small/300.3_full.png');
RGB_EBSD_Analysis(img, CS_copper, tgtStpSz);
%%
regen = imread('../data/result/300.3_full.png');
RGB_EBSD_Analysis(regen, CS_copper, tgtStpSz);
%%
regen = imread('../data/result/cleaned/300.3_full.png');
RGB_EBSD_Analysis(regen, CS_copper, tgtStpSz);
%%
img = imread('../data/small/200.3_full.png');
RGB_EBSD_Analysis(img, CS_al, tgtStpSz);
%%
regen = imread('../data/result/200.3_full.png');
RGB_EBSD_Analysis(regen, CS_al, tgtStpSz);

%%
regen = imread('../data/result/cleaned/200.3_full.png');
RGB_EBSD_Analysis(regen, CS_al, tgtStpSz);

%%
img = imread('../data/small/600.3_Al_full.png');
RGB_EBSD_Analysis(img, CS_al, tgtStpSz);
%%
regen = imread('../data/result/600.3_Al_full.png');
RGB_EBSD_Analysis(regen, CS_al, tgtStpSz);
%%
regen = imread('../data/result/cleaned/600.3_Al_full.png');
RGB_EBSD_Analysis(regen, CS_al, tgtStpSz);

%%
img = imread('../data/small/sim1_full.png');
RGB_EBSD_Analysis(img, CS_sim, tgtStpSz);
%%
regen = imread('../data/result/sim1_full.png');
RGB_EBSD_Analysis(regen, CS_sim, tgtStpSz);
%%
regen = imread('../data/result/sim1_full_cleaned.png');
RGB_EBSD_Analysis(regen, CS_sim, tgtStpSz);
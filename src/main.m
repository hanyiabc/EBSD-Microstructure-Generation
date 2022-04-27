pname = '../data/angs/';
pname_sim = '../data/sim_data/';
pname_al = '../data/angs/Al/';
pname_large = '../data/large/';
pname_ctf = '../data/ctf/';
pname_small = '../data/small/';
pname_cleaned = '../data/result/cleaned/';
pname_result = '../data/result/';
pname_comp= '../data/figures/comparison/';
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
fname1 = [pname '300.3.ang'];
fname2 = [pname '300.9.ang'];
fname3 = [pname '600.3.ang'];
fname4 = [pname '600.9.ang'];

fname_sim1 = [pname_sim 'ebsd1.ebsd.ctf'];
fname_sim2 = [pname_sim 'ebsd2.ebsd.ctf'];
fname_sim3 = [pname_sim 'ebsd3.ebsd.ctf'];

fname_al2003 = [pname_al '200.3 cleaned.ang'];
fname_al6003 = [pname_al '600.3 cleaned.ang'];
fname_al6009 = [pname_al '600.9 cleaned.ang'];
setMTEXpref('xAxisDirection','north');
setMTEXpref('zAxisDirection','outOfPlane');
%% 
% generate rgb images from EBSD orientations

angleM4 = plot_euler(fname1, CS_copper, stepSizeM4, targetStepSize, [pname_large, '300.3.png'], [pname_ctf, '300.3.ctf']);
angleM1 = plot_euler(fname2, CS_copper, stepSizeM1, targetStepSize, [pname_large, '300.9.png'], [pname_ctf, '300.9.ctf']);
angleM2 = plot_euler(fname3, CS_copper, stepSizeM2, targetStepSize, [pname_large, '600.3.png'], [pname_ctf, '600.3.ctf']);
angleM3 = plot_euler(fname4, CS_copper, stepSizeM3, targetStepSize, [pname_large, '600.9.png'], [pname_ctf, '600.9.ctf']);
%%
angleAl = plot_euler(fname_al2003, CS_al, stepSizeAl, targetStepSizeAl, [pname_large, '200.3.png'], [pname_ctf, '200.3.ctf']);
angleAl = plot_euler(fname_al6003, CS_al, stepSizeAl, targetStepSizeAl, [pname_large, '600.3_Al.png'], [pname_ctf, '600.3_Al.ctf']);
angleAl = plot_euler(fname_al6009, CS_al, stepSizeAl, targetStepSizeAl, [pname_large, '600.9_Al.png'], [pname_ctf, '600.9_Al.ctf']);
%%
angleSim1 = plot_euler(fname_sim1, CS_sim, stepSizeSim, targetStepSizeSim, [pname_small, 'sim1.png'], [pname_ctf, 'sim1.ctf']);
angleSim2 = plot_euler(fname_sim2, CS_sim, stepSizeSim, targetStepSizeSim, [pname_small, 'sim2.png'], [pname_ctf, 'sim2.ctf']);
angleSim3 = plot_euler(fname_sim3, CS_sim, stepSizeSim, targetStepSizeSim, [pname_small, 'sim3.png'], [pname_ctf, 'sim3.ctf']);
%%
%Crop generated rgb images randomly (full range)
img = imread([pname_large, '200.3.png']);
cropped = randomCrop(img, 448);
imwrite(cropped, [pname_small, '200.3.png'])

img = imread([pname_large, '300.3.png']);
cropped = randomCrop(img, 448);
imwrite(cropped, [pname_small, '300.3.png']);

img = imread([pname_large, '300.9.png']);
cropped = randomCrop(img, 448);
imwrite(cropped, [pname_small, '300.9.png']);

img = imread([pname_large, '600.3.png']);
cropped = randomCrop(img, 448);
imwrite(cropped,[pname_small, '600.3.png']);

img = imread([pname_large, '600.3_Al.png']);
cropped = randomCrop(img, 448);
imwrite(cropped, [pname_small, '600.3_Al.png']);

img = imread([pname_large, '600.9.png']);
cropped = randomCrop(img, 448);
imwrite(cropped, [pname_small, '600.9.png']);
%%
%convert small images to ctf files
img = imread([pname_small,'200.3.png']);
ebsd = image2EBSD(img, [targetStepSize * 2, targetStepSize * 2], CS_al);
ebsd.export([pname_ctf,'200.3.ctf']);

img = imread([pname_small,'600.3_Al.png']);
ebsd = image2EBSD(img, [targetStepSize * 2, targetStepSize * 2], CS_al);
ebsd.export([pname_ctf,'600.3_Al.ctf']);

img = imread([pname_small,'300.3.png']);
ebsd = image2EBSD(img, [targetStepSize * 2, targetStepSize * 2], CS_copper);
ebsd.export([pname_ctf,'300.3.ctf']);
s
img = imread([pname_small,'300.9.png']);
ebsd = image2EBSD(img, [targetStepSize * 2, targetStepSize * 2], CS_copper);
ebsd.export([pname_ctf,'300.9.ctf']);

img = imread([pname_small,'600.3.png']);
ebsd = image2EBSD(img, [targetStepSize * 2, targetStepSize * 2], CS_copper);
ebsd.export([pname_ctf,'600.3.ctf']);

img = imread([pname_small,'600.9.png']);
ebsd = image2EBSD(img, [targetStepSize * 2, targetStepSize * 2], CS_copper);
ebsd.export([pname_ctf,'600.9.ctf']);
%%
%perform analysis on the original data by reading a small RGB image


imgOri = imread([pname_small, 'sim1.png']);
imgGen = imread([pname_result, 'sim1.png']);
imgCle = imread([pname_cleaned, 'sim1.png']);
rgbEBSDAnalysis(imgOri, imgGen, imgCle, CS_sim, tgtStpSz, pname_comp, 'sim1');

imgOri = imread([pname_small, 'sim2.png']);
imgGen = imread([pname_result, 'sim2.png']);
imgCle = imread([pname_cleaned, 'sim2.png']);
rgbEBSDAnalysis(imgOri, imgGen, imgCle, CS_sim, tgtStpSz, pname_comp, 'sim2');

imgOri = imread([pname_small, 'sim3.png']);
imgGen = imread([pname_result, 'sim3.png']);
imgCle = imread([pname_cleaned, 'sim3.png']);
rgbEBSDAnalysis(imgOri, imgGen, imgCle, CS_sim, tgtStpSz, pname_comp, 'sim3');

imgOri = imread([pname_small, '300.3.png']);
imgGen = imread([pname_result, '300.3.png']);
imgCle = imread([pname_cleaned, '300.3.png']);
rgbEBSDAnalysis(imgOri, imgGen, imgCle, CS_copper, tgtStpSz, pname_comp, '300.3');

imgOri = imread([pname_small, '600.3.png']);
imgGen = imread([pname_result, '600.3.png']);
imgCle = imread([pname_cleaned, '600.3.png']);
rgbEBSDAnalysis(imgOri, imgGen, imgCle, CS_copper, tgtStpSz, pname_comp, '600.3');

imgOri = imread([pname_small, '300.9.png']);
imgGen = imread([pname_result, '300.9.png']);
imgCle = imread([pname_cleaned, '300.9.png']);
rgbEBSDAnalysis(imgOri, imgGen, imgCle, CS_copper, tgtStpSz, pname_comp, '300.9');

imgOri = imread([pname_small, '600.9.png']);
imgGen = imread([pname_result, '600.9.png']);
imgCle = imread([pname_cleaned, '600.9.png']);
rgbEBSDAnalysis(imgOri, imgGen, imgCle, CS_copper, tgtStpSz, pname_comp, '600.9');


imgOri = imread([pname_small, '200.3.png']);
imgGen = imread([pname_result, '200.3.png']);
imgCle = imread([pname_cleaned, '200.3.png']);
rgbEBSDAnalysis(imgOri, imgGen, imgCle, CS_al, tgtStpSz, pname_comp, '200.3');

imgOri = imread([pname_small, '600.3_Al.png']);
imgGen = imread([pname_result, '600.3_Al.png']);
imgCle = imread([pname_cleaned, '600.3_Al.png']);
rgbEBSDAnalysis(imgOri, imgGen, imgCle, CS_al, tgtStpSz, pname_comp, '600.3_Al');

%%
regen = imread('../data/result/300.3_full.png');
rgbEBSDAnalysis(regen, CS_copper, tgtStpSz);
%%
regen = imread('../data/result/cleaned/300.3_full.png');
rgbEBSDAnalysis(regen, CS_copper, tgtStpSz);
%%
img = imread('../data/small/200.3_full.png');
rgbEBSDAnalysis(img, CS_al, tgtStpSz);
%%
regen = imread('../data/result/200.3_full.png');
rgbEBSDAnalysis(regen, CS_al, tgtStpSz);
%%
regen = imread('../data/result/cleaned/200.3_full.png');
rgbEBSDAnalysis(regen, CS_al, tgtStpSz);
%%
img = imread('../data/small/600.3_Al_full.png');
rgbEBSDAnalysis(img, CS_al, tgtStpSz);
%%
regen = imread('../data/result/600.3_Al_full.png');
rgbEBSDAnalysis(regen, CS_al, tgtStpSz);
%%
regen = imread('../data/result/cleaned/600.3_Al_full.png');
rgbEBSDAnalysis(regen, CS_al, tgtStpSz);
%%
img = imread('../data/small/sim1_full.png');
rgbEBSDAnalysis(img, CS_sim, tgtStpSz);
%%
regen = imread('../data/result/sim1_full.png');
rgbEBSDAnalysis(regen, CS_sim, tgtStpSz);
%%
regen = imread('../data/result/sim1_full_cleaned.png');
rgbEBSDAnalysis(regen, CS_sim, tgtStpSz);
%%
img = imread('../data/small/sim3_full.png');
rgbEBSDAnalysis(img, CS_sim, tgtStpSz);
%%
regen = imread('../data/result/hist 1e-5.png');
rgbEBSDAnalysis(regen, CS_sim, tgtStpSz);
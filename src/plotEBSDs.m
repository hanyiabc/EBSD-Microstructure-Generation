pname = 'angs';
pname_sim = './sim_data';
pname_al = './angs/Al';
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
img = imread("full_range_large/200.3.png");
cropped = randomCrop(img, 448);
imwrite(cropped, 'small/200.3_full.png')

img = imread("full_range_large/300.3.png");
cropped = randomCrop(img, 448);
imwrite(cropped, 'small/300.3_full.png');

img = imread("full_range_large/300.9.png");
cropped = randomCrop(img, 448);
imwrite(cropped, 'small/300.9_full.png');

img = imread("full_range_large/600.3.png");
cropped = randomCrop(img, 448);
imwrite(cropped, 'small/600.3_full.png');

img = imread("full_range_large/600.3_Al.png");
cropped = randomCrop(img, 448);
imwrite(cropped, 'small/600.3_Al_full.png');

img = imread("full_range_large/600.9.png");
cropped = randomCrop(img, 448);
imwrite(cropped, 'small/600.9.png');

%%
img = imread("large/200.3.png");
cropped = randomCrop(img, 448);
imwrite(cropped, 'small/200.3.png')

img = imread("large/300.3.png");
cropped = randomCrop(img, 448);
imwrite(cropped, 'small/300.3.png');

img = imread("large/300.9.png");
cropped = randomCrop(img, 448);
imwrite(cropped, 'small/300.9.png');

img = imread("large/600.3.png");
cropped = randomCrop(img, 448);
imwrite(cropped, 'small/600.3.png');

img = imread("large/600.3_Al.png");
cropped = randomCrop(img, 448);
imwrite(cropped, 'small/600.3_Al.png');

img = imread("large/600.9.png");
cropped = randomCrop(img, 448);
imwrite(cropped, 'small/600.9.png');

%%
img = imread('small/200.3_filled.png');
ebsd = image2EBSD(img, [targetStepSize * 2, targetStepSize * 2], CS_al);
ebsdImg.export('200.3_filled.ctf');

img = imread('small/300.3_filled_small.png');
ebsd = image2EBSD(img, [targetStepSize * 2, targetStepSize * 2], CS_copper);
ebsdImg.export('300.3_filled_small.ctf');

img = imread('small/600.3_AlFilled.png');
ebsd = image2EBSD(img, [targetStepSize * 2, targetStepSize * 2], CS_al);
ebsdImg.export('600.3_AlFilled.ctf');
%%
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
RGB_EBSD_Analysis(angleM4, CS_copper, stepSizeM4);
% plot_to_EBSD(angleM1, CS_copper, stepSizeM1);
% plot_to_EBSD(angleM2, CS_copper, stepSizeM2);
% plot_to_EBSD(angleM3, CS_copper, stepSizeM3);
%%
RGB_EBSD_Analysis(angleSim1, CS_copper, stepSizeSim);
% plot_to_EBSD(angleSim2, CS_copper, stepSizeSim);
% plot_to_EBSD(angleSim3, CS_copper, stepSizeSim);
%%
% img = imread('300.3.png');
% plot_to_EBSD(img, CS_copper, stepSizeM4);
% 
% %%
% regen = imread('13.png');
% plot_to_EBSD(regen, CS_copper, stepSizeM4);
% regen = imread('1999.png');
% plot_to_EBSD(regen, CS_copper, stepSizeM4);
% %%
% regen = imread('1999_sim.png');
% plot_to_EBSD(regen, CS_copper, stepSizeSim);

%%
tgtStpSz = [targetStepSize targetStepSize];
%%
img = imread('./small/300.3_filled_small.png');
RGB_EBSD_Analysis(img, CS_copper, tgtStpSz);
%%
regen = imread('./result/300.3.png');
RGB_EBSD_Analysis(regen, CS_copper, tgtStpSz);
%%
regen = imread('./result/300.3_cleaned.png');
RGB_EBSD_Analysis(regen, CS_copper, tgtStpSz);
%%
img = imread('./small/200.3_filled.png');
RGB_EBSD_Analysis(img, CS_al, tgtStpSz);
%%
regen = imread('./result/al2003_filled_regen.png');
RGB_EBSD_Analysis(regen, CS_al, tgtStpSz);

%%
regen = imread('./result/al2003_filled_regen_cleaned.png');
RGB_EBSD_Analysis(regen, CS_al, tgtStpSz);

%%
img = imread('./small/600.3_AlFilled.png');
RGB_EBSD_Analysis(img, CS_al, tgtStpSz);
%%
regen = imread('./result/al6003_filled_regen.png');
RGB_EBSD_Analysis(regen, CS_al, tgtStpSz);
%%
regen = imread('./result/al6003_filled_regen_cleaned.png');
RGB_EBSD_Analysis(regen, CS_al, tgtStpSz);

%%
img = imread('./small/sim1.png');
RGB_EBSD_Analysis(img, CS_sim, tgtStpSz);
%%
regen = imread('./result/sim_tv.png');
RGB_EBSD_Analysis(regen, CS_sim, tgtStpSz);
%%
regen = imread('./result/sim_tv_cleaned.png');
RGB_EBSD_Analysis(regen, CS_sim, tgtStpSz);
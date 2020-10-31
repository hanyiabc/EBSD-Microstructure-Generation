%start by importing the original EBSD data so that there is a 
% "ebsd" variable stored as well as CS and all that

%establish grains
[grains, ebsd.grainId] = calcGrains(ebsd)
%can adjust this depending on target resolution
unitCell = [-.2 -.2; -.2 .2; .2 .2; .2 -.2];
%creates EBSDsquare variable for the ebsd data
ebsdS = ebsd('Aluminum').gridify('unitCell',unitCell);

plot(ebsdS,ebsdS.orientations)
hold on
plot(grains.boundary,'lineWidth',2)
hold off

%fills in the empty space for a higher resolution square grid
F = halfQuadraticFilter;
F.alpha = 0.5;
ebsdS2 = smooth(ebsdS,F,'fill',grains)

figure
plot(ebsdS2('indexed'),ebsdS2('indexed').orientations)
hold on
plot(grains.boundary,'lineWidth',2)
hold off

%regridify the data so it is saved on a square matrix rather than vector
ebsdSmain = ebsdS2('indexed').gridify('unitCell',unitCell);

%pull out the rotations/euler angles and save to new variable as a fraction
% 0-1, which will be converted to RGB values
eulers = ebsdSmain.rotations;
rgb(:,:,1) = eulers.phi1/(2*pi);
rgb(:,:,2) = eulers.Phi/(2*pi);
rgb(:,:,3) = eulers.phi2/(2*pi);

%display the rgb image 
figure
image(rgb)

%save the RGB image
imwrite(rgb,'rgbconstruct.tif','tif');
imfinfo('rgbconstruct.tif')

%now MATLAB reads the individual RGB values of the pixels and puts them
%into a matrix which I call the simulated euler values
rearray = imread('rgbconstruct.tif');
rearray = im2double(rearray);
eulerSIM = rearray(:,:,:)*(2*pi);


%NEEDS WORK
% Here I am trying to take the "generated" euler angles and put them into 
% a usable EBSD file format. Here the goal is to use the "generic" file 
% type referenced on MTEX, but I still can't get it to work...

% % % % x1 y1 phi1_1 Phi_1 phi2_1 phase_1
% % % % x2 y2 phi1_2 Phi_2 phi2_2 phase_2
% % % % x2 y3 phi1_3 Phi_3 phi2_3 phase_3
% % % % .      .       .       .
% % % % .      .       .       .
% % % % .      .       .       .
% % % % xM yM phi1_M Phi_M phi2_M phase_m

% Puts the euler angles into the correct columns
for i=1:3
    b=eulerSIM(:,:,i);
    b = b';
    c = b(:);
    j=i+2;
    generic(:,j) = c;
end
% 
% Creates the x position, for this example the step size after gridify was
%     0.4 microns
xvector = 0:0.4:212.8;
xvector = xvector(:);
% 
% Repeats the x steps through the full number of pixels and adds y steps
i=1;
for i = 1:683
    yvector = (i-1)*.4;
    j=(i*533);
    k=(i*533)-532;
    generic(k:j,1)=xvector;
    generic(k:j,2)=yvector;
end
% 
% Assign all data points to the same phase
generic(:,6) = 0;

% Save the file as a .txt
save('generated.txt','generic','-ASCII')


% Everything below here is various ways I was trying to import the data,
% but I always run into issues with the replotted data being really messed up
% to the point I think it is just being read wrong...

fname = fullfile(mtexDataPath,'EBSD','generated.txt');
SS = specimenSymmetry('triclinic');

ebsdSIM = EBSD.load(fname,CS,'interface','generic',...
  'ColumnNames', { 'x' 'y' 'phi1' 'Phi' 'phi2' 'Phase'}, 'Bunge', 'Radians');


ebsdSIM = loadEBSD_generic('generated.txt','CS',CS,'SS',SS,'ColumnNames',{'x','y','phi1','Phi','phi2','phase'},'Bunge')

ebsd5 = loadEBSD_generic('ebsdsim.txt','CS',CS,'ColumnNames',{'x','y','Euler1','Euler2','Euler3','phase'},'radians','Bunge')


ebsdSIM2 = ebsdSIM('Aluminum').gridify('unitCell',unitCell);

plot(ebsdSIM2,ebsdSIM2.orientations)
plot(ebsdSIM('indexed'),ebsdSIM('indexed').orientations)
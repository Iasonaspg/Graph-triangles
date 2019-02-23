%% CLEAN-UP

clear
close all


%% PARAMETERS

basePath  = 'https://sparse.tamu.edu/mat';
folderPath = '/home/johnfli/Code/PD_4/Data';
groupName = 'DIMACS10';
matName   = 'delaunay_n22'; % auto|great-britain_osm|delaunay_n22

%% (BEGIN)

fprintf('\n *** begin %s ***\n\n',mfilename);

%% LOAD INPUT GRAPH

fprintf( '...loading graph...\n' ); 
fprintf( '   - %s/%s\n', groupName, matName )
filesep = '/';

matFileName = [groupName '_' matName '.mat'];
csvFileName = [groupName '_' matName '_B_COO.csv'];
validationFileName = [groupName '_' matName '_validation_file.csv'];

if ~exist( matFileName, 'file' )
  fprintf('   - downloading graph...\n')
  matFileName = websave( [folderPath matFileName], [basePath filesep groupName filesep matName '.mat'] );
  fprintf('     DONE\n')
end

% Read file and data

ioData  = matfile( matFileName );
Problem = ioData.Problem;

% keep only adjacency matrix (logical values)
A = Problem.A > 0;

%% TRIANGLE COUNTING

fprintf( '...triangle counting...\n' ); 

ticCnt = tic;
B = A^2;
nT = full( sum( sum(B.* A) ) / 6 );
matlab_time = toc(ticCnt);

fprintf( '   - DONE: %d triangles found in %.5f sec\n', nT, matlab_time );


%% SAVE SPARSE MATRIX INTO COO FORMAT INTO .CSV FILE
   
[columns, rows, values] = find(B);

rows = rows - 1;
columns = columns - 1;

fprintf( '   - Writing CSV has started\n');

dlmwrite([folderPath csvFileName], [values rows columns], 'delimiter', ',', 'precision', 9);

N = length(A);
M = length(rows)/2;

clear Problem;

fprintf( '   - CSV Created\n');

%% SAVE RESULTS INTO VALIDATION FILE

dlmwrite([folderPath validationFileName], [N M nT matlab_time], 'delimiter', ',', 'precision', 9);


%% (END)

fprintf('\n *** end %s ***\n\n',mfilename);
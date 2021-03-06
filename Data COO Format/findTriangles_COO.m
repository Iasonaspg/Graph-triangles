%
% SCRIPT: FINDTRIANGLES
%
%   Download a graph from Sparse Matrix Collection and count the number of
%   triangles.
%

%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%   John Flionis
%   Michail Iason Pavlidis
%
% VERSION
%
%   0.1 - January 21, 2019
%


%% CLEAN-UP

clear
close all


%% PARAMETERS

basePath  = 'https://sparse.tamu.edu/mat';
folderPath = './';
groupName = 'DIMACS10';
matName   = 'great-britain_osm'; % auto|great-britain_osm|delaunay_n22

%% (BEGIN)

fprintf('\n *** begin %s ***\n\n',mfilename);

%% LOAD INPUT GRAPH

fprintf( '...loading graph...\n' ); 
fprintf( '   - %s/%s\n', groupName, matName )
filesep = '/';

matFileName = [groupName '_' matName '.mat'];
csvFileName = [groupName '_' matName '_COO.csv'];

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

% SAVE INTO .CSV FORMAT

[lines, columns] = find(A);

lines = lines - 1;
columns = columns - 1;
B = [columns lines];
tr = transpose(B);

fprintf( '   - Writing CSV has started\n');

fileID = fopen( [folderPath csvFileName], 'w');
fprintf(fileID,'%d,%d\n',tr);

N = length(Problem.A);
M = length(lines)/2;

clear Problem;
fclose(fileID);

fprintf( '   - CSV Created\n');

%% TRIANGLE COUNTING

fprintf( '...triangle counting...\n' ); 

ticCnt = tic;
nT = full( sum( sum( A^2 .* A ) ) / 6 );
matlab_time = toc(ticCnt);

fprintf( '   - DONE: %d triangles found in %.5f sec\n', nT, matlab_time );

%% (END)

fprintf('\n *** end %s ***\n\n',mfilename);


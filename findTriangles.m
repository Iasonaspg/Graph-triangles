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
folderPath = '/home/johnfli/Code/PD_4/Data';
groupName = 'DIMACS10';
<<<<<<< HEAD
matName   = 'delaunay_n12'; % auto|great-britain_osm|delaunay_n22
=======
matName   = 'delaunay_n10'; % auto|great-britain_osm|delaunay_n22
>>>>>>> origin/john

%% (BEGIN)

fprintf('\n *** begin %s ***\n\n',mfilename);

%% LOAD INPUT GRAPH

fprintf( '...loading graph...\n' ); 
fprintf( '   - %s/%s\n', groupName, matName )
filesep = '/';

matFileName = [groupName '_' matName '.mat'];
csvFileName = [groupName '_' matName '.csv'];
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

%% SAVE SPARSE MATRIX INTO CSR FORMAT INTO .CSV FILE
   
[columns, rows, values] = find(A);

<<<<<<< HEAD
[lines, columns, values] = find(A);

lines = lines - 1;
columns = columns - 1;
B = [columns lines values];
tr = transpose(B);

fprintf( '   - Writing CSV has started\n');

fileID = fopen( [folderPath csvFileName], 'w');
fprintf(fileID,'%d,%d,%d\n',tr);

N = length(Problem.A)
M = length(lines)/2;
=======
csrValA = values;
csrRowPtrA = [0; full(cumsum(sum(A,2)))];
csrColIndA = columns - 1;

fprintf( '   - Writing CSV has started\n');

dlmwrite([folderPath csvFileName], csrValA', 'delimiter', ',', 'precision', 9);
dlmwrite([folderPath csvFileName], csrRowPtrA', '-append', 'delimiter', ',', 'precision', 9);
dlmwrite([folderPath csvFileName], csrColIndA', '-append', 'delimiter', ',', 'precision', 9);

N = length(A);
M = length(rows)/2;
>>>>>>> origin/john

clear Problem;

fprintf( '   - CSV Created\n');

%% TRIANGLE COUNTING

fprintf( '...triangle counting...\n' ); 

ticCnt = tic;
nT = full( sum( sum( A^2 .* A ) ) / 6 );
matlab_time = toc(ticCnt);

fprintf( '   - DONE: %d triangles found in %.5f sec\n', nT, matlab_time );

<<<<<<< HEAD
dlmwrite([folderPath validationFileName], [N M nT matlab_time], 'delimiter', ',', 'precision', 9);
=======
%% SAVE RESULTS INTO VALIDATION FILE

dlmwrite([folderPath validationFileName], [N M nT matlab_time], 'delimiter', ',', 'precision', 9);

>>>>>>> origin/john

%% (END)

fprintf('\n *** end %s ***\n\n',mfilename);

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
folderPath = '~/Code/PD_4/Data/';
groupName = 'DIMACS10';
matName   = 'auto'; % auto|great-britain_osm|delaunay_n22

%% (BEGIN)

fprintf('\n *** begin %s ***\n\n',mfilename);

%% LOAD INPUT GRAPH

fprintf( '...loading graph...\n' ); 
fprintf( '   - %s/%s\n', groupName, matName )
fprintf( 'Filesep: %s\n', filesep)
filesep = '/';

matFileName = [groupName '_' matName '.mat'];
csvFileName = [groupName '_' matName '.csv'];
validationFileName = [groupName '_' matName '_validation_file.csv'];

if ~exist( matFileName, 'file' )
  fprintf('   - downloading graph...\n')
  matFileName = websave( [folderPath matFileName], [basePath filesep groupName filesep matName '.mat'] );
  fprintf('     DONE\n')
end

ioData  = matfile( matFileName );
Problem = ioData.Problem;

% keep only adjacency matrix (logical values)
A = Problem.A > 0;


%% SAVE INTO .CSV FORMAT

N = length(A);
M = full(sum(sum(A(1:end,1:end))));

B=[]
B(1,1:3) = [N N M];
k = 2;
for i = [1:N]
	for j = [1:N]
		if A(i,j)>0
			B(k,1:3) = [i j A(i,j)];
			k = k + 1;
		end
	end
end
dlmwrite([folderPath csvFileName], B, 'delimiter', ',', 'precision', 9);

clear Problem;

fprintf( '   - DONE\n');

%% TRIANGLE COUNTING

fprintf( '...triangle counting...\n' ); 

		ticCnt = tic;
nT = full( sum( sum( A^2 .* A ) ) / 6 );
		matlab_time = toc(ticCnt);

fprintf( '   - DONE: %d triangles found in %.5f sec\n', nT, matlab_time );

dlmwrite([folderPath validationFileName], [nT matlab_time], 'delimiter', ',', 'precision', 9);

%% (END)

fprintf('\n *** end %s ***\n\n',mfilename);



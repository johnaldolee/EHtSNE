function [c,ndx1] = coranking(hdpd,ldpd,ndx1)
% Function 
%
% c = coranking(hdpd,ldpd)
%
% computes the coranking matrix starting from the matrices of pairwise
% distances hdpd and ldpd (in the high- and low-dimensional spaces, 
% respectively). Be careful, the output is made of uint32 integers.
%
% References:
% [1] John A. Lee, Michel Verleysen.
%     Rank-based quality assessment of nonlinear dimensionality reduction.
%     Proc. 16th ESANN, Bruges, pp 49-54, April 2008.
% [2] John A. Lee, Michel Verleysen.
%     Quality assessment of nonlinear dimensionality reduction: 
%     rank-based  criteria.
%     Neurocomputing, 72(7-9):1431-1443, March 2009.
%
% Copyright J. A. Lee, March 24, 2014.

if nargin<3, ndx1 = []; end

if isempty(ndx1)
    if ~isdist(hdpd,0)
        error('Invalid distance matrix in hdpd.');
    end
end
if ~isdist(ldpd,0)
    error('Invalid distance matrix in ldpd.');
end

% check size equality
tmp1 = uint32(size(hdpd));
tmp2 = uint32(size(ldpd));
if any(tmp1~=tmp2)
    error('The matrices hdpd and ldpd do not have the same sizes.');
end
nbr = tmp1(1); % full size (N)
sss = tmp1(2); % subset size
if nbr<sss
    disp('Warning in coranking: transposing upper blocks into left blocks.')
    [c,ndx1] = coranking(hdpd',ldpd',ndx1');
    return
end

% GPU flag
try
    gpuf = (gpuDeviceCount>0) && (numel(hdpd)>7500^2); % sort is rather unefficient on GPU => for large matrices maybe...
catch
    gpuf = false;
end
if gpuf
    gpud = gpuDevice;
    gpuf = (gpud.FreeMemory>4*8*numel(hdpd)) || (isa(hdpd,'gpuArray') && isa(ldpd,'gpuArray')); % ... if memory is large enough
end

% compute sorting permutations
if gpuf
    if isempty(ndx1)
        [~,ndx1] = sort(gpuArray(hdpd),1);
        ndx1 = uint32(gather(ndx1));
    end
    [~,ndx2] = sort(gpuArray(ldpd),1);
    ndx2 = uint32(gather(ndx2));
else
    if isempty(ndx1)
        [~,ndx1] = sort(hdpd,1);
        ndx1 = uint32(ndx1);
    end
    [~,ndx2] = sort(ldpd,1);
    ndx2 = uint32(ndx2);
end

% compute the corresponding ranks (in LD space only)
ndx4 = zeros(nbr,sss,'uint32');
v1n = (1:nbr)';
for j = 1:sss
    ndx4(ndx2(:,j),j) = v1n;
end
clear ndx2;

% initialize the coranking matrix to zero
c = zeros(nbr,nbr,'uint32');

% compute the coranking matrix (successive vectorial increments)
for j = 1:sss
    h = v1n + nbr*(ndx4(ndx1(v1n,j),j) - 1);
    c(h) = c(h) + 1;
end

% remove useless first row and column
c = c(2:end,2:end);

% % older naive implementation
% 
% % compute sorting permutations
% [~,ndx1] = sort(hdpd,1);
% [~,ndx2] = sort(ldpd,1);
% ndx1 = uint32(ndx1);
% ndx2 = uint32(ndx2);
% 
% % compute the corresponding ranks
% % ndx3 = zeros(nbr,nbr,'uint32');
% ndx4 = zeros(nbr,sss,'uint32');
% v1n = (1:nbr)';
% for j = 1:sss
%     % ndx3(ndx1(i,j),j) = v1n; % not necessary, see next loop
%     ndx4(ndx2(:,j),j) = v1n;
% end
% clear ndx2;
% 
% % initialize the coranking matrix to zero
% c = zeros(nbr,nbr,'uint32');
% 
% % compute the coranking matrix (successive increments)
% for j = 1:sss
%     for i = 1:nbr
%         % more efficient 
%         % (this requires knowing the ranks in one space only)
%         h = ndx4(ndx1(i,j),j); % i replaces the rank given by ndx3
%         c(i,h) = c(i,h) + 1;
%     end
% end
% 
% % remove useless first row and column
% c = c(2:end,2:end);

% % older naive implementation
% 
% % compute sorting permutations
% [~,ndx1] = sort(hdpd,1);
% [~,ndx2] = sort(ldpd,1);
% ndx1 = uint32(ndx1);
% ndx2 = uint32(ndx2);
% 
% % compute the corresponding ranks
% % [~,ndx3] = sort(ndx1); % not necessary, see next loop
% [~,ndx4] = sort(ndx2);
% % ndx3 = uint32(ndx3);
% ndx4 = uint32(ndx4);
% clear ndx2;
% 
% % initialize the coranking matrix to zero
% c = zeros(nbr,nbr,'uint32');
% 
% % compute the coranking matrix (successive increments)
% for j = 1:sss
%     for i = 1:nbr
%         % more efficient 
%         % (this requires knowing the ranks in one space only)
%         h = ndx4(ndx1(i,j),j); % i replaces the rank given by ndx3
%         c(i,h) = c(i,h) + 1;
%     end
% end
% 
% % remove useless first row and column
% c = c(2:end,2:end);

% % oldest naive and straightforward implementation
% 
% % compute sorting permutations
% [~,ndx1] = sort(hdpd,1);
% [~,ndx2] = sort(ldpd,1);
% 
% % compute the corresponding ranks
% [~,ndx3] = sort(ndx1);
% [~,ndx4] = sort(ndx2);
% clear ndx1;
% clear ndx2;
% 
% % initialize the coranking matrix to zero
% c = zeros(nbr,nbr,'uint32');
% 
% % compute the coranking matrix (successive increments)
% for j = 1:sss
%     for i = 1:nbr
%         % strictly according the coranking definition 
%         % (this requires knowing the ranks in both spaces)
%         k = ndx3(i,j);
%         l = ndx4(i,j);
%         c(k,l) = c(k,l) + 1;
%     end
% end
% 
% % remove useless first row and column
% c = c(2:end,2:end);



function [n,x,p,b] = nx_trusion(c)
% Function
% 
% [n,x,p,b] = nx_trusion(c)
% 
% computes the intrusion and extrusion rates according to the coranking
% matrix in c. The outputs n and x denote the intruction and extrusion
% rates as a function of K, the size of the K-ary neighbourhoods. 
% The outputs p and b are the rate of perfectly preserved ranks and the
% baseline that corresponds to the overlap between two random K-ary
% neighbourhoods.
% 
% The local continuity meta-criterion [2] is obtained as follows:
% LCMC = n + x + p - b.
%
% The quality and behavior criteria [3] are obtained as follows:
% Q_NX = n + x + p    (the higher the better)
% B_NX = x - n        (positive = intrusive, negative = extrusive)
%
% References:
% [1] John A. Lee, Michel Verleysen.
%     Rank-based quality assessment of nonlinear dimensionality reduction.
%     Proc. 16th ESANN, April 2008, Bruges, pp 49-54.
% [2] L. Chen and A. Buja. 
%     Local multidimensional scaling for nonlinear dimensionality reduction,
%     graph layout, and proximity analysis. 
%     PhD thesis, University of Pennsylviana, July 2006.
% [3] John A. Lee, Michel Verleysen.
%     Quality assessment of nonlinear dimensionality reduction: 
%     rank-based  criteria.
%     Neurocomputing, 72(7-9):1431-1443, March 2009.
%
% Copyright J. A. Lee, December 12, 2024.

% check the coranking matrix
tmp = size(c);
if tmp(1)~=tmp(2)
    error('The coranking matrix is not square.');
end
nmo = tmp(1); % N-1
sss = sum(c,1); % M
if any(sss(1)~=sss) || any(sss>nmo+1)
    warning('The coranking matrix is not valid.');
end
sss = sss(1);

% compute K = [1,...,N-1]^T and K*M = M*[1,...,N-1]^T
v1 = (1:nmo)';
v2 = sss*v1;

% initialise outputs
n = zeros(nmo,1);
x = zeros(nmo,1);
p = cumsum(double(diag(c))) ./ v2;
b = v1 ./ nmo;

% intrusion and extrusion rates
%parfor k = 2:nmo % parfor does not improve on regular for on a multicore architecture
for k = 2:nmo
    v3 = 1:k-1;
    n(k) = sum(c(k,v3));
    x(k) = sum(c(v3,k));
end

% accumulation and normalisation
n = cumsum(n) ./ v2;
x = cumsum(x) ./ v2;



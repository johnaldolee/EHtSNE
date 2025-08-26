function b = isdist(D,lft,asy,vbs)
% Function
%
% b = isdist(D,lft,asy)
%
% checks that D is a valid (left block of a) distance matrix.
%
% Input: 
%   D  : D is the matrix to be tested
%   lft: if lsb is specified and true, isdist also considers left
%        sub-blocks as being valid
%   asy: if asy is specified and true, isdist also considers asymmetric
%        matrices
%   vbs: verbose (display warnings)
% Output: 
%   b  : Boolean flag indicating whether D is a valid distance matrix
%
% Copyright J. A. Lee, December 27, 2024.

% check args
if nargin<4, vbs = 0; end
if nargin<3, asy = 0; end
if nargin<2, lft = 0; end

% size
[nbr,sss] = size(D);

% positivity and diagonal
pty = all(D(:)>=0);
dnl = all(diag(D)<eps);
%if ~(pty && dnl), b = false; return; end

% depending on form factor...
if nbr<sss
    % D has more columns than rows
    if ~lft
        smy = false;
    else
        if vbs, disp('Warning in isdist: transposing upper block into left block.'); end
        S = D(1:nbr,1:nbr);
        smy = all(all(abs(S-S')<eps));
    end
elseif nbr>sss
    % D has more rows than columns
    if ~lft
        smy = false;
    else
        S = D(1:sss,1:sss);
        smy = all(all(abs(S-S')<eps));
    end
else
    % D is square
    smy = all(all(abs(D-D')<eps));
end

% warning messages
zrs = sum(D(:)<eps); % zeros
if zrs>min(nbr,sss) && vbs, disp('Warning in isdist: possible duplicates!'); end
if vbs
    if ~dnl, disp('Warning in isdist: diagonal is not all zero'); end
    if ~pty, disp('Warning in isdist: distance matrix is not positive'); end
    if ~smy, disp('Warning in isdist: distance matrix is not symmetric'); end
end

% output
b = (nbr==sss || ~lft) && dnl && (smy || asy) && pty;



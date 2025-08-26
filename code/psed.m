function DX2 = psed(X,ss)
% pairwise squared Euclidean distances
%
% Copyright J.A.Lee, February 6, 2013.

if nargin<2
    % % faster but inaccurate, since the same subtracted dot products are
    % % computed in two different ways
    % DX2 = sum(X.^2,2);
    % DX2 = bsxfun(@plus, DX2, DX2') - (2*X)*X';
    DX2 = X * X';
    DX2 = bsxfun(@minus,diag(DX2),DX2);
    DX2 = DX2 + DX2';
else
    DX2 = sum(X.^2,2);
    DX2 = bsxfun(@plus, DX2, DX2(ss)') - X*(2*X(ss,:)');
    %DX2 = X * X(ss,:)';
    %DX2 = bsxfun(@plus, sum(X.^2,2), diag(DX2)') - 2*DX2;
end

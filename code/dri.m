function [idx,DX,nbr,sss,L,shw,Y,dim,X] = dri(DX,dim,dtm,wht)
% Function
%
% [idx,DX,nbr,sss,L,shw,Y,dim,X] = dri(DX,dim,dtm,wht)
%
% initializes a few things (distances, embedded coordinates) 
% for dimensionality reduction
%
% Copyright J. A. Lee, December 1, 2024.

% check arguments
if nargin<3, dtm = true; end
if nargin<4, wht = false; end

% check whether data labels are presents
if iscell(DX)
    shw = true;
    L = DX{2};
    DX = DX{1};
else
    L = [];
    shw = false;
end

% ckeck the distance matrix
if isdist(DX,1,1) % allow incomplete matrices (left blocks) and asymmetry
    disp('Using pairwise distances')

    % PCA/MDS recovery of X

    % size of data set
    [nbr,sss] = size(DX);
    
    % compute scalar products from squared distances
    if nbr==sss
        % simultaneous double centering working for square matrices only
        S0 = (DX+DX').^2; % if asymmetric
        sS = sum(S0,1)./nbr;
        S0 = -1/2*(S0 - bsxfun(@plus, sS, sS') + sum(sS)/nbr); % perfectly symmetric
        
        % compute coordinates with MDS (eigenvalue decomposition of Gram)
        [V,D] = eigs(S0,dim);
    else
        % sequential double centering working for left blocks of distance matrix
        S0 = -1/2*DX.^2; % do not correct for asymmetry ???
        S0 = bsxfun(@minus, S0, sum(S0,1)/nbr);
        S0 = bsxfun(@minus, S0, sum(S0,2)/sss); % not perfectly symmetric
        
        % compute coordinates with SVD decomposition
        [V,D] = svds(S0,dim);
    end
    
    % sort eigenvalues and eigenvectors, truncate
    [D,P] = sort(diag(D),'descend');
    thr = find(cumsum(D) >= 0.999*sum(abs(D)));
    if isempty(thr)
        disp('Warning, PCoA Gram matrix might have negative eigenvalues! Disregarding...');
        D = diag(sqrt(abs(D)));
    else
        disp('Warning, PCoA Gram matrix with fewer significant eigenvalues than requested embedding dimension! Disregarding...');
        if thr(1)<2
            V = V(:,P(1:2));
            D = diag(sqrt(abs(D(1:2))));
        else
            dim = thr(1);
            V = V(:,P(1:dim));
            D = diag(sqrt(abs(D(1:dim))));
        end
    end
    
    % scaled coordinates (PCA/MDS)
    X = V * D;
else
    disp('Converting coordinates into pairwise distances')
    X = DX;
    DX = sqrt(psed(DX));
    
    % size of data set
    [nbr,sss] = size(DX);
end

% give up if there are more than one connected component
if any(DX(:)==inf)
    error('More than one connected component!');
end

%--------------------------------------------------------------------------

% make it deterministic

% sort everything according to mass center (computed from distances)
% (this makes the solution independent of the ordering of data)

if dtm
    % sort sums of squared distances between all "landmarks"
    [~,idx] = sort(sum(DX(1:sss,:).^2,2));
    
    % permute rows and columns for distances
    DX          = DX(:,idx); % permute columns
    DX(1:sss,:) = DX(idx,:); % permute rows
    
    % if incomplete distance matrix, then sort remaining rows
    if sss<nbr
        % sort sums of squared distances to all "landmarks"
        [~,tmp] = sort(sum(DX(sss+1:end,:).^2,2));
        
        % permute rows
        DX(sss+1:end,:) = DX(sss+tmp,:); % permute rows
        idx = [idx;sss+tmp];
    end
    
    % permute rows for coordinates
    X = X(idx,:);
    
    % permute labels accordingly
    if shw
        L = L(idx);
    end
else
    idx = 1:nbr;
end

%--------------------------------------------------------------------------

% check initialization type
if size(dim,1)==nbr
    disp('Initializing with given low-dimensional coordinates');
    Y = dim(idx,:); % with permutation determined above
    dim = size(Y,2);
else
    disp('Initializing with principal components');
    dim = ceil(abs(dim(1)));
    
    % PCA/MDS initialization

    % compute scalar products from squared distances
    if nbr==sss
        % simultaneous double centering working for square matrices only
        S0 = DX.^2;
        sS = sum(S0,1)./nbr;
        S0 = -1/2*(S0 - bsxfun(@plus, sS, sS') + sum(sS)/nbr); % perfectly symmetric
        
        % compute coordinates with MDS (eigenvalue decomposition of Gram)
        [V,D] = eigs(S0,dim);
    else
        % sequential double centering working for left blocks of distance matrix
        S0 = -1/2*DX.^2;
        S0 = bsxfun(@minus, S0, sum(S0,1)/nbr);
        S0 = bsxfun(@minus, S0, sum(S0,2)/sss); % not perfectly symmetric
        
        % compute coordinates with SVD decomposition
        [V,D] = svds(S0,dim);
    end
    
    % sort eigenvalues and eigenvectors, truncate
    [D,P] = sort(diag(D),'descend');
    thr = find(cumsum(D) >= 0.999*sum(abs(D)));
    if isempty(thr)
        disp('Warning, PCoA Gram matrix might have negative eigenvalues! Disregarding...');
        D = diag(sqrt(abs(D)));
    else
        dim = max(2,min(dim, thr(1))); % dim can be reduced here
        V = V(:,P(1:dim));
        D = diag(sqrt(abs(D(1:dim))));
    end

    if wht
        % whitened coordinates
        Y = sqrt(nbr)*V;
    else
        % scaled coordinates (PCA/MDS)
        Y = V * D;
    end
end


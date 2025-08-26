function [Y,T,E,Ys,Xs] = ehtsne(X,pxt,itr,dof,shw,own,reg)
% t-SNE with early hierarchization
% A space partitioning binary tree introduce of growing number of points
% in the embedding: 4, 8, 16, 32, ..., N with the specified perplexity pxt 
% (or lower in the first stages (4, 8, ...)). 
% In those first stages, the perplexity is thus comparatively larger wrt N,
% making it easier to preserve the global structure.
% The growing number of points is analogous to cells dividing themselves
% in the development of an embryo => the early embryo embedding ;-)
% The approach is compatible with regular or BH t-SNE, or even any other
% method of NE that accepts initial coordinates for the embedding Y.
% Inputs:
%   X  : (HD) data coordinates 
%   pxt: perplexity
%   itr: number of iterations
%   dof: degrees of freedom for the Student t hyperbolic function
%   shw: display scatter plots with intermediary results
%   own: calls basictsne instead of tsne (exact & B-H dep.on size)
%   reg: NEW (hidden)(works for own basictsne only)
% Outputs:
%   Y  : embedding coordinates
%   T  : space partitioning binary tree (obtained with binatree)
%   E  : energies (KL divergence values for all development stages)
%   Ys : cell array of intermediate embeddings
%
% Copyright John A. Lee, August 9, 2025.

if nargin<7, reg = 0; end
if nargin<6, own = false; end
if nargin<5, shw = false; end
if nargin<4, dof = 1; end
if nargin<3, itr = 1000; end
if nargin<2, pxt = 30; end

% sizes
[nbr,dim] = size(X);

% PCA
Xc = bsxfun(@minus, X, mean(X,1));
[U,S] = svds(Xc,min(dim,64));
X = U*S; % X gets modified
dim = size(X,2);

% space partitioning binary tree
T = binatree(X,1);

% determine prototype index for each tree branch
ptt = zeros(size(T.P,1),1);
for t = 2:size(T.P,1)
    v = T.P{t};
    Xt = X(v,:);
    aXt = mean(Xt);
    d = sum(bsxfun(@minus,Xt,aXt).^2,2);
    [~,i] = min(d);
    ptt(t) = v(i);
end

pw2 = T.pw2; %log2(size(T.P,1)/2);
if nargout>3
    Xs = cell(pw2,1);
    Ys = cell(pw2,1);
end
E = zeros(itr,pw2);
I = zeros(nbr,1); % index of permutation
for p = 2:pw2+1
    % hierarchical level
    if p==2
        Xp = X(ptt(2^p+1:2*2^p),:);
        Yp = 0.01*[-1,+1;+1,+1;+1,-1;-1,-1];%randn(2^p,2);
        M1p = zeros(2^p,2);
        M2p = zeros(2^p,2);
    elseif p<=pw2 % should be < not <=
        Xp = X(ptt(2^p+1:2*2^p),:);
        np = size(Yp,1);
        Yp = bsxfun(@minus, Yp, mean(Yp,1)); % recenter
        %vp = mean(sum(Yp.^2,2),1); % variance
        vv = [1:np;1:np];
        Yp = Yp(vv,:);% + sqrt(vp/np)/6*randn(2*np,2);
        M1p = M1p(vv,:);
        M2p = M2p(vv,:);
    else % p==pw2+1 uneven leafs?
        Yl = Yp;
        M1l = M1p;
        M2l = M2p;
        %vp = mean(sum(Yp.^2,2),1); % variance
        Xp = zeros(nbr,dim);
        Yp = zeros(nbr,2);
        M1p = zeros(nbr,2);
        M2p = zeros(nbr,2);
        j = 0;
        for i = 1:T.tp2
            k = numel(T.L{i});
            I(j+(1:k)) = T.L{i};
            Xp(j+(1:k),:) = X(T.L{i},:);
            Yp(j+(1:k),:) = bsxfun(@plus, Yl(i,:), zeros(k,2));%sqrt(vp/nbr)/6*randn(k,2));
            M1p(j+(1:k),:) = bsxfun(@plus, M1l(i,:), zeros(k,2));
            M2p(j+(1:k),:) = bsxfun(@plus, M2l(i,:), zeros(k,2));
            j = j + k;
        end
    end
    %Yp = 1e-4*Yp; % scale embedding slightly down to "reload" t-SNE 
    %Yp = sqrt(2)*Yp; % compensate for twice the population in 2D


    % run t-SNE for a few iterations
    pp = max(2,min(2^p/2,pxt)); % perplexity for this tree level
    %pp = 2*((pxt/2).^(1/(pw2+1))).^p
    scl = 1/4; %1/4; %1/2; %1/sqrt(2);
    % scl = 1e-4/sqrt(sum(var(Yp,[],1)));
    disp([sprintf('%4i/%4i: ',[size(Xp,1),nbr]),sprintf('perp.= %f',pp)]);
    if own
        alr = false;
        [Yp,Ep,M1p,M2p] = basictsne(Xp,scl*Yp,2,pp,{itr,scl*M1p,scl^2*M2p},dof,[1,0],shw,reg,alr); % run t-SNE w/o EE !
        %[Yp,Ep,M1p,M2p] = basictsne(Xp,scl*Yp,2,pp,{itr,scl*Mp,scl^2*M2p},dof,[4,1/5],shw,alr); % run t-SNE w/ EE ...
    else
        opt = struct('MaxIter',itr,'TolFun',0,'OutputFcn',[]);
        if p<12
            alg = 'exact';
        else
            alg = 'barneshut';
        end
        % learn.rate is 10 times smaller than recommended; momentum?
        [Yp,Ep] = tsne(Xp,'Algorithm',alg,'Distance','euclidean','NumDimensions',2,'NumPCAComponents',0,'InitialY',scl*Yp,'Perplexity',pp,'Exaggeration',1,'LearnRate',size(Xp,1),'Options',opt,'Verbose',2);
        if shw
            figure(99);
            plot(Yp(:,1),Yp(:,2),'k.');
            pause(1);
        end
        %pause;
    end
    if exist('Ys','var'), Ys{p} = Yp; end
    if exist('Xs','var'), Xs{p} = Xp; end
    E(:,p) = sum(Ep,2); % record KL div.
end

Y(I,:) = Yp;

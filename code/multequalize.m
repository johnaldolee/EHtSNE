function [nSX,lnSX,PX] = multequalize(DX,pxt,nms,nsd,dof,mvr,pwr)
% Function
%
%  [nSX,lnSX,PX] = multequalize(DX,pxt,nms,nsd,dof,mvr,pwr)
%
% compute multiscale similarities as in Ms.SNE, Ms.JSE, Ms.t-SNE and related methods.
% Switches allow disabling the multiple scales and to consider non-squared distances,
% as well as Student similarities instead of Gaussian ones.
%
% Copyright J.A.Lee 2013-2023

if nargin<2, pxt = 0; end % perplexity
if nargin<3, nms = false; end % non-multi-scale
if nargin<4, nsd = false; end % non-squared distances
if nargin<5, dof = 0; end % Student t P
if nargin<6, mvr = 1; end % Student t P
if nargin<6, pwr = 1; end % power of 2

pxt = abs(pxt(1));
pxt = max(2,pxt);

nbr = size(DX,1);
itr = 100;

% ensure zero diagonal
for i = 1:nbr
    DX(i,i) = 0;
end

% majorize and exponentiate distances in the HD space
if ~nsd
    DX = 1/2*DX.^2;
end

% multiscale neighborhood radii
disp('Soft neighborhood equalization:');

% Student exception
if 0<dof
    disp(['Student t equalization ! dof = ',num2str(dof),', mvr = ',num2str(mvr)]);

    % equalize
    PX = tequalize(DX,[],pxt,itr,dof,mvr);

    % output
    [nSX,lnSX] = tsimilarities(DX,PX,dof,mvr);
    return
end

% perplexities (exp. of entropy) or soft neighborhood sizes

if nms
    npl = 1;
    ens = pxt;
else
    npl = round(1/pwr*log2(nbr/pxt)); % number of perplexity levels (up to nbr/2)
    ens = pxt*(2^pwr).^(npl-1:-1:0);

    %ens = flip(pxt:(nbr-pxt)/log2(nbr)/3:nbr);
    %npl = length(ens);

    %npl = floor(log2(nbr/pxt)); % number of perplexity levels (up to nbr/2)
    %x = 1:-2/(npl-1):-1;
    %y = x+x.^3/3+x.^5/5+x.^7/7+x.^9/9+x.^11/11;
    %y = (npl-1)*(0.5+0.5*y/max(y));
    %ens = pxt*2.^y;
end

% compute precisions in HD space
PX = cell(npl,1);
PXi = [];
for i = 1:npl
    % entropy equalization
    PXi = equalize(DX,PXi,ens(i),itr);
    PX{i} = PXi;
end

% output
[nSX,lnSX] = multisimilarities(DX,PX);

function lgmn = logmin(x)
% compute the minimal logarithm value for the floating-point precision of x

% class of variable x
if isa(x,'gpuArray')
    fpt = classUnderlying(x);
else
    fpt = class(x);
end

% depending on the floating-point class
switch fpt
    case 'single', lgmn = feval(fpt,8e-46);  % SP: log(8e-46) = -103.2789
%   case 'double', lgmn = feval(fpt,8e-46);  % DP: mimic behavior of SP
    case 'double', lgmn = feval(fpt,3e-324); % DP: log(3e-324) = -744.4401
end

function [nSX,lnSX] = multisimilarities(DX,PX)
% multiscale similarities
% DX: matrix of half squared distances
% PX: cell of precisions vectors (one row vector per level)
% nSX: normalized similarities
% lnSX: logarithm of normalized similarities
% (gpu-compatible)

% number of levels
npl = size(PX,1);

if npl<=1
    % reduce to single-scale similarities (log.sim. more accurate)
    [nSX,lnSX] = similarities(DX,PX{1});
else
    % for all scales
    nSX = similarities(DX,PX{1});
    for i = 2:npl
        nSX = nSX + similarities(DX,PX{i});
    end
    nSX = (1/npl)*nSX; % average

    % log(similarities)
    if nargout>1
        lnSX = log(max(logmin(DX),nSX));
    end
end

function [nSX,lnSX] = similarities(DX,PX)
% similarities
% DX: matrix of half squared distances
% PX: row vector of precisions
% nSX: normalized similarities
% lnSX: logarithm of normalized similarities
% (gpu-compatible)

% zero and negligible entries
zni = (abs(DX)<=1e-64); % abs not necessary if positive entries only

% exploit shift invariance to improve num. accuracy
tmp = DX;
tmp(zni) = inf; % much slower with NaN, inf not slower than any regular value
DX = bsxfun(@minus, DX, min(tmp,[],1)); % subtract minimum distance

% exponential/Gaussian similarity
lSX = bsxfun(@times, -PX, DX);
lSX(zni) = -746; % largest integer n such that exp(n)==0 in double precision (simple precision too)
SX = exp(lSX);

% normalize similarities
sSX = sum(SX,1); % "marginal" normalization factor
nSX = bsxfun(@rdivide, SX, sSX);

% log(similarities)
if nargout>1, lnSX = bsxfun(@minus, lSX, log(sSX)); end

function [nSX,lnSX] = tsimilarities(DX,PX,dof,mvr)
% tsimilarities
% DX: matrix of half squared distances
% PX: row vector of precisions
% nSX: normalized similarities
% lnSX: logarithm of normalized similarities
% (gpu-compatible)

if nargin<3, dof = 1; end
if nargin<4, mvr = 1; end

% zero and negligible entries
zni = logical(eye(size(DX,1))); % diagonal only %(abs(DX)<=1e-64); % abs not necessary if positive entries only

% exploit shift invariance to improve num. accuracy
%lSX = log(bsxfun(@plus, PX, bsxfun(@times, 2/dof, DX)));
lSX = log1p(bsxfun(@times, 2/dof*PX, DX));
tmp = lSX;
tmp(zni) = inf; % much slower with NaN, inf not slower than any regular value
lSX = bsxfun(@minus, lSX, min(tmp,[],1)); % subtract minimum logdistance
lSX = -(dof+mvr)/2*lSX;

% exponential/Student similarity
lSX(zni) = -746; % largest integer n such that exp(n)==0 in double precision (simple precision too)
SX = exp(lSX);

% normalize similarities
sSX = sum(SX,1); % "marginal" normalization factor
nSX = bsxfun(@rdivide, SX, sSX);

% log(similarities)
if nargout>1, lnSX = bsxfun(@minus, lSX, log(sSX)); end

function [PX,nSX,lnSX] = equalize(mDXp,PX,pxt,itr)
% baseline equalization
% (gpu-compatible)

nbr = size(mDXp,1);

% initialize precisions
if isempty(PX)
    tmp = mDXp;
    tmp(abs(tmp)<=1e-64) = inf; % much slower with NaN, inf not slower than any regular value
    tmp = nbr/(nbr-1)*mean(mDXp,1) - min(tmp,[],1); % shift invariance !scaling to avoid negative value!
    PX = 1./tmp;
end

% Shannon entropy (pxt is the perplexity)
Htgt = log(abs(pxt));

disp(['Target perplexity = ',num2str(pxt)]);

% equalization with Newton's method
% (precisions are initialized above)
for t = 1:itr
    % compute normalized similarities in the HD space
    [nSX,lnSX] = similarities(mDXp,PX);

    % compute all entropies
    Htmp = -sum(nSX.*lnSX,1);

    % show progress
    disp([sprintf('%3i/%3i: ',[t,itr]),sprintf('Sh.ent.(-> %2.4f) in ',Htgt),sprintf('%2.3f %2.3f %2.3f %2.3f ',min(Htmp),median(Htmp),mean(Htmp),max(Htmp))]);

    % delta H
    Htmp = Htmp - Htgt;

    % stop or update
    if all(abs(Htmp)<1e-3*abs(Htgt)), break; end

    % update
    PX = PX - max(-PX/2, min(PX/2, Htmp./sum(nSX.*(1+lnSX).*bsxfun(@minus,mDXp,sum(nSX.*mDXp,1)),1) ) ); % allow PX to be multiplied by min. 0.5 or max. 1.5
end

disp(['         precs. in '         ,sprintf('%1.4g %1.4g %1.4g %1.4g ',min(PX),median(PX),mean(PX),max(PX))]);

function [PX,nSX,lnSX] = tequalize(mDXp,PX,pxt,itr,dof,mvr)
% baseline equalization
% (gpu-compatible)

% initialize precisions
if isempty(PX)
    PX = 10./max(mDXp,[],1);
end

% Shannon entropy (pxt is the perplexity)
Htgt = log(abs(pxt));

disp(['Target perplexity = ',num2str(pxt)]);

% equalization with Newton's method
% (precisions are initialized above)
for t = 1:itr
    % compute normalized similarities in the HD space
    [nSX,lnSX] = tsimilarities(mDXp,PX,dof,mvr);

    % compute t factor
    tfc = 2/dof*(dof+mvr)*mDXp ./ (1 + bsxfun(@times, 2/dof*PX, mDXp));
    %tfc = 2/dof*(dof+mvr) ./ bsxfun(@plus, PX, bsxfun(@times, 2/dof, mDXp));

    nbr = size(nSX,1);
    zdg = (ones(nbr) - eye(nbr));
    tfc = tfc .* zdg;
    nSX = nSX .* zdg;

    % compute all entropies
    Htmp = -sum(nSX.*lnSX,1);

    % show progress
    disp([sprintf('%3i/%3i: ',[t,itr]),sprintf('Sh.ent.(-> %2.4f) in ',Htgt),sprintf('%2.3f %2.3f %2.3f %2.3f ',min(Htmp),median(Htmp),mean(Htmp),max(Htmp))]);
    disp(['         precs. in '         ,sprintf('%1.4g %1.4g %1.4g %1.4g ',min(PX),median(PX),mean(PX),max(PX))]);

    % delta H
    Htmp = Htmp - Htgt;

    % stop or update
    if all(abs(Htmp)<1e-3*abs(Htgt)), break; end

    % update
    PX = PX - max(-PX/2, min(PX/2, Htmp./sum(nSX.*(1+lnSX).*bsxfun(@minus,tfc,sum(nSX.*tfc,1)),1) ) ); % allow PX to be multiplied by min. 0.5 or max. 1.5
end

disp(['         precs. in '         ,sprintf('%1.4g %1.4g %1.4g %1.4g ',min(PX),median(PX),mean(PX),max(PX))]);

tmp = sort(mDXp,1);
disp(['         min.dist. in '         ,sprintf('%1.4g %1.4g %1.4g %1.4g ',min(tmp(  2,:)),median(tmp(  2,:)),mean(tmp(  2,:)),max(tmp(  2,:)))]);
disp(['         max.dist. in '         ,sprintf('%1.4g %1.4g %1.4g %1.4g ',min(tmp(end,:)),median(tmp(end,:)),mean(tmp(end,:)),max(tmp(end,:)))]);

figure;
stp = Htmp./Htgt;
subplot(1,2,1); plot(tmp(  2,:),stp,'k.'); xlabel('min.dist.'); ylabel('conv.ratio');
subplot(1,2,2); plot(tmp(end,:),stp,'k.'); xlabel('max.dist.'); ylabel('conv.ratio');

%nSX(abs(mDXp)<1e-64) = 0;
%nSX = bsxfun(@rdivide, nSX, sum(nSX,1));




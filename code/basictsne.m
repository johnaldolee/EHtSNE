function [Y,E,M1,M2] = basictsne(X,Y,dim,pxt,itr,dof,ee,shw,reg,alr)
% Basic t-SNE to 2D:
%
% [Y,E,M1,M2] = basictsne(X,Y,dim,pxt,itr,dof,ee,shw)
%
% Inputs:
%   X  : HD coordinates
%   Y  : LD coordinates (before) (default: []) (scalar: scale of PCA init.) 
%   dim: target dimensionality (integer; default: 2; superseded by prov. Y 
%   pxt: perplexity (default: 32)
%   itr: #iterations (default: 1000) or {itr,M1,M2} (see ehtsne)
%   dof: #degrees of freedom (default: 1)
%   ee : early exaggeration (default: [12,1/4] => 250/1000 iter.)
%        legacy Lvdm [4,1/10]; next: [4,1/5]; now [12,1/4]
%   shw: animated scatter plot (slower)
%   reg: regularization to scaled initialization 
%        (PCA or provided Y; NA for random)
%        (positive is L_2^2; negative is L_2,1 aka TV) (default: 0 => none)
%        (typical: +/- 0.05 quite strong already)
% Outputs:
%   Y  : LD coordinates (after)
%   E  : KL divergence value
%   M1 : 1st-order momentum (after)
%   M2 : 2nd-order momentum (after)
%
% Copyright J.A.Lee, August 9, 2025.

% default arguments
if nargin<10, alr = true; end % switch on accelerations (NAG and/or AdaM)
if nargin<9, reg = 0; end % hidden arg for now: regularization with init.
if nargin<8, shw = false; end
if nargin<7, ee = [12,1/4]; end
if nargin<6, dof = 1; end
if nargin<5, itr = 1000; end
if nargin<4, pxt = 32; end
if nargin<3, dim = 2; end
if nargin<2, Y = []; end

% size (N)
nbr = size(X,1);

% check nature of X
if isdist(X)
    % distances are provides
    disp('Use provided distances => PCoA/CMDS');
    DX = X; % distance as provided

    % PCoA
    SX = DX.^2;
    SX = SX - mean(SX,1) - mean(SX,2) + mean(SX(:)); % auto-bsxfun?
    [V,D] = eigs(-1/2*SX,min(dim,64)); % 64 PCs max.
    X = 1/sqrt(nbr)*V*D.^(1/2);
else
    % coordinates are provided
    disp('Use coordinates => Euclidean distances');

    % compute distances (Euclidean by default)
    DX = pairwisedistances(X);
end

% check target dimension
dim = ceil(abs(dim));

% embedding initialization
if isempty(Y)
    % random
    disp('Random initialization');
    Y = 1e-4*randn(nbr,dim);
    reg = 0; % no regularization possible here
elseif isscalar(Y)
    % (scaled) PCA
    disp('Scaled PCA initialization');
    Xc = bsxfun(@minus, X, mean(X,1));
    [U,S,~] = svds(Xc,dim);
    scaf = 1; % scaling factor with momentum
    Y = Y/sqrt(sum(diag(S).^2))*U*S; % Y is a scalar at first!
    Yreg = Y; % keep a record for regularization
    mrts = 0.9;
else
    % as provided in arg Y
    disp('Provided initialization');
    if size(Y,1)~=nbr, error('size Y >< size X'); end
    dim = size(Y,2);
    scaf = 1; % scaling factor with momentum
    Yreg = Y; % keep a record for regularization
    mrts = 0.9;
end

% first simple checks
dof = max(1,dof);

% initialize momenta or recover inherited momenta
if iscell(itr)
    % recover from hidden arg
    disp('Inheriting momenta');
    M1  = itr{2};
    M2  = itr{3};
    itr = itr{1};
else
    % initialize
    M1 = zeros(nbr,dim);
    M2 = zeros(nbr,dim);
end

% early exaggeration factor and relative duration
if isscalar(ee), ee = [ee,1/5]; end
ee = abs(ee);
if ee(2)<=1, ee(2) = ee(2)*itr; end

% optimizer (all cond.on alr==true)
nag = alr; % Nesterov accelerated gradient (look ahead momentum): always
if reg
    adm = 0; % no AdaM (it scales the gradient)
else
    adm = alr; % AdaM
end

% display a few things
disp(['Target dimension: ',num2str(dim)]);
disp(['Degrees of freedom: ',num2str(dof)]);
disp(['Number of iterations: ',num2str(itr)]);
disp(['Early exaggeration: ',num2str(ee(1)),' for ',num2str(floor(ee(2))),' iterations']);
if adm
    disp('NAG AdaM');
else
    disp('NAG');
end
if reg<0
    disp(['Regularization to initialization is L_1,2 with lambda = ',num2str(-reg)]);
elseif 0<reg
    disp(['Regularization to initialization is L_2^2 with lambda = ',num2str(reg)]);
else
    disp('No regularization to initialization');
end

% pairwise distances and entropic affinities
nms = true; % non-multi-scale
nsd = false; % squared distance
nSX = multequalize(DX,pxt,nms,nsd);
nSX = 0.5/nbr*(nSX+nSX'); % t-SNE symmetrization
lnSX = log(eps+nSX); % log.entr.affinities
for i = 1:nbr, lnSX(i,i) = 0; end % force diagonal to zero

% iterate
E = zeros(itr,2); % energy track record (=>KL)
tol = 1e-6; % tolerance for stopping criterion
lrt = 1.0; %nbr; % one if AdaM (M1&M2), else N if NAG (M1) empiric learning rate
mrt1 = 0.9; % momenta parameter order 1
mrt2 = 0.99; % momenta parameter order 2
gY = zeros(nbr,dim); % gradient of Y
for t = 1:itr
    if nag
        % look ahead of current estimate
        if adm
            Ym = Y - lrt*(1/(1-mrt1^t)*M1./(1e-4+sqrt(1/(1-mrt2^t)*M2))); % NAG & AdaM: not recommended
        else
            Ym = Y - lrt*(1/(1-mrt1^t))*M1; % unbiased 1st-order momentum only
        end
    else
        Ym = Y;
    end
    DY2 = psed(Ym); % pairwise squared Euclidean distance
    AY = 1 + 1/dof*DY2;
    SY = AY.^(-(dof+1)/2);
    for i = 1:nbr, SY(i,i) = 0; end % diagonal
    sSY = sum(sum(SY));
    nSY = SY./sSY;
    lnSY = bsxfun(@minus, -(1+dof)/2*log(AY), log(sSY));
    E(t,1) = sum(sum(nSX.*(lnSX-lnSY))); % KL over all N^2 entries
    if t<=ee(2)
        % specified early exaggeration (EE>=1)
        EE = ee(1);
        tmp = (ee(1)*nSX-nSY).*SY;
    else
        % no early exaggeration (EE=1)
        EE = 1;
        tmp = (nSX-nSY).*SY;
    end

    % learning rate: the "partial Hessian" à la Vladymyrov (check!)
    % plus compensation for EE in EE phase (otherwise EE==1)
    if alr && (reg==0) % ATTENTION test
        pHs = EE.*sum(nSX.*SY); % inv.diag. of postive (attractive) component of Hessian
    else
        pHs = EE./nbr; % inv. scalar of naive constant Hessian = sum(nSX) = nbr
    end

    % gradient (with 'naive' learning rate = nbr (i.e., N))
    for d = 1:dim
        gY(:,d) = 4./pHs'.*sum(1/dof*tmp.*bsxfun(@minus, Ym(:,d)', Ym(:,d)) ,1)';
    end

    % regularization to scaled initialization
    % putting L2 reg. here includes it in Adam/Nesterov but raises scaling
    % issues of the two gradient components (hence here scaled by 1/nbr,
    % as nbr (N) seems to be a good step size for t-SNE KL 'raw' GD
    if 0<reg
        % with norm L_2^2
        % [~,Ysca] = procrustes(Ym,Yreg);
        scaf = mrts*scaf + (1-mrts)*sqrt(mean(sum(bsxfun(@minus, Ym, mean(Ym,1)).^2,2)) ./ mean(sum(bsxfun(@minus, Yreg, mean(Yreg,1)).^2,2)));
        Ysca = (1-mrts^t)*scaf.*Yreg;
        gYr = reg*(Ym-Ysca);
        %gYr = bsxfun(@minus, gYr, mean(gYr,1));
        gY = gY + gYr;
        E(t,2) = reg/2*mean(sum((Ym-Ysca).^2,2));
    end

    % update with modified gradient (by learning rate and momentum; NAG ADAM)
    M1 = mrt1*M1 + (1-mrt1)*gY; % update 1st-order momentum (see debiasing factor against cold start)
    M2 = mrt2*M2 + (1-mrt2)*gY.^2; % update 2nd-order momentum (see debiasing factor against cold start)
    if adm
        Y = Y - lrt*(1/(1-mrt1^t)*M1./(1e-4+sqrt(1/(1-mrt2^t)*M2))); % learning rate and (negative) "velocity" (gradient for GD or momentum for NAG ADAM)
    else
        Y = Y - lrt*(1/(1-mrt1^t))*M1;
    end

    % 2-step optmization for L_2,1 with shrinkage proximal operator
    % incompatible with AdaM on KL grad. because of unpredicatable scaling
    if reg<0
        % with norm L_2,1 aka TV
        %[~,Ysca] = procrustes(Y,Yreg); % works but messes the optimization
        scaf = mrts*scaf + (1-mrts)*mean(sqrt(sum(bsxfun(@minus, Y, mean(Y,1)).^2,2))) ./ mean(sqrt(sum(bsxfun(@minus, Yreg, mean(Yreg,1)).^2,2)));
        Ysca = (1-mrts^t)*scaf.*Yreg;
        Dreg = sqrt(sum((Y-Ysca).^2,2)); % Euclidean distance (non-squared)
        Davg = mean(Dreg); % mean Eucl.dist.
        %Y = Y - bsxfun(@times, min(-reg*(1-mrts^t)*scaf/nbr, Dreg), bsxfun(@rdivide, (Y-Ysca), max(eps,Dreg))); % grad. from sum, not avg.
        Y = Y - bsxfun(@times, min(-reg*Davg, Dreg), bsxfun(@rdivide, (Y-Ysca), max(eps,Dreg))); % grad. from sum, not avg.
        % 2 variants of the shrinkage operator: the shrinkage is modulated
        % by the (momentum-damped) scaling factor of Yreg or by the current
        % Davg (which varies more?)
        E(t,2) = - reg*Davg; % obj.fun. is average to be compatible w/KL (also avg., see l.rate=nbr for grad.)
    end

    % % centering (unnecessay and not compatible with reg. w/scaling only)
    % Y = bsxfun(@minus, Y, mean(Y,1));

    % display
    if mod(t,10)==0
        disp([sprintf('%4i/%4i: ',[t,itr]),sprintf('EE=%2.4f ',EE),sprintf('KL div.=%2.4f ',E(t,1)),sprintf('reg.=%2.4f ',E(t,2)),sprintf('l.rate=%2.4f ',lrt),sprintf('pHs=%2.4f ',mean(1./pHs)),sprintf('Grad.mag.=%2.4f',2*sqrt(sum(gY(:).^2)))]);
    end
    if t>ee(2)+30
        % stopping criterion
        c1 = abs(1 - E(t,1)/(eps+E(t-10,1))); % KL div.
        c2 = abs(1 - E(t,2)/(eps+E(t-10,2))); % regul.
        if c1<tol && c2<tol, break; end % both
    end

    if shw && dim==2 && mod(t,10)==0
        figure(99);
        subplot(1,2,1);
        if reg
            plot(Ysca(:,1),Ysca(:,2),'g.',Y(:,1),Y(:,2),'k.');
            title(['Reg.Sca.Fac. = ',num2str(scaf)]);
        else
            plot(Y(:,1),Y(:,2),'k.');
        end
        subplot(1,2,2);
        vec = 10:itr;
        plot(vec,E(vec,1),'r-',vec,E(vec,2),'b-',vec,E(vec,1)+E(vec,2),'g-');
        %set(gca,'YScale','log');
        legend('KL','Reg.','All')
        pause(0.1);
    end
end

function [Y,YY,FF,idm] = sbdr_abd_ms(DX,dim,pxt,kba,itr,tol,fpt,att)
% Function
% 
% [Y,YY,FF,idm] = sbdr_abd_ms(DX,dim,pxt,kba,itr,tol,fpt,att)
% 
% performs similarity-based dimensionality reduction,
% with shift-invariant softmax-normalized similarities and
% sums of alpha-beta-divergences as a cost function.
% The similarities can be single-scale (unique perplexity value [1,2]) or 
% multiscale (mean of single-scale similarities with several different 
% perplexities [3]).
%
% Scaled principal component are used to initialize embedding Y.
% The algorithm is fully deterministic.
% The optimization involves a line search with backtracking for the step
% size adjustment under the strong Wolfe conditions. The search direction 
% is the product of the gradient with a diagonal estimate of the Hessian, 
% which is refined by the limited-memory BFGS algorithm (m=7).
% The optimization is multiscale, in the sense that BFGS is run several 
% times, starting with the largest perplexity value and then introducing
% or switching to smaller ones.
%
% Inputs:
%   DX  : symmetric matrix of non-squared pairwise distances in HD space
%   dim : embedding dimensionality (scalar integer; 2)
%         if dim is non-scalar, both multiscale optimization and multiscale
%         similarities are disabled;
%         if dim is a matrix with the same number of rows as DX,
%         then it serves as initialization for Y (instead of PCA init.)
%         the embedding dimension is then given by the number of columns
%   pxt : perplexity value (scalar; 0)
%         pxt determines perplexity values in the HD (and LD) spaces
%         these value range from p_1 = max(2,abs(pxt)) to p_i = p_1*2^(i-1)
%         with i s.t. p_i is as larger as about 1/2 of the data set size;
%         if abs(pxt)>=2, then single-scale similarities are used and pxt
%         is the smallest, targeted perplexity;
%         if abs(pxt)<2, then multiscale similarities are used
%         if pxt is negative, then the precisions in the LD space are set
%         to be proportional to p_i^(2/dim), otherwise they are computed 
%         like HD precisions, using entropy equalization on Y
%   kba : values of kappa, beta, and alpha in the type I or II mixture
%         of alpha-beta-divergences (a matrix with 1,2 or 3 rows and
%         1 (constant), 2 (linear), or itr values)
%         1st row: kappa controls 
%                    the mixture type:
%                      negative is type 1 => NeRV
%                      zero is no mixture => SNE
%                      positive is type 2 => JSE - scal.gen. Jensen-Shannon 
%                    the mixture weight (magnitude between 0 and 1)
%         2nd row: beta (0)
%         3rd row: alpha (1; must differ from zero if any kappa>0)
%         kba can also be a string whose first character determines the 
%         method ('s' for SNE, 'n' for NeRV, 'j' for JSE, 'h' for HE, 
%         'e' for EE); the optional rest of the string specifies k 
%         for NeRV and JSE, or a==b for EE
%   itr : maximum number of iterations per perplexity value
%         (integer; 30; overwritten by #cols in kba)
%         if itr<0, then multiscale optimization is disabled
%         (it naturally overwrites att to abs(att))
%   tol : tolerance for stopping criterion (tiny scalar float; 1e-4)
%   fpt : specifies the floating-point type (scalar; 2)
%         positive => CPU, negative => GPU (no guarantee for GPU!)
%         abs(fpt)<2 => single precision, double otherwise
%   att : enable adaptive tail thickness (real; 1)
%         (<1 => lighter; >1 => heavier; <0 => auto)
%
% Outputs:
%   Y   : coordinates in the LD space of the final embedding
%   YY  : cell with all embeddings used as initialization at each scale 
%   FF  : cell with all values of the cost function after each iteration
%   idm : intrinsic dimensionality estimator based on multiscale precisions
%         if nargout>3, then the function returns idm without embedding
%         (Y is left equal to its initialisation value, YY and FF empty)
%
% References:
%
% [1] John A. Lee
%     Type 1 and 2 mixtures of divergences for stochastic neighbor embedding
%     Proc. ESANN 2012, 20:525-530.
% [2] J. A. Lee, E. Renard, G. Bernard, P. Dupont, M. Verleysen
%     Type 1 and 2 mixtures of Kullback-Leibler divergences
%     as cost functions in dimensionality reduction
%     based on similarity preservation
%     Neurocomputing 2013, 112: 92-108.
% [3] John A. Lee, Diego H. Peluffo, Michel Verleysen
%     Multi-scale similarities in stochastic neighbour embedding:
%     Reducing dimensionality while preserving both local and global structure
%     Neurocomputing 2015, 169:246-261.
%     http://dx.doi.org/10.1016/j.neucom.2014.12.095
%
% Copyright J.A.Lee, April 17, 2017.

%% process arguments ------------------------------------------------------

% default arguments
if nargin<8, att = 1; end
if nargin<7, fpt = 2; end
if nargin<6, tol = 1e-4; end
if nargin<5, itr = 30; end
if nargin<4, kba = [0;0;1]; end
if nargin<3, pxt = 0; end
if nargin<2, dim = 2; end
if nargin<1, error('Insufficient number of arguments!'); end

% multiscale optimisation
if itr<0
    mso = false; % single scale
    att = abs(att);
else
    mso = true; % multiple scale
end

% check distance and target dimensionality; initialize a few things
[idx,DX,nbr,sss,L,shw,Y,dim,X] = dri(DX,dim,1,1);
ss = 1:sss;

% check perplexity
if nargout>3, pxt = -1; end
pxt = pxt(1);
if pxt<0, PYq = false; else PYq = true; end
pxt = abs(pxt);
if pxt<2, mss = true; else mss = false; end
pxt = max(2,pxt);

% check other parameters
itr = ceil(abs(itr(1)));
if itr==0, itr= 30; end
tol = abs(tol(1));
if tol==0; tol = 1e-4; end
fpt = fpt(1);
att = att(1);

% floating-point type and GPU computation
try
    gpu = (fpt<0) && (1<=gpuDeviceCount);
catch
    gpu = false;
end
if abs(fpt)<2
    fpt = 'single';
else
    fpt = 'double';
end

% check kappa, beta, and alpha parameters
if isempty(kba), kba = [0;0;1]; end
if ischar(kba)
    tmp = str2double(kba(2:end));
    switch lower(kba(1))
        case 's'
            kba = [0;0;1];
        case 'n'
            if isnan(tmp), tmp = 0.5; end
            kba = [-tmp;0;1];
        case 'j'
            if isnan(tmp), tmp = 0.5; end
            kba = [tmp;0;1];
        case 'h'
            kba = [0.5;0.5;0.5];
        case 'e'
            if isnan(tmp), tmp = 1; end
            kba = [0.5;tmp;tmp];
        otherwise
            error(['Unrecognized method name ',kba,'in kba']);
    end
end
if size(kba,1)==1, kba = [kba;zeros(1,size(kba,2))]; end
if size(kba,1)==2, kba = [kba; ones(1,size(kba,2))]; end
switch size(kba,2)
    case 1
        kba = kba*ones(1,itr);
    case 2
        if aeq(kba(1,1),kba(1,2))
            k = kba(1,1)*ones(1,itr);
        else
            k = kba(1,1):(kba(1,2)-kba(1,1))/(itr-1):kba(1,2);
        end
        if aeq(kba(2,1),kba(2,2))
            b = kba(2,1)*ones(1,itr);
        else
            b = kba(2,1):(kba(2,2)-kba(2,1))/(itr-1):kba(2,2);
        end
        if aeq(kba(3,1),kba(3,2))
            a = kba(3,1)*ones(1,itr);
        else
            a = kba(3,1):(kba(3,2)-kba(3,1))/(itr-1):kba(3,2);
        end
        kba = [k;b;a];
    otherwise
        itr = size(kba,2);
end
kba = [max(-1,min(1,kba(1,:)));kba(2:3,:)];
if any(aeq(kba(2,:),kba(3,:)) & kba(3,:)<0), error('alpha must be non-negative if alpha==beta'); end
if any(kba(1,:)>0 & aeq(kba(3,:),0) & kba(2,:)~=kba(3,:)), error('alpha must differ from zero if kappa>0'); end

% convert to specified floating-point type
DX = feval(fpt,DX);
Y = feval(fpt,Y);

%% v NEW - July 2018

mahalocal = 0;
if mahalocal
    size(X)
    MDX = DX;
    txp = 64;
    K = ceil(3*txp);
    mm = cell(nbr,1);
    for t = 1:mahalocal
        disp('MahaLocal');
        
        % find KNN
        [~,xdi] = sort(MDX);
        
        % weights
        nSX = multequalize(MDX,txp,true);
        
        % compute Mahalanobis multiplier (K NN only)
        for i = 1:nbr
            tmp = bsxfun( @times, nSX(xdi(1:K,i),i), bsxfun( @minus, X(xdi(1:K,i),:), X(i,:) ) );
            [U,S,V] = svd(tmp,'econ');
            mm{i} = V*diag(diag(1./S))*U';
            %mm{i} = V*S.^2*U';
        end
        
        % compute Mahalanobis distances (all)
        for i = 1:nbr
            tmp = bsxfun( @minus, X, X(i,:) );
            mm{i} = mm{i}.*sqrt( sum(sum(tmp.^2,2)) ./ sum(sum((tmp*mm{i}).^2,2)) );
            dst = sqrt(sum((tmp*mm{i}).^2,2));
            MDX(:,i) = dst;
        end
    end
    DX = MDX;
end

%% ^ NEW - July 2018

% transfer to GPU if requested
if gpu
    DX = gpuArray(DX);
    Y = gpuArray(Y);
end

%% precisions in the HD space ---------------------------------------------

% majorize and exponentiate distances in the HD space
mDXp = 1/2*max(DX,0).^2;

%% multiscale neighborhood radii
disp('Soft neighborhood equalization:');

% perplexities (exp. of entropy) or soft neighborhood sizes
if mss || mso
%    npl = round(log2(nbr/2/pxt)); % number of perplexity levels (up to nbr/4)
    npl = round(log2(nbr/pxt)); % number of perplexity levels (up to nbr/2)
else
    npl = 1;
end
ens = pxt*2.^(npl-1:-1:0);

% compute precisions in HD space
PX = cell(npl,1);
PXi = [];
for i = 1:npl
    % entropy equalization
    PXi = equalize(mDXp,PXi,ens(i),itr);
    PX{i} = PXi;
    if i>2
        PXi = PXi.*(PX{i-1}./PX{i-2}); % guess based on previous ratio => spare at least one iteration
    end
end

% evaluate intrinsic dimensionality
if 1<size(PX,1)
    dy = zeros(npl-1,sss);
    for i = 1:npl-1
        dy(i,:) = bringback(2./(log2(PX{i+1}) - log2(PX{i})));
        %dy(i,:) = bringback((log2(PX{i+1}) - log2(PX{i}))./2);
    end
    dys = [min(dy,[],2),mean(dy,2),max(dy,[],2)];
    ids =     mean(dy(end,:),2) ; % small scale
    idm = max(mean(dy       ,2)); % mid   scale pretty good estimate of the intrinsic dimensionality
    idl =     mean(dy(  1,:),2) ; % large scale
    %dys = 1./[min(dy,[],2),mean(dy,2),max(dy,[],2)];
    %ids = 1./    mean(dy(end,:),2) ; % small scale
    %idm = 1./min(mean(dy       ,2)); % mid   scale pretty good estimate of the intrinsic dimensionality
    %idl = 1./    mean(dy(  1,:),2) ; % large scale
    if ~gpu && shw
        figure(98);
        subplot(2,2,1); plot(log2(ens'),-0.5*log2(cat(1,PX{:}))); title('Raw'); xlabel('log2(Perplexity)'); ylabel('log2(Bandwidth)');
        dx = 0.5*(log2(ens(1:end-1)') + log2(ens(2:end)'));
        subplot(2,2,2); plot(dx,dy); title('1/Derivative'); xlabel('log2(Perplexity)'); ylabel('d log2(Perplexity) / d log2(Bandwidth)');
        %subplot(2,2,2); plot3(repmat(dx,1,nbr),dy,repmat(L',length(dx),1)); title('1/Derivative'); xlabel('log2(Perplexity)'); ylabel('d log2(Perplexity) / d log2(Bandwidth)');
        %subplot(2,2,2); hold on; for j = 1:length(dx), scatter(dx(j)*ones(1,nbr),dy(j,:),5,L,'o','filled'); end; title('1/Derivative'); xlabel('log2(Perplexity)'); ylabel('d log2(Perplexity) / d log2(Bandwidth)');
        subplot(2,2,4); plot(dx,dys); title(['Med. mean inv. der. = ',num2str(idm)]); xlabel('log2(Perplexity)'); ylabel('d log2(Perplexity) / d log2(Bandwidth)');
        %subplot(2,2,3); plot(dx, -dim*cumsum( bsxfun(@rdivide, dy, max(dy,[],1)) ,1) );
        subplot(2,2,3); plot(log2(ens'),-log2(cat(1,PX{:}))./idm); title(num2str([ids,idm,idl])); xlabel('log2(Perplexity)'); ylabel('log2(Bandwidth)');
    end
    
    % display estimated intrinsic dimensionality
    disp(['Estimated intrinsic dimensionality: ',num2str(idm)]);

    % adaptive tail thickness
    if att<0, att = min(2,max(1,idm/dim)); end
end
if nargout>3, YY = {}; FF = {}; return; end

% indicate which tails are used
if mss
    if aeq(att,1)
        disp( 'Using regular tails (1)');
    else
        disp(['Using heavier tails (',num2str(att),')']);
        disp(num2str([ids,idm,idl]));
    end
end

% precisions in LD space
PY = cell(numel(ens),1);
if PYq
    % entropy equalization
    % see further below
else
    % simpler approach
    tmp = ens;
    if mss
        tmp = tmp.^(2*att/dim);
    else
        tmp = tmp.^(2/dim); % do not use att if single-scale
    end
    
    tmp = 2^(1+2/dim)/mean(var(Y)) * max(tmp)./tmp; % make precision inversely proportional to mean variance
    for i = 1:npl
        PY{i} = tmp(i)*ones(1,sss,'like',Y);
    end
    
%    PY{1} = 2^(1+2/dim)/mean(var(Y))*ones(1,sss,class(Y));
%    %PY{1} = 2^(1+2/dim)/mean(var(Y))/mean(PX{1})*PX{1};
%    for i = 2:npl
%        PXr = PX{i} ./ PX{i-1}; % ratio for PX corresponding to K_i and K_{i-1} = 0.5*K_i
%        idi = 2*log(ens(i)/ens(i-1))./log(PXr); % intrinsic dimensionality on scale i for datum j
%        midi = mean(idi); % mean intrinsic dimensionality on scale i
%        PY{i} = PY{i-1}.*PXr.^(att/dim*(0.1*midi+0.9*idi)); % growth factor is corrected with regularized dim. ratio times att parameter
%    end
end

%% main part: iterate until convergence -----------------------------------
disp('Cost function minimization:')

% embedding optimization

YY = cell(npl,3);
FF = cell(npl,1);

stp = 1;
if mso, fpl = 1; else fpl = npl; end % first perplexity level => number of optimization levels
for cpl = fpl:npl
    disp(['Level ',num2str(cpl),'/',num2str(npl),': perplexity ',num2str(ens(cpl)),' to ',num2str(ens(1))]);
    
    if PYq
        % half majorized squared Euclidean distances in LD space
        if sss<nbr
            mDYp = 1/2*max(psed(Y,ss),0);
        else
            mDYp = 1/2*max(psed(Y),0); % much faster on GPU
        end
        
        % entropy equalization
        PYi = [];
        for i = 1:cpl
            PYi = equalize(mDYp,PYi,ens(i),itr);
            PY{i} = att^max(0,cpl-npl+1)*PYi;
        end
    end

    % perplexity index range
    if mss, pir = 1:cpl; else pir = cpl; end

    % scale weights
    msw = 1/numel(pir)*ones(numel(pir),1,'like',mDXp);
    %msw = numel(pir):-1:1; msw = msw./sum(msw);
    %msw = 1:numel(pir); msw = msw./sum(msw);
    
    % compute HD similarities
    [nSX,lnSX] = multisimilarities(mDXp,PX(pir),msw);
    
%     % new!!!
%     if ~isempty(L)
%         zni = (nSX<=1e-64);
%         [nSL,lnSL] = categsimilarities(L);
%         disp('Mean ShEntropies');
%         eSX = mean(-sum(nSX.*lnSX))
%         eSL = mean(-sum(nSL.*lnSL))
%         nSX = 1/eSX/(1/eSX+1/eSL)*nSX + 1/eSL/(1/eSX+1/eSL)*nSL;
%         lnSX = log(nSX);
%         lnSX(zni) = -746;
%         eSX = mean(-sum(nSX.*lnSX))
%     end
%     % new!!!
    
    % collect intermediate results
    if nargout>1
        YY{cpl,1} = Y; 
        YY{cpl,2} = PX{cpl};
        YY{cpl,3} = PY{cpl};
    end

    % first function evaluation
    F = zeros(1,itr,'like',Y); % records of all objective function values
    ncfe = zeros(1,itr); % number of cost function evaluation
    px = Y(:);
    [pf,pg,ph,Fs] = cfe(px,PY(pir),nSX,lnSX,dim,kba(:,1),msw); % first cost function evaluation
    ncfe(1) = 1;
    
    % diagonal approximation of Hessian (regularized)
    dH = mean(abs(ph));
    Hg = pg./dH; % first search direction
    
    bfgs = true;
    if bfgs
        % limited memory BFGS
        
        % lmBFGS variables
        hlm = 7; % history length
        xdifs = zeros(numel(px),hlm,'like',px);
        gdifs = zeros(numel(px),hlm,'like',px);
        
        % Wolfe conditions and backtracking variables
        b1 = 1e-2; % Wolfe: tiny gains are allowed
        b2 = 0.9; % Wolfe
        nst = 10; % maximum number of steps
        ys = zeros(nst,1,'like',pf);
        xs = zeros(nst,1,'like',pf);
        
        % iterate
        tbis = 0;
        for t = 1:itr
            % record current cost function value
            F(t) = pf;
            
            % show current result
            if ~rem(t,10) || t<10
                fprintf(1,'%3i/%3i #eval.=%3i F=%1.6f step=%1.6f dH=%1.6f %s (k=%1.2f,b=%1.2f,a=%1.2f)\n',t,itr,sum(ncfe),F(t)/F(1),stp,dH,Fs,kba(1,t),kba(2,t),kba(3,t));
            end
            
            % show scatter plot
            if ~gpu && (dim==2 || dim==3) && shw
                lts = ['(',num2str(cpl),'/',num2str(npl),', ',num2str(t),'/',num2str(itr),')'];
                
                Y = reshape(double(px),nbr,dim);
                figure(99);
                
                subplot(2,2,1);
                if dim==2
                    scatter(Y(:,1),Y(:,2),20,L,'o','filled');
                elseif dim==3
                    scatter3(Y(:,1),Y(:,2),Y(:,3),20,L,'o','filled');
                end
                title(['Embedding ',lts]);
                axis equal;
                
                subplot(2,2,2); cla; hold on;
                [l2K,~,~,lpR_NX] = nx_trusion_lp(DX,psed(Y));
                for l = pir, plot(log2(ens(l))*[1,1],[0,1],'k:'); end
                plot(l2K(1:end-1),lpR_NX,'ro-');
                axis([0,l2K(end),0,1]);
                title(['Approx. Quality ',num2str(mean(lpR_NX)),' ',lts]);
                xlabel('log_2(K)');
                ylabel('R_{NX}(K)');
                
                subplot(2,2,4);
                plot(1:t,F(1:t),'b-',cumsum(ncfe(1:t)),F(1:t),'r-');
                xlabel('#iter./eval.');
                ylabel('F');
                title(['Cost fun. ',Fs,' ',lts]);
                
                drawnow;
            end
            
            % (loose) line search with backtracking
            ostp = stp;
            for i = 1:nst
                % new x (descent along search direction)
                x = px - stp*(Hg-mean(Hg)); % remove mean to avoid translational freedom
                
                % cost function evaluation
                [f,g,h,Fs] = cfe(x,PY(pir),nSX,lnSX,dim,kba(:,t),msw);
                ncfe(t) = ncfe(t) + 1;
                
                % record
                ys(i) = f - pf;
                xs(i) = stp;
                
                % Wolfe conditions (Armijo + strong condition on curvature)
                tmp = pg'*Hg;
                if f<=pf+stp*b1*tmp && abs(g'*Hg)<=abs(b2*tmp)
                    stp = 1.25*stp; % confident for next search ;-)
                    break; % stop on first success
                else
                    % oops! went to far... continue cautiously
                    if i<2
                        % dichotomic decrease
                        stp = 0.5*stp;
                    elseif ys(i)/ys(i-1)<(xs(i)/xs(i-1))^2
                        % quadratic interpolation
                        qa = ys(i)*xs(i-1)         - ys(i-1)*xs(i)      ;
                        qb = ys(i)*xs(i-1)*xs(i-1) - ys(i-1)*xs(i)*xs(i);
                        stp = max(0.125*stp, qb/(2*qa) ); % between 1/8 and 1/2
                    else
                        % faster decrease
                        stp = 0.125*stp;
                    end
                end
            end
            
            % workarounds
            if i==nst
                disp(['l-BFGS reinitialization at it. ',num2str(t),' (unsuccessful line search)']);
                tbis = 1;
                stp = min(1,2*ostp);
            else
                tbis = tbis + 1;
            end
            
            % stopping criterion
            if t>=5 && abs(1 - F(t)/F(t-4))<tol
                disp(['Stop at it. ',num2str(t),' (tolerance reached)']);
                break;
            end
            
            % determine Hessian approximation for next line search
            
            % positive definite diagonal approximation of Hessian (regularized)
            dH = mean(abs(h));
            % the Hessian is rank-deficient and prod(h) can be null!!!
            % (due to plasticity: gradient and curvature are null in some directions
            % if outliers are present; this issue is not corrected by lmBFGS
            
            % approximation of the inverse Hessian with lmBFGS "double recursion"
            xdifs = [x-px,xdifs(:,1:hlm-1)];
            gdifs = [g-pg,gdifs(:,1:hlm-1)];
            r = 1./(sum(gdifs.*xdifs,1));
            r = min(max(r,-1/epsmax),1/epsmax); % avoid too large coefficients
            a = zeros(1,hlm,'like',x);
            b = zeros(1,hlm,'like',x);
            q = g;
            for i = 1:+1:min(t,hlm) % backward loop (if hlm>0)
                a(i) = r(i)*xdifs(:,i)'*q;
                q = q - a(i)*gdifs(:,i);
            end
            Hg = q./dH; % search direction
            for i = min(t,hlm):-1:1 % forward loop (if hlm>0)
                b(i) = r(i)*gdifs(:,i)'*Hg;
                Hg = Hg + (a(i)-b(i))*xdifs(:,i);
            end
            % (hlm==0) => (Hg==g)
            
            % updates
            pf = f;
            px = x;
            pg = g;
        end
    else
        % gradient descent with momentum ( !!! DOES NOT WORK PROPERLY YET !!! )
        
        % iterate
        tbis = 0;
        for t = 1:itr
            % record current cost function value
            F(t) = pf;
            
            % show current result
            if ~rem(t,10) || t<10
                fprintf(1,'%3i/%3i #eval.=%3i F=%1.6f step=%1.6f dH=%1.6f %s (k=%1.2f,b=%1.2f,a=%1.2f)\n',t,itr,sum(ncfe),F(t)/F(1),stp,dH,Fs,kba(1,t),kba(2,t),kba(3,t));
            end
            
            % show scatter plot (same as above)
            if ~gpu && (dim==2 || dim==3) && shw
                lts = ['(',num2str(cpl),'/',num2str(npl),', ',num2str(t),'/',num2str(itr),')'];
                
                Y = reshape(double(px),nbr,dim);
                figure(99);
                
                subplot(2,2,1);
                if dim==2
                    scatter(Y(:,1),Y(:,2),20,L,'o','filled');
                elseif dim==3
                    scatter3(Y(:,1),Y(:,2),Y(:,3),20,L,'o','filled');
                end
                title(['Embedding ',lts]);
                axis equal;
                
                subplot(2,2,2); cla; hold on;
                [l2K,~,~,lpR_NX] = nx_trusion_lp(DX,psed(Y));
                for l = pir, plot(log2(ens(l))*[1,1],[0,1],'k:'); end
                plot(l2K(1:end-1),lpR_NX,'ro-');
                axis([0,l2K(end),0,1]);
                title(['Approx. Quality ',num2str(mean(lpR_NX)),' ',lts]);
                xlabel('log_2(K)');
                ylabel('R_{NX}(K)');
                
                subplot(2,2,4);
                plot(1:t,F(1:t),'b-',cumsum(ncfe(1:t)),F(1:t),'r-');
                xlabel('#iter./eval.');
                ylabel('F');
                title(['Cost fun. ',Fs,' ',lts]);
                
                drawnow;
            end
            
            % new x (descent along search direction)
            x = px - stp*(Hg-mean(Hg)); % remove mean to avoid translational freedom
            
            % cost function evaluation
            [f,g,h,Fs] = cfe(x,PY(pir),nSX,lnSX,dim,kba(:,t),msw);
            if f<pf, stp = 1.25*stp; else stp = 0.50*stp; end % step size
            ncfe(t) = ncfe(t) + 1;
                
            % stopping criterion
            if t>=5 && abs(1 - F(t)/F(t-4))<tol
                disp(['Stop at it. ',num2str(t),' (tolerance reached)']);
                break;
            end
            
            % determine Hessian approximation for next line search
            
            % positive definite diagonal approximation of Hessian (regularized)
            dH = mean(abs(h));
            % the Hessian is rank-deficient and prod(h) can be null!!!
            % (due to plasticity: gradient and curvature are null in some directions
            % if outliers are present; this issue is not corrected by lmBFGS
            
            mtm = 0.2; % momentum
            Hg = (1-mtm)*Hg + mtm*g./dH; % search direction
            
            % updates
            pf = f;
            px = x;
            pg = g;
        end
    end
    
    % collect cost function values
    if nargout>2, FF{cpl} = F; end
    
    % rebuild coordinates from generic unknown vector x
    Y = reshape(x,nbr,dim);
end

%% finalize outputs -------------------------------------------------------

% PCA decorrelation (to avoid translational and rotational freedom)
Y = bsxfun(@minus, Y, sum(Y,1)./nbr); % mean subtraction
[U,S] = svd(Y,0); % SVD PCA on coordinates
Y = bringback(U * S); % back from GPU to CPU and double precision after rotation

% rearrange in initial order
Y(idx,:) = Y;

% other optional output arguments
if nargout>1
    for l = 1:size(YY,1)
        % back from GPU to CPU and double precision if necessary
        YY{l,1} = bringback(YY{l,1}); % LD coordinates
        YY{l,2} = bringback(YY{l,2}); % HD precisions
        YY{l,3} = bringback(YY{l,3}); % LD precisions
        
        % rearrange in initial order
        YY{l,1}(idx,:) = YY{l,1};
        YY{l,2}(idx) = YY{l,2}';
        YY{l,3}(idx) = YY{l,3}';
    end
    
    % minimize discrepancies with Procrustes' transform
    [~,tmp] = procrustes(Y,YY{end,1});
    YY{end,1} = tmp;
    for l = size(YY,1)-1:-1:1
        [~,tmp] = procrustes(YY{l+1,1},YY{l,1});
        YY{l,1} = tmp;
    end
end
if nargout>2
    for l = 1:size(FF,1)
        FF{l} = bringback(FF{l});
    end
end

%% additional functions

function [f,g,h,Fs] = cfe(x,PY,nSX,lnSX,dim,kba,msw)
% cost function evaluation (with gradient and diagonal of Hessian)

% sizes
nbr = size(nSX,1);
sss = size(nSX,2);
ss = 1:sss;

% reconstruct output coordinates from generic BFGS vector x
Y = reshape(x,nbr,dim);

% half majorized squared Euclidean distances in the LD space
if sss<nbr
    mDYp = 1/2*max(0,psed(Y,ss));
else
    mDYp = 1/2*max(0,psed(Y)); % much faster on GPU    
end

% compute LD similarities
[nSY,lnSY] = multisimilarities(mDYp,PY,msw);

% compute terms of gradient weights
k = abs(kba(1)); b = kba(2); a = kba(3);
if aeq(a,b)
    % symmetric Euclidean
    if aeq(a,0)
        % log. Eucl. dist.
        bd0 = lnSX.^2./2;
        bpcd0 = (lnSY - lnSX).^2./2;
        u1cd1 = lnSY - lnSX;
        u2cd2 = 1 - lnSY + lnSX;
        Fs = 'E0 sym';
    elseif aeq(a,1/2)
        % Hellinger distance
        bd0 = nSX./2;
        bpcd0 = 2*abs(nSX-nSY);
        u1cd1 = 2*(nSY - sqrt(nSY).*sqrt(nSX));
        u2cd2 = sqrt(nSY).*sqrt(nSX);
        Fs = 'E.5 sym';
    elseif aeq(a,1)
        % Euclidean distance
        bd0 = nSX.^2./2;
        bpcd0 = (nSY-nSX).^2./2;
        u1cd1 = nSY.*(nSY-nSX);
        u2cd2 = nSY.*nSY;
        Fs = 'E1 sym';
    else
        % T-log_{1-a}. Euclidean distance
        bd0 = exp(2*a*lnSX)./(2*a^2);
        bpcd0 = (exp(a*lnSY)-exp(a*lnSX)).^2./(2*a^2);
        u1cd1 = (exp(2*a*lnSY) - exp(a*(lnSY+lnSX)))./a;
        u2cd2 = (2-1/a)*exp(2*a*lnSY) - (1-1/a)*exp(a*(lnSY+lnSX));
        Fs = 'Ea sym';
    end
elseif 0<kba(1) && kba(1)<1
    lgmn = logmin(x);
    
    % type 2 mixture
    if aeq(b,0)
        if aeq(a,1)
            % Jensen-Shannon (Kullback-Leibler type 2)
            SZ = (1-k)*nSY+k*nSX;
            lSZ = log(max(lgmn,SZ));
            
            bd0 = -nSX.*lnSX;
            bpcd0 = 1/k*nSY.*lnSY + 1/(1-k)*nSX.*lnSX - 1/k/(1-k)*SZ.*lSZ;
            u1cd1 = 1/k*nSY.*(lnSY-lSZ);
            u2cd2 = exp(lnSY-lSZ+lnSX);
            Fs = 'JS1 t2';
        else
            % a-"scaled" Jensen-Shannon (Kullback-Leibler type 2)
            SZa = (1-k)*exp(a*lnSY)+k*exp(a*lnSX); % not normalized if a~=1
            lSZ = 1/a*log(max(lgmn,SZa));
            
            bd0 = -1/a*exp(a*lnSX).*lnSX;
            bpcd0 = (1/k*exp(a*lnSY).*lnSY + 1/(1-k)*exp(a*lnSX).*lnSX - 1/k/(1-k)*SZa.*lSZ)./a;
            u1cd1 =  1/k*exp(a*lnSY).*(lnSY-lSZ);
            u2cd2 = (a-1)*u1cd1 + exp(a*(lnSY-lSZ+lnSX));
            Fs = 'JSa t2';
        end
    else
        % a-"scaled" mixture
        SZa = (1-k)*exp(a*lnSY)+k*exp(a*lnSX); % not normalized if a~=1
        lSZ = 1/a*log(max(lgmn,SZa));
        
        if aeq(a,-b)
            % Itakura-Saito
            bd0 = 1/b*lnSX;
            bpcd0 = ( 1/k/(1-k)*lSZ - 1/k*lnSY - 1/(1-k)*lnSX )./a;
            u1cd1 = 1/k/a*expm1(a*(lnSY-lSZ));
            u2cd2 = (a-1)*u1cd1 + ( 1/k - (1/k-1)*exp(2*a*(lnSY-lSZ)) );
            Fs = 'ISa t2';
        else
            % general case of AB-div.
            bd0 = 1/b/(a+b)*exp((a+b)*lnSX);
            bpcd0 = ( 1/k*exp((a+b)*lnSY) + 1/(1-k)*exp((a+b)*lnSX) - 1/k/(1-k)*exp((a+b)*lSZ) )./(b*(a+b));
            u1cd1 = 1/k/b*exp(a*lnSY).*(exp(b*lnSY)-exp(b*lSZ));
            u2cd2 = (a-1)*u1cd1 + 1/k*( exp((a+b)*lnSY) - (1-k)*exp(2*a*lnSY).*exp((b-a)*lSZ) );
            Fs = 'AB t2';
        end
    end
else
    % type 1 mixture
    if aeq(b,0)
        if aeq(a,1)
            % gen. Kullback-Leibler
            bd0 = -nSX.*lnSX; % sum( nSY-nSX ) == 0 !!!
            bpcd0 = (1-2*k)*(nSY-nSX) + ((1-k)*nSX-k*nSY).*(lnSX-lnSY); % first term useless due to normalization
            u1cd1 = (1-k)*(nSY-nSX) + k*nSY.*(lnSY-lnSX);
            u2cd2 = (1-k)*nSX + k*nSY;
            Fs = 'gKL1 t1';
        else
            % a-"scaled" gen. Kullback-Leibler
            bd0 = -1/a*exp(a*lnSX).*lnSX;
            bpcd0 = ( (1-2*k)*(exp(a*lnSY)-exp(a*lnSX)) + a*((1-k)*exp(a*lnSX)-k*exp(a*lnSY)).*(lnSX-lnSY) )./a^2;
            u1cd1 = (1-k)/a*(exp(a*lnSY)-exp(a*lnSX)) + k*exp(a*lnSY).*(lnSY-lnSX);
            u2cd2 = (a-1)*((1-k)/a*(exp(a*lnSY)-exp(a*lnSX))+k*exp(a*lnSY).*(lnSY-lnSX)) + (1-k)*exp(a*lnSX) + k*exp(a*lnSY);
            Fs = 'gKLa t1';
        end
    elseif aeq(a,0)
        if aeq(b,1)
            % gen. Kullback-Leibler
            bd0 = -nSX.*lnSX;
            bpcd0 = (1-2*k)*(nSX-nSY) + ((1-k)*nSY-k*nSX).*(lnSY-lnSX); % first term useless due to normalization
            u1cd1 = (1-k)*nSY.*(lnSY-lnSX) + k*(nSY-nSX);
            u2cd2 = (1-k)*nSY + k*nSX;
            Fs = 'gLK1 t1';
        else
            % b-"scaled" gen. Kullback-Leibler
            bd0 = -1/b*exp(b*lnSX).*lnSX;
            bpcd0 = ( (1-2*k)*(exp(b*lnSX)-exp(b*lnSY)) + b*((1-k)*exp(b*lnSY)-k*exp(b*lnSX)).*(lnSY-lnSX) )./b^2;
            u1cd1 = (1-k)*exp(b*lnSY).*(lnSY-lnSX) + k/b*(exp(b*lnSY)-exp(b*lnSX));
            u2cd2 = (b-1)*((1-k)*exp(b*lnSY).*(lnSY-lnSX)+k/b*(exp(b*lnSY)-exp(b*lnSY))) + (1-k)*exp(b*lnSY) + k*exp(b*lnSX);
            Fs = 'gLKb t1';
        end
    elseif aeq(a,-b)
        % Itakura-Saito
        alnSYoSX = a*(lnSY-lnSX);
        bd0 = -1/a*lnSX;
        bpcd0 = ( (1-2*k)*alnSYoSX + (1-k)*expm1(-alnSYoSX) + k*expm1(alnSYoSX) )./a^2;
        u1cd1 = ( (k-1)*expm1(-alnSYoSX) + k*expm1(alnSYoSX) )./a;
        u2cd2 = 2*k-1 + (1-k)*(1+a)*exp(-alnSYoSX) + k*(1-a)*exp(alnSYoSX);
        Fs = 'ISa t1';
    else
        % general case of AB-div.
        bd0 = ( (1-k)/b + k/a )/(a+b)*exp((a+b)*lnSX);
        bpcd0 = ( (1-k)*(a*exp((a+b)*lnSX)+b*exp((a+b)*lnSY)-(a+b)*exp(a*lnSX+b*lnSY)) + k*(a*exp((a+b)*lnSY)+b*exp((a+b)*lnSX)-(a+b)*exp(a*lnSY+b*lnSX)) )/a/b/(a+b);
        u1cd1 = (1-k)/a*(exp((b+a)*lnSY)-exp(b*lnSY+a*lnSX)) + k/b*(exp((a+b)*lnSY)-exp(a*lnSY+b*lnSX));
        u2cd2 = (1-k)*(exp((b+a)*lnSY)+(b-1)/a*(exp((b+a)*lnSY)-exp(b*lnSY+a*lnSX))) + k*(exp((a+b)*lnSY)+(a-1)/b*(exp((a+b)*lnSY)-exp(a*lnSY+b*lnSX)));
        Fs = 'AB t1';
    end
end

% non-negligible entries
nni = (0<mDYp);

% cancel all discarded terms and multiply by weights
W0 = sum(nni,1)./sum(abs(bd0.*nni),1);
bpcd0 = bsxfun(@times, W0, bpcd0.*nni);
u1cd1 = bsxfun(@times, W0, u1cd1.*nni);
u2cd2 = bsxfun(@times, W0, u2cd2.*nni);

% compute cost function value
f = sum(sum(bpcd0,1),2);

% compute gradient and Hessian weights
W1 = zeros(nbr,sss,'like',Y);
W2 = zeros(nbr,sss,'like',Y);
lvl = size(PY,1);
mswe = msw; mswe(end) = mswe(end) + eps; [~,j] = max(mswe); % find max of weights, favor last one if ties
for i = 1:lvl
    if 0<msw(i)
        % recompute the similarities at level i
        nSYi = similarities(mDYp,PY{i});
        
        if i==j
            % approx of second term of second derivative
            % (replace the weighted mean with a single term)
            tmp0 = bsxfun(@times, PY{i}, u2cd2);
            W2 = W2 - ((2*nSYi-1).*tmp0 + bsxfun(@times, sum(tmp0,1), nSYi.^2));
        end
        
        tmp0 = u1cd1.*(bsxfun(@times, PY{i}, nSYi)./max(epsmax,nSY)); % avoid division with tiny numbers
        W1 = W1 + msw(i)*(bsxfun(@times, sum(tmp0,1), nSYi) - tmp0);
        W2 = W2 + msw(i)*(bsxfun(@times, sum(tmp0,1), nSYi) - tmp0).*(2*nSYi-1);
    end
end

% symmetrization
if sss<nbr
    tmp0 = 0.5*W1(ss,ss);
    W1(ss,ss) = tmp0 + tmp0'; % "late" symmetrization for "marginal"
    tmp0 = 0.5*W2(ss,ss);
    W2(ss,ss) = tmp0 + tmp0'; % "late" symmetrization for "marginal"
else
    W1 = 0.5*(W1 + W1');
    W2 = 0.5*(W2 + W2');
end

% compute the gradient and approximate diagonal Hessian
Ypd1 = zeros(size(Y),'like',Y); % gradient w.r.t. Y
Ypd2 =  ones(size(Y),'like',Y); % diag. Hessian w.r.t. Y (crude approximation)
for d = 1:dim
    Yd = Y(:,d);
    if sss<nbr
        dif = bsxfun(@minus, Yd, Yd(ss)');
    else
        dif = bsxfun(@minus, Yd, Yd'); % much faster on GPU
    end
    Ypd1(:,d) = sum( W1.*dif              ,2);
    Ypd2(:,d) = sum( W1      + W2.*dif.^2 ,2);
end

% reshape for BFGS
g = Ypd1(:);
h = Ypd2(:);

function bln = aeq(u,v,e)
% approximately equal

if nargin<3, e = 1e-2; end
bln = abs(u-v)<e;

function x = bringback(x)
try
    % try to transfer from GPU to CPU
    x = double(gather(x));
catch
    % nothing to do
end

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
    case 'double', lgmn = feval(fpt,8e-46);  % DP: mimic behavior of SP
%   case 'double', lgmn = feval(fpt,3e-324); % DP: log(3e-324) = -744.4401
end

function epmx = epsmax
% largest epsilon
epmx = 2^(-23); % this is equivalent to eps(single(1.0)), without forcing the class to be 'single'

function [nSX,lnSX] = multisimilarities(DX,PX,W)
% multiscale similarities
% DX: matrix of half squared distances
% PX: cell of precisions vectors (one row vector per level)
% W : weights (for of each scale)
% nSX: normalized similarities
% lnSX: logarithm of normalized similarities
% (gpu-compatible)

% number of levels
npl = size(PX,1);

% optional arg
if nargin<3, W = ones(npl,1,'like',DX); end
W = W./sum(W);

if npl<=1
    % reduce to single-scale similarities (log.sim. more accurate)
    [nSX,lnSX] = similarities(DX,PX{1});
else
    % for all scales
    nSX = zeros(size(DX),'like',DX);
    for i = 1:npl
        if 0<W(i), nSX = nSX + W(i)*similarities(DX,PX{i}); end
    end
    
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
zni = (DX<=1e-64);

% exploit shift invariance to improve num. accuracy
tmp = DX;
tmp(zni) = inf; % much slower with NaN, inf not slower than any regular value
DX = bsxfun(@minus, DX, min(tmp,[],1)); % subtract minimum distance

% exponential/Gaussian similarity
lSX = bsxfun(@times, -PX, DX);
lSX(zni) = -746; % smallest integer n such that exp(n)==0 in double precision (simple precision too)
SX = exp(lSX);

% normalize similarities
sSX = sum(SX,1); % "marginal" normalization factor
nSX = bsxfun(@rdivide, SX, sSX);

% log(similarities)
if nargout>1, lnSX = bsxfun(@minus, lSX, log(sSX)); end

function [PX,nSX,lnSX] = equalize(mDXp,PX,pxt,itr)
% baseline equalization
% (gpu-compatible)

% initialize precisions
if isempty(PX)
    tmp = mDXp;
    tmp(tmp<=1e-64) = inf; % much slower with NaN, inf not slower than any regular value
    tmp = mean(mDXp,1) - min(tmp,[],1); % shift invariance
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
    disp([sprintf('%3i/%3i: ',[t,itr]),sprintf('Sh.ent.(-> %2.4f) in ',Htgt),sprintf('%2.3f %2.3f %2.3f ',min(Htmp),mean(Htmp),max(Htmp))]);

    % delta H
    Htmp = Htmp - Htgt;

    % stop or update
    if all(abs(Htmp)<1e-3*abs(Htgt)), break; end

    % update
    PX = PX - max(-PX/2, min(PX/2, Htmp./sum(nSX.*(1+lnSX).*bsxfun(@minus,mDXp,sum(nSX.*mDXp,1)),1) ) ); % allow PX to be multiplied by min. 0.5 or max. 1.5
end

disp(['         precs. in '         ,sprintf('%1.4g %1.4g %1.4g ',min(PX),mean(PX),max(PX))]);


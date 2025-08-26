function [Rauc,Qauc,Bauc,R_NX,Q_NX,B_NX] = nx_scores(k,pts,X,varargin)
% Function
%
% [Rauc,Qauc,Bauc,R_NX,Q_NX,B_NX] = 
% nx_scores(k,pts,X,Y1,M1,L1,Y2,M2,L2,...,YT,MT,LT)
% nx_scores(k,pts,DX,DY1,M1,L1,DY2,M2,L2,...,DYT,MT,LT)
% nx_scores(k,pts,X,Y_M_L_cells)
% nx_scores(k,pts,DX,DY_M_L_cells)
%
% calls T times the functions pairwisedistances, coranking, and nx_trusion 
% in order to compute the Euclidean distances, the coranking matrices, and
% the intrusion and extrusion (NX) rates for each embedding Y1, Y2, ..., YT
% of  data set X. X can contain coordinates or distances. If X is a cell, 
% then averages over multiple views X{1}, X{2}, ... are considered.
% Strings M1 to MT contain each a pair of valid color and marker characters 
% (e.g. bgrcmykw followed by sdhp^v<>.ox*+). 
% Strings L1 to LT contain the legend or name of the embeddings.
% If X, Y1, ..., YT are square matrices, they are considered as distances.
% Arguments Yt, Mt, and Lt can also be passed in a cell matrix {Yt;Mt;Lt}.
% With the NX rates, the Q_NX and B_NX curves are computed.
% From the Q_NX criterion, the random baseline is subtracted to obtain
% the local continuity meta-criterion (LCMC), which is then normalized
% by the range between perfect (1) and random (K/N-1) to get R_NX.
% The R_NX criterion is the relative improvement w.r.t. a random embedding.
%
% The various curves (Q_NX, R_NX, etc.) can be displayed in diagrams
% w.r.t. neighborhood size K.
% Parameter k specifies the range for the abscissae, from 1 to k.
% Any k>1 means a linear scale for the abscissae between 1 and k.
% The cases isempty(k) and k==1 are equivalent to specifying k=nbr.
% Any k<=0 means a logarithmic scale for the abscissae, from 1 to N.
% If numel(k)>1, then k(2) is used to draw a vertical line in the diagram.
% Parameter pts is a string that specifies the demanded diagrams.
% Any character in the following list generates a new figure:
% q: Q_NX(K) only
% b: Q_NX(K) and B_NX(K) in a single plot
% l: LCMC(K)
% r: R_NX(K) only
% p: R_NX(K) and percentiles of R_NX(K)
%    The percentiles of each R_NX curve are displayed as vertical bars in
%    a second diagram. The horizontal position of the bars is given by
%    the weighted average (sum_K K*R_NX(K)) / (sum_K R_NX(K)) .
% If pts is empty, then several plots comparing various scalar scores
% (AUCs, stars) are displayed.
%
% References:
% [1] John A. Lee, Michel Verleysen.
%     Quality assessment of nonlinear dimensionality reduction: 
%     rank-based criteria.
%     Neurocomputing 2009, 72(7-9):1431-1443.
% [2] J. A. Lee, E. Renard, G. Bernard, P. Dupont, M. Verleysen
%     Type 1 and 2 mixtures of Kullback-Leibler divergences
%     as cost functions in dimensionality reduction
%     based on similarity preservation
%     Neurocomputing 2013, 112: 92-108.
% [3] John A. Lee, Diego H. Peluffo, Michel Verleysen
%     Multiscale stochastic neighbor embedding: 
%     Towards parameter-free dimensionality reduction
%     Proc. ESANN 2014, 22:177-182.
%
% Copyright J.A.Lee, May 18, 2025.

% small-scale threshold
sst = 6144;

% views (versions of X)
if iscell(X)
    % several views
    vws = length(X);
    nbr = size(X{1},1);
    for s = 1:vws
        if size(X{s},1)~=nbr, error('View sizes do not match!'); end
    end
else
    % one view
    vws = 1;
    nbr = size(X,1);
    X = {X};
end

% numbers related to data size
nmo = nbr - 1; % number minus one
nmt = nbr - 2; % number minus two

% weights
wmo = 1./(1:nmo);
wmo = wmo/sum(wmo);
wmt = 1./(1:nmt);
wmt = wmt/sum(wmt);

% get the number of embeddings
if isscalar(varargin) && iscell(varargin{1}), varargin = varargin{1}; end
rpt = floor(numel(varargin)/3);
if 3*rpt~=numel(varargin), error('Incorrect number of arguments!'); end

% default value for k
if isempty(k), k = nbr; end
if numel(k)>2, lgnd = k(3); else, lgnd = []; end
if numel(k)>1, kopt = k(2); else, kopt = []; end
k = k(1);
if k<=0
    k = nbr; % full range
    loab = true; % logarithmic scale for abscissae
else
    loab = false; % linear scale for abscissae
end
k = abs(k);
if k<1, k = k*nbr; end
k = min(nbr,abs(k));

% Euclidean distances for the data set
DX = cell(vws,1);
for s = 1:vws
    if isdist(X{s})
        DX{s} = X{s};
        disp(['Interpreting X{',num2str(s),'} as pairwise distances']);
    else
        if nbr<sst, DX{s} = psed(X{s}); end % distances only in small scale
        disp(['Interpreting X{',num2str(s),'} as coordinates']);
    end
end

% initialize outputs
Q_NX = zeros(nmo,3,rpt,vws);
B_NX = zeros(nmo,3,rpt,vws);
LCMC = zeros(nmo,3,rpt,vws);
R_NX = zeros(nmt,3,rpt,vws);
Qauc = zeros(1,3,rpt,vws);
Bauc = zeros(1,3,rpt,vws);
Rauc = zeros(1,3,rpt,vws);

% colormap
cmp = colorcube(12);

% colors, markers, and labels
clr = zeros(rpt,3);
mkr = cell(rpt,1);
lbl = cell(rpt,1);

% for each view
for s = 1:vws
    tmp1 = [];

    % for each repetition of an embedding (e.g., different methods)
    for t = 1:rpt
        % extract embedding from varargin
        Yt = varargin{3*t-2}; % t-th embedding
        if size(Yt,1)~=nbr, error(['The ',num2str(t),'-th embedding has not the right size']); end
        smp = size(Yt,3); % sample size

        % initialise
        q_nx = zeros(nmo,smp);
        b_nx = zeros(nmo,smp);
        lcmc = zeros(nmo,smp);
        r_nx = zeros(nmo,smp); % size nmo now, will be nmt

        % for each sampled embedding (e.g., repetitions for uncertainty)
        for u = 1:smp
            if nbr<=sst
                if isdist(Yt(:,:,u))
                    DYt = Yt(:,:,u);
                else
                    DYt = pairwisedistances(Yt(:,:,u)); % psed(Yt); % it does not matter that the distances are squared
                    % distance computation is cheaper than sorting just below
                end

                % compute the rates
                [tmp2,tmp1] = coranking(DX{s},DYt,tmp1); % use secret 3rd arg to avoid repetitive sorting of DX
                [nt,xt,pt,bt] = nx_trusion(tmp2);
            else
                % fast version on coordinates only (all vs subsampling)
                idx = randperm(nbr,sst); % fresh permutation
                [nt,xt,pt,bt] = nx_trusion_subs(X{s},Yt(:,:,u),idx);
            end

            % quality curves
            q_nx(:,u) = nt + xt + pt;
            b_nx(:,u) = xt - nt;
            lcmc(:,u) = q_nx(:,u) - bt; % b==(1:nmo)'./nmo
            r_nx(:,u) = lcmc(:,u) ./ (1-bt); % beware: division by zero for k==nmo
            %r_nx(:,u) = lcmc(1:end-1,u) ./ (1-bt(1:end-1)); % avoid division by zero for k==nmo
            %r_nx = bsxfun(@times, lcmc(1:end-1,u), nmo./(nmo - (1:nmo-1)')); % same result, different way
        end

        % scalar quality criteria
        tmp = wmo*q_nx;
        Qauc(1,:,t,s) = [mean(tmp,2),min(tmp,[],2),max(tmp,[],2)]; % area under Q_NX in a logplot
        tmp = wmo*b_nx;
        Bauc(1,:,t,s) = [mean(tmp,2),min(tmp,[],2),max(tmp,[],2)]; % area under B_NX in a logplot
        r_nx = r_nx(1:end-1,:); % crop size from nmo down to nmt
        tmp = wmt*r_nx;
        Rauc(1,:,t,s) = [mean(tmp,2),min(tmp,[],2),max(tmp,[],2)]; % area under R_NX in a logplot

        % quality curves
        Q_NX(:,:,t,s) = [mean(q_nx,2),min(q_nx,[],2),max(q_nx,[],2)];
        B_NX(:,:,t,s) = [mean(b_nx,2),min(b_nx,[],2),max(b_nx,[],2)];
        LCMC(:,:,t,s) = [mean(lcmc,2),min(lcmc,[],2),max(lcmc,[],2)];
        R_NX(:,:,t,s) = [mean(r_nx,2),min(r_nx,[],2),max(r_nx,[],2)];
    end
end

% average over views (useless average? for noise sampling on data?)
Q_NX = mean(Q_NX,4);
B_NX = mean(B_NX,4);
LCMC = mean(LCMC,4);
R_NX = mean(R_NX,4);
Qauc = mean(Qauc,4);
Bauc = mean(Bauc,4);
Rauc = mean(Rauc,4);

% percentiles of R_NX
Kavg = ((1:nmt) * squeeze(R_NX(:,1,:))) ./ sum(squeeze(R_NX(:,1,:)),1);
 pct = [5,10,25,50,75,90,95,100];
Rpct = prctile(squeeze(R_NX(:,1,:)),pct');

% "twice ranks" (across neighbors and then across methods)
[~,TR] = sort(squeeze(R_NX(:,1,:)),2); % avoid last row of Q_NX and LCMC
[~,TR] = sort(  TR,2); % ranks from last to first for all K
AR = wmt*(rpt+1-TR); % weighted average in a logplot
FS = wmt*5/max(1,rpt-1)*(TR-1); % five stars system (from 0 to 5)

% colors and markers
for t = 1:rpt
    % extract color from varargin
    c = varargin{3*t-1}(1);
    if isnan(str2double(c))
        switch c
            case 'r', clr(t,:) = [1,0,0];
            case 'g', clr(t,:) = [0,1,0];
            case 'b', clr(t,:) = [0,0,1];
            case 'k', clr(t,:) = [0,0,0];
            case 'w', clr(t,:) = [1,1,1];
            case 'c', clr(t,:) = [0,1,1];
            case 'm', clr(t,:) = [1,0,1];
            case 'y', clr(t,:) = [1,1,0];
            case 'o', clr(t,:) = [1,0.5,0];
            case 'p', clr(t,:) = [1,0.5,1];
            case 'R', clr(t,:) = [1,0,0]/2;
            case 'G', clr(t,:) = [0,1,0]/2;
            case 'B', clr(t,:) = [0,0,1]/2;
            case 'K', clr(t,:) = [0,0,0]/2;
            case 'W', clr(t,:) = [1,1,1]/2;
            case 'C', clr(t,:) = [0,1,1]/2;
            case 'M', clr(t,:) = [1,0,1]/2;
            case 'Y', clr(t,:) = [1,1,0]/2;
            case 'O', clr(t,:) = [1,0.5,0]/2;
            case 'P', clr(t,:) = [1,0.5,1]/2;
            otherwise
                error('Wrong color character!');
        end
    else
        clr(t,:) = cmp(str2double(c)+2,:);
    end 
    
    % extract marker and label from varargin
    mkr{t} = varargin{3*t-1}(2); % t-th color and marker
    lbl{t} = varargin{3*t};      % t-th label + blank space
end

% show ranks and stars
for t = 1:rpt
    disp([lbl{t},': av.rank= ',sprintf('\t%2.1f',AR(t)),' ',sprintf('\t%1.2f',FS(t)),' stars']);
end

% show various comparisons of scalar scores (AUCs, stars)
if isempty(pts) && nargout==0
    figure;
    
    subplot(2,2,1);
    hold on;
    for t = 1:rpt
        set(plot(mean(Qauc(1,:,t)),mean(Bauc(1,:,t)),mkr{t}),'Color',clr(t,:),'LineWidth',2);
    end
    xlabel('Qauc');
    ylabel('Bauc');
    
    subplot(2,2,2);
    hold on;
    for t = 1:rpt
        set(plot(mean(Qauc(1,:,t)),mean(Rauc(1,:,t)),mkr{t}),'Color',clr(t,:),'LineWidth',2);
    end
    xlabel('Qauc');
    ylabel('Rauc');
    
    subplot(2,2,3);
    hold on;
    for t = 1:rpt
        set(plot(mean(Qauc(1,:,t)),FS(t),mkr{t}),'Color',clr(t,:),'LineWidth',2);
    end
    xlabel('Qauc');
    ylabel('FS');
    grid on;
    
    subplot(2,2,4);
    hold on;
    for t = 1:rpt
        set(plot(mean(Rauc(1,:,t)),FS(t),mkr{t}),'Color',clr(t,:),'LineWidth',2);
    end
    xlabel('Ravg');
    ylabel('FS');
    grid on;
    
    return
end

% show curves
for c = 1:length(pts)
    cc = lower(pts(c));
    
    % initialise the figure (or reuse for single plot)
    if length(pts)>1, figure; else, cla; end
    
    if cc=='p'
        % raise axes
        subplot(2,1,1);
    end
    
    % K range, Y label, Y coordinates, Y bounds, legend position
    switch cc
        case 'l'
            kra = nmo;
            yla = 'LCMC$(K)$';
            yco = squeeze(LCMC(:,1,:));
            ymn = squeeze(LCMC(:,2,:));
            ymx = squeeze(LCMC(:,3,:));
            ylb = 0;
            yub = 100;
            lpo = 'NorthEast';
        case 'q'
            kra = nmo;
            yla = '$100 Q_{\mathrm{NX}}(K)$';
            yco = squeeze(Q_NX(:,1,:));
            ymn = squeeze(Q_NX(:,2,:));
            ymx = squeeze(Q_NX(:,3,:));
            ylb = 0;
            yub = 100;
            lpo = 'SouthEast';
        case 'b'
            kra = nmo;
            yla = '$100 B_{\mathrm{NX}}(K)$, $100 Q_{\mathrm{NX}}(K)$';
            yco = squeeze(Q_NX(:,1,:));
            ymn = squeeze(B_NX(:,2,:));
            ymx = squeeze(Q_NX(:,3,:));
            ylb = 5*floor(20*min(ymn(:)));
            yub = 5* ceil(20*max(ymx(:)));
            lpo = 'East';
        case {'r','p'}
            kra = nmt;
            yla = '$100 R_{\mathrm{NX}}(K)$';
            yco = squeeze(R_NX(:,1,:));
            ymn = squeeze(R_NX(:,2,:));
            ymx = squeeze(R_NX(:,3,:));
            ylb = 0;
            yub = 5* ceil(20*max(ymx(:)));
            lpo = 'South'; % not always optimal when loab is true
        otherwise
            disp('Incorrect plot type character, skipping...');
    end
    switch lgnd
        case -4
            lpo = 'North';
        case -3
            lpo = 'NorthWest';
        case -2
            lpo = 'West';
        case -1
            lpo = 'SouthWest';
        case 0
            lpo = 'South';
        case 1
            lpo = 'SouthEast';
        case 2
            lpo = 'East';
        case 3
            lpo = 'NorthEast';
        case 4
            lpo = 'North';
        otherwise
            
    end
    
    % prepare axes
    set(gca,'FontName','Times','FontSize',16); % 10
    hold on;
    
    % draw optional vertical line
    if ~isempty(kopt)
        if 0<kopt
            h = plot([kopt,kopt],[0,100],'k-');
            %set(h,'LineWidth',0.5);
        end
    end
    
    % draw baselines
    v1 = (1:kra)';
    if cc=='l'
        % LCMC
        h = plot(v1,100/v1(end)*v1(end:-1:1),'k-');
        set(h,'LineWidth',1);
    elseif cc=='q' || cc=='b'
        % Q_NX
        h = plot(v1,100/v1(end)*v1,'k-');
        set(h,'LineWidth',1);
    end
    if cc=='b'
        h = plot(v1,zeros(size(v1)),'k-');
        set(h,'LineWidth',1);
    end
    
    % draw isolevels
    stp = 0.1;
    for lvl = stp:stp:1-stp
        if cc=='l'
            % LCMC
            h = plot(v1,100*lvl/v1(end)*v1(end:-1:1),'k:');
            set(h,'LineWidth',1);
            
            % Q_NX in LCMC diagram
            h = plot(v1,100*(lvl-v1/v1(end)),'k:');
            set(h,'LineWidth',1);
        else
            % Q_NX or R_NX (horizontal)
            h = plot(v1,100*lvl*ones(size(v1)),'k:');
            set(h,'LineWidth',1);
            
            if cc=='q'
                % LCMC in Q_NX diagram
                h = plot(v1,100*(lvl+(1-lvl)/v1(end)*v1),'k:');
                set(h,'LineWidth',1);
            elseif cc=='r' || cc=='p'
                % Q_NX in R_NX diagram
                h = plot(v1,100*(1-(1-lvl)*nmo./(nmo-v1)),'k:');
                set(h,'LineWidth',1);
            end
        end        
    end

    % draw curves and bands
    for t = 1:rpt
        if cc=='b'
            h = plot(v1,100*B_NX(:,1,t),'--');
            set(h,'Color',clr(t,:),'LineWidth',1); % 1.5
            h = fill([v1;flipud(v1)],100*[B_NX(:,2,t);flipud(B_NX(:,3,t))],clr(t,:));
            set(h,'EdgeColor',clr(t,:),'FaceAlpha',0.25,'EdgeAlpha',0.5);
        end
        h = plot(v1,100*yco(:,t),'-');
        set(h,'Color',clr(t,:),'LineWidth',1); % 1.5
        h = fill([v1;flipud(v1)],100*[ymn(:,t);flipud(ymx(:,t))],clr(t,:));
        set(h,'EdgeColor',clr(t,:),'FaceAlpha',0.25,'EdgeAlpha',0.5);
    end
    
    % draw small markers
    if loab
        v2 = 2.^(1:floor(log2(kra)));
    else
        v2 = floor(nmt/20):floor(kra/10):nmt;
    end
    v3 = v1(v2);
    for t = 1:rpt
        if cc=='b'
            h(t) = plot(v3,100*B_NX(v2,t),mkr{t});
            set(h(t),'Color',clr(t,:),'LineWidth',1); % 2
        end
        h(t) = plot(v3,100*yco(v2,t),mkr{t});
        set(h(t),'Color',clr(t,:),'LineWidth',1); % 2
    end

    % legend
    lgd = cell(rpt,1);
    for t = 1:rpt
        tmp = [lbl{t},' $\,\,\,\,\,\,$'];
        if cc=='l'
            lgd{t} = tmp;
        elseif cc=='q' || cc=='b'
            lgd{t} = [sprintf(' $% 2.1f$ ',abs(100*mean(Qauc(1,:,t)))),tmp];
            %if Bauc(t)>0,
            %    lgd{t} = [sprintf(' $+% 2.1f$ ',abs(100*mean(Bauc(1,:,t)))),lbl{t},' $\,\,$'];
            %else
            %    lgd{t} = [sprintf(' $-% 2.1f$ ',abs(100*mean(Bauc(1,:,t)))),lbl{t},' $\,\,$'];
            %end
        else % 'r'
            lgd{t} = [sprintf(' $% 2.1f$ ',abs(100*mean(Rauc(1,:,t)))),tmp];
        end
    end
    set(legend(h,lgd,'Location',lpo),'Interpreter','LaTeX');

    % finalise the figure
    hold off;
    axis([0,k,max(-20,ylb),min(100,yub)]);
    if loab
        set(gca,'Xscale','log');
    end
    xlabel('$K$','Interpreter','LaTeX');
    ylabel( yla ,'Interpreter','LaTeX');

    if cc=='p'
        % raise and prepare axes
        subplot(2,1,2);
        set(gca,'FontName','Times','FontSize',16);
        hold on;
        
        % draw R_NX isolevels
        for lvl = stp:stp:1-stp
            h = plot(v1,100*lvl*ones(size(v1)),'k:');
            set(h,'LineWidth',1);
        end
        
        % draw bars and "whiskers" (markers)
        for t = 1:rpt
            set(plot(Kavg(t)*ones(8,1),100*Rpct( :     ,t),[mkr{t}    ]),'Color',clr(t,:),'LineWidth',2);
            set(plot(Kavg(t)*ones(5,1),100*Rpct(2:end-2,t),[mkr{t},'-']),'Color',clr(t,:),'LineWidth',2);
            set(text(Kavg(t),100*Rpct(end,t),['$\;$ ',lbl{t}]),'Rotation',45,'HorizontalAlignment','left','VerticalAlignment','bottom','FontSize',16,'Interpreter','LaTeX');
        end
        
        % finalise the figure
        xlabel( '$K^{\mathrm{avg}}$' ,'Interpreter','LaTeX');
        ylabel(['\%=[ ',sprintf('%i ',pct(1)),sprintf('%i--',pct(2:end-3)),sprintf('%i ',pct(end-2:end)),'] of $100 R_{\mathrm{NX}}(K)$'],'Interpreter','LaTeX');
        axis([max(0,10*floor(min(Kavg)/10)),min(nbr,10*ceil(max(Kavg)/10)),5*floor(20*min(Rpct(:))),5*ceil(20*max(Rpct(:)))]);
    end
end



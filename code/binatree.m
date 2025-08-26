function T = binatree(X,btt,pw2,brs)
% Binary tree building
%
% X   = coordinates
% btt = binary tree type (VP==0 by default, or KD==1)
% pw2 = tree size (p)
% brs = binary randomization switches 
%       0 = farthest from central point
%       1 = farthest from farther from central point
%       (0 or 1 for each tree level)
%
% T is a structure with fields:
% L = cell array of 2^p VP leaves
% I = leaf index
% B = full binary tree with 2*2^p branches, root in T(2)
%
% Copyright J.A.Lee, May 8, 2024.

% size
nbr = size(X,1);
dim = size(X,2);
adv = 1:dim;

% defaults
if nargin<4, brs = 0; end
if nargin<3, pw2 = floor(log2(nbr)); end % one or two per leaf (see below also)
if nargin<2, btt = 0; end % VP by default = 0

% check n
pw2 = min(abs(pw2(1)),floor(log2(nbr))); % one or two per leaf
tp2 = pow2(pw2);

% check binary randomization switches
if length(brs)<pw2+1
    brs = [brs(:);zeros(pw2+1-length(brs),1)];
else
    brs = brs(:);
end
brs = logical(brs);

% initialise tree
L = cell(tp2,1); % leaves
P = cell(2*tp2,1); % branch populations
B = zeros(2*tp2,8); % branches
I = zeros(nbr,1); % index of points in leaves (and second half of branches)

% iterative tree construction
Q = cell(2*tp2,2); % queue
Qt = 1; % queue top
Qb = Qt; % queue bottom
Q{Qt,1} = 2; % root in 2nd row of T
Q{Qt,2} = (1:nbr)'; % all points
while Qb<=Qt
    % queue
    cbi = Q{Qb,1}; % current branch index
    v = Q{Qb,2};
    
    % subset
    Xv = X(v,:);
    
    % central point (closest to average)
    [~,cpi] = min(sum(bsxfun(@minus, Xv, mean(Xv,1)).^2,2)); cpi = cpi(1);
    
    if btt==0
        % all dimensions
        sdi = 0;
        sds = adv;
    else
        % dimension with maximum stan.dev.
        [~,sdi] = max(std(Xv,[],1));
        sds = sdi;
    end
    
    % vantage point (farthest from central point)
    [mxd,vpi] = max(sum(bsxfun(@minus, Xv(:,sds), Xv(cpi,sds)).^2,2)); vpi = vpi(1);
    %[mxd,vpi] = lpsort_mex(-sum(bsxfun(@minus, Xv(:,sds), Xv(cpi,sds)).^2,2)); vpi = vpi(ceil(length(v)/4*rand(1)));

    % vantage point (farthest from farthest, to generate randomness)
    if brs(round(log2(cbi))) % rand(1)>0.5 % 0 
        [mxd,vpi] = max(sum(bsxfun(@minus, Xv(:,sds), Xv(vpi,sds)).^2,2)); vpi = vpi(1);
    end
    
    % find median of distances to vantage point
    [va0,id0,mi0] = minmedmax_mex(sum(bsxfun(@minus, Xv(:,sds), Xv(vpi,sds)).^2,2)); % remove _mex if necessary
    id1 = id0(    1:mi0);
    id2 = id0(mi0+1:end);
    
    if 2<size(va0)
        % find quartiles of distances
        [va1,~,mi1] = minmedmax_mex(va0(    1:mi0)); % remove _mex if necessary
        [va2,~,mi2] = minmedmax_mex(va0(mi0+1:end)); % remove _mex if necessary
    else
        va1 = va0(mi0);
        va2 = va0(mi0);
        mi1 = 1;
        mi2 = 1;
    end

    % build tree
    P{cbi,1} = v;
    B(cbi,:) = [numel(va0),v(cpi),v(vpi),va1(mi1),va0(mi0),va2(mi2),mxd(1),sdi];
    % branch population, centroid index, vantage point index, 1st quartile,
    % median, 3rd quartile, max.dist.to centroid, dimension index
    
    % depending on n
    if cbi<=tp2
        % two branches
        
        % left and right branches
        Qt = Qt + 1; Q{Qt,1} = 2*cbi-1; Q{Qt,2} = v(id1);
        Qt = Qt + 1; Q{Qt,1} = 2*cbi  ; Q{Qt,2} = v(id2);
    else
        % single leaf
        
        % population index
        L{cbi-tp2} = v;

        % record leaf indices
        for i = 1:length(v)
            I(v(i)) = cbi-tp2;
        end
    end
    
    % queue
    Q{Qb,2} = []; % free memory
    Qb = Qb + 1;
end

% finalise tree structure
T = struct('L',[],'P',[],'B',B,'X',X,'I',I,'nbr',nbr,'btt',btt,'pw2',pw2,'tp2',tp2);
T.L = L;
T.P = P;
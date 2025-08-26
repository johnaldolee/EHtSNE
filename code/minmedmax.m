function [y,i,mid,cnt] = minmedmax(x)
% Minimum-median-maximum:
% 
% [Y,I,mid,cnt] = minmedmax(X)
% 
% Minimum-median-maximum consists in finding the minimum, median, and 
% maximum in a column vector of length N.
% minmedmax is a specific case of quickselect.
% The time complexity is thus linear on average (O(N)), 
% in contrast with regular complete sorting (O(N*log(N))).
%
% The function shares more or less the same output syntax as that of 
% [Y,I] = sort(X,1).
% However, matrix X cannot have more than two dimensions.
% Also, argument DIM of these functions is not supported.
% The only elements that are guaranteed are the minima (Y(1,:)), 
% the maxima (Y(end,:)) and the medians (Y(mid,:).
% Also: Y(i,:)<=Y(mid,:)<=Y(j,:) for all i<=mid<=j .
%
% Optional outputs are:
%  I  : index to get Y(i,j) as X(I(i,j),j)
%  mid: index of the median in Y and I
%  cnt: count of passages in the innermost loop
%
% Copyright J.A.Lee, April 11, 2015.

% check dimension
if ~ismatrix(x), error('lpsort does not support ND arrays'); end

% size
R = uint32(size(x,1));
C = uint32(size(x,2));

% index of median
M = uint32(ceil(size(x,1)/2));

% initialize outputs
y = x;
i = repmat((1:double(R))',1,C);
cnt = 0; % operation count

% return if nothing to do
if R<=1
    y = [y;y;y];
    i = [i,i,i];
    return
end

% find min and swap
[~,idx] = min(y);
for j = 1:size(x,2)
    tmp = y(idx(j),j); y(idx(j),j) = y(1,j); y(1,j) = tmp;
    tmp = i(idx(j),j); i(idx(j),j) = i(1,j); i(1,j) = tmp;
end

% find max and swap
[~,idx] = max(y);
for j = 1:size(x,2)
    tmp = y(idx(j),j); y(idx(j),j) = y(end,j); y(end,j) = tmp;
    tmp = i(idx(j),j); i(idx(j),j) = i(end,j); i(end,j) = tmp;
end

% for each column
for j = 1:C
    % chunk bounds
    t = zeros(2,1,'uint32'); % no need for a queue, always reuse the same 
    t(1) =   2; % lower bound
    t(2) = R-1; % upper bound
    
    % while chunks are left
    while 1
        % lower and upper bounds
        lb = t(1);
        ub = t(2);
        
        % assert: in the first iteration, we have 
        % lb==1<ub and therefore lb<=M<ub
        
        % three possible random pivot values => take the intermediate one
        lpv = y(lb,j); % first
        upv = y(ub,j); % last
        if upv<lpv, pv = lpv; lpv = upv; upv = pv; end % sort
        pv = y(floor((lb+ub)/2),j); % middle
        if pv<lpv, pv = lpv; elseif pv>upv, pv = upv; end
        
        % swap around pivot
        g = lb;
        h = ub;
        while g<h
            cnt = cnt + 1;
            
            % compare left value to pivot
            yg = y(g,j);
            ig = i(g,j);
            if yg<pv
                % OK left : keep in place
                g = g + 1;
            elseif yg>pv
                % KO left : swap
                y(g,j) = y(h,j); y(h,j) = yg;
                i(g,j) = i(h,j); i(h,j) = ig;
                h = h - 1;
            end
            
            % compare right value to pivot
            yh = y(h,j);
            ih = i(h,j);
            if yh>pv
                % OK right: keep in place
                h = h - 1;
            elseif yh<pv
                % KO right: swap
                y(h,j) = y(g,j); y(g,j) = yh;
                i(h,j) = i(g,j); i(g,j) = ih;
                g = g + 1;
            elseif yg==pv,
                % yh==pv
                if g-lb<ub-h, g = g + 1; else h = h - 1; end
            end
        end
        % assert: g==h is the pivot location
        
        if M<g
            % insert left chunk if the median is in it
            t(1) =  lb; % new lower bound
            t(2) = g-1; % new upper bound
        elseif g<M
            % insert right chunk if the median is in it
            t(1) = g+1; % new lower bound
            t(2) =  ub; % new upper bound
        else
            % nothing more to do
            break
        end
        
        % assert: the new chunk (zero, one, or two) have a cumulated
        % length that is 1 or 2 elements shorter than the current chunk

        %[lb,p2,ub,pv]
        %[lb,p2,ub,pv,g,h], y, t'
    end
end

% third output
mid = double(M);

% average number of operations per column
cnt = cnt/size(x,2);

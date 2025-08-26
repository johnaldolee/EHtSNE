function [n,x,p,b] = nx_trusion_subs(X,Y,L)
% Subsampling of nx_trusion for faster computation
% Works on coordinates only, not distance matrices
% All versus subsampling
%
% Copyright J.A.Lee, May 13, 2025.

% sizes
nbr = size(X,1);
nmo = nbr - 1;
len = length(L);

% compute K = [1,...,N-1]^T and K*M = M*[1,...,N-1]^T
v1 = (1:nmo)';
v2 = len*v1;

% initialize outputs
n = zeros(nmo,1); % intrusions (off-diagonal)
x = zeros(nmo,1); % extrusions (off-diagonal)
b = v1 ./ nmo; % baseline

if nbr*len<=2^25
    % mid-scale

    % distances
    DX = zeros(nbr,len);
    DY = zeros(nbr,len);
    for l = 1:len
        DX(:,l) = sum( bsxfun(@minus, X, X(L(l),:)).^2, 2 );
        DY(:,l) = sum( bsxfun(@minus, Y, Y(L(l),:)).^2, 2 );
    end

    % sort distances
    [~,IX] = sort(DX,1);
    [~,IY] = sort(DY,1);

    % diagonal of co-ranking
    p = sum(IX(2:end,:)==IY(2:end,:),2); % diagonal of coranking

    % for each landmark
    for l = 1:len
        BB = zeros(nbr,2); % off-diagonal of coranking matrix

        % for each neighbor
        for i = 1:nmo
            ipo = i+1; % ipo = i + 1 (i plus one) => self is not a neighbor
            BB(IX(ipo,l),1) = 1; % flag as met already in X index
            BB(IY(ipo,l),2) = 1; % flag as met already in Y index
            %if IX(ipo,l)==IY(ipo,l), continue; end % already in p
            %if BB(IX(ipo,l),2), n(i) = n(i) + 1; end % the one   pair is met
            %if BB(IY(ipo,l),1), x(i) = x(i) + 1; end % the other pair is met
            inc = (IX(ipo,l)~=IY(ipo,l)); % already in p
            n(i) = n(i) + (BB(IX(ipo,l),2)&&inc); % the one   pair is met
            x(i) = x(i) + (BB(IY(ipo,l),1)&&inc); % the other pair is met
            % x & n might be swapped; to be checked?
        end
    end
else
    % large-scale

    % diagonal of co-ranking
    p = zeros(nmo,1);

    % for each landmark
    for l = 1:len
        % distances
        DXl = sum( bsxfun(@minus, X, X(L(l),:)).^2, 2 );
        DYl = sum( bsxfun(@minus, Y, Y(L(l),:)).^2, 2 );

        % sort distances
        [~,IXl] = sort(DXl,1);
        [~,IYl] = sort(DYl,1);

        BB = zeros(nbr,2); % off-diagonal of coranking matrix

        % for each neighbor
        for i = 1:nmo
            ipo = i+1; % ipo = i + 1 (i plus one) => self is not a neighbor
            BB(IXl(ipo),1) = 1; % flag as met already in X index
            BB(IYl(ipo),2) = 1; % flag as met already in Y index
            %if IXl(ipo)==IYl(ipo), continue; end % already in p
            %if BB(IXl(ipo),2), n(i) = n(i) + 1; end % the one   pair is met
            %if BB(IYl(ipo),1), x(i) = x(i) + 1; end % the other pair is met
            inc = (IXl(ipo)~=IYl(ipo)); % not in p here
            p(i) = p(i) + (~inc);
            n(i) = n(i) + (BB(IXl(ipo),2)&&inc); % the one   pair is met
            x(i) = x(i) + (BB(IYl(ipo),1)&&inc); % the other pair is met
            % x & n might be swapped; to be checked?
        end
    end
end

% accumulation and normalisation
p = cumsum(p) ./ v2;
n = cumsum(n) ./ v2;
x = cumsum(x) ./ v2;

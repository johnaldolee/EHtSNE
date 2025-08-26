function [d,p,q,r,s,g,a,t] = pairwisedistances(x,gbr,gbp,udg,aem,vbs)
% Function
%
% [d,p,q,r,s,g,a,t] = pairwisedistances(x,gbr,gbp,udg,ibd,aem)
%
% computes the pairwise distances from the coordinates in x
% (one row per observation).
% If gbr is equal to 'k', an undirected graph is build based on K-ary
% neighborhoods, where K is specified by ceil(gbp). As the graph is
% undirected with udg==true, there might actually be more than K neighbors.
% if gbp is a vector, then K is specified for several possible runs in 
% order to merge disconnected components in a hierarchical way.
% If gbr is equal to 'e', the graph construction relies on epsilon-balls
% centered on each point.
% If gbr is equal to 'd', the Delaunay triangulation is built;
% this works only in 2D or 3D.
% Distances are computed according to the shortest (weighted) paths
% in the resulting geometric graph.
% Any other value of gbr, or if it is left unspecified, leads
% to the computation of Euclidean distances (without a graph).
% With the gbr and gbp options it is also possible to specify previously
% computed distances instead of coordinates in x; those distance need not 
% be symmetric.
% Boolean flag udg converts the directed KNN or eps geometric graph into
% an undirected graph by symetrizing the adjencency matrix and taking the 
% minimum edge length of both (possibly one infinite length in the pair).
% Boolean flag udg is true by default (symmetrize to an undirected graph; 
% use udg false with care.
% Parameter aem is the angle estimation method; positive, it is the angle
% in degrees that is used as an approximation to the true angle; the cosine
% of the angle is used except for special case 60, 90, and 180; if null
% or negative, the true angle is estimated on the fly; default aem is 180.
% Boolean flag vbs controls verbosity (default is true)
% Matrix x must then be square with zero diagonal elements.
% Matrix d contains the pairwise distances d_(i<=j).
% Matrix p gives the paths backward (from j back towards i).
% Matrix q gives the paths forward (from i towards j).
% Matrix r corresponds to the ranks (number of edges in the paths).
% Matrix s contains the "stretched" distances. % NEED DETAILS !!!
% Integer g is the index of the graph gravity center.
% Matrix a is the adjacency matrix.
% Logical matrix t indicates the paths' terminal vertices
%
% Copyright J.A.Lee, June 15, 2024.

% check arguments
if nargin<2, gbr = 'a'; end
if nargin<3, gbp = []; end
if nargin<4, udg = true; end
if nargin<5, aem = 180; end
if nargin<6, vbs = true; end

% check argument values
if isempty(gbr), gbr = 'a'; end
gbr = lower(gbr(1));
if isempty(gbp)
    if gbr=='k', gbp = 6; elseif gbr=='e', gbp = 0.1; end
end

% recover or compute distances
if iscell(x) % from sneodistances
    d = x{2}; % recover precomputed (discounted) distances first
    x = x{1}; % get coordinates for angle calculation
    ed = sqrt(psed(x));
    edf = false; % assume non-Eucl. (precaution)
else
    if vbs, disp('Computing Euclidean distances...'); end
    g = x*x'; % Gram matrix of  inner products and corresponding symmetrized Euclidean distances
    d = bsxfun(@minus,diag(g),g); % distance matrix from Gram
    d = sqrt(max(0,d + d')); % max avoids negative num. inaccuracies
    ed = d;
    edf = true;
end
od = d; % keep track of original distances ATTENTION!!!

% size and dimension of data set
[nbr,dim] = size(x);

% vectors of ones
n1 = ones(nbr,1);

% build a graph and compute the shortest paths
if isempty(gbr) || (gbr~='k' && gbr~='e' && gbr~='d')
    p = []; % all empty to avoid storage of trivial values
    q = [];
    r = [];
    s = [];
    g = [];
    a = [];
    t = [];
    return;
else
    % adjacency matrix and rank matrix
    a = zeros(nbr,nbr,'uint32');
    r = uint32(nbr+1)*(ones(nbr,nbr,'uint32') - eye(nbr,'uint32'));

    % build graph edges
    if gbr == 'd' % Delaunay
        if vbs, disp('Computing Delaunay graph...'); end
        if dim<=3
            dt = delaunayTriangulation(x); % just "delaunay" in Octave
            e = edges(dt);
            for j = 1:size(e,1)
                a(e(j,1),e(j,2)) = 1;
                a(e(j,2),e(j,1)) = 1;
                r(e(j,1),e(j,2)) = 1;
                r(e(j,2),e(j,1)) = 1;
                % symmetric already (cfr symmetrization below)
            end
            d(~(a+eye(nbr,'uint32'))) = inf;
        else
            gbr = 'k';
            gbp = 2*(dim+1);
            % delaunayn is possible in Octave
        end
    end
    if gbr == 'k' % K-ary
        if 0<gbp(1) && gbp(1)<nbr
            if vbs, disp(['Computing KNN graph with K = ',num2str(gbp)]); end
            [~,idx] = sort(d);
            for j = 1:nbr
                tmp = idx(2:(1+gbp(1)),j);
                a(tmp,j) = 1;
                r(tmp,j) = 1;
                d(idx((2+gbp(1)):end,j),j) = inf;
            end
        else
            if vbs, disp('Having KNN graph with K = N (all points are neighbours)'); end
            % reinitialize adjacency matrix and rank matrix; d remains
            a = ones(nbr,nbr,'uint32') - eye(nbr,'uint32'); % all but self
            r = ones(nbr,nbr,'uint32') - eye(nbr,'uint32'); % back to self
        end
    elseif gbr == 'e' % epsilon ball
        if vbs, disp(['Computing epsilon-ball graph with ',num2str(gbp(1))]); end
        % gbp = gbp * mD; % makes everything proportional
        for j = 1:nbr
            for i = 1:nbr
                if d(i,j)>gbp(1)
                    d(i,j) = inf;
                elseif i~=j
                    a(i,j) = 1;
                    r(i,j) = 1;
                end
            end
        end
    end

    % convert into an undirected graph (symmetrize adjacency, not distances)
    if udg
        disp('Preliminary symmetrization of the KNN graph');
        a = max(a,a'); % impose adjacency both ways if one way
        r = min(r,r'); % minimum of ranks
        d = od; d(~a & ~eye(nbr)) = inf; % recompose distances
        % inf.mask is symmetric but distances are not necessarily so!!!
    end

    if aem<0
        if vbs, disp('Adding squared edge lengths...'); end
    elseif aem==0
        if vbs, disp('Estimating path edge angles on the fly...'); end
        if ~edf, disp('Warning: no guarantee for non-Euclidean distances!'); end
    elseif aem==60
        if vbs, disp('Assuming path edge angles of  60°'); end
    elseif aem==90
        if vbs, disp('Assuming path edge angles of  90°'); end
    elseif aem==180
        if vbs, disp('Assuming path edge angles of 180°'); end
    else % 0<aem<=180
        aem = min(aem,180);
        if vbs, disp(['Assuming path edge angles of ',num2str(aem),'°']); end
    end

    % first run of Floyd
    disp(['Floyd-Warshall level 1, K or eps = ',num2str(gbp(1))]);
    if nargout==1 && aem==180
        % Floyd algorithm to compute (only) the length of all shortest paths
        % no symmetrization
        for j = 1:nbr
            d = min(d,bsxfun(@plus,d(:,j),d(j,:))); % matricial form of Floyd
        end
        s = d; % distances are stretched already
    else
        % Floyd algorithm to compute all shortest paths (paths, lengths, ranks)
        if aem<180
            d = 1/2*(d+d'); % symmetrize: go in and out
        end
        s = d; % stretched geodesic distances

        % path matrix
        ni = uint32(1:nbr); % 1,2,3,...,N
        p = ni(n1,:); % row (1...N) replicated N times (last edge back) (this is j)
        q = ni(n1,:)'; % column (1...N)' replicated N times (first edge forward) (this is i)
        % d_ij is distance from j to i
        % p_ij is path from i back to j ((1...N) as rows)
        % q_ij is path from i towards j ((1...N) as columns)

        % Floyd iterations (for each vertex)
        % tks = zeros(nbr,6);
        for j = 1:nbr
            tms = bsxfun(@plus,s(:,j),s(j,:)); % simple addition for stretched geodesic distances

            % length of the combined paths to i from k through j
            if aem<0 % not recommended => it reconcentrates
                % quadratic addition <=> zig-zags with 90° edge angles
                tmd = sqrt(bsxfun(@plus,d(:,j).^2,d(j,:).^2));
                % could spare the ^2 sqrt alternances
            elseif aem==180
                % linear addition <=> straight paths with 180° edge angles
                tmd = bsxfun(@plus,d(:,j),d(j,:)); % works for asymmetric
            elseif aem==90
                tmd = bsxfun(@plus,d(:,j),d(j,:)); % works for asymmetric
                df = od(q(:,j),j); % refer to original distances to get untouched edges (forward and backward)
                db = od(p(j,:),j)'; % same; transpose otherwise it is a column, weirdly enough
                tmd = tmd + 1/2*(sqrt(bsxfun(@plus,df.^2,db.^2)) - bsxfun(@plus,df,db));                
            elseif 0<aem && aem<180
                tmd = bsxfun(@plus,d(:,j),d(j,:)); % works for asymmetric
                df = od(q(:,j),j); % refer to original distances to get untouched edges (forward and backward)
                db = od(p(j,:),j)'; % same; transpose otherwise it is a column, weirdly enough
                tmd = tmd + 1/2*(sqrt(max(0,bsxfun(@plus,df.^2,db.^2) - 2*bsxfun(@times,df,db)*cos(pi/180*aem))) - bsxfun(@plus,df,db));                
            else % aem==0
                % Pythagorean addition <=> zig-zags exact angle
                tmd = bsxfun(@plus,d(:,j),d(j,:)); % works for asymmetric

                % discount for angle p&q first and last edges
                % use p and q; compute all?
                % angle triplet around j is (p(j,:),j,q(:,j))
                % forward starting from j is q(:,j)
                % backward ending to j is p(j,:)
                % ac^2 = ab^2 + bc^2 - 2 ab bc cos
                % cos = (ab^2 + bc^2 - ac^2) / (2 ab bc)
                % angle discount: -2 dist.ab dist.bc cos
                df = ed(q(:,j),j); % refer to original distances to get untouched edges
                db = ed(p(j,:),j)'; % same; transpose otherwise it is a column, weirdly enough
                ca = max(-1, (bsxfun(@plus, df.^2, db.^2) - ed(q(:,j),p(j,:)).^2) ./ (eps + bsxfun(@times, 2*df, db))); % max(-1,x) to prevent failure of triangle inequality; defaults at 180°
                df = od(q(:,j),j); % refer to original distances to get untouched edges
                db = od(p(j,:),j)'; % same; transpose otherwise it is a column, weirdly enough
                tmd = tmd + 1/2*(sqrt(max(0,bsxfun(@plus,df.^2,db.^2) - 2*bsxfun(@times,df,db)*ca)) - bsxfun(@plus,df,db));
            end
            % d_(i<=j) + d(j<=k) = d(i<=j<=k) = d(i<=k)

            % rank of the combined paths to i from k through j
            tmu = bsxfun(@plus,r(:,j),r(j,:)); % works for asymmetric

            % logical indices (combined shorter than initial paths)
            lidx = tmd<d; % keep track of minimum

            % update lengths and ranks (logical indexing)
            d(lidx) = tmd(lidx); % realize min. on distances
            r(lidx) = tmu(lidx); % realize min. on ranks
            s(lidx) = tms(lidx); % realize min. on stretched distances

            % update paths backward (p)
            pj = p(:,j); % path indices when passing through j
            tmu = pj(:,n1); % replicate in N columns
            p(lidx) = tmu(lidx); % realize min. on paths (p)

            % update paths forward  (q) NEW!!!
            qj = q(j,:); % path indices when passing through j
            tmu = qj(n1,:); % replicate in N columns
            q(lidx) = tmu(lidx); % realize min. on paths (q)

            % tks(j,:) = [d(701,707),d(117,707),double([p(701,707),p(117,707),q(701,707),q(117,707)])];
        end
%         figure;
%         subplot(1,3,1);
%         plot(tks(:,1:2));
%         title('d');
%         subplot(1,3,2);
%         plot(tks(:,3:4));
%         title('p');
%         subplot(1,3,3);
%         plot(tks(:,5:6));
%         title('q');
%         %tks
    end
    
    if gbr=='k' % multi-scale only for K nearest component (not epsilon!)
        % iterate over "scales"
        for t = 2:numel(gbp)
            % mention that there are more than one connected component
            mid = isinf(d); % mask of infinite distances
            dcf = any(any(mid));
            if dcf
                disp('  More than one connected component! Merging...');
                disp(['FW level ',num2str(t),', K = ',num2str(gbp(t))]);
            else
                disp('  Single fully-connected component! Exiting FW...');
                break;
            end
            
            % count and label the disconnected components
            ndc = 1; % number of disconnected components
            sgr = zeros(nbr,nbr); % signatures of disconnected components
            idc = ones(1,nbr); % index of disconnected components
            sgr(:,1) = mid(:,1);
            for i = 1:nbr
                idx = find(all(bsxfun(@eq,sgr(:,1:ndc),mid(:,i)))); % compare to existing signatures
                if idx>0
                    % register in existing
                    idc(i) = idx; % label
                else
                    % create new signature
                    ndc = ndc + 1; % increase count
                    sgr(:,ndc) = mid(:,i); % record new signature
                    idc(i) = ndc; % label
                end
            end

            disp(['  Number of disconnected components: ',num2str(ndc)])
            %figure; histogram(idc);

            % find K nearest components and compute min.distance in blocks
            DDC = zeros(ndc); % distances to disconnected components
            iDC = zeros(ndc); % index i in block
            jDC = zeros(ndc); % index j in block
            for i = 1:ndc
                for j = 1:ndc % ASYMMETRIC; i-1 % ASSUMES SYMMETRY !!!!!
                    imk = idc==i; % logical mask of component i
                    jmk = idc==j; % logical mask of component j
                    ios = find(imk); % indices in the whole matrix
                    jos = find(jmk); % indices in the whole matrix
                    blk = od(imk,jmk); % build block; keep track of Eucl.dist.
                    [dmn,idx] = min(blk(:)); % grand minimum in block
                    [ibk,jbk] = ind2sub(size(blk),idx); % compute i and j in block
                    iDC(i,j) = ios(ibk); % i in the whole matrix 
                    %iDC(j,i) = ios(ibk); % symmetric
                    jDC(i,j) = jos(jbk); % j in the whole matrix
                    %jDC(j,i) = jos(jbk); % symmetric
                    DDC(i,j) = dmn; % record grand minimum
                    %DDC(j,i) = dmn; % symmetric
                end
            end

            % prepare for Floyd (subset of iterations for new distances)
            ifw = zeros(1,nbr);
            [~,idx] = sort(DDC,1); % sort distances to identify KNComp
            for j = 1:ndc
                tmp = idx(2:min(1+gbp,ndc),j); % exclude self-component, go up to min(1+K,#Comp)
                for k = 1:min(gbp,ndc-1)
                    bi = iDC(tmp(k),j); % index i
                    bj = jDC(tmp(k),j); % index j
                    ifw(bi) = 1; % index to be revisited by FW
                    ifw(bj) = 1; % index to be revisited by FW
                    
                    % distance
                    d(bi,bj) = od(bi,bj);
                    d(bj,bi) = od(bj,bi); % Eucl.dist. is symmetric
                    
                    % adjacency
                    a(bi,bj) = 1;
                    a(bj,bi) = 1;

                    % path backward
                    p(bi,bj) = bj;
                    p(bj,bi) = bi;

                    % path forward
                    q(bi,bj) = bi; % wild guess !!!
                    q(bj,bi) = bj; % wild guess !!!

                    % rank
                    r(bi,bj) = 1;
                    r(bj,bi) = 1;

                    % stretched
                    s(bi,bj) = d(bi,bj);
                    s(bj,bi) = d(bj,bi);
                end
            end
            %figure; imagesc(d);

            % rerun Floyd (check consistency with first run above!)
            if nargout==1 && 0<aem
                % subset of FW iterations
                for j = 1:nbr
                    if ifw(j)
                        d = min(d,bsxfun(@plus,d(:,j),d(j,:))); % matricial form of Floyd
                    end
                end
                s = d; % distances are stretched already
            else
                % subset of FW iterations
                for j = 1:nbr
                    if ifw(j)
                        tms = bsxfun(@plus,s(:,j),s(j,:));

                        % length of the combined paths from i to k through j
                        if aem<0
                            tmd = sqrt(bsxfun(@plus,d(:,j).^2,d(j,:).^2)); % works for asymmetric
                        elseif aem==180
                            tmd = bsxfun(@plus,d(:,j),d(j,:)); % works for asymmetric
                        elseif aem==90
                            tmd = bsxfun(@plus,d(:,j),d(j,:)); % works for asymmetric
                            df = od(q(:,j),j);
                            db = od(p(j,:),j)'; % transpose otherwise it is a column, weirdly enough
                            tmd = tmd + 1/2*(sqrt(bsxfun(@plus,df.^2,db.^2)) - bsxfun(@plus,df,db));                
                        elseif 0<aem && aem<180
                            tmd = bsxfun(@plus,d(:,j),d(j,:)); % works for asymmetric
                            df = od(q(:,j),j);
                            db = od(p(j,:),j)'; % transpose otherwise it is a column, weirdly enough
                            tmd = tmd + 1/2*(sqrt(max(0,bsxfun(@plus,df.^2,db.^2) - 2*bsxfun(@times,df,db)*cos(pi/180*aem))) - bsxfun(@plus,df,db));                
                        else % aem==0
                            tmd = bsxfun(@plus,d(:,j),d(j,:)); % works for asymmetric
                            df = ed(q(:,j),j);
                            db = ed(p(j,:),j)'; % transpose otherwise it is a column, weirdly enough
                            ca = max(-1, (bsxfun(@plus, df.^2, db.^2) - ed(q(:,j),p(j,:)).^2) ./ (eps + bsxfun(@times, 2*df, db)));
                            df = od(q(:,j),j);
                            db = od(p(j,:),j)'; % transpose otherwise it is a column, weirdly enough
                            tmd = tmd + 1/2*(-bsxfun(@plus,df,db) + sqrt(max(0,bsxfun(@plus,df.^2,db.^2) - 2*bsxfun(@times,df,db)*ca)));
                        end
                        % d_(i<=j) + d(j<=k) = d(i<=j<=k) = d(i<=k)

                        % rank of the combined paths from i to k through j
                        tmu = bsxfun(@plus,r(:,j),r(j,:)); % works for asymmetric

                        % logical indices (combined shorter than initial paths)
                        lidx = tmd<d; % keep track of minimum

                        % update lengths and ranks (logical indexing)
                        d(lidx) = tmd(lidx); % realize min. on distances
                        r(lidx) = tmu(lidx); % realize min. on ranks
                        s(lidx) = tms(lidx); % realize min. on stretched

                        % update paths
                        pj = p(:,j); % path indices when passing through j
                        tmu = pj(:,n1); % replicate in N columns
                        p(lidx) = tmu(lidx); % realize min. on paths

                        % update paths forward  (q) NEW!!!
                        qj = q(j,:); % path indices when passing through j
                        tmu = qj(n1,:); % replicate in N columns
                        q(lidx) = tmu(lidx); % realize min. on paths (q)
                    end
                end
            end
            figure; imagesc(d);
        end
    end

    % identify terminal nodes
    if nargout>1 || dim==2
        t = true(nbr,nbr); % path terminal nodes
        for j = 1:nbr
            % could use indexing with path column (beware of diagonal)
            for i = [1:j-1,j+1:nbr] % skip diagonal
                if p(i,j)>0
                    t(p(i,j),j) = false; % cancel all nodes that are pointed to
                end
            end
        end
    end

    % center of mass
    if any(any(isinf(d))) % retest for disconnected
        g = [];
    else
        % compute mass center (computed from distances)
        [~,idx] = sort(sum(d.^2));
        g = idx(1);
    end

    % scalar products % name clash on s
    %s = zeros(nbr,nbr);
    % for j = 1:nbr
    %     for i = 1:nbr
    %         if i==j || p(i,j)==0 || p(p(i,j),j)==0, continue; end % avoid self and zero-length edges
    %         s(i,j) = ( x(i,:) - x(p(i,j),:) )*( x(p(i,j),:)-x(p(p(i,j),j),:) )'; % scalar product
    %         s(i,j) = s(i,j) ./ max( sum(( x(i,:) - x(p(i,j),:) ).^2), sum(( x(p(i,j),:)-x(p(p(i,j),j),:) ).^2) ); % normalize (max.norm^2)
    %         %s(i,j) = s(i,j) ./ (sqrt(sum(( x(i,:) - x(p(i,j),:) ).^2))*sqrt(sum(( x(p(i,j),:)-x(p(p(i,j),j),:) ).^2)));
    %     end
    % end
    % min(s(:))
    % max(s(:))

    % various graphs...
    if ~isempty(g) && dim==2 && vbs
        figure;
        hold on;

        % all edges
        if nbr<=200
            for i = 1:nbr
                for j = i+1:nbr
                    if a(i,j)
                        plot([x(i,1),x(j,1)],[x(i,2),x(j,2)],'c-');
                    end
                end
            end
        else
            disp('Not drawing full graph (too many edges)');
        end

        % points
        scatter(x(:,1),x(:,2),30,d(:,g),'o','filled'); % points
        colormap jet;
        colorbar;
        plot(x(g,1),x(g,2),'ro'); % g center / root

        % path edges
        for i = 1:nbr
            j = p(i,g);
            if j
                plot([x(i,1),x(j,1)],[x(i,2),x(j,2)],'b-');
                %[i,g,s(i,g)]
                %text(x(i,1),x(i,2),[' ',num2str(round(s(i,g)))]);
            end
        end
        plot(x(t(:,g),1),x(t(:,g),2),'b*'); % terminal points

        title(['SP tree for G center ',num2str(g)]);
        axis equal;
    end
    if nbr<=3000 && vbs
        figure;

        subplot(2,2,1);
        plot(d(:),s(:),'k.');
        axis equal;
        xlabel('Raw SP');
        ylabel('Stretched SP');
        title('Shepard diagram');

        subplot(2,2,2);
        plot(od(:),s(:),'k.');
        axis equal;
        xlabel('Orig.dist.');
        ylabel('Stretched SP');
        title('Shepard diagram');

        subplot(2,2,4);
        plot(od(:),d(:),'k.');
        axis equal;
        xlabel('Orig.dist.');
        ylabel('Raw SP');
        title('Shepard diagram');

    end
end

%% Create/load data

% data identifier
if ~exist('did','var'), did = 'c'; end % Sphere/Digits/Coil/Frey
spd = 3; % Sphere dimension
rld = false; % reload

% perplexity (default)
if ~exist('pxt','var'), pxt = 32; end 

% for reproducibility
stream = RandStream.getGlobalStream;
reset(stream);

% Data generation
nbr = 8192;%4*8192; % for ESANN 2024
switch did
    case 'H'
        % Penta hierarchies
        nrb = 5;
        if ~exist('lvl','var'), lvl = 6; end
        nbr = nrb^lvl;
        dim = nrb*lvl;
        X = zeros(nbr,dim);
        for l = 1:lvl
            X(:,l*nrb-nrb+1:l*nrb) = repmat(2^l*randn(nbr/nrb^(l-1),nrb),nrb^(l-1),1);
        end
        L = repmat((1:nrb)',nbr/nrb,1);
        str = ['HierBlob_',num2str(nrb),'_',num2str(lvl),'_',num2str(pxt)];
        colmap = jet;
        %L = (1:nbr)';
        %colmap = X(:,end-2:end);
        %colmap = (colmap-min(colmap))./(max(colmap)-min(colmap));
    case 'o'
        % Circle and noise as in Kobak
        nbr = 6000;
        dim = 10;
        r = rand(nbr,1);
        X = [cos(2*pi*r),sin(2*pi*r),0.01*randn(size(r,1),dim-2)];
        L = round(nbr*r/72);
        str = 'NoisyCircle';
        colmap = hsv;
        figure;
        plot(X(:,1),X(:,2),'k.');
        axis equal;
        axis off;
    case 'u'
        % Uniform random distribution in a sphere
        idm = 3;
        str = ['Unif',num2str(idm)];
        if rld
            load(['data_',str]);
        else
            rbn = ceil(nbr*( (2/sqrt(pi))^idm *gamma(idm/2+1) ));
            X = rand(rbn,idm) - 0.5; % centered, R = 0.5
            L = 1 + (X>0)*2.^(1:spd)'; % quadrants
            K = ceil(nbr/3);
            save(['data_',str],'X','L','str');
        end            
        colmap = jet(64);
        figure;
        if idm<3
            scatter(X(:,1),X(:,2),30,L,'o','filled');
        else
            scatter3(X(:,1),X(:,2),X(:,3),30,L,'o','filled');
            view(15,75); axis equal;
        end
        colormap(colmap);
        set(gca,'Fontname','Times','Fontsize',12);
    case 's'
        % Sphere
        nbr = 3000;
        str = ['Sph',num2str(spd)];
        if rld
            load(['data_',str]);
        else
            X = randn(nbr,3);
            X = bsxfun(@rdivide, X, sqrt(sum(X.^2,2)));
            if spd>3
                X = [X,0.5*randn(nbr,spd-3)./sqrt(spd-3)];
            end
            L = 1 + (X>0)*2.^(1:spd)'; % quadrants
            %L = 32+64/180*atan2(X(:,1),X(:,2));
            K = ceil(nbr/3);
            save(['data_',str],'X','L','str');
        end
        colmap = 1/3*[3,2,1;2,1,0;1,2,3;0,1,2;2,3,1;1,2,0;3,1,2;2,0,1];
        figure;
        scatter3(X(:,1),X(:,2),X(:,3),30,L,'o','filled');
        view(15,75); axis equal;
        colormap(colmap);
        set(gca,'Fontname','Times','Fontsize',12);
    case 'r'
        % Swiss roll
        nbr = 6000;
        str = 'SwissRoll';
        if rld
            load(['data_',str]);
        else
            revo = 1;
            step = 1;
            Z = 1 - 2*rand(nbr,2); % latent space
            X = zeros([nbr,3]);
            X(:,1) = (step*sqrt(2+2*Z(:,1))) .* cos(2*pi*revo*sqrt(2+2*Z(:,1)));
            X(:,2) = (step*sqrt(2+2*Z(:,1))) .* sin(2*pi*revo*sqrt(2+2*Z(:,1)));
            X(:,3) = 2*Z(:,2);
            L = ceil(8+8*(Z(:,2)));
        end
        colmap = hsv(64);
        figure;
        scatter3(X(:,1),X(:,2),X(:,3),30,L,'o','filled');
        view(15,75); axis equal;
        colormap(colmap);
        set(gca,'Fontname','Times','Fontsize',12);
    case 't'
        % Toroidal helix
        str = ['Tor',num2str(spd)];
        if rld
            load(['data_',str]);
        else
            nl = 30; % number of loops
            U = (0:nbr-1)';
            %U = (nbr-1)*rand(nbr,1);
            L = rem(U,nbr/nl);
            U = 2*pi/nbr*U;
            X = [(2+cos(nl*U)).*cos(U),(2+cos(nl*U)).*sin(U),sin(nl*U)];
            %X = X + 0.001*randn(size(X));
            %X = bsxfun(@plus, X, 0.01/nbr*((1:1:nbr)+(nbr:-1:1)-(nbr+1)/2)');
            K = round(nbr/nl);
            save(['data_',str],'X','L','str');
        end
        colmap = hsv(64);
        figure;
        scatter3(X(:,1),X(:,2),X(:,3),30,L,'o','filled');
        view(10,45); axis equal;
        colormap(colmap);
        set(gca,'Fontname','Times','Fontsize',12);
    case 'd'
        % MNIST digits
        nbr = 60000;
        if rld
            load('data_MNIST');
        else
            load 'mnist_train_1.mat'
            if nbr<60000
                sel = randperm(60000);
                sel = sel(1:nbr);
                %sel = 1:60:60000;
                X = train_X(sel,:);
                L = train_labels(sel);
            else
                X = train_X;
                L = train_labels;
            end
            K = ceil(nbr/10);
            str = 'MNIST';
            save(['data_',str],'X','L','str');
        end
        colmap = jet(64); colmap = colmap(5:6:64,:);
        nr = 10;
        nc = 10;
        Xs = X(1:100,:);
        for i = 1:10
            tmp = X(L==i,:);
            Xs(nc*(i-1)+(1:nc),:) = tmp(1:nc,:);
        end
        [r,c] = meshgrid(1:nr,1:nc);
        figure;
        Xss = Xs;
        iss = reshape(1:nr*nc,nr,nc)';%iss = 1:nr*nc;
        mosaic1([c(:),r(:)],Xss(iss,:),28,28,nc,nr);%mosaic1([c(:),r(:)],Xss(iss,:),28,28,nr,nc);
        scatter((1.5:(nc-2)/(nc-1):nc-0.5)',(nr+0.5)*ones(nc,1),150,colmap,'o','filled');%scatter((nc+0.5)*ones(nr,1),(1.5:(nr-2)/(nr-1):nr-0.5)',150,colmap,'o','filled');
        axis([0,nc,0,nr+1]);%axis([0,nc+1,0,nr]);
        colormap(flipdim(gray,1));
        axis equal;
    case 'h'
        % Fashion MNIST clothes
        nbr = 60000;
        rld = true; % to recover fmssne run CdeBodt
        if rld
            load('data_fMNIST');
        else
            T = readtable('FashionMNIST/fashion-mnist_train.csv');
            L = T{:,1};
            X = T{:,2:end};
            if nbr<60000
                sel = randperm(60000);
                sel = sel(1:nbr);
                %sel = 1:60:60000;
                X = X(sel,:);
                L = L(sel);
            end
            K = ceil(nbr/10);
            str = 'fMNIST';
            save(['data_',str],'X','L','str');
        end
        colmap = jet(64); colmap = colmap(5:6:64,:);
        nr = 10;
        nc = 10;
        Xs = X(1:100,:);
        for i = 1:10
            tmp = X(L==(i-1),:);
            Xs(nc*(i-1)+(1:nc),:) = tmp(1:nc,:);
        end
        [r,c] = meshgrid(1:nr,1:nc);
        figure;
        Xss = Xs;
        iss = reshape(1:nr*nc,nr,nc)';%iss = 1:nr*nc;
        mosaic1([c(:),r(:)],Xss(iss,:),28,28,nc,nr);%mosaic1([c(:),r(:)],Xss(iss,:),28,28,nr,nc);
        scatter((1.5:(nc-2)/(nc-1):nc-0.5)',(nr+0.5)*ones(nc,1),150,colmap,'o','filled');%scatter((nc+0.5)*ones(nr,1),(1.5:(nr-2)/(nr-1):nr-0.5)',150,colmap,'o','filled');
        axis([0,nc,0,nr+1]);%axis([0,nc+1,0,nr]);
        colormap(flipdim(gray,1));
        axis equal;
    case 'c'
        % COIL-20
        load 'coil_1440.mat'
        L = reshape(bsxfun(@plus, (1:20), zeros(72,1)), [1440,1]);
        stp = 1;%2;
        X = X(1:stp:end,:);
        L = L(1:stp:end);
        colmap = jet(20);
        str = 'Coil20';
        save(['data_',str],'X','L','str');
        nbr = size(X,1);
        nr = 9;%20;
        nc = 20;%9;
        [r,c] = meshgrid(1:nr,1:nc);
        figure;
        Xss = X(1:(nr-1)/stp:1440/stp,:);
        iss = reshape(1:nr*nc,nr,nc)';%iss = 1:nr*nc;
        mosaic1([c(:),r(:)],Xss(iss,:),128,128,nc,nr);%mosaic1([c(:),r(:)],Xss(iss,:),128,128,nr,nc);
        scatter((1.5:(nc-2)/(nc-1):nc-0.5)',(nr+0.5)*ones(nc,1),150,colmap,'o','filled');%scatter((nc+0.5)*ones(nr,1),(1.5:(nr-2)/(nr-1):nr-0.5)',150,colmap,'o','filled');
        axis([0,nc,0,nr+1]);%axis([0,nc+1,0,nr]);
        axis equal;
        colormap gray;

        % latent space
        nc = 72/stp;
        nr = 20;
        [r,c] = meshgrid(1:nr,1:nc);
        Z = [c(:),r(:) + 2*(c(:)/nc-0.5).^2];        
    case 'C'
        % COIL-100
        load coil100.mat
        str = 'Coil100';
        colmap = jet(100);
        stp = 1;
        X = X(1:stp:end,:);
        L = L(1:stp:end);
    case 'f'
        % Frey faces
        load frey_rawface.mat
        stp = 1;
        X = double(ff(:,1:stp:end)');
        nbr = size(X,1);
        L = ceil(131/nbr*(1:nbr)'); % 131 segments
        colmap = jet(64); colmap = colmap(4:4:64,:);
        str = 'FreyF';
        nr = 16;
        nc = 16;
        [r,c] = meshgrid(1:nr,1:nc);
        figure;
        Xss = X;
        iss = reshape(1:nr*nc,nr,nc)';%iss = 1:nr*nc;
        mosaic1([c(:),r(:)],Xss(iss,:),20,28,nc,nr);%mosaic1([c(:),r(:)],Xss(iss,:),20,28,nr,nc);
        scatter((1.5:(nc-2)/(nc-1):nc-0.5)',(nr+0.5)*ones(nc,1),150,colmap,'o','filled');%scatter((nc+0.5)*ones(nr,1),(1.5:(nr-2)/(nr-1):nr-0.5)',150,colmap,'o','filled');
        axis([0,nc,0,nr+1]);%axis([0,nc+1,0,nr]);
        colormap(gray);
        axis equal;
    case 'g'
        % Google
        T = readtable('google_review_ratings.csv'); 
        X = T{:,2:end-1};
        X(isnan(X)) = 0;
        L = 1 + sum(X==0,2);
        stp = 1;
        X = X(1:stp:end,:);
        L = L(1:stp:end);
        str = 'Google';
    case 'p'
        % phoneme
        load phoneme.mat
        stp = 1;
        X = X(1:stp:end,:);
        L = 1 + L(1:stp:end)';
        str = 'Phoneme';
        colmap = hsv;
    case 'm'
        X = readNPY('X.npy');
        L = readNPY('Y.npy');
        stp = 1;
        X = X(1:stp:end,:);
        L = 1 + L(1:stp:end);
        str = 'MouseRNA';
        colmap = jet(133);
    case 'M'
        % Mammoth
        T = readtable('mammoth_a.csv');
        stp = 165;
        X = T{1:stp:end,:};
        L = X(:,2);
        str = 'Mammoth';
        colmap = hsv;
        figure;
        %plot3(X(:,1),X(:,2),X(:,3),'k.');
        scatter3(X(:,1),X(:,2),X(:,3),20,L,'o','filled');
        colormap(colmap);
        axis equal;
        axis off;
    case 'v'
        % covertype
        load covtype_dataset.mat
        X = covtype(:,1:54);
        X = bsxfun(@minus, X, mean(X,1));
        X = bsxfun(@rdivide, X, std(X,[],1));
        L = covtype(:,55);
        colmap = lines(7);
        str = 'CoverType';
    otherwise
        error('Unknown data set');
end
%fnm = [str,'_dat'];
%exportgraphics(gcf,[fnm,'.pdf']);
%exportgraphics(gcf,[fnm,'.eps']);
%exportgraphics(gcf,[fnm,'.png']);

% size
nbr = size(X,1);
dim = size(X,2);

%% ESANN 2024 NeuCom ext. (this is the one!)

% PCA down to tractable dimension (but still HD >>2) 
Xc = bsxfun(@minus, X, mean(X,1));
if size(X,2)>32
    [U,S,~] = svds(Xc,32);
    Xc = U*S;
    figure;
    plot(diag(S));
    title('Eigenvalue spectrum (1 to 32)');
end

% #repetitions
rep = 5;
tmg = zeros(rep,3); % columns are t-SNE rnd, t-SNE pca, EH t-SNE 

% reset seed
stream = RandStream.getGlobalStream;
reset(stream);

% shared parameters
ee = 12; % EE
own = (nbr<=6144); % own t-SNE or Matlab's
itr = 1000;
pcs = 10.^(-5:-1); % spread of PCA/rand initialization
dof = 1;

% run all methods
if own
    % specific parameters
    shw = false;

    Yrr = zeros(nbr,2,rep);
    Yrp = zeros(nbr,2,rep);
    for r = 1:rep
        % random init.
        tic
        Yrr(:,:,r) = basictsne(Xc,    [],2,pxt,itr,dof,ee,shw); % regular random init.
        tmg(r,1) = toc;

        % PCA init.
        tic
        Yrp(:,:,r) = basictsne(Xc,pcs(r),2,pxt,itr,dof,ee,shw); % regular PCA init.
        tmg(r,2) = toc;
    end
else
    % specific parameters
    opt = struct('MaxIter',itr,'TolFun',1e-10,'OutputFcn',[]);
    if nbr<1024
        alg = 'exact';
    else
        alg = 'barneshut';
    end

    Yrr = zeros(nbr,2,rep);
    Yrp = zeros(nbr,2,rep);
    for r = 1:rep
        % random initi.
        tic
        Y = pcs(r)*randn(nbr,2);
        std(Y)
        Yrr(:,:,r) = tsne(Xc,'Algorithm',alg,'Distance','euclidean','NumDimensions',2,'NumPCAComponents',0,'InitialY',Y,'Perplexity',pxt,'Exaggeration',ee,'LearnRate',nbr,'Options',opt,'Verbose',2);
        tmg(r,1) = toc;

        % PCA init.
        tic
        [U,S] = svds(Xc,2);
        Y = nbr*pcs(r)/sqrt(sum(diag(S).^2))*U*S;
        std(Y)
        Yrp(:,:,r) = tsne(Xc,'Algorithm',alg,'Distance','euclidean','NumDimensions',2,'NumPCAComponents',0,'InitialY',Y,'Perplexity',pxt,'Exaggeration',ee,'LearnRate',nbr,'Options',opt,'Verbose',2);
        tmg(r,2) = toc;
    end
end

Yeh = zeros(nbr,2,rep);
for r = 1:rep
    tic
    Yeh(:,:,r) = ehtsne(Xc,pxt,itr,dof,0,own);
    tmg(r,3) = toc;
end
%%
if own || nbr<8192
    Yms = sbdr_abd_ms(Xc,2,0,'s',30);
else
    if strcmp(str,'MouseRNA')
        load 'fmssne_tasic 1.mat'
        Yms = X_LD;
    elseif strcmp(str,'MNIST')
        load 'fmssne_mnist 1.mat'
        Yms = X_LD;
    elseif strcmp(str,'fMNIST')
        load 'fmssne_fashion-mnist 1.mat'
        Yms = X_LD;
    else
        disp('Unavailable Fast MsSNE !');
    end
end

if own || nbr<8192 || strcmp(str,'MNIST') || strcmp(str,'fMNIST') || strcmp(str,'MouseRNA')
    Y0 = {Yrr,Yrp,Yeh,Yms;'g<','b>','r*','ko';'t-SNE Rand.init.','t-SNE PCA init.','Early Hierarchization t-SNE','Multi-Scale SNE'};
else
    Y0 = {Yrr,Yrp,Yeh    ;'g<','b>','r*'     ;'t-SNE Rand.init.','t-SNE PCA init.','Early Hierarchization t-SNE'                  };
end

fnm = strcat(str,'_Exp0_dat');
if exist('Y0','var')
    save(strcat(fnm,'.mat'),'X','Xc','Y0','tmg');
else
    load(strcat(fnm,'.mat'),'X','Xc','Y0','tmg');
end

figure;
kop = -1;
nx_scores([-nbr,pxt,kop],'r',Xc,Y0);
title('Relative neighborhood preservation w.r.t. K');

fnm = strcat(str,'_Exp0_Rnx');
print(gcf,'-depsc',strcat(fnm,'.eps'));
print(gcf,'-dpdf' ,strcat(fnm,'.pdf'));
print(gcf,'-dpng' ,strcat(fnm,'.png'));
%exportgraphics(gca,[fnm,'.eps']);
%exportgraphics(gca,[fnm,'.pdf']);
%exportgraphics(gca,[fnm,'.png']);

figure;
msz = 5;

subplot(2,2,1); 
xyp = Yrr(:,:,1);
scatter(xyp(:,1),xyp(:,2),msz,L,'o','filled');
xy9 = prctile(xyp,[1,99],1);
axis(1.2*[xy9(1,1),xy9(2,1),xy9(1,2),xy9(2,2)]); axis equal; axis off;
colormap(colmap);
title(Y0{3,1});

subplot(2,2,2);
xyp = Yrp(:,:,1);
scatter(xyp(:,1),xyp(:,2),msz,L,'o','filled');
xy9 = prctile(xyp,[1,99],1);
axis(1.2*[xy9(1,1),xy9(2,1),xy9(1,2),xy9(2,2)]); axis equal; axis off;
colormap(colmap);
title(Y0{3,2});

subplot(2,2,3);
xyp = Yeh(:,:,1);
scatter(xyp(:,1),xyp(:,2),msz,L,'o','filled');
xy9 = prctile(xyp,[1,99],1);
axis(1.2*[xy9(1,1),xy9(2,1),xy9(1,2),xy9(2,2)]); axis equal; axis off;
colormap(colmap);
title(Y0{3,3});

if own || nbr<8192 || strcmp(str,'MNIST') || strcmp(str,'fMNIST') || strcmp(str,'MouseRNA')
    subplot(2,2,4);
    xyp = Yms(:,:,1);
    scatter(xyp(:,1),xyp(:,2),msz,L,'o','filled');
    xy9 = prctile(xyp,[1,99],1);
    axis(1.2*[xy9(1,1),xy9(2,1),xy9(1,2),xy9(2,2)]); axis equal; axis off;
    colormap(colmap);
    title(Y0{3,4});
end

fnm = strcat(str,'_Exp0_Emb');
print(gcf,'-depsc',strcat(fnm,'.eps'));
print(gcf,'-dpdf' ,strcat(fnm,'.pdf'));
print(gcf,'-dpng' ,strcat(fnm,'.png'));

% timings
tmg
mean(tmg,1)
std(tmg,[],1)


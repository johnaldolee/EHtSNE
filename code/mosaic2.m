function mosaic2(Y,X,nhp,nvp,L,abt)

nbr = size(Y,1);

if nargin<6, abt = false; end
if nargin<5, L = zeros(nbr,1); end

DY = pairwisedistances(Y);
DY = DY + 1e100*(DY<=0);
K1 = min(DY,[],1);

tmp = sort(K1);
K1 = max(K1,tmp( ceil(0.05*length(K1))));
K1 = min(K1,tmp(floor(0.95*length(K1))));

tmp = max(K1);

xmax = max(Y(:,1)) + tmp;
xmin = min(Y(:,1)) - tmp;
ymax = max(Y(:,2)) + tmp;
ymin = min(Y(:,2)) - tmp;

if all(L==round(L))
    L = L - min(L) + 1;
    if max(L)==1
        colmap = [1,1,1];
    else
        colmap = jet(max(L));
    end
end

hold on;

for i = 1:nbr
    img = reshape(X(i,:),nhp,nvp)';
    img = img./max(img(:));
    rgb = bsxfun(@times,img,cat(3,colmap(L(i),1),colmap(L(i),2),colmap(L(i),3)));
    hdl = image(Y(i,1)+K1(i)/2*[-1,1],Y(i,2)+K1(i)/2*[-1,1],rgb,'CDataMapping','scaled');
    if abt, set(hdl,'AlphaData',floor(10*img)); end    
end

axis([xmin,xmax,ymin,ymax]);
set(gca,'Ydir','reverse');
axis off;

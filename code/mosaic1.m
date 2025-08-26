function nbr = mosaic1(map,dat,nhp,nvp,nhc,nvc,abt)

if nargin<7, abt = false; end
if nargin<6, nvc = 16; end
if nargin<5, nhc = 12; end

xmax = max(map(:,1));
xmin = min(map(:,1));
xdiv = (xmax-xmin)./nhc;
ymax = max(map(:,2));
ymin = min(map(:,2));
ydiv = (ymax-ymin)./nvc;

nbr = zeros(nhc,nvc);

hold on;

for i = 1:nhc
    for j = 1:nvc
        here = find( xmin+xdiv*(i-1)<=map(:,1) & map(:,1)<=xmin+xdiv*i & ymin+ydiv*(j-1)<=map(:,2) & map(:,2)<=ymin+ydiv*j );
        nbr(i,j) = length(here);
        if (0<size(here,1))
            img = reshape(mean(dat(here,:),1),nhp,nvp)';
            hdl = image(xmin+[xdiv*(i-1),xdiv*i],ymin+[ydiv*(j-1),ydiv*j],img,'CDataMapping','scaled');
            if abt, set(hdl,'AlphaData',img); end
        end
    end
end

axis([xmin,xmax,ymin,ymax]);
set(gca,'Ydir','reverse');
%colormap(flipdim(gray,1));
axis off;

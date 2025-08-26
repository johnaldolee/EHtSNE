% run all ESANN_art_2024_ext experiments

nrb = 5;
rld = true;
tictocall = zeros(5,3,4,3);

% artificial hierarchical
did = 'H';
% small size n2
for l_ = 1:2
    lvl = l_ + 3;
    for p_ = 1:3
        pxt = 5^p_;
        if rld
            str = ['HierBlob_',num2str(nrb),'_',num2str(lvl),'_',num2str(pxt)];
            load([str,'_Exp0_dat.mat']);
        else
            ESANN_art_2024_ext;
        end
        tictocall(:,:,l_,p_) = tmg;
    end
end

%pause; % for tsne license

% large size nlogn
for l_ = 3:4
    lvl = l_ + 3;
    for  p_ = 1:3
        pxt = 5^p_;
        if rld
            str = ['HierBlob_',num2str(nrb),'_',num2str(lvl),'_',num2str(pxt)];
            load([str,'_Exp0_dat.mat']);
        else
            ESANN_art_2024_ext;
        end
        tictocall(:,:,l_,p_) = tmg;
    end
end

tictocavg = squeeze(mean(tictocall,1));
tictocavg


figure;
hold on;
linecodes = {':','--','-'};
for p_ = 1:3
    plot([5^4,5^5],squeeze(tictocavg(1,1:2,p_)),['r+',linecodes{p_}]);
end
for p_ = 1:3
    plot([5^4,5^5],squeeze(tictocavg(2,1:2,p_)),['bx',linecodes{p_}]);
end
for p_ = 1:3
    plot([5^4,5^5],squeeze(tictocavg(3,1:2,p_)),['k*',linecodes{p_}]);
end
for p_ = 1:3
    plot([5^6,5^7],squeeze(tictocavg(1,3:4,p_)),['r+',linecodes{p_}]);
end
for p_ = 1:3
    plot([5^6,5^7],squeeze(tictocavg(2,3:4,p_)),['bx',linecodes{p_}]);
end
for p_ = 1:3
    plot([5^6,5^7],squeeze(tictocavg(3,3:4,p_)),['k*',linecodes{p_}]);
end
hold off;
xlabel('Data size (log)');
ylabel('Time (log)');
set(gca,'XScale','log');
set(gca,'YScale','log');
axis([500,1000000,1,300]);
lgd = {...
    'Rand. O(N^2), K* = 5',...
    'Rand. O(N^2), K* = 25',...
    'Rand. O(N^2), K* = 125',...
    'PCA O(N^2), K* = 5',...
    'PCA O(N^2), K* = 25',...
    'PCA O(N^2), K* = 125',...
    'EH O(N^2), K* = 5',...
    'EH O(N^2), K* = 25',...
    'EH O(N^2), K* = 125',...
    'Rand. O(NlogN), K* = 5',...
    'Rand. O(NlogN), K* = 25',...
    'Rand. O(NlogN), K* = 125',...
    'PCA O(NlogN), K* = 5',...
    'PCA O(NlogN), K* = 25',...
    'PCA O(NlogN), K* = 125',...
    'EH O(NlogN), K* = 5',...
    'EH O(NlogN), K* = 25',...
    'EH O(NlogN), K* = 125',...
    };
legend(lgd,'Location','southeast');

fnm = 'HierBlobtiming';
print(gcf,'-depsc',strcat(fnm,'.eps'));
print(gcf,'-dpdf' ,strcat(fnm,'.pdf'));
print(gcf,'-dpng' ,strcat(fnm,'.png'));


%%

clear
did = 'c';
pxt = 24;
ESANN_art_2024_ext;

clear
did = 's';
pxt = 32;
ESANN_art_2024_ext;

clear
did = 'r';
pxt = 32;
ESANN_art_2024_ext;

clear
did = 'f';
pxt = 32;
ESANN_art_2024_ext;

clear
did = 'p';
pxt = 32;
ESANN_art_2024_ext;
%%
clear
did = 'o';
pxt = 32;
ESANN_art_2024_ext;

clear
did = 'M';
pxt = 32;
ESANN_art_2024_ext;
%%
clear
did = 'C';
pxt = 32;
ESANN_art_2024_ext;

clear
did = 'm';
pxt = 32;
ESANN_art_2024_ext;
%%
clear
did = 'd';
pxt = 32;
ESANN_art_2024_ext;
%%
clear
did = 'h';
pxt = 32;
ESANN_art_2024_ext;
%%
clear
did = 'v';
pxt = 32;
ESANN_art_2024_ext;



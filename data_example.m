clear; clc; close all;

%% fit CCA and pCCA to MATLAB's carbig dataset
load('sample_cca_data.mat')

x = X(:,1:3); y = X(:,4:5);
ccaResults = vanillaCCA(x',y');
[pccaResults, suffStats, LL] = maxLL_pCCA(x',y',2);

f=figure; pos=get(f,'Position'); set(f,'Position',pos.*[1 1 2 1]);

subplot(1,2,1); hold on;
plot(ccaResults.canonVarsX(1,:),ccaResults.canonVarsY(1,:),'.')
plot(suffStats.Zx_mean(1,:),suffStats.Zy_mean(1,:),'.')
xlabel('X dim 1 scores'); ylabel('Y dim 1 scores')
xlim([-4 4]),ylim([-4 4])
title(sprintf('r_1 = %.3f',ccaResults.canonCorr(1)))
legend('CCA','prob CCA');

subplot(1,2,2); hold on;
plot(ccaResults.canonVarsX(2,:),ccaResults.canonVarsY(2,:),'.')
plot(suffStats.Zx_mean(2,:),suffStats.Zy_mean(2,:),'.')
xlabel('X dim 2 scores'); ylabel('Y dim 2 scores')
xlim([-4 4]),ylim([-4 4])
title(sprintf('r_2 = %.3f',ccaResults.canonCorr(2)))
legend('CCA','prob CCA');

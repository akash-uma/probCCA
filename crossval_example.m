clear; clc; close all;

%% fit pCCA on synthetically generated dataset

% generate fake data
rng(1000);
xDim = 15;
yDim = 10;
zDim = 5;
nTrials = 10000;

params.W_x = randn(xDim,zDim);
params.W_y = randn(yDim,zDim);
x_sqrt = randn(xDim,xDim);
y_sqrt = randn(yDim,yDim);
params.psi_x = x_sqrt*x_sqrt';
params.psi_x = params.psi_x ./ max(max(params.psi_x));
params.psi_y = y_sqrt*y_sqrt';
params.psi_y = params.psi_y ./ max(max(params.psi_y));
params.mu_x = randn(xDim,1);
params.mu_y = randn(yDim,1);
[X,Y] = simdata_pcca(params, nTrials);

% fit using maximum likelihood (very fast)
zDimList = 0:10;
dim_maxLL = crossval_pCCA(X,Y,'zDimList',zDimList,'showPlots',false);
LLs_maxL = [dim_maxLL.sumLL];

% fit using em algorithm (a bit slower)
dim_em = crossval_em_pCCA(X,Y,'zDimList',zDimList,'showPlots',false);
LLs_em = [dim_em.sumLL];

% Identify optimal latent dimensionality
istar_em = (LLs_em == max(LLs_em));
istar_maxL = (LLs_maxL == max(LLs_maxL));
estParams_em = dim_em(istar_em);
estParams_maxL = dim_maxLL(istar_maxL);
optimalDim_em = dim_em(istar_em).zDim
optimalDim_maxLL = dim_maxLL(istar_maxL).zDim

% plot cv-LL as a function of dimensionality
figure;
hold on
em_plot=plot(zDimList,LLs_em,'bo-');
plot(optimalDim_em,LLs_em(istar_em),'r*','MarkerSize',10)
maxL_plot=plot(zDimList,LLs_maxL,'ko-');
plot(optimalDim_maxLL,LLs_maxL(istar_maxL),'r*','MarkerSize',10)
xlabel('CCA dimensions')
ylabel('Cross-validated LL')
legend([em_plot,maxL_plot],'EM algorithm','max likelihood','Location','Best')

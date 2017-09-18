function [ X, Y ] = simdata_pcca( params, N )
%SIMDATA_PCCA Summary of this function goes here
%   Detailed explanation goes here
    
    %rng(0);
    W_x = params.W_x; W_y = params.W_y;
    psi_x = params.psi_x; psi_y = params.psi_y;
    mu_x = params.mu_x; mu_y = params.mu_y;
    
    [xDim,zDim] = size(W_x);
    [yDim,zDim] = size(W_y);

    Z  = randn(zDim, N);
    Xmean = bsxfun(@plus,W_x*Z,mu_x);
    Ymean = bsxfun(@plus,W_y*Z,mu_y);
%     ns_x = sqrtm(psi_x)*randn(xDim,N);
%     ns_y = sqrtm(psi_y)*randn(yDim,N);
%     
%     X = bsxfun(@plus, W_x*Z + ns_x, mu_x);
%     Y = bsxfun(@plus, W_y*Z + ns_y, mu_y);
    X = zeros(xDim,N);
    Y = zeros(yDim,N);
    for ii=1:N
        X(:,ii) = mvnrnd(Xmean(:,ii),psi_x);
        Y(:,ii) = mvnrnd(Ymean(:,ii),psi_y);
    end
    
end


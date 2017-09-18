function [ suffStats, LL ] = pCCA_estep( X, Y, params )
%PCCA_ESTEP Summary of this function goes here
%   Detailed explanation goes here

    N = size(X,2);
    meanX = mean(X,2);
    meanY = mean(Y,2);
    centeredX = bsxfun(@minus,X,meanX);
    centeredY = bsxfun(@minus,Y,meanY);
    covX = 1/N * (centeredX*centeredX');
    covY = 1/N * (centeredY*centeredY');
    covXY = 1/N * (centeredX*centeredY');
    sampleCov = [covX covXY; covXY' covY];
    xDim = size(X,1);
    yDim = size(Y,1);
    
    if params.zDim ==0
        suffStats = [];

        C = [params.psi_x zeros(xDim,yDim); zeros(yDim,xDim) params.psi_y];

        % Log likelihood
        term1 = (xDim+yDim)*log(2*pi);
        E = eig(C); % for calculating determinant (product of eigenvalues)
        term2 = sum(log(E));
        term3 = trace(C\sampleCov);
        estMu_testMu = [params.mu_x; params.mu_y] - [meanX; meanY];
        term4 = estMu_testMu'/C*estMu_testMu;
        LL = -N/2 * (term1 + term2 + term3);
    else
        W_x = params.W_x;
        W_y = params.W_y;
        psi_x = params.psi_x;
        psi_y = params.psi_y;

        % posterior expectations and variances
        I = eye(size(W_x,2));
        
        % E(z|x,y)
        W = [W_x; W_y];
        C = [W_x*W_x'+psi_x W_x*W_y'; W_y*W_x' W_y*W_y'+psi_y];
        centeredData = [centeredX; centeredY];
        suffStats.Zxy_mean = W'/C*centeredData;
        suffStats.Zxy_cov = diag(diag(I - W'/C*W)); % diag to get rid of tiny values

        % E(z|x) and E(z|y)
        C_x = C(1:xDim,1:xDim);
        C_y = C((xDim+1):end,(xDim+1):end);
        suffStats.Zx_mean = W_x'/C_x*centeredX;
        suffStats.Zy_mean = W_y'/C_y*centeredY;
        postCov = diag(diag(I - W_x'/C_x*W_x));
        suffStats.Zx_cov = postCov;
        suffStats.Zy_cov = postCov;
        r = diag(corr(suffStats.Zx_mean',suffStats.Zy_mean'));
        
        % Log likelihood
        term1 = (xDim+yDim)*log(2*pi);
        E = eig(C); % for calculating determinant (product of eigenvalues)
        term2 = sum(log(E));
        term3 = trace(C\sampleCov);
        estMu_testMu = [params.mu_x; params.mu_y] - [meanX; meanY];
        term4 = estMu_testMu'/C*estMu_testMu;
        LL = -N/2 * (term1 + term2 + term3);
    end


end


function [ params, suffStats, LL ] = maxLL_pCCA( X, Y, zDim )
%MYPCCA Summary of this function goes here
%   X - first dataset (xDim x N)
%   Y - second dataset (yDim x N)
    
    params.mu_x = mean(X,2);
    params.mu_y = mean(Y,2);
    params.zDim = zDim;
    
    N = size(X,2);
    centeredX = bsxfun(@minus,X,params.mu_x);
    centeredY = bsxfun(@minus,Y,params.mu_y);
    covX = 1/N * (centeredX*centeredX');
    covY = 1/N * (centeredY*centeredY');
    covXY = 1/N * (centeredX*centeredY');
    sampleCov = [covX covXY; covXY' covY];
    xDim = size(X,1);
    yDim = size(Y,1);
    
    if rank(sampleCov) ~= (xDim+yDim)
        error('ERROR: sample covariance is not full rank.');
    end
    
    if zDim==0
        params.W_x = [];
        params.W_y = [];
        params.psi_x = covX;
        params.psi_y = covY;
        params.canonCorr = nan;
        suffStats = [];
        
        C = [covX zeros(xDim,yDim); zeros(yDim,xDim) covY];
    else
        ccaStruct = vanillaCCA(X,Y);
        U1 = ccaStruct.canonVecsX(:,1:zDim);
        U2 = ccaStruct.canonVecsY(:,1:zDim);
        Pd = ccaStruct.canonCorr(1:zDim);
        scoreX = ccaStruct.canonVarsX(1:zDim,:);
        scoreY = ccaStruct.canonVarsY(1:zDim,:);
        M = diag(sqrt(Pd));
        r = Pd;

        % pCCA parameters
        W_x = covX * U1 * M;
        W_y = covY * U2 * M;
        psi_x = covX - W_x*W_x';
        psi_y = covY - W_y*W_y';

        params.W_x = W_x;
        params.W_y = W_y;
        params.psi_x = psi_x;
        params.psi_y = psi_y;
        params.canonCorr = r;

        % posterior expectations and variances
        I = eye(size(M,1));

        % E(z|x) and E(z|y)
        suffStats.Zx_mean = M'*scoreX;
        suffStats.Zy_mean = M'*scoreY;
        postCov = I - M*M';
        suffStats.Zx_cov = postCov;
        suffStats.Zy_cov = postCov;

        % E(z|x,y)
        W = [W_x; W_y];
        C = [covX W_x*W_y'; W_y*W_x' covY];
        centeredData = [centeredX; centeredY];
        suffStats.Zxy_mean = W'/C*centeredData;
        % suffStats.Zxy_cov = diag(diag(I - W'/C*W)); % diag to get rid of tiny values
        suffStats.Zxy_cov = I - W'/C*W;
    end

    % Log likelihood
    E = eig(C); % for calculating determinant (product of eigenvalues)
    LL = -N/2 * ((xDim+yDim)*log(2*pi) + sum(log(E)) + trace(C\sampleCov));

end


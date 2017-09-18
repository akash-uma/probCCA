function [ params, LL ] = fast_em_pCCA( X, Y, zDim )
%EM_PCCA Summary of this function goes here
%   X - first dataset (xDim x N)
%   Y - second dataset (yDim x N)
    
    % optimization options
    maxIter = 1000;
    tol = 1e-6;
    verbose = false;
    
    % calculate some matrices
    params.mu_x = mean(X,2);
    params.mu_y = mean(Y,2);
    params.zDim = zDim;
    [xDim,~] = size(X);
    [yDim,N] = size(Y);
    centeredX = bsxfun(@minus,X,params.mu_x);
    centeredY = bsxfun(@minus,Y,params.mu_y);
    centeredData = [centeredX; centeredY];
    covX = 1/N * (centeredX*centeredX');
    covY = 1/N * (centeredY*centeredY');
    covXY = 1/N * (centeredX*centeredY');
    sampleCov = [covX covXY; covXY' covY];
    
    if rank(sampleCov) ~= (xDim+yDim)
        error('ERROR: sample covariance is not full rank.');
    end

    % initialize parameters
    scale = exp(2*sum(log(diag(chol(sampleCov))))/(xDim+yDim));
    W_xy = randn((xDim+yDim),zDim)*sqrt(scale/zDim);
    psi_xy = [covX zeros(xDim,yDim); zeros(yDim,xDim) covY];
    Ix = eye(xDim);
    Iy = eye(yDim);
    Iz = eye(zDim);
    const = (xDim+yDim)*log(2*pi);
    LL=[];
    
    
    if zDim==0
        params.W_x = [];
        params.W_y = [];
        params.psi_x = covX;
        params.psi_y = covY;
        params.canonCorr = nan;
        suffStats = [];
        
        C = [covX zeros(xDim,yDim); zeros(yDim,xDim) covY];

        % Log likelihood
        logDet = 2*sum(log(diag(chol(C))));
        LL = -N/2 * (const + logDet + trace(C\sampleCov));
        return
    end

    iter = 1;
    stop = false;
    while iter<=maxIter && ~stop
        % =======
        % E-step - set q(z) = p(z|x)
        % =======
        iPsi = [psi_xy(1:xDim,1:xDim)\Ix zeros(xDim,yDim); ...
            zeros(yDim,xDim) psi_xy((xDim+1):end,(xDim+1):end)\Iy];
        iPsiW = iPsi*W_xy;
        iSig = iPsi - iPsiW / (Iz+W_xy'*iPsiW) * iPsiW';
        iSigW = iSig*W_xy;
        cov_iSigW = sampleCov*iSigW;
        E_ZZ = Iz - W_xy'*iSigW + iSigW'*cov_iSigW;
        
        % Compute LL
        logDet = 2*sum(log(diag(chol(iSig))));
        curr_LL = -N/2 * (const - logDet + trace(iSig*sampleCov));
        LL = [LL curr_LL];
        
        % =======
        % M-step - calculate new W and psi
        % =======
        W_xy = cov_iSigW / E_ZZ;
        psi_xy = sampleCov - W_xy*cov_iSigW' - cov_iSigW*W_xy' + W_xy*E_ZZ*W_xy';
        psi_xy = (psi_xy+psi_xy')/2;
        psi_xy = [psi_xy(1:xDim,1:xDim) zeros(xDim,yDim); ...
            zeros(yDim,xDim) psi_xy((xDim+1):end,(xDim+1):end)];

        if verbose
            fprintf('EM iteration %5i lik %8.1f \r', iter, curr_LL);
        end
        
        % check convergence
        if (iter > 1) && (LL(iter)-LL(iter-1) < tol)
            stop = true;
        end
        
        iter = iter+1;
    end

    params.psi_x = psi_xy(1:xDim,1:xDim);
    params.psi_y = psi_xy((xDim+1):end,(xDim+1):end);
    params.W_x = W_xy(1:xDim,:);
    params.W_y = W_xy((xDim+1):end,:);
    
end


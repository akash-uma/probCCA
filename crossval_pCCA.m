function [ dim ] = crossval_pCCA( X, Y, varargin )
%CROSSVAL_PCCA Summary of this function goes here
% INPUTS:
%   X - 1st data matrix (xDim x N)
%   Y - 2nd data matrix (yDim x N)
%
% OUPUTS:
%   dim - structure whose ith entry (corresponding to the ith latent
%         dimensionality) has fields 
%           zDim      -- latent dimensionality 
%           sumLL     -- cross-validated log likelihood
%           estParams -- pCCA parameters estimated using all data
%
% OPTIONAL INPUTS:
%   numFolds    - number of cross-validation folds (default: 10)
%   zDimList    - latent dimensionalities to compare (default: [1:10])
%                 Note: dimensionality 0 corresponds to an independent
%                 Gaussian model, where all variance is private. 
%   showPlots   - logical specifying whether to show CV plots
%                 (default: true)
    
    numFolds = 10;
    zDimList = 1:10;
    showPlots = true;
    extraOpts = assignopts(who, varargin);
    
    N = size(X,2);
    
    % Randomly reorder data points
    %rng(0);
    randIdx = randperm(N);
    X = X(:,randIdx);
    Y = Y(:,randIdx);
    
    % set cross-validation folds
    fdiv = floor(linspace(1,N+1,numFolds+1));
    
    for ii=1:length(zDimList);
        zDim = zDimList(ii);
        fprintf('Processing latent dimensionality = %d\n', zDim);
    
        dim(ii).zDim  = zDim;
        dim(ii).sumLL = 0;
        
        for cvf = 0:numFolds
            if cvf == 0
                fprintf('  Training on all data.\n');
            else
                fprintf('  Cross-validation fold %d of %d.\n', cvf, numFolds);
            end
            
            % Set cross-validation masks
            testMask = false(1, N);
            if cvf > 0
                testMask(fdiv(cvf):fdiv(cvf+1)-1) = true;
            end
            trainMask = ~testMask;
            Xtrain = X(:,trainMask);
            Xtest  = X(:,testMask);
            Ytrain = Y(:,trainMask);
            Ytest  = Y(:,testMask);
                  
            % Check if training data covariance is full rank
            if rcond(cov(Xtrain')) < 1e-10
                fprintf('ERROR: X training data covariance matrix ill-conditioned.\n');
                keyboard
            end
            if rcond(cov(Ytrain')) < 1e-10
                fprintf('ERROR: Y training data covariance matrix ill-conditioned.\n');
                keyboard
            end
            
            % Fit model parameters to training data
            [estParams,~,trainLL] = maxLL_pCCA(Xtrain, Ytrain, zDim);
            
            if cvf == 0
                % Save parameters
                dim(ii).estParams = estParams;       
            else
                % test likelihood
                [~,LL] = pCCA_estep(Xtest,Ytest,estParams);
                
                dim(ii).sumLL = dim(ii).sumLL + LL;
            end     
        end
        
    end
    
    if showPlots
        figure; hold on;
    
        % LL versus latent dimensionality
        sumLL = [dim.sumLL];
        plot(zDimList, sumLL,'b-');
        xlabel('Latent dimensionality');
        ylabel('Cross-validated LL');
        istar = find(sumLL == max(sumLL));
        plot(zDimList(istar), sumLL(istar), 'ro', 'MarkerSize', 5);
        title(sprintf('Optimal dim = %d\n', zDimList(istar)));
    end
    
end


function [ resultStruct ] = vanillaCCA( X, Y )
%MYCCA Summary of this function goes here
%   X - first dataset (xDim x N)
%   Y - second dataset (yDim x N)
    
    N = size(X,2);
    centeredX = bsxfun(@minus,X,mean(X,2));
    centeredY = bsxfun(@minus,Y,mean(Y,2));
    covX = 1/N * (centeredX*centeredX');
    covY = 1/N * (centeredY*centeredY');
    covXY = 1/N * centeredX*centeredY';
    
    CovX_sqrt = sqrtm(covX);
    CovY_sqrt = sqrtm(covY);
    K = CovX_sqrt \ covXY / CovY_sqrt;
    
    [UU,DD,VV] = svd(K);
    diagDD = diag(DD);
    
    r = rank(K);
    resultStruct.canonVecsX = CovX_sqrt \ UU(:,1:r);
    resultStruct.canonVecsY = CovY_sqrt \ VV(:,1:r);
    resultStruct.canonCorr = diagDD(1:r);
    resultStruct.canonVarsX = resultStruct.canonVecsX'*centeredX;
    resultStruct.canonVarsY = resultStruct.canonVecsY'*centeredY;
    
end


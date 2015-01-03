function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

poolFilter = ones(poolDim, poolDim) / (poolDim * poolDim);
for imageNum = 1:numImages
    for filterNum = 1:numFilters
        pooledImage = conv2(convolvedFeatures(:,:,filterNum,imageNum), poolFilter, 'valid');
        pooledImage = pooledImage(1:poolDim:end, 1:poolDim:end);
        pooledFeatures(:,:,filterNum,imageNum) = pooledImage;
    end
end

end


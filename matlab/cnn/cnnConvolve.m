function convolvedFeatures = cnnConvolve(images, W, b)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  images - large images to convolve with, matrix in the form
%           images(r, c, numInplanes, image number)
%  W, b - W, b for features from the sparse autoencoder
%         W is of shape (filterDim,filterDim,numInplanes, numOutplanes)
%         b is of shape (numInplanes, numOutplanes, 1)
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)


filterDim = size(W, 1);
numInplanes = size(W, 3);
numOutplanes = size(W, 4);
assert (numInplanes == size(images, 3))

numImages = size(images, 4);
imageDim = size(images, 1);
convDim = imageDim - filterDim + 1;

convolvedFeatures = zeros(convDim, convDim, numOutplanes, numImages);

% ---- Convolutions ----
for imageNum = 1:numImages
  for outplane = 1: numOutplanes
    convolvedImage = zeros(convDim, convDim);
    for inplane = 1: numInplanes
      filter = W(:,:,inplane,outplane);
      filter = rot90(squeeze(filter), 2);
      im = squeeze(images(:,:,inplane,imageNum));
      convolvedImageLoop = conv2(im, filter, 'valid') + b(outplane);
      convolvedImageLoop = 1 ./ (1+exp(-convolvedImageLoop));
      convolvedImage = convolvedImage + convolvedImageLoop;
    end
    convolvedFeatures(:,:,outplane,imageNum) = convolvedImage;
  end
end

end

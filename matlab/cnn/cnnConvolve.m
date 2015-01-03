function convolvedFeatures = cnnConvolve(images, W, b)
%
% Parameters:
%  images - large images to convolve with, matrix in the form
%           images(r, c, inplane number, image number)
%  W, b - W, b for features from the sparse autoencoder
%         W is of shape (filterDim,filterDim,numInplanes,numOutplanes)
%         b is of shape (numOutplanes,1)
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
% Important:
%  Summation across input planes should be performed before activation,
%  i.e., for each output plane computation, only activate once.


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
      convolvedImage = convolvedImage + convolvedImageLoop;
    end
    % Summation across input planes is prior to being activated!
    convolvedImage = 1 ./ (1+exp(-convolvedImage)); 
    convolvedFeatures(:,:,outplane,imageNum) = convolvedImage;
  end
end

end

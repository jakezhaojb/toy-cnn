function theta = cnnInitParams(imageDim0,filterDim0,numInplane0,numOutplane0,poolDim0,...
                               imageDim1,filterDim1,numInplane1,numOutplane1,poolDim1,...
                               numClasses)
% Initialize parameters for a single layer convolutional neural
% network followed by a softmax layer.
%                            
% Parameters:
%  imageDim   -  height/width of image
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  numClasses -  number of classes to predict
%
%
% Returns:
%  theta      -  unrolled parameter vector with initialized weights

%% Initialize parameters randomly based on layer sizes.
assert(filterDim0 < imageDim0,'filterDim must be less that imageDim');
assert(filterDim1 < imageDim1,'filterDim must be less that imageDim');

Wc0 = 1e-1*randn(filterDim0,filterDim0,numInplane0,numOutplane0);
Wc1 = 1e-1*randn(filterDim1,filterDim1,numInplane1,numOutplane1);

outDim0 = imageDim0 - filterDim0 + 1; % dimension of convolved image
outDim1 = imageDim1 - filterDim1 + 1; % dimension of convolved image

% assume outDim is multiple of poolDim
assert(mod(outDim0,poolDim0)==0,...
       'poolDim must divide imageDim - filterDim + 1');
assert(mod(outDim1,poolDim1)==0,...
       'poolDim must divide imageDim - filterDim + 1');

outDim0 = outDim0/poolDim0;
outDim1 = outDim1/poolDim1;
hiddenSize = outDim1^2*numOutplane1;

% we'll choose weights uniformly from the interval [-r, r]
r  = sqrt(6) / sqrt(numClasses+hiddenSize+1);
Wd = rand(numClasses, hiddenSize) * 2 * r - r;
%Wd(end, :) = 0;

%bc0 = zeros(numInplane0, numOutplane0, 1);
%bc1 = zeros(numInplane1, numOutplane1, 1);
bc0 = zeros(numOutplane0, 1);
bc1 = zeros(numOutplane1, 1);
bd = zeros(numClasses, 1);
%bd(end) = 0;

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [Wc0(:) ; Wc1(:) ; Wd(:) ; bc0(:) ; bc1(:) ; bd(:)];

end


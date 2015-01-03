function [Wc0, Wc1, Wd, bc0, bc1, bd] = cnnParamsToStack(theta,imageDim0,filterDim0,numInplane0,numOutplane0,poolDim0,...
                                             imageDim1,filterDim1,numInplane1,numOutplane1,poolDim1,...
                                             numClasses)
% Converts unrolled parameters for a single layer convolutional neural
% network followed by a softmax layer into structured weight
% tensors/matrices and corresponding biases
%                            
% Parameters:
%  theta      -  unrolled parameter vectore
%  imageDim   -  height/width of image
%  numInplane -  number of input planes
%  numOutplane -  number of output planes
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  numClasses -  number of classes to predict
%
%
% Returns:
%  Wc0      -  filterDim x filterDim x numInplanes0 x numOutplane0 parameter tensor
%  Wc1      -  filterDim x filterDim x numInplanes1 x numOutplane1 parameter tensor
%  Wd      -  numClasses x hiddenSize parameter matrix, hiddenSize is
%             calculated as numFilters*((imageDim1-filterDim1+1)/poolDim1)^2 
%  bc0      -  bias for convolution layer of size numOutplane0 x 1
%  bc1      -  bias for convolution layer of size numOutplane1 x 1
%  bd      -  bias for dense layer of size hiddenSize x 1

outDim0 = (imageDim0 - filterDim0 + 1)/poolDim0;
outDim1 = (imageDim1 - filterDim1 + 1)/poolDim1;
hiddenSize = outDim1^2*numOutplane1;

%% Reshape theta
indS = 1;
indE = filterDim0^2*numInplane0*numOutplane0;
Wc0 = reshape(theta(indS:indE),filterDim0,filterDim0,numInplane0,numOutplane0);
indS = indE+1;
indE = indE + filterDim1^2*numInplane1*numOutplane1;
Wc1 = reshape(theta(indS:indE),filterDim1,filterDim1,numInplane1,numOutplane1);
indS = indE+1;
indE = indE+hiddenSize*numClasses;
Wd = reshape(theta(indS:indE),numClasses,hiddenSize);
indS = indE+1;
indE = indE+numOutplane0;
bc0 = reshape(theta(indS:indE),numOutplane0,1);
indS = indE+1;
indE = indE+numOutplane1;
bc1 = reshape(theta(indS:indE),numOutplane1,1);
bd = theta(indE+1:end);


end

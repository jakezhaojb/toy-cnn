function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim0,numInplane0,numOutplane0,poolDim0,...
                                filterDim1,numInplane1,numOutplane1,poolDim1,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
% Added Paramaters:
%  filterDim0
%  numFilters0 
%  poolDim0 
%  
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim0 = size(images,1); % height/width of image
imageDim1 = (imageDim0-filterDim0+1)/poolDim0;
numImageChannel = size(images, 3);
numImages = size(images,4); % number of images

%% Reshape parameters and setup gradient matrices

% Wc1 is filterDim x filterDim x numFilters parameter matrix
% bc1 is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc0, Wc1, Wd, bc0, bc1, bd] = cnnParamsToStack(theta,imageDim0,filterDim0,numInplane0,numOutplane0,poolDim0,...
                                                imageDim1,filterDim1,numInplane1,numOutplane1,poolDim1,...
                                                numClasses);

%Wd(end, :) = 0;
%bd(end) = 0;

% Same sizes as Wc1,Wd,bc1,bd. Used to hold gradient w.r.t above params.
Wc1_grad = zeros(size(Wc1));
Wc0_grad = zeros(size(Wc0));
Wd_grad = zeros(size(Wd));
bc1_grad = zeros(size(bc1));
bc0_grad = zeros(size(bc0));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim0 = imageDim0-filterDim0+1; % dimension of convolved output
outputDim0 = (convDim0)/poolDim0; % dimension of subsampled output
convDim1 = outputDim0-filterDim1+1; % dimension of convolved output
outputDim1 = (convDim1)/poolDim1; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations0 = zeros(convDim0,convDim0,numOutplane0,numImages);
activations1 = zeros(convDim1,convDim1,numOutplane1,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled0 = zeros(outputDim0,outputDim0,numOutplane0,numImages);
activationsPooled1 = zeros(outputDim1,outputDim1,numOutplane1,numImages);

activations0 = cnnConvolve(images, Wc0, bc0);
activationsPooled0 = cnnPool(poolDim0, activations0);
activations1 = cnnConvolve(activationsPooled0, Wc1, bc1);
activationsPooled1 = cnnPool(poolDim1, activations1);
    
% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled1,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);

%%% YOUR CODE HERE %%%
probs = exp(bsxfun(@plus, Wd * activationsPooled, bd));
sumProbs = sum(probs, 1);
probs = bsxfun(@times, probs, 1 ./ sumProbs);

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

probsL = log(probs);
I = sub2ind(size(probs), reshape(labels, 1, []), 1:numImages);
probsL = probsL(I);
cost = -sum(probsL);

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

t = zeros(size(probs));
t(I) = 1;
t = t - probs;
Dd = -t; % softmax delta
DcPooled1 = reshape(Wd' * Dd, [outputDim1, outputDim1, numOutplane1, numImages]); % TODO verify
Dc1 = zeros(convDim1, convDim1, numOutplane1, numImages); % TODO subject to change
for i = 1:numImages
    for j = 1:numOutplane1
        Dc1(:,:,j,i) = kron(squeeze(DcPooled1(:,:,j,i)), ones(poolDim1)) * (1 / poolDim1^2); % Upsample
        activated = squeeze(activations1(:,:,j,i));
        Dc1(:,:,j,i) = squeeze(Dc1(:,:,j,i)) .* activated .* (1 - activated);
    end
end

% TODO critical step here
DcPooled0 = zeros(outputDim0, outputDim0, numOutplane0, numImages);
for i = 1:numImages
    for j = 1: numOutplane0
        for k = 1: numOutplane1
            filt = squeeze(Wc1(:,:,j,k));
            DcPooled0(:,:,j,i) = DcPooled0(:,:,j,i) + conv2(squeeze(Dc1(:,:,k,i)), filt,'full'); % TODO summarize
        end
    end
end
Dc0 = zeros(convDim0, convDim0, numOutplane0, numImages);
for i = 1:numImages
    for j = 1:numOutplane0
        Dc0(:,:,j,i) = kron(squeeze(DcPooled0(:,:,j,i)), ones(poolDim0)) * ( 1 / poolDim0^2);
        activated = squeeze(activations0(:,:,j,i));
        Dc0(:,:,j,i) = squeeze(Dc0(:,:,j,i)) .* activated .* (1 - activated);
    end
end
%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

Wd_grad_ = zeros(numel(Wd), numImages);
for i = 1:numImages
    Wd_grad_(:, i) = -kron(t(:, i), activationsPooled(:, i));
end
Wd_grad = reshape(sum(Wd_grad_, 2), size(Wd'));
Wd_grad = Wd_grad';
bd_grad = sum(Dd, 2);

for i = 1: numImages
    for j = 1: numOutplane1
        DcR = rot90(squeeze(Dc1(:,:,j,i)), 2);
        for k = 1: numInplane1
            im = squeeze(activationsPooled0(:,:,k,i));
            Wc1_grad(:,:,k,j) = Wc1_grad(:,:,k,j) + conv2(im, DcR, 'valid');
            bc1_grad(j) = bc1_grad(j) + sum(sum(squeeze(Dc1(:,:,j,i))));
        end
    end
end

for i = 1:numImages
    for j = 1:numOutplane0
        DcR = rot90(squeeze(Dc0(:,:,j,i)), 2);
        for k = 1:numInplane0
            im = squeeze(images(:,:,k,i));
            Wc0_grad(:,:,k,j) = Wc0_grad(:,:,k,j) + conv2(im, DcR, 'valid');
            bc0_grad(j) = bc0_grad(j) + sum(sum(squeeze(Dc0(:,:,j,i))));
        end
    end
end

%Wd_grad(end, :) = 0;
%bd_grad(end) = 0;

%% Unroll gradient into grad vector for minFunc
grad = [Wc0_grad(:) ; Wc1_grad(:) ; Wd_grad(:) ; bc0_grad(:) ; bc1_grad(:) ; bd_grad(:)];

end

function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim0,numInplane0,numOutplane0,poolDim0,...
                                filterDim1,numInplane1,numOutplane1,poolDim1,pred)
% Calcualte cost and gradient for a two layers convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numChannels x numImges,
%                a 4D tensor
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numInplane -  number of input planes
%  numOutplane-  number of output planes
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim0 = size(images,1);
imageDim1 = (imageDim0-filterDim0+1)/poolDim0;
numImageChannel = size(images, 3);
numImages = size(images,4); % number of images

% Wc1 is filterDim1 x filterDim1 x numInplane1 x numOutplane1 parameter tensor
% bc1 is the corresponding bias
% Wc0 is filterDim0 x filterDim0 x numInplane0 x numOutplane0 parameter tensor
% bc0 is the corresponding bias
% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc0, Wc1, Wd, bc0, bc1, bd] = cnnParamsToStack(theta,imageDim0,filterDim0,numInplane0,numOutplane0,poolDim0,...
                                                imageDim1,filterDim1,numInplane1,numOutplane1,poolDim1,...
                                                numClasses);

% How about leaving out the last class theta in Softmax?
%Wd(end, :) = 0;
%bd(end) = 0;

Wc1_grad = zeros(size(Wc1));
Wc0_grad = zeros(size(Wc0));
Wd_grad = zeros(size(Wd));
bc1_grad = zeros(size(bc1));
bc0_grad = zeros(size(bc0));
bd_grad = zeros(size(bd));

%% ---------- Forward Propagation ----------
convDim0 = imageDim0-filterDim0+1;
outputDim0 = (convDim0)/poolDim0;
convDim1 = outputDim0-filterDim1+1;
outputDim1 = (convDim1)/poolDim1;

% convDim x convDim x numOutplane x numImages tensor for storing activations
activations0 = zeros(convDim0,convDim0,numOutplane0,numImages);
activations1 = zeros(convDim1,convDim1,numOutplane1,numImages);

% outputDim x outputDim x numOutplane x numImages tensor for storing
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

%% --------- Softmax Layer ---------
probs = zeros(numClasses,numImages);

probs = exp(bsxfun(@plus, Wd * activationsPooled, bd));
sumProbs = sum(probs, 1);
probs = bsxfun(@times, probs, 1 ./ sumProbs);

% --------- Calculate Cost ----------
cost = 0;

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

%% --------- Backpropagation ----------
% softmax layer
t = zeros(size(probs));
t(I) = 1;
t = t - probs;
Dd = -t;
% second convolutional layer and its post pooling layer
DcPooled1 = reshape(Wd' * Dd, [outputDim1, outputDim1, numOutplane1, numImages]);
Dc1 = zeros(convDim1, convDim1, numOutplane1, numImages);
for i = 1:numImages
    for j = 1:numOutplane1
        Dc1(:,:,j,i) = kron(squeeze(DcPooled1(:,:,j,i)), ones(poolDim1)) * (1 / poolDim1^2); % Upsample
        activated = squeeze(activations1(:,:,j,i));
        Dc1(:,:,j,i) = squeeze(Dc1(:,:,j,i)) .* activated .* (1 - activated);
    end
end
% first convolutional layer and its pooling layer
DcPooled0 = zeros(outputDim0, outputDim0, numOutplane0, numImages);
for i = 1:numImages
    for j = 1: numOutplane0
        for k = 1: numOutplane1
            filt = squeeze(Wc1(:,:,j,k));
            % IMPORTANT -- backprop across convolution
            % this step, adopt 'full' conv2 as opposed to 'valid' applied during feed-forward pass.
            DcPooled0(:,:,j,i) = DcPooled0(:,:,j,i) + conv2(squeeze(Dc1(:,:,k,i)), filt,'full');
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
%% ---------- Gradient Calculation ----------
% softmax layer
Wd_grad_ = zeros(numel(Wd), numImages);
for i = 1:numImages
    Wd_grad_(:, i) = -kron(t(:, i), activationsPooled(:, i));
end
Wd_grad = reshape(sum(Wd_grad_, 2), size(Wd'));
Wd_grad = Wd_grad';
bd_grad = sum(Dd, 2);
% second convolutional layer
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
% first convolutional layer
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

% If previously leave out the last class theta in softmax...
%Wd_grad(end, :) = 0;
%bd_grad(end) = 0;

%% Unroll gradient into grad vector for minFunc
grad = [Wc0_grad(:) ; Wc1_grad(:) ; Wd_grad(:) ; bc0_grad(:) ; bc1_grad(:) ; bd_grad(:)];

end

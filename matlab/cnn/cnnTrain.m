%% Convolution Neural Network Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started in building a single.
%  layer convolutional nerual network. In this exercise, you will only
%  need to modify cnnCost.m and cnnminFuncSGD.m. You will not need to 
%  modify this file.

%%======================================================================
%% STEP 0: Initialize Parameters and Load Data
%  Here we initialize some parameters used for the exercise.

% Configuration
imageDim0 = 28;
imageDim1 = 10;
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
filterDim0 = 9;    % Filter size for conv layer
filterDim1 = 5;
numInplane0 = 1;
numOutplane0 = 20;
numInplane1 = 20;
numOutplane1 = 10;
poolDim0 = 2;      % Pooling dimension, (should divide imageDim-filterDim+1)
poolDim1 = 2;      % Pooling dimension, (should divide imageDim-filterDim+1)

% Load MNIST Train
addpath ../common/;
images = loadMNISTImages('../common/train-images-idx3-ubyte');
images = reshape(images,imageDim0,imageDim0,1,[]);
labels = loadMNISTLabels('../common/train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

% Initialize Parameters
theta = cnnInitParams(imageDim0,filterDim0,numInplane0,numOutplane0,poolDim0,...
                      imageDim1,filterDim1,numInplane1,numOutplane1,poolDim1,numClasses);

%%======================================================================
%% STEP 1: Implement convNet Objective
%  Implement the function cnnCost.m.

%%======================================================================
%% STEP 2: Gradient Check
%  Use the file computeNumericalGradient.m to check the gradient
%  calculation for your cnnCost.m function.  You may need to add the
%  appropriate path or copy the file to this directory.

DEBUG=true;  % set this to true to check gradient
if DEBUG
    % To speed up gradient checking, we will use a reduced network and
    % a debugging data set

    db_imageDim0 = 28;
    db_imageDim1 = 5;
    db_filterDim0 = 9;    % Filter size for conv layer
    db_filterDim1 = 2;
    db_numInplane0 = 1;
    db_numOutplane0 = 2;
    db_numInplane1 = 2;
    db_numOutplane1 = 2;
    db_poolDim0 = 4;      % Pooling dimension, (should divide imageDim-filterDim+1)
    db_poolDim1 = 2;      % Pooling dimension, (should divide imageDim-filterDim+1)
    db_images = images(:,:,:,1:10);
    db_labels = labels(1:10);
    db_theta = cnnInitParams(db_imageDim0,db_filterDim0,db_numInplane0,db_numOutplane0,db_poolDim0,...
                             db_imageDim1,db_filterDim1,db_numInplane1,db_numOutplane1,db_poolDim1,numClasses);
    
    [cost grad] = cnnCost(db_theta,db_images,db_labels,numClasses,db_filterDim0,db_numInplane0,...
                          db_numOutplane0,db_poolDim0,db_filterDim1,db_numInplane1,db_numOutplane1,...
                          db_poolDim1);

    % Check gradients
    numGrad = computeNumericalGradient( @(x) cnnCost(x,db_images,db_labels,numClasses,db_filterDim0,db_numInplane0,...
                          db_numOutplane0,db_poolDim0,db_filterDim1,db_numInplane1,db_numOutplane1,...
                          db_poolDim1), db_theta);
 
    % Use this to visually compare the gradients side by side
    disp([numGrad grad]);
    
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    % Should be small. In our implementation, these values are usually 
    % less than 1e-9.
    disp(diff); 
 
    assert(diff < 1e-9,...
        'Difference too large. Check your gradient computation again');
    
end;

%%======================================================================
%% STEP 3: Learn Parameters
%  Implement minFuncSGD.m, then train the model.

options.epochs = 3;
options.minibatch = 256;
options.alpha = 1e-1;
options.momentum = .95;

opttheta = minFuncSGD(@(x,y,z) cnnCost(x,y,z,numClasses,filterDim0,numInplane0,numOutplane0,poolDim0,...
                                filterDim1,numInplane1,numOutplane1,poolDim1),theta,images,labels,options);

%%======================================================================
%% STEP 4: Test
%  Test the performance of the trained model using the MNIST test set. Your
%  accuracy should be above 97% after 3 epochs of training

testImages = loadMNISTImages('../common/t10k-images-idx3-ubyte');
testImages = reshape(testImages,imageDim0,imageDim0,1,[]);
testLabels = loadMNISTLabels('../common/t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

[~,cost,preds]=cnnCost(opttheta,testImages,testLabels,numClasses,...
                       filterDim0,numInplane0,numOutplane0,poolDim0,...
                       filterDim1,numInplane1,numOutplane1,poolDim1,true);

acc = sum(preds==testLabels)/length(preds);

% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f\n',acc);

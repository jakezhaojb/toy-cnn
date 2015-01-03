function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialization
  f = 0;
  g = zeros(size(theta));

  % cost computation
  theta(:, end) = 0; % leave out the last theta
  prob = exp(theta' * X);
  sumProb = sum(prob, 1);
  prob = bsxfun(@times, prob, 1 ./ sumProb);
  probL = log(prob);
  I = sub2ind(size(prob), reshape(y, 1, []), 1:m);
  probL = probL(I);
  f = -sum(probL);

  % gradients computation
  t = zeros(size(prob));
  t(I) = 1;
  t = t - prob;
  g_ = zeros(numel(theta), m);
  % TODO subject to optimizing?
  for i = 1:m
      g_(:, i) = -kron(t(:, i), X(:, i));
  end
  g = sum(g_, 2);
  g = reshape(g, size(theta));
  g(:, end) = 0; % leave out the last theta, keep it to 0.
  
  g=g(:); % make gradient a vector for minFunc


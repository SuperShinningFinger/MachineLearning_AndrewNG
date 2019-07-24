function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
predictions = X * theta; 
predictions = sigmoid(predictions);
delta = -y .* log(predictions) - (1 - y) .* log(1-predictions);
J = sum(delta) / m;
% 在原基础上加上正则化项
J = J + lambda * sum(theta(2:size(theta)).^2) / (2 * m);

% 也可以先全部计算再加上正则项
grad(1) = sum(X'(1) * (predictions - y)) / m;
grad(2:size(theta)) = (X'(2:size(X', 1), :) * (predictions - y)) / m + lambda / m * theta(2:size(theta));



% =============================================================

end

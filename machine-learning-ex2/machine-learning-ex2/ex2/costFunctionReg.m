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

hypothesis = sigmoid(X * theta);	% hypothesis is m*1

% calculation of cost non-regularized
%J = 1 / m * sum(-y .* log(hypothesis) - (1-y) .* log(1 - hypothesis));

theta_dash = theta(2:size(theta));
theta_reg = [0;theta_dash];


% calculation of cost regularized
J = (1/m) * (-y' * log(hypothesis) - (1-y)' * log(1-hypothesis)) + (lambda / (2*m)) * theta_reg' * theta_reg;

% gradient non-regularized
% for i = 1 : m
%	grad = grad + (hypothesis(i) - y(i)) * X(i,:)'; 	
	%since X(i,:) is a row vector, we transpose it to get a column vector
% end
% grad = (1 / m) * grad;

% gradient regularized
grad = (1 / m) * (X' * (hypothesis - y) + lambda * theta_reg);


% =============================================================

end

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%delta = zeros(1,2);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	
	% X -> 97*2			   ; theta -> 2*1			=> X*theta -> 97*1
	% y -> 97*1			   ; (X*theta-y)-> 97*1 	=> (X*theta-y)' -> 1*97
	% (X*theta-y)' -> 1*97 ; X -> 97*2				=> (X*theta-y)' * X -> 1*2
	% delta -> 1*2
	
	delta = (1/m) * (X*theta-y)' * X; 
	theta = theta - alpha * delta';
	
	
	

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	fprintf('J for %d iteration: %0.6f \n ',iter,J_history(iter));
	
end

end

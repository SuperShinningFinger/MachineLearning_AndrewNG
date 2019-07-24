function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% X(5000x401) Theta1 10x401 Theta2 10x26
%添加偏置单元
X = [ones(size(X, 1), 1) X];

% hidden layer 5000x25
layer2 = sigmoid(X * Theta1');

% 添加偏置单元
layer2 = [ones(size(layer2, 1), 1) layer2];

% output layer 5000x10
layer3 = sigmoid(layer2 * Theta2');

% 每个样本最有可能的数字预测结果
[max_value, max_index] = max(layer3, [], 2);
p = max_index;








% =========================================================================


end

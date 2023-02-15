function [output] = inner_product_forward(input, layer, param)

d = size(input.data, 1);
k = size(input.data, 2); % batch size
n = size(param.w, 2);

% Replace the following line with your implementation.

% Initalizing the height, width, channel and batch size

% Height
output.height = n;

% Width
output.width = 1;

% Others
output.channel = 1;

% Batch size is already given
output.batch_size = k;

% Calculatin the output using the formula w*x+b
w = transpose(param.w);
x = input.data;
b = transpose(param.b);

% Calculate the output
output.data = w * x + b;


end

function [output] = conv_layer_forward(input, layer, param)
% Conv layer forward
% input: struct with input data
% layer: convolution layer struct
% param: weights for the convolution layer

% output: 

h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
pad = layer.pad;
stride = layer.stride;
num = layer.num;
% resolve output shape
h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;

assert(h_out == floor(h_out), 'h_out is not integer')
assert(w_out == floor(w_out), 'w_out is not integer')
input_n.height = h_in;
input_n.width = w_in;
input_n.channel = c;

%% Fill in the code
% Iterate over the each image in the batch, compute response,
% Fill in the output datastructure with data, and the shape. 

% initalizing the preknown data
output.height = h_out;
output.width = w_out;

% the channel is the layer number 
output.channel = num;
output.batch_size = batch_size;

% Output datasrtucture
output.data = zeros([h_out, w_out, num, batch_size]);

% Now we have all the data to start convolutions
% loop to do the convolutions on the image
for imageBatch = 1:batch_size

    % Get the data for the current kernal  
    inputImage = input.data(:, imageBatch);

    % Reshapt the image like in the pooling layer
    inputImage = reshape(inputImage, [h_in, w_in, c]);
    % Same as pooling layer
    inputImage = padarray(inputImage, [pad pad]);

    % Loop over the image to do the convalution
    for r = 1:h_out
        for c1 = 1:w_out
            for filter = 1:num

                convArea = inputImage(1+(r-1)*stride:k+(r-1)*stride, 1+(c1-1)*stride:k+(c1-1)*stride, :);

                % getting the filter 
                convFilter = param.w(:, filter);
                convFilter = reshape(convFilter, [k k c]);

                % We need the bias which is already probided
                convBias = param.b(:, filter);

                % do convolution on current image kernal 
                convres = sum(convArea .* convFilter, 'all') + convBias;
                output.data(r, c1, filter, imageBatch) = convres;
            end
        end
    end
end

% Reshape the the image data and save in output
output.data = reshape(output.data, [h_out * w_out * num, batch_size]);

end


function [output] = pooling_layer_forward(input, layer)

    h_in = input.height;
    w_in = input.width;
    c = input.channel;
    batch_size = input.batch_size;
    k = layer.k;
    pad = layer.pad;
    stride = layer.stride;
    
    h_out = (h_in + 2*pad - k) / stride + 1;
    w_out = (w_in + 2*pad - k) / stride + 1;
    
    
    output.height = h_out;
    output.width = w_out;
    output.channel = c;
    output.batch_size = batch_size;

    % Replace the following line with your implementation.
    output.data = zeros([h_out, w_out, c, batch_size]);

    % Running a loop from 1 to the batch size
    % We already have the batch size
    for batchImage = 1:batch_size

        % imageInput has the data of the current image
        imageInput = input.data(:, batchImage);

        % We need to resize the current image using the height, width and channel 
        imageInput = reshape(imageInput, [h_in, w_in, c]);

        % paddding the image in pooling layer
        % Using the image processing toolbox to used the padarray functions
        imageInput = padarray(imageInput, [pad pad]);

        % max pooling
        for r_idx = 1:h_out
            for c_idx = 1:w_out
                for channel_idx = 1:c

                    % max pooling on the kernal area
                    maxPoolingKernal = imageInput(1+(r_idx-1)*stride:k+(r_idx-1)*stride, 1+(c_idx-1)*stride:k+(c_idx-1)*stride, channel_idx);
                    % save the image data after pooling
                    output.data(r_idx, c_idx, channel_idx, batchImage) = max(maxPoolingKernal, [], 'all');
                end
            end
        end
    % ending the loop after the whole image has been processed
    end

    % Reshaping the whole image and saving it in data
    output.data = reshape(output.data, [h_out * w_out * c], batch_size);

end


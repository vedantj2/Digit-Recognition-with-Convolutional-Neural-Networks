layers = get_lenet();
load lenet.mat
% load data
% Change the following value to true to load the entire dataset.
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);
xtrain = [xtrain, xvalidate];
ytrain = [ytrain, yvalidate];
m_train = size(xtrain, 2);
batch_size = 64;
 
 
layers{1}.batch_size = 1;
img = xtest(:, 1);
img = reshape(img, 28, 28);
imshow(img')
 
%[cp, ~, output] = conv_net_output(params, layers, xtest(:, 1), ytest(:, 1));
output = convnet_forward(params, layers, xtest(:, 1));
output_1 = reshape(output{1}.data, 28, 28);
imshow(output_1);
% Fill in your code here to plot the features.

% code for the second layer

img2 = output{2}.data;
% This layer is 24*24 hence reshaping it to 24*24*20
img2 = reshape(img2, [24, 24, 20]);
% running the loops from the first to the twentieth instance
for i = 1:20
    subplot(4, 5, i);
    img2Output = transpose(img2(:, :, i));
    imshow(img2Output);
end

% Same for layer 3
img3 = output{3}.data;
% This layer is also 24*24 hence reshaping it to 24*24*20
img3 = reshape(img3, [24, 24, 20]);
figure;
for i = 1:20
    subplot(4, 5, i);
    img3Output = transpose(img3(:, :, i));
    imshow(img3Output);
end
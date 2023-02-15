input = zeros(784, 100);
for i=0:4
    filename = sprintf('./img/%d.png', i);
    img = imread(filename);
    img = rgb2gray(img);
    img = imresize(img, [28, 28]);
    img = 255 - img;
    img = im2double(img);
    img = transpose(img);
    img = reshape(img, [784, 1]);
    input(:, i+1) = img;
end
load lenet.mat
layers = get_lenet();
[output, P] = convnet_forward(params, layers, input);
for i=0:4
    P(:, i+1)
end
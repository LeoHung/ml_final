function [ normlalized_img ] = normalize_img( img )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    [N, M, d] = size(img);
    tmp_img = reshape(img, 1, N * M * d);
    tmp_img = tmp_img - mean(tmp_img);
    tmp_img = tmp_img / std(tmp_img);
    normlalized_img = reshape(tmp_img, N, M, d );
end


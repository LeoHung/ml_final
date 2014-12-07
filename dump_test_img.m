load('features/data.mat');

N = size(X_test, 1);
h = figure;
for i = 1:200
    imshow(imresize(reshape(X_test(i,:),32,32,3),[256,256]))
    saveas(h,sprintf('../test_img/%d.jpg', i));
end
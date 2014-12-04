load('../data.mat');

N = size(X_train, 1);
h = figure;
for i = 1:N
    imshow(imresize(reshape(X_train(i,:),32,32,3),[256,256]))
    j = y_train(i);
    saveas(h,sprintf('../img/%d/%d.jpg', j, i));
end
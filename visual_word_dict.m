function visual_word_dict( dataset, k, w_size ,output )
%VISUAL_WORD Summary of this function goes here
%   Detailed explanation goes here
    load(dataset);
%     X = [X_train; X_test];
    X = [X_train];
    
    N = size(X, 1);
    img_width = 256;
    sample_factor = 100;
    
    num_w = (img_width/ w_size)^2;
    l_w = img_width/ w_size;
    
    partial_imgs = zeros(N/sample_factor, w_size * w_size *3 );
    
    subpatch_i = 1;
    partial_img_i = 1;
    for img_i = 1:N
        img = normalize_img(imresize(reshape(X(img_i,:) , 32, 32, 3), [img_width, img_width]));
        disp(img_i);
        for i = 1:w_size:(img_width-w_size+1)
            for j = 1:w_size:(img_width-w_size+1)
                subpatch_i = subpatch_i +1;
                if mod(subpatch_i, sample_factor) > 0
                    continue
                end
                
                partial_imgs(partial_img_i, :) = reshape(img(i:(i+w_size-1),j:(j+w_size-1),:), 1, w_size*w_size*3); 
                partial_img_i = partial_img_i + 1;
            end
        end
    end
    
    [visual_word_idx, centroids] = kmeans(partial_imgs, k );
    
    save(output, 'centroids');
end



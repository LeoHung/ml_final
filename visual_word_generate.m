function visual_word_generate(dataset, dictfile, k, w_size ,output)
    load(dataset);
    load(dictfile);
    
%     img_width = 256;
    img_width = 32;
    num_w = (img_width/ w_size)^2;
    l_w = img_width/ w_size;
    
    [X_train_N, M] = size(X_train);
    X_train_partial_imgs = reshaped_data(X_train, w_size, img_width);
    X_train_v_docs = gen_v_doc( X_train_partial_imgs, centroids, X_train_N, num_w);
%     X_train_v_w_dists = gen_v_w_dist(X_train_v_docs, X_train_N, k); 
    disp('gen_v_w_expand(X_train_v_docs, k);');
    X_train_v_w_expand = gen_v_w_expand(X_train_v_docs, k);
    
    
    [X_test_N, M] = size(X_test);
    X_test_partial_imgs = reshaped_data(X_test, w_size, img_width);
    X_test_v_docs = gen_v_doc( X_test_partial_imgs, centroids, X_test_N, num_w);
%     X_test_v_w_dists = gen_v_w_dist(X_test_v_docs, X_test_N, k); 
    disp('gen_v_w_expand(X_test_v_docs, k);');
    X_test_v_w_expand = gen_v_w_expand(X_test_v_docs, k);

    
%     X_train = X_train_v_w_dists;
%     X_test = X_test_v_w_dists;

    X_train = X_train_v_w_expand ;
    X_test = X_test_v_w_expand;

%     [X_train, X_test] = rescale(X_train, X_test);
    
    save(output, 'X_train', 'X_test', 'y_train');
    
end

function v_w_dists = gen_v_w_dist(v_docs,N, k)
    v_w_dists = zeros(N, k); 
    for i = 1:N
        v_w_dists(i,:) = hist(v_docs(i,:), k);
    end
     
end

function v_w_expand = gen_v_w_expand(v_docs, k)
    [N, M] = size(v_docs);
    
    v_w_expand = zeros(N, M *k);
    
    for i = 1:N
        for j = 1:M
            v_w_expand(i, (j-1) * k + v_docs(i, j)) = 1;
        end
    end

end

function v_docs = gen_v_doc(partial_imgs, centroids, N,  num_w)
    D = pdist2(partial_imgs, centroids);
    [M, I] = min(D,[],2);
    
    v_docs = reshape(I, N, num_w);
end

function partial_imgs = reshaped_data(X, w_size, img_width)
    [N, M] = size(X);

    num_w = (img_width/ w_size)^2;
    l_w = img_width/ w_size;
    
    
    partial_imgs = zeros(N * num_w, w_size * w_size*3);
    
    partial_img_i = 1;
    for img_i = 1:N
%         img = normalize_img(imresize(reshape(X(img_i,:) , 32, 32, 3), [img_width, img_width]));
        img = reshape(X(img_i,:) , 32, 32, 3);

       
        for part_i = 1:l_w
            for part_j = 1:l_w
                start_i = (part_i -1) * w_size +1;
                end_i = (part_i) * w_size ;
                start_j = (part_j -1) * w_size + 1;
                end_j = (part_j) * w_size;
                               
                partial_img = reshape(img(start_i:end_i,start_j:end_j, :), 1, w_size * w_size *3); 
          
                
                partial_imgs(partial_img_i,:) = partial_img;
                partial_img_i = partial_img_i +1;
            end
        end
    end
    
end

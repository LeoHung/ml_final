function [rescaled_X_train, rescaled_X_test] = rescale(X_train, X_test)
    X = [X_train; X_test];
    X_train_N = size(X_train, 1);
    rescaled_X = rescale_one(X);
    
    rescaled_X_train = rescaled_X(1:X_train_N, :);
    rescaled_X_test = rescaled_X(X_train_N+1:end, :);
end

function [rescaled_X ] = rescale_one(X)
    min_Xs = min(X);
    max_Xs = max(X);
    
    [N, M] = size(X);
    rescaled_X = zeros(N, M);
    
    for j = 1:M
        min_X = min_Xs(j);
        max_X = max_Xs(j);
        rescaled_X(:, j) = (X(:, j) - min_X) / (max_X - min_X); 
    end

end


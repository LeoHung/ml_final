function [ acc ] = grid( dataset, classifier)
    c_range = power(10, -2:1:1);
    g_range = power(10, 0:-1:-4);
    
    c_N = size(c_range, 2);
    g_N = size(g_range, 2);
    
    performance = zeros(c_N, g_N);
    
    for c_i = 1:c_N
        c = c_range(c_i);
        parfor g_i = 1:g_N
            g = g_range(g_i);
            
            performance(c_i, g_i) = cvClassfier(dataset, struct('lambda', c, 'loss', classifier, 'dual', true, 'kernelfn', 'rbf', 'gamma', g));
            
            fprintf('c= %f, g=%f, acc=%f\n', c, g, performance(c_i, g_i));
        end
    end
    
    [max_acc,ind]=max(performance(:));
    [y,x] = ind2sub(size(performance),ind);
    
    max_c = c_range(y);
    max_g = g_range(x);
    disp(c_range);
    disp(g_range);
    disp(performance);
    fprintf('c= %f, g=%f, acc=%f\n', max_c, max_g, max_acc);
    
end


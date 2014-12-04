function [ acc ] = grid( dataset, classifier, isKernel)
    c_range = power(10, 0:1:3);
    g_range = power(10, 0:-1:-4);
    
    if isKernel
        grid_kernel(c_range, g_range, dataset, classifier)
    else
        grid_linear(c_range, dataset, classifier)
    end
end

function grid_linear(c_range, dataset, classifier)
    c_N = size(c_range, 2);
    
    performance = zeros(c_N);
    
    for c_i = 1:c_N
        c = c_range(c_i);
        performance(c_i) = cvClassfier(dataset, struct('lambda', c, 'loss', classifier));
            
        fprintf('c= %f,  acc=%f\n', c, performance(c_i));
    end
    
    [max_acc,ind]=max(performance(:));
    
    max_c = c_range(ind);
    disp(c_range);
    disp(performance);
    fprintf('c= %f, acc=%f\n', max_c, max_acc);

end

function grid_kernel(c_range, g_range, dataset, classifier)
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

function acc =  cvClassfier( dataset, opt )
%RUNCLASSIFIER Runs a simple SVM or MLR classifier.
% dataset - either 'random' or './path/to/dataset/' containing
%           entries X_train, X_test, y_train, (y_test - optional).
% opt     - options to run with:
%     .loss     - 'mlr' for softmax regression and 'l2svm' for L2 SVM.
%                 Default is 'mlr'.
%
%     .lambda   - regularization parameter. Default is 0.
%
%     .dual     - optimize in the dual if true. Default is false. If false
%                 then a linear kernel is used.
%
%     .kernelfn - kernel function - Either a string 'rbf' for RBF kernel or
%                 'poly' for a polynomial kernel.
%                 Alternatively, kernelfn can be a function kernelfn(x, y)
%                 which should return an m1 x m2 gram matrix between 
%                 x and y, where there are m1 examples in x and m2 in y.
%                 For example you can implement a tanh kernel with params
%                 a and b as opt.kernelfn = @(X1, X2) tanh(a*X1*X2' - b).
%                 Default is 'rbf'.
%
%     .gamma    - RBF kernel width. Larger gamma => smaller variance.
%                 gaussian. Default is 1.
%
%     .order    - Polynomial order. Default is 3.
%
    cv_round = 10;

    if nargin < 1, dataset = 'random'; end

    if nargin < 2
        % parameters you can play with.
        opt.lambda = 1;        % regularization
        opt.loss = 'mlr';      % 'mlr' for Multinomial Logistic Regression
                               % (softmax) or 'l2svm' for L2 SVM.
        opt.dual = false;      % optimize dual problem
                               % (must be true to use kernels)
        opt.kernelfn = 'rbf';  % kernel to use (either rbf or poly)
        opt.gamma = 1e-2;      % Kernel parameter for RBF kernel.
        opt.order = 2;         % Kernel parameter for polynomial kernel.
    end
    
    % type the following into the matlab terminal to compile minFunc:
    % >> addpath ./minFunc/
    % >> mexAll
    addpath(genpath('./minFunc/'));
    addpath ./tinyclassifier/    
    addpath ./helpers
    
    disp(opt);
    load(dataset);
    y_train = double(y_train);
    n = size(X_train, 1);
    ymin = min(y_train(:));
    y_train = y_train - ymin + 1;
    K = max(y_train(:));
                
    cv_round_accuracys = zeros(cv_round,1);
    cv_N = n/ cv_round;
    for cv_i = 1:cv_round
        if cv_i < cv_round
            start_i = (cv_i - 1) *cv_N+1;
            end_i = cv_i * cv_N;
        else
            start_i = (cv_i - 1) *cv_N+1;
            end_i = n;
        end
        train_list = true(n, 1);
        train_list(start_i:end_i, 1) =false;
                
        test_list = false(n,1);
        test_list(start_i:end_i, 1) = true;
                
        cv_X_train = X_train(train_list,:);
        cv_y_train = y_train(train_list,:);
        cv_X_test = X_train(test_list, :);
        cv_y_test = y_train(test_list, :);
        
        disp(size(cv_X_train));
        disp(size(cv_X_test));
        
        params = trainClassifier(cv_X_train(1:end,:), cv_y_train(1:end), opt);
        preds = predictClassifier(params, cv_X_test);
        cv_round_accuracys(cv_i, 1) = 100 * mean(preds(:) == cv_y_test(:));
    end
    
    disp(cv_round_accuracys);
    fprintf('CV accuracy = %.2f%%\n', mean(cv_round_accuracys));
    
    acc = cv_round_accuracys;
    
end


% Authors: Tatsiana Palikarpava, Daniil Chyzhou

% Loading data from file 
load('ex04plain.mat', 'Adata', 'Alab')
% Constructing dataset
a = prdataset(Adata, Alab)
% Displaying it on graph
figure(1),clf
scatterd(a), hold on
pause
% ---------------Crossvalidation------------------
% Calculating errors for KNN-classifiers
Errors_c = [];
% Standard deviation
Stdev_c = [];
er = zeros(1,40);
for i = 1:40 
    % Constructing KNN-classifier for different k
    wc_crossval = knnc([], i)
    % Calculating error and standard deviation
    
    [Errors_c(i), Stdev_c(i)] = prcrossval(a, wc_crossval, 10, 2);
    % Here we deal with separate folds and manually calculate error rates for them
    R  = prcrossval(a, [], 10, 0);
    for j = 1:10
        curr_fold = a(R == j, :);
        other_folds = a(R~=j,:);
        er(i) = er(i) + testc(curr_fold,knnc(other_folds,i)) ;
    end
    er(i) = er(i)/10;
end
[min_err, best_k] = min(Errors_c);
% Displaying this data on the curves
figure(2), clf
plot (Errors_c, 'b.-'), hold on
plot (Stdev_c, 'r.-')
plot (er, 'g.- ')
xlabel('Values of k'),ylabel('Error')
title('10-fold cross validation')
legend({'Error rate','Standard deviation','Error for separate '},'Location','SouthEast')
% For little k like 1 or 2 error rate is very high, because the classifier
% uses creates "islands" that will never appear in test data. But too high
% values of k are also useless, because they will work slowly, giving
% similar error rate. It is because we take into account entities that are
% located too far from the current object.

% ---------------Resubstitution------------------ 
% Calculating errors for KNN-classifiers
Errors_r = [];
%Stdev_r = std(Ts);
for i = 1:40
    % Constructing KNN-classifier for different k using training data
    wc_resubst = knnc(a, i) 
    % Calculating error and standard deviation for test data
    Errors_r(i) = testc(a*wc_resubst);
end
% Displaying this data on the curves
figure(3), clf
plot (Errors_r, 'b.-'), hold on
% plot (Stdev_r, 'r.-')
xlabel('Values of k'),ylabel('Error')
title('Resubstitution')
legend({'Error rate'},'Location','SouthEast')
% For resubstitution we have another situation. In the case of 1NN
% classifier error is 0, and then it starts increasing, reaching more or less
% similar rate as in cross validation. But it's still a bit lower because
% resubstitution is a bit biased.


% Searching for the best k value

% Constructing a classifier for this k
wc_best = knnc(a, best_k);
% ROC-curve (we change it for FP/TP)
err = roc(a*wc_best);

tp = 1 - err.xvalues; 
err.xvalues = err.error
err.error = tp
err.xlabel = 'False Positive';
err.ylabel = 'True Positive';
err.title = 'ROC-curve'
figure(4), clf
plote(err,'g--'),hold on
pause

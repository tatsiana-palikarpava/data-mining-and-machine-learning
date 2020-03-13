% AUTHORS:
% Tatsiana Palikarpava
% Irena Bytyci


%
% initial code for Exercise01 
% (have a look at the corresponding task in the teaching system
%
prwaitbar off
A=breast                              % a 2-class dataset
wc=ldc(A)                                  % a classifier

Res = A*wc                        % classification result
testc(Res)                         % resubstitution error

confmat(Res)                 % shows the confusion matrix
cf = confmat(Res);                              % save it
ww = 1.0 ./ classsizes(A);
ncf = bsxfun(@times,cf,ww');                  % normalize

% performance measures
[tp,tn]=testc(Res,'TP',classnames(A,1));
[fn,fp]=testc(Res,'FN',classnames(A,1));

% normalized confmat
ncf

[tp fn; fp tn]

% compute ROC curve
Npoints=61
Thresholds = [ linspace(0,1,Npoints) 1];
e = roc(A*wc,[],Npoints)                    % a ROC curve
                % using Npoints thresholds on likelihoods

% display it                
figure(1),clf
e.xlabel = 'False Negative';
e.ylabel = 'False Positive';
plote(e,'gd-')
hold on
plot(fn,fp,'ro','MarkerSize',15)

% display separately the two errors in the curve
figure(2),clf,hold on 
plot(Thresholds,e.xvalues,'r.-')
plot(.5,fn,'ro','MarkerSize',15)
plot(Thresholds,e.error,'b.-')
plot(.5,fp,'bd','MarkerSize',15)
xlabel('Threshold'),ylabel('Error rate')
legend({'FN','','FP',''},'Location','SouthEast')
%======================================================
% E.XVALUES contains the errors in the first class (FN)
% E.ERROR contains the errors in the second class (FP)
e1 = e
TP = 1 - e.xvalues; 
e.xvalues = e.error
e.error = TP
% Now we have FP  in the first class and TP in the second

figure(3),clf
e.xlabel = 'False Positive';
e.ylabel = 'True Positive';
plote(e,'gd-')
hold on
plot(fp,tp,'ro','MarkerSize',15)

% display separately the two errors in the curve
figure(4),clf,hold on 
plot(Thresholds,e.xvalues,'r.-')
plot(.5,fp,'ro','MarkerSize',15)
plot(Thresholds,e.error,'b.-')
plot(.5,tp,'bd','MarkerSize',15)
xlabel('Threshold'),ylabel('Error rate')
legend({'FP','','TP',''},'Location','SouthEast')

n_A = cf(1,1)+cf(1,2);
n_B = size(Res,1) - n_A;
prec = zeros(1,Npoints)
for i = 1:(Npoints)
    
    %n_A = (1-Thresholds(i))*size(Res);
    %n_B = size(Res) - n_A;
    
    tpr = (1-e1.xvalues(i))*n_A;
    fpr = e1.error(i)*n_B;
    prec(i) = tpr/(tpr + fpr);
end
%disp(prec);
figure(5),clf,hold on 
plot(Thresholds(1:Npoints),prec,'r.-')
plot(.5,prec(31),'ro','MarkerSize',15)
plot(Thresholds,e.error,'b.-')
plot(.5,tp,'bd','MarkerSize',15)
xlabel('Threshold'),ylabel('prec/rec rate')
legend({'prec','','rec',''},'Location','SouthEast')





acc = zeros(1,Npoints)
for i = 1:(Npoints)
    %n_A = (1-Thresholds(i))*size(Res);
    %n_B = size(Res) - n_A;
  
    tpr = (1-e1.xvalues(i))*n_A;
    tnr = (1-e1.error(i))*n_B;
    fpr = e1.error(i)*n_B;
    fnr = (e1.xvalues(i))*n_A;
    acc(i) = (tpr + tnr)/(tpr + fpr + tnr + fnr);
end
[m, i] = max(acc)

figure(7),clf,hold on 
plot(Thresholds(1:Npoints),acc,'r.-')
plot(.5,acc(31),'ro','MarkerSize',15)
plot(i*1/Npoints,acc(i),'bd','MarkerSize',15)
xlabel('Threshold'),ylabel('accuracy')






N=150;
noises = 5*[0 0.1 0.2];

ppos=get(gcf,'Position');
ppos(3:4)= [1120 420] ;
set(gcf, 'Position',  ppos)
for n = 1:3
  % Generating data with appropriate noise level
  B_data = gendatb(N,noises(n))
  % Displaying data
  figure(1),clf
  scatterd(B_data),axis([-10 6 -10 6]),grid on, axis equal
  title(['var=' num2str(noises(n)) ', std=' num2str(sqrt(noises(n)))])
  % Means and covariance for Bayes classifier
  [U, covs] = meancov(B_data);
  % Displaying Bayes classifier
  wb = nbayesc(U,covs)
  plotc(wb)
  % Quadratic discriminant classifier
  wbqdc=qdc(B_data) 
  % Linear discriminant classifier
  wbldc=ldc(B_data)
  % Displaying them
  plotc(wbqdc, 'r:')
  plotc(wbldc, 'b:')
   legend({'Class 1','Class 2','Bayes','QDC','LDC'},'Location','SouthEast')
  
  
  % Calculating TP/FP roc-curves
  N_points = 101
  Thresholds = [ linspace(0,1,N_points) 1];
  
  res_B = B_data*wb
  res_Q = B_data*wbqdc
  res_L = B_data*wbldc
  
  figure(2), clf
  
  e_B = roc(res_B,[], N_points)
  tp = 1 - e_B.xvalues; 
  e_B.xvalues = e_B.error
  e_B.error = tp
  [TPB,TNB]=testc(res_B,'TP',classnames(B_data,1));
  [FNB,FPB]=testc(res_B,'FN',classnames(B_data,1));
  
  e_Q = roc(res_Q,[], N_points)
  tp = 1 - e_Q.xvalues; 
  e_Q.xvalues = e_Q.error
  e_Q.error = tp
  [TPQ,TNQ]=testc(res_Q,'TP',classnames(B_data,1));
  [FNQ,FPQ]=testc(res_Q,'FN',classnames(B_data,1));
  
  e_L = roc(res_L,[], N_points)
  tp = 1 - e_L.xvalues; 
  e_L.xvalues = e_L.error
  e_L.error = tp
  [TPL,TNL]=testc(res_L,'TP',classnames(B_data,1));
  [FNL,FPL]=testc(res_L,'FN',classnames(B_data,1));
  % Displaying roc-curves
  plote(e_B,'g:'),hold on
  plote(e_Q,'r:'),hold on
  plote(e_L,'b:'),hold on
  plot(FPB,TPB,'go','MarkerSize',10)
  plot(FPQ,TPQ,'ro','MarkerSize',10)
  plot(FPL,TPL,'bo','MarkerSize',10)
  legend({'Bayes','QDC','LDC'},'Location','SouthEast')
  xlabel('FP'),ylabel('TP')
  % Accuracy plots
  cf_B = confmat(res_B); 
  acc = zeros(1, N_points)
  n_A = cf_B(1,1) + cf_B(1,2)
  n_B = N - n_A
  for i = 1:N_points
    tpr = e_B.error(i)*n_A;
    tnr = (1-e_B.xvalues(i))*n_B;
    fpr = e_B.xvalues(i)*n_B;
    fnr = (1 - e_B.error(i))*n_A;
    acc(i) = (tpr + tnr)/(tpr + fpr + tnr + fnr);
  end
  [m, i] = max(acc)
  figure(3),clf, hold on
  plot(Thresholds(1:N_points),acc,'r.-')
  plot(.5,acc(51),'ro','MarkerSize',5)
  plot(i*1/N_points,acc(i),'bo','MarkerSize',5)
  xlabel('Threshold'),ylabel('Accuracy')
  
  
  cf_Q = confmat(res_Q); 
  acc = zeros(1, N_points)
  n_A = cf_Q(1,1) + cf_Q(1,2)
  n_B = N - n_A
  for i = 1:N_points
    tpr = e_Q.error(i)*n_A;
    tnr = (1-e_Q.xvalues(i))*n_B;
    fpr = e_Q.xvalues(i)*n_B;
    fnr = (1 - e_Q.error(i))*n_A;
    acc(i) = (tpr + tnr)/(tpr + fpr + tnr + fnr);
  end
  [m, i] = max(acc)
  plot(Thresholds(1:N_points),acc,'g.-')
  plot(.5,acc(51),'ro','MarkerSize',5)
  plot(i*1/N_points,acc(i),'bo','MarkerSize',5)
  xlabel('Threshold'),ylabel('Accuracy')
  
  cf_L = confmat(res_L); 
  acc = zeros(1, N_points)
  n_A = cf_L(1,1) + cf_L(1,2)
  n_B = N - n_A
  for i = 1:N_points
    tpr = e_L.error(i)*n_A;
    tnr = (1-e_L.xvalues(i))*n_B;
    fpr = e_L.xvalues(i)*n_B;
    fnr = (1 - e_L.error(i))*n_A;
    acc(i) = (tpr + tnr)/(tpr + fpr + tnr + fnr);
  end
  [m, i] = max(acc)
  plot(Thresholds(1:N_points),acc, 'b.-')
  plot(.5,acc(51),'ro','MarkerSize',5)
  plot(i*1/N_points,acc(i),'bo','MarkerSize',5)
  legend({'acc_b','0.5 threshold','max accuracy','acc_q','','','acc_l','',''},'Location','SouthEast')
  xlabel('Threshold'),ylabel('Accuracy')
  pause
end

 




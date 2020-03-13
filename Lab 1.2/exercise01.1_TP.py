#!/usr/bin/env python2
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Data Mining and Machine Learning, 2019-20

Created on Tue Sep 26 21:34:02 2017

@author: pfr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This is the translation to Python/scikit-learn of exercise01.m
% initial code for Exercise01 
% (have a look at the corresponding task in the teaching system)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

print(__doc__)

# You almost always will need numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

# and some of the modules from SCIKIT-LEARN 
from sklearn import datasets,metrics

# At this point we decide whether we're going to use linear classifier or quadratic
# bool linear = true stands for LDA, false - for QDA
linear = True
if linear:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as DiscriminantAnalysis
else:
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as DiscriminantAnalysis


breast = datasets.load_breast_cancer()  # load dataset

# Scikit-learn uses objects with a variable number of attributes

print('The fields in the dataset structure are {}.'.format(dir(breast)))

# but almost always you've got at least the attributes 'data' and 'target'


(N,D)=breast.data.shape
C = breast.target_names.size

print('{} {}-dimensional examples from {} classes'.format(N,D,C))

#print breast.DESCR
print('\n%% Labels are: {}'.format(list(breast.target_names)))
print('\n%% Features are: {}'.format(list(breast.feature_names)))

print('\n%% Labels in the target field are: {}'.format(np.unique(breast.target)))

classsiz = ()
for c in np.unique(breast.target):
    classsiz = classsiz + (np.size(np.nonzero(breast.target==c)),)    
print('\n%% Class frequencies are: {}'.format(classsiz))

#%%


some = [1, 19, 35]    # some database objects

print('')
print(breast.data[some])
print(breast.target[some])
print(breast.target_names[breast.target[some]])


#%%
print('')
print('Learn an arbitrary classifier...')
# Using a Linear Discriminant Analysis:
# https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
wc = DiscriminantAnalysis()
print('    ....done.')

# This call to fit using data is what really does the learning
wc.fit(breast.data, breast.target)

#%%
print('')
print('')
print('And test de classifier')
# and the call to predict is what gives the outputs as labels (as in the target field)
pred = wc.predict(breast.data)
print('')
print('Prediction:\n{}'.format(pred[some]))

# but there are other ways to obtain other outputs. In particular, a
# continuous output can be obtained using decision_function
# But in general you need to have a look at the documentation for each predictor
scores = wc.decision_function(breast.data)
print('')
print('Scores (distances to the classifier)\n{}'.format(scores[some]))

# in some cases, probabilities for each class can also be obtained
probs = wc.predict_proba(breast.data)
print('')
print('Probabilities of being in each class:\n{}'.format(probs[some]))



#%%


# Accuracy
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# Overall accuracy: ratio of properly classified samples
Acc = metrics.accuracy_score(breast.target,pred)
print('Overall accuracy is {}'.format(Acc))
# If normalize is set to False, returns the number, not the ratio
n_ok = metrics.accuracy_score(breast.target,pred,normalize=False)
print('Number of hits {}'.format(n_ok))

# Average error. Equiv. to sum(breast.target!=pred)/float(N)
print('Average error is {}'.format((N-n_ok)/float(N)))
print(' ')


#%%

# Since some is a list of intergers, probs[some] is the probabilities at
# the indexes contained in some
print('Some output probabilities...\n{}'.format(probs[some]))
print('and label predictions...\n{}'.format(pred[some]))
print('with corresponding decision function...\n{}'.format(scores[some]))


#%%


cf = metrics.confusion_matrix(breast.target,pred,labels=[0, 1])
print('')
print('Confusion matrix:\n{}'.format(cf))

# cf.sum(axis=1)[:, np.newaxis] Does the same in the denominator, but
# is less explicit than reshape:
ncf = cf.astype('float') / cf.sum(axis=1).reshape(2,1)
print('Normalized confusion matrix:\n{}'.format(ncf))

Precision = metrics.precision_score(breast.target,pred,pos_label=1)
Recall = metrics.recall_score(breast.target,pred,pos_label=1)

print('')
print ('Precision and Recall: {}, {}'.format(Precision,Recall))

TP = Recall
FN = 1 - TP
TN = metrics.recall_score(breast.target,pred,pos_label=0)
FP = 1 - TN

print('')
print( np.array([[TP, FN],[FP,TN]]) )

#%%
# AUTHOR: Tatsiana Palikarpava

# compute the ROC curve

# Here we have a bit different funcions for LDA and QDA, because for QDA
# too much points are dropped,
# so we change the parameter drop_intermediate
if linear == False:
    fpr, tpr, thres = metrics.roc_curve(breast.target, scores, drop_intermediate = False)
else:
    fpr, tpr, thres = metrics.roc_curve(breast.target, scores)


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = {:0.4f})'.format(metrics.auc(fpr,tpr)))

# plot the TP-FP value for thers=0.5
plt.plot(FP,TP,'ro')

# plot the roc curve
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0+.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()

#%%
# plot the values of TP and FP for all thresholds
plt.figure()
plt.plot(thres,fpr, color='darkorange', lw=lw, label='FPr')
plt.plot(thres,tpr, color='navy', lw=lw, label='TPr')

plt.plot([0,0],[FP,TP],'ro')

# We also define specific limits of plot in X axis, because the range of thresholds is too big
if linear == False:
    plt.xlim([-100, 250])
plt.ylim([0.0, 1.0+.01])
plt.xlabel('Thresholds')
plt.ylabel('FP, TP')
plt.title('FP,TP / threshold')
plt.legend(loc="lower right")
plt.show()

#######################################################


#%%


plt.figure()
lw = 2
plt.plot(1 - tpr, fpr, color='darkorange',
         lw=lw, label='ROC curve (area = {:0.4f})'.format(metrics.auc(1 - tpr, fpr)))

# plot the FP-FN value for thers=0.5
plt.plot(FN, FP,'ro')

# plot the roc curve
plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0+.01])
plt.xlabel('False Negative Rate')
plt.ylabel('False Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()
#%%
# plot the values of FP and FN for all thresholds
plt.figure()
plt.plot(thres,fpr, color='darkorange', lw=lw, label='FPr')
plt.plot(thres,1 - tpr, color='navy', lw=lw, label='FNr')

plt.plot([0,0],[FP, FN],'ro')
if linear == False:
    plt.xlim([-50, 50])
plt.ylim([0.0, 1.0+.01])
plt.xlabel('Thresholds')
plt.ylabel('FP, FN')
plt.title('FP, FN / threshold')
plt.legend(loc="lower right")
plt.show()

#######################################################
# We can compute precision & recall curve using this funcion:
# prec, rec, thres = metrics.precision_recall_curve(breast.target, probs[:,1])
# But also we can compute it manually
n_A = classsiz[0];
n_B = classsiz[1];
prec = np.zeros(thres.size)
rec = tpr
for i in range(0, thres.size): 
    tp = tpr[i] * n_A;
    fp = fpr[i] * n_B;
    prec[i] = tp / (tp + fp);
plt.figure()
lw = 2
plt.plot(prec, rec, color='darkorange',
         lw=lw, label='precision-recall curve)')

# plot the Precision-Recall value for thers=0.5
plt.plot(Precision, Recall,'ro')

# plot the roc curve
plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0+.01])
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-recall curve')
plt.legend(loc="lower right")

plt.show()

#%%

# plot the values of Precision and Recall for all thresholds
plt.figure()
plt.plot(thres,prec, color='darkorange', lw=lw, label='Precision')
plt.plot(thres,rec, color='navy', lw=lw, label='Recall')
plt.plot([0,0],[TP*n_A/(TP*n_A+FP*n_B), TP],'ro')
if linear == False:
    plt.xlim([-100, 100])
plt.ylim([0.0, 1.0+.01])
plt.xlabel('Thresholds')
plt.ylabel('Precision, Recall')
plt.title('Precision, Recall / threshold')
plt.legend(loc="lower right")
# =============================================================================
plt.show()
# =============================================================================
plt.figure()
#And now we compute average accuracy
acc = (n_B*(1-fpr)+n_A*tpr)/N
plt.plot(thres,acc, color='red', lw=lw, linestyle='--', label='Accuracy')
# We calculate maximum value of accuracy and corresponding threshold value and display it as a blue point
maxa = max(acc)
print("The best accuracy is ")
print(maxa)
inda = np.argmax(acc)
plt.plot([0],[Acc],'ro')
plt.plot([thres[inda]],[maxa],'bo')
if linear == False:
    plt.xlim([-100, 100])
plt.ylim([0.0, 1.0+.01])
plt.xlabel('Thresholds')
plt.ylabel('Error Rates')
plt.title('Error curves')
plt.legend(loc="lower right")
plt.show()

#COMPARING OF CLASSIFIERS
#As far as I noticed the highest accuracy for LDA was 0.9796 and for QDA 0.9716, so for QDA it's 
# a bit lower, but from the other side area of ROC curve is 0.9963 for QDA, TP = 0.986, TN = 0.953;
# for LDA roc curve area is 0.9965, TP = 0.994, TN = 0.915
#So, the first class is better predicted by QDA, but the second is much better predicted by LDA,
#Nevertheless, average accuracy is almost the same
#I would prefer using QDA, because it's more flexible tool and the second class is predicted much better



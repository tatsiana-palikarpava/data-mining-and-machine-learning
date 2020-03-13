#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:45:04 2019

@author: pfr
"""

from sklearn import datasets, metrics
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle, islice
from matplotlib import colors

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red':[(0, 0.7, 0.7), (1, 1, 1)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue':  [(0, 1, 1), (1, 0.7, 0.7)]})
plt.cm.register_cmap(cmap=cmap)

N = 600

noises = [0.9, 1.0, 1.1]

data_sets = [datasets.make_moons(n_samples=N, noise=noises[0]),
             datasets.make_moons(n_samples=N, noise=noises[1]),
             datasets.make_moons(n_samples=N, noise=noises[2])]

datas = [data_sets[0][0],data_sets[1][0],data_sets[2][0]]
targets = [data_sets[0][1],data_sets[1][1],data_sets[2][1]]

names = ['one', 'two', 'three']



####################################################################################
#   Define the type of classifier
#   If linear is True then it will be Linear Discriminant Classifier, else Quadratic
####################################################################################
linear = True
for j in range (0,2):
    if linear == True:
        wc = [LDA(), LDA(), LDA()]
    else:
        wc = [QDA(), QDA(), QDA()]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    i = 0
    for ax, data, noise in zip(axes.ravel(), data_sets, noises):
        
        color = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(data[1]) + 1))))
        
        ax.set_title('std='+str(noise))
        ax.scatter(data[0][:, 0], data[0][:, 1], s=10, color=color[data[1]])
        ax.grid()
        ax.set_xlim([-1.5, 2.5])
        ax.set_ylim([-1.5, 2.])
        
        # This call to fit using data is what really does the learning
        wc[i].fit(datas[i], targets[i])
        
        n_A = N/2
        n_B = N/2
        x_min, x_max = [-1.5, 2.5]
        y_min, y_max = [-1.5, 2.]
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_A),
                                 np.linspace(y_min, y_max, n_B))
        Z = wc[i].predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape)
        ax.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                      norm=colors.Normalize(0., 1.), zorder=0)
        ax.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')
        i = i + 1
        
    plt.show()
    
    #   Do the prediction
    pred = [wc[0].predict(datas[0]),wc[1].predict(datas[1]),wc[2].predict(datas[2])]
    Acc = [metrics.accuracy_score(targets[0],pred[0]),
           metrics.accuracy_score(targets[1],pred[1]),
           metrics.accuracy_score(targets[2],pred[2])]
    
    
    
    # calculate the roc curve
    
    Precision = [metrics.precision_score(targets[0],pred[0],pos_label=1),
                 metrics.precision_score(targets[1],pred[1],pos_label=1),
                 metrics.precision_score(targets[2],pred[2],pos_label=1)]
    Recall = [metrics.recall_score(targets[0],pred[0],pos_label=1),
              metrics.recall_score(targets[1],pred[1],pos_label=1),
              metrics.recall_score(targets[2],pred[2],pos_label=1)]
    
    #print('')
    #print ('Precision and Recall: {}, {}'.format(Precision,Recall))
    
    TP = Recall
    FN = np.ones(len(TP)) - TP
    TN = [metrics.recall_score(targets[0],pred[0],pos_label=0),
          metrics.recall_score(targets[1],pred[1],pos_label=0),
          metrics.recall_score(targets[2],pred[2],pos_label=0)]
    FP = np.ones(len(TN)) - TN
    
    scores = [wc[0].decision_function(datas[0]),
              wc[1].decision_function(datas[1]),
              wc[2].decision_function(datas[2])]
    fpr0, tpr0, thres0 = metrics.roc_curve(targets[0], scores[0], drop_intermediate = False)
    fpr1, tpr1, thres1 = metrics.roc_curve(targets[1], scores[1], drop_intermediate = False)
    fpr2, tpr2, thres2 = metrics.roc_curve(targets[2], scores[2], drop_intermediate = False)
    
    # plot TP/FP roc-curve
    plt.figure()
    lw = 2
    plt.plot(fpr0, tpr0, color='darkorange',lw=lw, label = 'TP/FP roc curve, noise = 0')
    plt.plot(fpr1, tpr1, color='green', lw=lw, label = 'TP/FP roc curve, noise = 0.1')
    plt.plot(fpr2, tpr2, color='blue', lw=lw, label = 'TP/FP roc curve, noise = 0.2')
    # plot the TP-FP value for thers=0.5
    plt.plot(FP,TP,'ro', label = 'threshold = 0.5')
    
    # plot the roc curve
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0+.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()
    
    
    plt.figure()
    
    #And now we compute average accuracy
    
    acc0 = (n_B * (np.ones(len(fpr0)) - fpr0) + n_A * tpr0) / N
    plt.plot(thres0,acc0, color='red', lw=lw, linestyle='--', label='Accuracy, noise = 0')
    acc1 = (n_B * (np.ones(len(fpr1)) - fpr1) + n_A * tpr1) / N
    plt.plot(thres1,acc1, color='green', lw=lw, linestyle='--', label='Accuracy, noise = 0.1')
    acc2 = (n_B * (np.ones(len(fpr2)) - fpr2) + n_A * tpr2) / N
    plt.plot(thres2,acc2, color='blue', lw=lw, linestyle='--', label='Accuracy, noise = 0.2')
    # We calculate maximum value of accuracy and corresponding threshold value and display it as a blue point
    maxa = [max(acc0), max(acc1), max(acc2)]
    print("The best accuracies are ")
    print(maxa)
    inda = [np.argmax(acc0),np.argmax(acc1),np.argmax(acc2)]
    plt.plot([0],[Acc],'ro')
    plt.plot([thres0[inda[0]]],[maxa[0]],'bo', label = 'max accuracy')
    plt.plot([thres1[inda[1]]],[maxa[1]],'bo')
    plt.plot([thres2[inda[2]]],[maxa[2]],'bo')
    plt.ylim([0.0, 1.0+.01])
    plt.xlabel('Thresholds')
    plt.ylabel('Accuracy')
    plt.title('Accuracies')
    plt.legend(loc="lower right")
    plt.show()
    
    
    probs0 = wc[0].predict_proba(datas[0])
    probs1 = wc[1].predict_proba(datas[1])
    probs2 = wc[2].predict_proba(datas[2])
    prec0, rec0, thres0 = metrics.precision_recall_curve(targets[0], probs0[:,1])
    prec1, rec1, thres1 = metrics.precision_recall_curve(targets[1], probs1[:,1])
    prec2, rec2, thres2 = metrics.precision_recall_curve(targets[2], probs2[:,1])
    plt.figure()
    lw = 2
    plt.plot(prec0, rec0, color='darkorange',
             lw=lw, label='PR curve, noise = 0')
    plt.plot(prec1, rec1, color='green',
             lw=lw, label='PR curve, noise = 0.1')
    plt.plot(prec2, rec2, color='blue',
             lw=lw, label='PR curve, noise = 0.2')
    
    # plot the Precision-Recall value for thers=0.5
    plt.plot(Precision[0], Recall[0],'ro', label = 'threshold = 0.5')
    plt.plot(Precision[1], Recall[1],'ro')
    plt.plot(Precision[2], Recall[2],'ro')
    # plot the roc curve
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0+.01])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-recall curve')
    plt.legend(loc="lower right")
    
    plt.show()
    # =============================================================================
    linear = False
    # =============================================================================
    


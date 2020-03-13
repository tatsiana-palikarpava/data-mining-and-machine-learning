#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Tatsiana Palikarpava

Example mostly taken from scikit-learn    
"""

from time import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.io import loadmat


from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier

def get_digit(X,i,dim=8):
    dig = X[i].reshape(dim,dim)
    if dim == 16: dig = dig.T
    return dig

def show_digit(X,i,dim=8):
    """ Auxiliary function to show a digit """
    plt.gray()
    plt.matshow(get_digit(X,i,dim))
    plt.title("A sample digit: "+str(y[i]))
    plt.show()
#


#%% Load a dataset and plot some samples
d = 16 # choose between using 8x8 or 16x16 digits.
if d == 8:
    digits = datasets.load_digits()
    X, y = digits.data/16, digits.target
else:
    mat = loadmat('mnist16.mat', squeeze_me=True, struct_as_record=False)
    X, y = mat['A'].data, mat['A'].nlab - 1


show_digit(X,13,dim=d)
show_digit(X,280,dim=d)


# Plot images of the digits
n_img_per_row = 10
h = d+2 # height/width of each digit 
img = np.zeros(( h * n_img_per_row , h * n_img_per_row ))
for i in range(n_img_per_row):
    ix = h * i + 1
    for j in range(n_img_per_row):
        iy = h * j + 1
        k = np.random.randint(len(X))
        img[ix:ix + d, iy:iy + d] = get_digit(X,k,dim=d)

plt.imshow(img, cmap=plt.cm.gray)
plt.xticks([])
plt.yticks([])
plt.title('A selection from the 64-dimensional digits dataset')
plt.show()

#%%
# Create a list of classifiers
# To add a classifier ad a tuple to the list 'classifiers'.
# The tuple must be of the form:
# ( "NAME" , lw , clf )
# where "NAME" is the name you want to appear in the plots, lw is
# the line width for the plots and clf is the classifier itself, a
# scikit-lear object.

classifiers = [
    ("SGDp",1, SGDClassifier(loss='perceptron', eta0=1, learning_rate='constant', penalty=None)),
    ("Perceptron", 1,Perceptron(tol=1e-5,max_iter=100,eta0=1)),
    ("SGD",1, SGDClassifier(loss='squared_hinge', penalty=None))
]


#%% ---------------------------------------
## Exercise 0
###########################################

heldout = [0.95, 0.9,0.75, 0.50, 0.25, 0.01] # Ratio of samples left out from training, for error estimation
rounds = 5 # Number of repetitions to compute average error


xx = 1. - np.array(heldout)
seed = np.random.randint(100)

for name, lws, clf in classifiers:
    print("\n   Training %s" % name)
    rng = np.random.RandomState(seed)  #to have the same for all classifiers
    yyTr = []
    yyTs = []
    
    for i in heldout:
        tr_time = 0
    
        ssumTr = 0
        ssumTs = 0
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=i, random_state=rng)

            t_ini = time()
            clf.fit(X_train, y_train)
            tr_time += time() - t_ini

            y_pred = clf.predict(X_test)

            ssumTr += clf.score(X_train,y_train)
            ssumTs += clf.score(X_test,y_test)

        yyTr.append(ssumTr/rounds)
        yyTs.append(ssumTs/rounds)

        print("Average training time after {} rounds: {}".format(rounds,tr_time/rounds))
        print("average accuracy: {}".format(yyTs[-1]))
    
    plt.plot(xx, yyTs, '-o',lw=lws ,label=name+' (test)')
    plt.plot(xx, yyTr, '--o',lw=lws, label=name+' (train)')

plt.legend(loc="lower right")
plt.xlabel("Relative training set size")
plt.ylabel("Accuracy")
plt.show()

#%%----------------------------------------
## Exercise 1
###########################################

""" I select SGD classifier with percentage of 0.99 for training and 0.01 for testing """


classifiers = [
    ("SGD", 1, SGDClassifier(loss='squared_hinge', penalty=None)),
    ("MLP", 1, MLPClassifier(hidden_layer_sizes=(5), max_iter=130, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1))
]

heldout = [0.01] # Ratio of samples left out from training, for error estimation
rounds = 5 # Number of repetitions to compute average error


xx = 1. - np.array(heldout)
seed = np.random.randint(100)
"""Making a loop for different number of neurons at the first layer"""

    
accTr = []
accTs = []
for name, lws, clf in classifiers:
    
    rng = np.random.RandomState(seed)  #to have the same for all classifiers
    yyTr = []
    yyTs = []
    if name == "MLP":
        for n in range(5, 21, 5):
            clf.set_params(hidden_layer_sizes=(n))
            for i in heldout:
                tr_time = 0
               
                ssumTr = 0
                ssumTs = 0
                for r in range(rounds):
                    X_train, X_test, y_train, y_test = \
                        train_test_split(X, y, test_size=i, random_state=rng)
            
                    t_ini = time()
                    clf.fit(X_train, y_train)
                    tr_time += time() - t_ini
            
                    y_pred = clf.predict(X_test)
            
                    ssumTr += clf.score(X_train,y_train)
                    ssumTs += clf.score(X_test,y_test)
            accTr.append(ssumTr/rounds)
            accTs.append(ssumTs/rounds)
    else:
        for i in heldout:
            tr_time = 0
           
            ssumTr = 0
            ssumTs = 0
            for r in range(rounds):
                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=i, random_state=rng)
        
                t_ini = time()
                clf.fit(X_train, y_train)
                tr_time += time() - t_ini
        
                y_pred = clf.predict(X_test)
        
                ssumTr += clf.score(X_train,y_train)
                ssumTs += clf.score(X_test,y_test)
    
        accTr.append(ssumTr/rounds)
        accTs.append(ssumTs/rounds)


print("SGD train accuracy:"+str(accTr[0]))
print("SGD test accuracy:"+str(accTs[0]))
inds = [1, 2, 3, 4]
for i in inds:
    print("Number of neurons: "+ str(i * 5))
    print("MLP train accuracy:"+str(accTr[i]))
    print("MLP test accuracy:"+str(accTs[i]))
    
    
""" The results obtained for MLP classifier are very bad, we have accuracies below 0.1 """
  #%%  
"""That is why we apply normalizing"""
from sklearn import preprocessing
d = 16 # choose between using 8x8 or 16x16 digits.

if d == 8:
    digits = datasets.load_digits()
    X, y = digits.data/16, digits.target
else:
    mat = loadmat('mnist16.mat', squeeze_me=True, struct_as_record=False)
    X, y = mat['A'].data, mat['A'].nlab - 1
    X = preprocessing.normalize(X)
classifiers = [
    ("SGD", 1, SGDClassifier(loss='squared_hinge', penalty=None)),
    ("MLP", 1, MLPClassifier(hidden_layer_sizes=(5), max_iter=400, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1))
]

heldout = [0.01] # Ratio of samples left out from training, for error estimation
rounds = 5 # Number of repetitions to compute average error


xx = 1. - np.array(heldout)
seed = np.random.randint(100)
    
accTr = []
accTs = []
times = []
for name, lws, clf in classifiers:
    
    rng = np.random.RandomState(seed)  #to have the same for all classifiers
    yyTr = []
    yyTs = []
    if name == "MLP":
        for n in range(5, 21, 5):
            clf.set_params(hidden_layer_sizes=(n))
            for i in heldout:
                tr_time = 0
               
                ssumTr = 0
                ssumTs = 0
                for r in range(rounds):
                    X_train, X_test, y_train, y_test = \
                        train_test_split(X, y, test_size=i, random_state=rng)
            
                    t_ini = time()
                    clf.fit(X_train, y_train)
                    tr_time += time() - t_ini
            
                    y_pred = clf.predict(X_test)
            
                    ssumTr += clf.score(X_train,y_train)
                    ssumTs += clf.score(X_test,y_test)
            accTr.append(ssumTr/rounds)
            accTs.append(ssumTs/rounds)
            times.append(tr_time/rounds)
    else:
        for i in heldout:
            tr_time = 0
           
            ssumTr = 0
            ssumTs = 0
            for r in range(rounds):
                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=i, random_state=rng)
        
                t_ini = time()
                clf.fit(X_train, y_train)
                tr_time += time() - t_ini
        
                y_pred = clf.predict(X_test)
        
                ssumTr += clf.score(X_train,y_train)
                ssumTs += clf.score(X_test,y_test)
    
        accTr.append(ssumTr/rounds)
        accTs.append(ssumTs/rounds)
        times.append(tr_time/rounds)


print("SGD train accuracy:"+str(accTr[0]))
print("SGD test accuracy:"+str(accTs[0]))
print("SGD average time:"+str(times[0]))
inds = [1, 2, 3, 4]
for i in inds:
    print("Number of neurons: "+ str(i * 5))
    print("MLP train accuracy:"+str(accTr[i]))
    print("MLP test accuracy:"+str(accTs[i]))
    print("MLP average time:"+str(times[i]))
    
   
    
"""Now we see excellent result with MLP classifier"""
#%%
# Exercise 2
classifiers = [
    ("SGD", 1, SGDClassifier(loss='squared_hinge', penalty=None)),
    ("MLP", 1, MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=200, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1))
]

heldout = [0.01] # Ratio of samples left out from training, for error estimation
rounds = 5 # Number of repetitions to compute average error

xx = 1. - np.array(heldout)
seed = np.random.randint(100)
    
accTr = []
accTs = []
times = []
for name, lws, clf in classifiers:
    
    rng = np.random.RandomState(seed)  #to have the same for all classifiers
    yyTr = []
    yyTs = []
    if name == "MLP":
        for n in range(5, 21, 5):
            for m in range(5, 21, 5):
                clf.set_params(hidden_layer_sizes=(n, n))
                for i in heldout:
                    tr_time = 0
                   
                    ssumTr = 0
                    ssumTs = 0
                    for r in range(rounds):
                        X_train, X_test, y_train, y_test = \
                            train_test_split(X, y, test_size=i, random_state=rng)
                
                        t_ini = time()
                        clf.fit(X_train, y_train)
                        tr_time += time() - t_ini
                
                        y_pred = clf.predict(X_test)
                
                        ssumTr += clf.score(X_train,y_train)
                        ssumTs += clf.score(X_test,y_test)
                accTr.append(ssumTr/rounds)
                accTs.append(ssumTs/rounds)
                times.append(tr_time/rounds)
    else:
        for i in heldout:
            tr_time = 0
           
            ssumTr = 0
            ssumTs = 0
            for r in range(rounds):
                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=i, random_state=rng)
        
                t_ini = time()
                clf.fit(X_train, y_train)
                tr_time += time() - t_ini
        
                y_pred = clf.predict(X_test)
        
                ssumTr += clf.score(X_train,y_train)
                ssumTs += clf.score(X_test,y_test)
    
        accTr.append(ssumTr/rounds)
        accTs.append(ssumTs/rounds)
        times.append(tr_time/rounds)


print("SGD train accuracy:"+str(accTr[0]))
print("SGD test accuracy:"+str(accTs[0]))
print("SGD average time:"+str(times[0]))
inds = [1, 2, 3, 4]
for i in inds:
    for j in inds:
        print("Number of neurons: "+ str(i * 5) + ", " + str(j * 5))
        print("MLP train accuracy:"+str(accTr[(i - 1)*4 + j]))
        print("MLP test accuracy:"+str(accTs[(i - 1)*4 + j]))
        print("MLP average time:"+str(times[(i - 1)*4 + j]))
    
#%%
# Exercise 3
classifiers = [
    ("SGD", 1, SGDClassifier(loss='squared_hinge', penalty=None)),
    ("MLP", 1, MLPClassifier(hidden_layer_sizes=(10, 20), max_iter=130, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1))
]   
heldout = [0.95, 0.9,0.75, 0.50, 0.25, 0.01] # Ratio of samples left out from training, for error estimation
rounds = 5 # Number of repetitions to compute average error


xx = 1. - np.array(heldout)
seed = np.random.randint(100)
for it in [130, 200, 250]:
    for name, lws, clf in classifiers:
        print("\n   Training %s" % name)
        rng = np.random.RandomState(seed)  #to have the same for all classifiers
        yyTr = []
        yyTs = []
        if name == "MLP":
            clf.set_params(max_iter = it)
        for i in heldout:
            tr_time = 0
        
            ssumTr = 0
            ssumTs = 0
            for r in range(rounds):
                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=i, random_state=rng)
    
                t_ini = time()
                clf.fit(X_train, y_train)
                tr_time += time() - t_ini
    
                y_pred = clf.predict(X_test)
    
                ssumTr += clf.score(X_train,y_train)
                ssumTs += clf.score(X_test,y_test)
    
            yyTr.append(ssumTr/rounds)
            yyTs.append(ssumTs/rounds)
    
            print("Average training time after {} rounds: {}".format(rounds,tr_time/rounds))
            print("average accuracy: {}".format(yyTs[-1]))
        
        plt.plot(xx, yyTs, '-o',lw=lws ,label=name+' (test)')
        plt.plot(xx, yyTr, '--o',lw=lws, label=name+' (train)')
    
    plt.legend(loc="lower right")
    plt.xlabel("Relative training set size")
    plt.ylabel("Accuracy")
    plt.show()

#%%
# Exercise 4

classifiers = [
   ("MLP1", 1, MLPClassifier(hidden_layer_sizes=(10, 20), max_iter=200, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1)),
    ("MLP2", 1, MLPClassifier(hidden_layer_sizes=(10, 20), max_iter=200, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.2)),
    ("MLP3", 1, MLPClassifier(hidden_layer_sizes=(10, 20), max_iter=200, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.21)),
    ("MLP4", 1, MLPClassifier(hidden_layer_sizes=(10, 20), max_iter=200, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.23)),
    ("MLP5", 1, MLPClassifier(hidden_layer_sizes=(10, 20), max_iter=200, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.25)),
    ("MLP6", 1, MLPClassifier(hidden_layer_sizes=(10, 20), max_iter=200, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.3)),
    ("MLP7", 1, MLPClassifier(hidden_layer_sizes=(10, 20), max_iter=200, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.4)),
    ("MLP8", 1, MLPClassifier(hidden_layer_sizes=(10, 20), max_iter=200, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.8))
]   
rounds = 5 # Number of repetitions to compute average error


#xx = 1. - np.array(heldout)

seed = np.random.randint(100)
yyTr = []
yyTs = []
xx = []
for name, lws, clf in classifiers:
    print("\n   Training %s" % name)
    rng = np.random.RandomState(seed)  #to have the same for all classifiers
    
    tr_time = 0
    
    ssumTr = 0
    ssumTs = 0
    for r in range(rounds):
        X_train, X_test, y_train, y_test = \
              train_test_split(X, y, test_size=0.01, random_state=rng)

        t_ini = time()
        clf.fit(X_train, y_train)
        tr_time += time() - t_ini

        y_pred = clf.predict(X_test)

        ssumTr += clf.score(X_train,y_train)
        ssumTs += clf.score(X_test,y_test)

    yyTr.append(ssumTr/rounds)
    yyTs.append(ssumTs/rounds)
    xx.append(tr_time/rounds)  
plt.plot(xx, yyTs, '-o',lw=lws ,label = 'test')
plt.plot(xx, yyTr, '--o',lw=lws, label= 'train')

plt.legend(loc="lower right")
plt.xlabel("Time")
plt.ylabel("Accuracy")
plt.show()

#%%
# Exercise 5
classifiers = [
    ("MLP1", 1, MLPClassifier(hidden_layer_sizes=(10, 20), max_iter=130, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1)),
    ("MLP2", 1, MLPClassifier(early_stopping = True, validation_fraction = 0.1, hidden_layer_sizes=(10, 20), max_iter=130, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1)),
    ("MLP3", 1, MLPClassifier(early_stopping = True, validation_fraction = 0.2, hidden_layer_sizes=(10, 20), max_iter=130, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1)),
    ("MLP4", 1, MLPClassifier(early_stopping = True, validation_fraction = 0.4, hidden_layer_sizes=(10, 20), max_iter=130, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1))
]   
heldout = [0.95, 0.9,0.75, 0.50, 0.25, 0.01] # Ratio of samples left out from training, for error estimation
rounds = 5 # Number of repetitions to compute average error


xx = 1. - np.array(heldout)
seed = np.random.randint(100)
for name, lws, clf in classifiers:
    print("\n   Training %s" % name)
    rng = np.random.RandomState(seed)  #to have the same for all classifiers
    yyTr = []
    yyTs = []

    for i in heldout:
        tr_time = 0
        
        ssumTr = 0
        ssumTs = 0
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=i, random_state=rng)

            t_ini = time()
            clf.fit(X_train, y_train)
            tr_time += time() - t_ini
    
            y_pred = clf.predict(X_test)
    
            ssumTr += clf.score(X_train,y_train)
            ssumTs += clf.score(X_test,y_test)
    
        yyTr.append(ssumTr/rounds)
        yyTs.append(ssumTs/rounds)
    
        print("Average training time after {} rounds: {}".format(rounds,tr_time/rounds))
        print("average accuracy: {}".format(yyTs[-1]))
        
    plt.plot(xx, yyTs, '-o',lw=lws ,label=name+' (test)')
    plt.plot(xx, yyTr, '--o',lw=lws, label=name+' (train)')
    
plt.legend(loc="lower right")
plt.xlabel("Relative training set size")
plt.ylabel("Accuracy")
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:44:55 2019

@author: Tatsiana Palikarpava
"""

#%%
""" Ïmport libraries... """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

from sklearn.model_selection import cross_val_score
from time import time
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict

from sklearn.model_selection import RandomizedSearchCV


#%% 
""" Reading data from .csv files """

_train = pd.read_csv("train.csv")
print(_train.info())

_test = pd.read_csv("test.csv")
""" Showing Saleprice distribution... """
plt.figure(1)
sns.distplot(_train['SalePrice']).set_title('Sale Price Distribution Training Dataset');
plt.show()
_train['LSalePrice'] = np.log(_train['SalePrice'])
plt.figure(2)
sns.distplot(_train['LSalePrice']).set_title('Log Sale Price Distribution Training Dataset');
plt.show()

#%%

""" Work with missing values """

# Obtain columns with missing values
null_columns=_train.columns[_train.isnull().any()]
#_train[null_columns].isnull().sum()
""" Showing missing values statistics """
labels = []
values = []
for col in null_columns:
    labels.append(col)
    values.append(_train[col].isnull().sum())
labels = [i for i in labels]
values = [i for i in values]
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(5,10))
rects = ax.barh(ind, np.array(values), color='violet')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_ylabel("Column Names")
ax.set_title("Variables with missing values");
plt.show()

#%%
""" Some more data transformation """
# 'Id' is useless for learning column, so we drop it
train0 = _train.drop('Id', axis = 1)
# Saving correct values of price
y = train0['LSalePrice']
train0 = train0.drop('SalePrice', axis = 1)

test0 = _test.drop('Id', axis = 1)

""" Showing heatmap """
corrmat = train0.corr()
k = 20 #number of variables for heatmap
cols = corrmat.nlargest(k, 'LSalePrice')['LSalePrice'].index
nulls =(train0.isnull().sum()>0)
for c in cols:
    if nulls[c] == True:
        cols = cols.drop(c)

cm = np.corrcoef(train0[cols].values.T)
plt.figure(figsize = (20,7))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
train0 = train0.drop('LSalePrice', axis = 1)
#print(_train.shape)
#print(y)

#%% 
""" Eliminating columns with much missed data """

percent_to_delete = 15

""" Merge train & test """
comb = train0.append(test0, sort=False)
val_to_delete = comb.shape[0]*percent_to_delete/100

#print(comb.shape)
nulls =(comb.isnull().sum()>val_to_delete)
cols = comb.columns
print("Features to delete:")
for c in cols:
    if nulls[c] == True:
        print(c)
        comb = comb.drop(c, axis = 1)

#%%
""" Imputing missing values in numerical data """
cols = comb.columns
# Here you can choose between Simple Imputer and Iterative Imputer
iterative = True
if not iterative:
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
else:
    imp = IterativeImputer()
""" Dividing numerical and categorical data """
not_num = []
for c in cols:
    if comb[c].dtype == object:
        not_num.append(c)
num = cols.drop(not_num)

imp.fit(comb[num])
comb[num] = imp.transform(comb[num])

""" Imputing missing values in categorical data """
imp2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp2.fit(comb[not_num])
comb[not_num] = imp2.transform(comb[not_num])

# Transform few numerical features to categorical because of their meaning
comb['MSSubClass'] = comb['MSSubClass'].astype(str)
comb['MoSold'] = comb['MoSold'].astype(str)
#%% 
""" Dimensionality reduction """
# ATTENTION: TIME CONSUMING
# DO NOT USE TOGETHER WITH FEATURE IMPORTANCE ANALYSIS
dim_red = False
if dim_red:
    import prince
    # Here you can choose between PCA and FAMD
    pca = True
    if pca:
        # One-hot encoding
        dummies = pd.get_dummies(comb)
        pca = prince.PCA(n_components=50)
        pca = pca.fit(dummies)
        expl = (pca.explained_inertia_)
        cum = (np.cumsum(expl))[-1]
        print("Explained variance " + str(cum))
        dummies = pca.transform(dummies)
    else:
        famd = prince.FAMD(n_components=50)
        famd = famd.fit(comb)
        expl = (famd.explained_inertia_)
        cum = (np.cumsum(expl))[-1]
        print("Explained variance " + str(cum))
        comb = famd.transform(comb)
        # One-hot encoding
        dummies = pd.get_dummies(comb)
   
#%%
""" Split combined data """
if not dim_red:
    dummies = pd.get_dummies(comb)
X_train = dummies.iloc[:len(train0)]
    #print(X_train.shape)
y_train = y
    #print(y_train.shape)
X_test = dummies.iloc[len(train0):]
    #print(X_test.shape)

#%%
""" Performance analysis """
# ATTENTION: TIME CONSUMING
perf_an = False
if perf_an:
    estimators = [200, 500, 1000, 1500, 2000, 2500] 
    depths = [5, 15, 25, 35, 50, 75, 100] 
    features = [0.25, 0.33, 0.5, 'sqrt', 'log2'] 
    leaf = [1, 2, 5, 10]
    split = [2, 4, 6, 8]
    rounds = 10 # Number of repetitions to compute average error

    xx = np.array(estimators)
    #xx = np.array(depths)
    #xx = np.array(features)
    #xx = np.array(leaf)
    #xx = np.array(split)
    
    seed = np.random.randint(100)
    
    rng = np.random.RandomState(seed)  #to have the same for all classifiers
    yyTr = []
    yyTs = []
        
    for i in estimators:
        tr_time = 0
        ssumTs = 0
        for r in range(rounds):
            X_train_t, X_test_t, y_train_t, y_test_t = \
            train_test_split(X_train, y_train, test_size=0.3, random_state=rng)
            t_ini = time()
            #clf = RandomForestRegressor(n_estimators = i, min_samples_split = 2, min_samples_leaf = 1, max_depth = 125, max_features = 0.33)
            #clf = RandomForestRegressor(n_estimators = 2000, min_samples_split = 2, min_samples_leaf = 1, max_depth = i, max_features = 0.33)
            #clf = RandomForestRegressor(n_estimators = 2000, min_samples_split = 2, min_samples_leaf = 1, max_depth = 25, max_features = i)
            #clf = RandomForestRegressor(n_estimators = 2000, min_samples_split = 2, min_samples_leaf = i, max_depth = 25, max_features = 0.33)
            clf = RandomForestRegressor(n_estimators = i, min_samples_split = 2, min_samples_leaf = 1, max_depth = 25, max_features = 0.33)
            
            clf.fit(X_train_t, y_train_t)
            tr_time += time() - t_ini
            
            ssumTs += clf.score(X_test_t,y_test_t)
    

        yyTs.append(ssumTs/rounds)
    
        print("Average training time after {} rounds: {}".format(rounds,tr_time/rounds))
        print("average accuracy: {}".format(yyTs[-1]))
       
    plt.plot(xx, yyTs, '-o',lw=2 ,label='(test)')
   
    plt.legend(loc="lower right")
    #plt.xlabel("Relative training set size")
    plt.ylabel("Accuracy")
    plt.show()
   
#%%
"""  Tuning hyperparameters of random forest regressor """
# ATTENTION: TIME CONSUMING
hyp_tune = False
if hyp_tune:
    # Setting grid of parameters to explore
    # Number of trees in random forest
    n_estimators = [50, 200, 500, 1000, 1500, 2000, 2500]
    # Number of features to consider at every split
    max_features = [0.2, 0.33, 0.5, 'sqrt', 'log2']
    # Maximum number of levels in tree
    max_depth = [5, 15, 25, 35, 50, 75, 100]
    
    # Minimum number of samples required to split a node
    min_samples_split = [2, 4, 6, 8]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 5, 10]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    print (random_grid)
    # Use the random grid to search for best hyperparameters
    rf = RandomForestRegressor()
    # Random search of parameters, using 5 fold cross validation, 
    # search across 100 different combinations
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    
    rf_random.fit(X_train, y_train)
    print(rf_random.best_params_)
   
    rf_model = rf_random.best_estimator_
#%%
if not hyp_tune:
    rf_model = RandomForestRegressor(n_estimators = 1600, min_samples_split = 2, min_samples_leaf = 1, max_depth = 25, max_features = 0.33)
    rf_model.fit(X_train, y_train) 
    
    
""" Tree visualization """
# ATTENTION: TIME CONSUMING
vis = False
if vis:   
    from sklearn.tree import export_graphviz
    import pydot
    # Pull out one tree from the forest
    tree = rf_model.estimators_[5]
    # Export the image to a dot file
    export_graphviz(tree, out_file = 'tree.dot', feature_names = dummies.columns, rounded = True, precision = 1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    graph.write_png('tree.png')
    

#%%
""" Default feature importances """
importances = list(rf_model.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(cols, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
#[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

important = []
numbs = []
for f in feature_importances:
    if f[1] >= 0.005:
        important.append(f[0])
        numbs.append(f[1])
print(important)
# list of x locations for plotting
x_values = list(range(len(important)))
plt.figure(3)
# Make a bar chart
plt.bar(x_values, numbs, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
# Tick labels for x axis
plt.xticks(x_values, important, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
plt.show()

#%%
""" Permutation feature importances """
# ATTENTION: TIME CONSUMING
perm = False
if dim_red:
    perm = False
if perm:
    from rfpimp import permutation_importances
    X = X_train
    Y = y_train
    rf = rf_model
    def r2(rf, X_train, y_train):
        return r2_score(y_train, rf.predict(X_train))
    def acc(rf, X_train, y_train):
        return rf.score(X_train, y_train)
    
    perm_imp_rfpimp = permutation_importances(rf, X_train, y_train, r2)
    #print(perm_imp_rfpimp)
    
    not_important = []
    ni_numbs = []
    i = np.size(X_train,1) - 1
    while perm_imp_rfpimp.iloc[i]['Importance'] < 0.0001:
        important.append(perm_imp_rfpimp.iloc[i].name)
        ni_numbs.append(perm_imp_rfpimp.iloc[i]['Importance'])
        i-=1 
    
    important = []
    i_numbs = []
    i = 0
    while perm_imp_rfpimp.iloc[i]['Importance'] > 0.005:
        important.append(perm_imp_rfpimp.iloc[i].name)
        i_numbs.append(perm_imp_rfpimp.iloc[i]['Importance'])
        i+=1   
  
# list of x locations for plotting
    X_train = X_train.drop(not_important, axis = 1)
    X_test = X_test.drop(not_important, axis = 1)
    x_values = list(range(len(important)))
    plt.figure(4)
    # Make a bar chart
    plt.bar(x_values, i_numbs, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
    # Tick labels for x axis
    plt.xticks(x_values, important, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
    plt.show()


#%% 
""" Generating a submission """
X_train_t, X_test_t, y_train_t, y_test_t = \
            train_test_split(X_train, y_train, test_size=0.2, random_state=42)
rf_model.fit(X_train_t, y_train_t)
print("Estimated score: " + str(rf_model.score(X_test_t, y_test_t) ))
rf_model.fit(X_train, y_train) 
predicted_labels = rf_model.predict(X_test)
predicted_labels = pd.Series(predicted_labels)
submission = _test['Id']
submission = submission.to_frame()
submission['SalePrice'] = np.exp(predicted_labels)
submission.to_csv('submission.csv', index=False)















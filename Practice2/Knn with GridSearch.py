from sklearn import tree
import graphviz
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict,GridSearchCV,RandomizedSearchCV
from sklearn import  neighbors
import sklearn.metrics as met
import matplotlib.pyplot as plt
import time

#Insert data set
data=pd.read_csv('../Test/DataSets/tae.csv',sep=',',header=None)
train=data.ix[:,0:4]
target=data.ix[:,5]

# construct the set of hyperparameters to tune
params = {"n_neighbors": np.arange(3, 31, 2),"metric": ["euclidean", "cityblock"]}

# tune the hyperparameters via a cross-validated grid search
print("[INFO] tuning hyperparameters via grid search")
model = neighbors.KNeighborsClassifier()
grid = GridSearchCV(model, params)
start = time.time()
grid.fit(train, target)

# evaluate the best grid searched model on the testing data
print("[INFO] grid search took {:.2f} seconds".format( time.time() - start))
acc = grid.score(train, target)
print("[INFO] grid search accuracy: {:.2f}%".format(acc * 100))
print("[INFO] grid search best parameters: {}".format( grid.best_params_))

"""for i in range(5, 40):
    clf = neighbors.KNeighborsClassifier(n_neighbors=i)
    print('--------------------{}---------------------------------'.format(i))
    print('cross_val_predict')
    predicted = cross_val_predict(clf, train, target, cv=10, )  # predict y values for the test fold

    print('mean recall in all classes:')
    print(met.recall_score(target, predicted, average=None))

    print('mean precision in all classes')
    print(met.precision_score(target, predicted, average=None))

    print('mean accuracy in all classes:')
    print(met.accuracy_score(target, predicted))
    print('----------------------------------------')

    fig, ax = plt.subplots()
    ax.scatter(target, predicted, edgecolors=(0, 0, 0))
    ax.plot([target.min(), target.max()], [target.min(), target.max()], 'k--', lw=4)
    ax.set_xlabel('Measured k {}'.format(i))
    ax.set_ylabel('Predicted')
    plt.show()"""

"""for i in range(5,12):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    clf = clf.fit(train, target)
    pre=clf.predict(X=train)
    #print(pre)
    print('depth ',i,'\n',met.classification_report(target,pre))"""

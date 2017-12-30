
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict,GridSearchCV,RandomizedSearchCV
from sklearn import  neighbors
import time

#Insert data set
data=pd.read_csv('tae.csv',sep=',',header=None)
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


from sklearn import tree
import graphviz
import pandas as pd
from sklearn.model_selection import KFold,cross_val_predict
from sklearn import svm, tree, neighbors, neural_network, naive_bayes
import sklearn.metrics as met

data=pd.read_csv('DataSets/tae.csv',sep=',',header=None)
#print(data)
train=data.ix[:,0:4]
target=data.ix[:,5]
#print(x)
#print(target)

kf=KFold(n_splits=10)
print(kf)
print(kf.get_n_splits(train))
for train_index, test_index in kf.split(train):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = train.ix[train_index,:], train.ix[test_index,:]
   print(X_train)
   y_train, y_test = target[train_index], target[test_index]
   print(y_train)
for i in range(5, 40):
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

"""for i in range(5,12):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    clf = clf.fit(train, target)
    pre=clf.predict(X=train)
    #print(pre)
    print('depth ',i,'\n',met.classification_report(target,pre))"""

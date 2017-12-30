import sys
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
# from sklearn.metrics import recall_score
from sklearn import metrics
from sklearn import datasets
from sklearn import svm, tree, neighbors, neural_network, naive_bayes

print('---------------------------------------------')
iris = datasets.load_iris()
print('shape: ', iris.data.shape)
print('Target: ' , iris.target_names)
#------------------------------------------------

# create svm classifier
clf = svm.SVC(kernel='linear', C=1)
#clf = tree.DecisionTreeClassifier(criterion='gini')
#clf = tree.DecisionTreeRegressor()
#clf = neighbors.KNeighborsClassifier()
#clf = naive_bayes.GaussianNB()

fold_no = 10
scores = cross_val_score(clf, iris.data, iris.target, cv=fold_no, scoring='f1_macro')  # return score of test fold
print('Mean Accuracy for each fold:' , scores)
print("Mean Accuracy for all fold: %0.2f :" %scores.mean())
print("Std for all fold: %0.2f " %scores.std())

# ----------------------------------------------------------------------------------------

scoring = ['precision_macro', 'recall_macro', 'accuracy']

scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,cv=fold_no, return_train_score=True)
print(scores.keys()) # show all scoring keys

print('precision in train & test for each class')
print(scores['train_precision_macro'])
print(scores['test_precision_macro'])

print('train_precision_macro.mean(): ', scores['train_precision_macro'].mean())
print('test_precision_macro.mean():',scores['test_precision_macro'].mean())


print('recall in train & test for each class:')
print(scores['train_recall_macro'])
print(scores['test_recall_macro'])
print('train_recall_macro.mean(): ',scores['train_recall_macro'].mean())
print('test_recall_macro.mean(): ',scores['test_recall_macro'].mean())



print('accuracy in train & test for each class:')
print(scores['train_accuracy'])
print(scores['test_accuracy'])

print('-----------------------------------------------------')
print('cross_val_predict')
predicted = cross_val_predict(clf, iris.data, iris.target, cv=10,)  # predict y values for the test fold

print ('mean recall in all classes:')
print( metrics.recall_score(iris.target, predicted, average='macro'))

print('mean precision in all classes')
print( metrics.precision_score(iris.target,predicted, average='macro'))

print('mean accuracy in all classes:')
print(metrics.accuracy_score(iris.target, predicted))
print('----------------------------------------')

print('mean f1 in all classes:')
print(metrics.f1_score(iris.target, predicted,average='weighted'))
print('----------------------------------------')

# loading libraries
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import datasets,svm, tree, neighbors, neural_network, naive_bayes
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
df = datasets.load_iris()
# create design matrix X and target vector y
X = df.data 	# end index is exclusive
y = df.target 	# another way of indexing a pandas df

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)
print(lambda x: x % 2 != 0, myList)
# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
# changing to misclassification error
MSE = [1 - x for x in cv_scores]
print(MSE)
# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print ("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
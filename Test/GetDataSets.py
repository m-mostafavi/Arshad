import  pandas as pa
from sklearn import tree
f=pa.read_excel('DataSets/default of credit card clients.xls')
#print(f.)
#print(f['Y'])
#print(f.shape)
print(f.columns)
print(f.info())
print(f.ix[23])
#f.to_csv('asfdasdf.csv')
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(f.ix[1:,:24], f.ix[1:,25:25])



#######################################
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)



##################################
from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
iris.items()
#print(digits)
#print(digits.target)
#print(digits.images[0])



file = open('testSaveFile', 'w')
file.write(str(iris))
file.close()

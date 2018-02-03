import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from itertools import cycle, islice
from sklearn.metrics import pairwise_distances
from sklearn import metrics
#Insert data set
data=pd.read_csv('tae.csv',sep=',',header=None)
X=data.ix[:,0:4]

X = StandardScaler().fit_transform(X)

plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
plot_num = 1

target=data.ix[:,5]
#print(target)
silhouetteScoreValue=[]
FMIValue=[]
homogeneityValue=[]
completenessValue=[]
v_measureValue=[]
for i in np.arange(2, 10):
    clf =KMeans(n_clusters=i).fit(X)
    #print(clf)
    if hasattr(clf, 'labels_'):
        y_pred = clf.labels_.astype(np.int)
        #print(y_pred)
    else:
        y_pred = clf.predict(X)
       # print(y_pred)
   # plt.subplot(2, 2, plot_num)
   #  colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
   #                                           '#f781bf', '#a65628', '#984ea3',
   #                                           '#999999', '#e41a1c', '#dede00']),
   #                                    int(max(y_pred) + 1))))
   # print(X[:, 1],X[:, 2])
    #plt.scatter(X[:, 1], X[:, 2], s=2, color=colors[y_pred])

    labels = clf.labels_
    #print(labels)
    silhouetteScoreValue.append(metrics.silhouette_score(X, labels, metric='euclidean'))
    FMIValue.append(metrics.fowlkes_mallows_score(target,labels))
    homogeneityValue.append(metrics.homogeneity_score(target,labels))
    completenessValue.append(metrics.completeness_score(target,labels))
    v_measureValue.append(metrics.v_measure_score(target,labels))
    #plot_num+=1
#plt.subplot(2, 2, 1)
plt.plot(np.arange(2,10),silhouetteScoreValue,marker='*', linestyle='-.',color='g',label='silhouette_score')
plt.plot(np.arange(2, 10),FMIValue,marker='o', linestyle='--',color='r',label='fowlkes_mallows_score')
plt.plot(np.arange(2, 10),homogeneityValue,marker='s', linestyle='-',color='c',label='homogeneity_score')
plt.plot(np.arange(2, 10),completenessValue,marker='+', linestyle='-.',color='m',label='completeness_score')
plt.plot(np.arange(2, 10),v_measureValue,marker='D', linestyle=':',color='k',label='v_measure_score')
plt.legend(loc='upper right')
plt.show()
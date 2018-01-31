import numpy as np
from sklearn import cluster as clus
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from itertools import cycle, islice
#Insert data set
data=pd.read_csv('tae.csv',sep=',',header=None)
X=data.ix[:,0:4]
X = StandardScaler().fit_transform(X)
plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
plot_num = 1
plt.subplot(4, 4, plot_num)
target=data.ix[:,5]
for i in np.arange(1, 20):
    clf =clus.k_means(X,n_clusters=i)
    if hasattr(clf, 'labels_'):
        print(1)
        y_pred = clf.labels_.astype(np.int)
    else:
        print(22)
        y_pred = clf.predict(X)
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
    print(y_pred)
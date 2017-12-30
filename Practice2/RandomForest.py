import pandas as pd
import numpy as np
from matplotlib import gridspec
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import sklearn.metrics as met


#Insert data set
data=pd.read_csv('tae.csv',sep=',',header=None)
train=data.ix[:,0:4]
target=data.ix[:,5]

recall=[]
precision=[]
accuracy=[]
f1score=[]


#random_state: 0 , 5 , 10
#n_estimators: 5 , 10 , 15
for i in np.arange(2, 20):
    clf =  RandomForestClassifier(n_estimators=15,max_depth=i, random_state=5 )
    print('--------------------{}---------------------------------'.format(i))
    print('cross_val_predict')
    predicted = cross_val_predict(clf, train, target, cv=10, )  # predict y values for the test fold
    print('mean recall in all classes:')
    re=met.recall_score(target, predicted, average='macro')
    recall.append([i,re])
    print(re)
    print('mean precision in all classes')
    pre=met.precision_score(target, predicted, average='macro')
    precision.append([i,pre])
    print(pre)
    print('mean accuracy in all classes:')
    acc=met.accuracy_score(target, predicted)
    accuracy.append([i,acc])
    print(acc)
    print('mean f1score in all classes:')
    f1=met.f1_score(target, predicted, average='macro')
    f1score.append([i,f1])
    print(f1)
    print('----------------------------------------')
gs = gridspec.GridSpec(3, 2)
fig = plt.figure()
ax1 = plt.subplot(gs[0,:-1])
r = np.array(recall)
ax1.plot(r[:,0],  r[:,1], 'ro')
ax1.set_xlabel('measure depth')
ax1.set_ylabel('recall')
# ------------------------------------
ax2 = plt.subplot(gs[0,-1:])
p=np.array(precision)
ax2.plot(p[:,0],  p[:,1], 'ro')
ax2.set_xlabel('measure depth ')
ax2.set_ylabel('precision')
# ------------------------------------
ax3 = plt.subplot(gs[1,:])
a=np.array(accuracy)
ax3.plot(a[:,0],  a[:,1], 'ro')
ax3.set_xlabel('measure depth ')
ax3.set_ylabel('accuracy')
# ------------------------------------
ax4 = plt.subplot(gs[2,:])
f=np.array(f1score)
ax4.plot(f[:,0],  f[:,1], 'ro')
ax4.set_xlabel('measure depth ')
ax4.set_ylabel('f1score')
#------------------------------------
plt.show()
print(max( f[:,1]))
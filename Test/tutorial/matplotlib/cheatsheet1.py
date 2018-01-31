import numpy as np
#درست کردن یک آرایه به تعداد صد بین مقادیر 0 و 10
x=np.linspace(0,10,100)
#print(x)
#-----------------------------------------------
y=np.cos(x)
z=np.sin(x)
#print(z)
#2D Data or Images
#-----------------------------------------------
#یک آرایه دوبعدی که تابع رندوم را 2 برابر میکند
data=2*np.random.random((10,10))
#یک آرایه دوبعدی که تابع رندوم را 3 برابر میکند
data2=3*np.random.random((10,10))
#print(data2)
#-----------------------------------------------
#آرایه 100*100 که با مقادیر -3 تا 3 پ ر شده اند.
Y, X = np.mgrid[-3:3:100j, -3:3:100j]
#print(X**2)
U = -1 - X ** 2 + Y
#print(U)
V = 1 + X - Y ** 2
print(V)

from matplotlib.cbook import get_sample_data
img = np.load(get_sample_data('axes_grid/bivariate_normal.npy'))
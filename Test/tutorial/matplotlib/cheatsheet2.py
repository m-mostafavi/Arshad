import matplotlib.pyplot as plt
#figure
fig=plt.figure()
fig2=plt.figure(figsize=plt.figaspect(-1.5))

#َAxes
fig.add_axes()
#اولی تعدا سطر - دومی تعداد ستون - سومی ضرب سطر و ستون و جاسگاه در این ضرب است
ax1 = fig.add_subplot(221) # row-col-num
ax3 = fig.add_subplot(235)
fig3, axes = plt.subplots(nrows=2,ncols=5)
fig4, axes2 = plt.subplots(ncols=3)
plt.show()

"""
=========================================================
Comparing different clustering algorithms on toy datasets
=========================================================

This example shows characteristics of different
clustering algorithms on datasets that are "interesting"
but still in 2D. With the exception of the last dataset,
the parameters of each of these dataset-algorithm pairs
has been tuned to produce good clustering results. Some
algorithms are more sensitive to parameter values than
others.

The last dataset is an example of a 'null' situation for
clustering: the data is homogeneous, and there is no good
clustering. For this example, the null dataset uses the
same parameters as the dataset in the row above it, which
represents a mismatch in the parameter values and the
data structure.

While these examples give some intuition about the
algorithms, this intuition might not apply to very high
dimensional data.
"""
print(__doc__)
#متن اول صفحه را در خروجی چاپ میکند

import time
#ماژول تایم را به کد اضافه میکند

import warnings
#ماژول خطا را به کد اضافه میکند


import numpy as np
#ماژول numpy را با نام مستعار np  به کد اضافه میکند


import matplotlib.pyplot as plt
#ماژول pyplot را به جهت رسم دیاگرام با نام مستعار plt  به کد اضافه میکند


from sklearn import cluster, datasets, mixture
#ماژول های مربوط به خوشه بندی، دیتاست ها و ترکیب گوسی را به کد اضافه میکند


from sklearn.neighbors import kneighbors_graph
#ماژول گراف KNN را به کد اضفه میکند


from sklearn.preprocessing import StandardScaler
#این ماژول برای نرمال سلزی داده ها استفاده می شود


from itertools import cycle, islice
#تکرر کننده های cycle , islice جهت ساختن ساختمان داده های تکرارکننده


np.random.seed(0)
#initialiser  برای تابع رندوی numpy است


# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1500
#تعداد نمونه هایی که در دیتاست های بعدی استفاده می شود را برابر 1500 نمونه در نظر میگیرد


noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)
#یک دیتاست به شکل دایره دار درست می کند


noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
#دیتاست به شکل ماه درست میکند


blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
#دیتاست به شکل حباب حباب درست میکند


no_structure = np.random.rand(n_samples, 2), None
#دیتاست بدون اهیچ شکلی و به صورت رندوم درست میکند

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
#1500 نمونه  دوبعدی درست میکند


transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
#برای آرایه های دو بعدی معادل ضرب برداری است

aniso = (X_aniso, y)
#تشکیل یک آرایه دو بعدی از X_aniso , y


# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)
#ایجاد دیتاست حباب با واریانس های متفاوت

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 12.5))
#یک شکل با انداز تعیین شده مشخص میکند
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
#طرح subplot را تنظیم میکند

plot_num = 1

default_base = {'quantile': .3,
                'eps': .3, #مقدار اپسیلون در الگوریتم های مبتنی بر چگالی
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10, #تعداد همسایه ها
                'n_clusters': 3} #تعداد خوشه ها
#ایجاد یک لیست برای پارامترهای دیفالت در دیتاست ها


datasets = [
    (noisy_circles, {'damping': .77, 'preference': -240,'quantile': .2, 'n_clusters': 2}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (varied, {'eps': .18, 'n_neighbors': 2}),
    (aniso, {'eps': .15, 'n_neighbors': 2}),
    (blobs, {}),
    (no_structure, {})]
#ایجاد یک لیست از دیتاست های ایجاد شده به همرا  پارامترهای اولیه متفاوت

#در اینجا با ایجاد دو حلقه تو در تو انواع دیتاست ها را با الگوریتم های مختلف ترسیم میکند
for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # متغییر اختصاص داده شده در هر دور
    #dataset  موجود در  هر مرحله از datasets را بر میگرداند
    #algo_params پارامترهای هر دیتاست را بر میگرداند


    # update parameters with dataset-specific values
    params = default_base.copy()
    #یک کپی از پارامترهای دیفالت برای دیتاست ها را بر میگرداند

    params.update(algo_params)
    #پارامترها با مقادیر پارامترهای هر مرحله آپدیت میکند

    X, y = dataset
    #X: نمونه های دوبعدی دیتا ست که در آرایه قرار دارد
   #y: تارگت هر نمونه که البته در آرایه قراردارد

    X = StandardScaler().fit_transform(X)
#نرمال کردن مجموعه داده ها برای انتخاب پارامتر ساده تر

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
#برآورد پهنای باند جهت استفاده از شیفت میانگین

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=params['n_neighbors'], include_self=False)
    #ماتریس اتصال برای بخش ساختار یافته

    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
# انتصاب الگوریتم شیفت میانگین که از یک کرنل فلت استفاده میکند به متغیر ms -
#  جهت عملکرد بهتر connectivity  قبلا محاسبه شده است که به جهت استفاده از کرنل RBF مهم است

    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
#انتساب الگوریتمMiniBatchKMeans با متغیر two_means

    ward = cluster.AgglomerativeClustering( n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
#یک نوع الگوریتم سلسله مراتبی هستش که اشیا خوشه بندی سلسله مراتبی را با استفاده از رویکرد پایین به بالا انجام می ده
    # که متغیر linkage مجموع اختلاف مربع در تمام خوشه ها را کاهش می دهد

    spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack',affinity="nearest_neighbors")
# استفاده از الگوریتم SpectralClustering

    dbscan = cluster.DBSCAN(eps=params['eps'])
    # استفاده از الگوریتم DBSCAN

    affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
# الگوریتم AffinityPropagation خوشه ها را با ارسال پیام ها بین جفت نمونه تا همگرایی ایجاد می کند

    average_linkage = cluster.AgglomerativeClustering( linkage="average", affinity="cityblock",n_clusters=params['n_clusters'], connectivity=connectivity)
    # یک نوع الگوریتم سلسله مراتبی هستش که اشیا خوشه بندی سلسله مراتبی را با استفاده از رویکرد پایین به بالا انجام می ده
    # که متغیر linkage مجموع اختلاف مربع در تمام خوشه ها را کاهش می دهد
    #در اینجا برای فاصله از فاصله منهتن استفاده کرده است
    birch = cluster.Birch(n_clusters=params['n_clusters'])
#الگوریتم Birch یک درخت به نام خصوصیت ویژگی درخت (CFT) برای داده های داده شده ایجاد می کند

    gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')
#استفاده از الگوریتم ماتریس گوسی

    clustering_algorithms = (
        ('MiniBatchKMeans', two_means),
        ('AffinityPropagation', affinity_propagation),
        ('MeanShift', ms),
        ('SpectralClustering', spectral),
        ('Ward', ward),
        ('AgglomerativeClustering', average_linkage),
        ('DBSCAN', dbscan),
        ('Birch', birch),
        ('GaussianMixture', gmm)
    )
# ایجاد متغیر

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.show()

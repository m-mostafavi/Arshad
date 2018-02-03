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
#متن اول صفحه را در خروجی چاپ میکند
print(__doc__)

#ماژول تایم را به کد اضافه میکند
import time

#ماژول خطا را به کد اضافه میکند
import warnings


#ماژول numpy را با نام مستعار np  به کد اضافه میکند
import numpy as np


#ماژول pyplot را به جهت رسم دیاگرام با نام مستعار plt  به کد اضافه میکند
import matplotlib.pyplot as plt


#ماژول های مربوط به خوشه بندی، دیتاست ها و ترکیب گوسی را به کد اضافه میکند
from sklearn import cluster, datasets, mixture


#ماژول گراف KNN را به کد اضفه میکند
from sklearn.neighbors import kneighbors_graph


#این ماژول برای نرمال سلزی داده ها استفاده می شود
from sklearn.preprocessing import StandardScaler


#تکرر کننده های cycle , islice جهت ساختن ساختمان داده های تکرارکننده
from itertools import cycle, islice


#initialiser  برای تابع رندوی numpy است
np.random.seed(0)



# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
#تعداد نمونه هایی که در دیتاست های بعدی استفاده می شود را برابر 1500 نمونه در نظر میگیرد
n_samples = 100


#یک دیتاست به شکل دایره دار درست می کند
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)

#دیتاست به شکل ماه درست میکند
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)


#دیتاست به شکل حباب حباب درست میکند
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)


#دیتاست بدون اهیچ شکلی و به صورت رندوم درست میکند
no_structure = np.random.rand(n_samples, 2), None


# Anisotropicly distributed data
#تعریف متغیر رندوم سایز
random_state = 170

#1500 نمونه  دوبعدی درست میکند
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)

#تعریف متغیر برای استفاده در ضرب برداری
transformation = [[0.6, -0.6], [-0.4, 0.8]]

#برای آرایه های دو بعدی معادل ضرب برداری است
X_aniso = np.dot(X, transformation)

#تشکیل یک آرایه دو بعدی از X_aniso , y
aniso = (X_aniso, y)



# blobs with varied variances
#ایجاد دیتاست حباب با واریانس های متفاوت
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)


# ============
# Set up cluster parameters
# ============
#یک شکل با انداز تعیین شده مشخص میکند
plt.figure(figsize=(9 * 2 + 3, 12.5))

#طرح subplot را تنظیم میکند
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)


plot_num = 1

#ایجاد یک لیست برای پارامترهای دیفالت در دیتاست ها
default_base = {'quantile': .3,
                'eps': .3, #مقدار اپسیلون در الگوریتم های مبتنی بر چگالی
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10, #تعداد همسایه ها
                'n_clusters': 3} #تعداد خوشه ها


#ایجاد یک لیست از دیتاست های ایجاد شده به همرا  پارامترهای اولیه متفاوت
datasets = [
    (noisy_circles, {'damping': .77, 'preference': -240,'quantile': .2, 'n_clusters': 2}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (varied, {'eps': .18, 'n_neighbors': 2}),
    (aniso, {'eps': .15, 'n_neighbors': 2}),
    (blobs, {}),
    (no_structure, {})]


#در اینجا با ایجاد دو حلقه تو در تو انواع دیتاست ها را با الگوریتم های مختلف ترسیم میکند
# متغییر اختصاص داده شده در هر دور
# dataset  موجود در  هر مرحله از datasets را بر میگرداند
# algo_params پارامترهای هر دیتاست را بر میگرداند
for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # متغییر اختصاص داده شده در هر دور
    #dataset  موجود در  هر مرحله از datasets را بر میگرداند
    #algo_params پارامترهای هر دیتاست را بر میگرداند


    # update parameters with dataset-specific values
    #یک کپی از پارامترهای دیفالت برای دیتاست ها را بر میگرداند
    params = default_base.copy()

    #پارامترها با مقادیر پارامترهای هر مرحله آپدیت میکند
    params.update(algo_params)

    # X: نمونه های دوبعدی دیتا ست که در آرایه قرار دارد
    # y: تارگت هر نمونه که البته در آرایه قراردارد
    X, y = dataset

    # نرمال کردن مجموعه داده ها برای انتخاب پارامتر ساده تر
    X = StandardScaler().fit_transform(X)


    # estimate bandwidth for mean shift
    # برآورد پهنای باند جهت استفاده از شیفت میانگین
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])


    # connectivity matrix for structured Ward
    #ماتریس اتصال برای بخش ساختار یافته
    connectivity = kneighbors_graph(X, n_neighbors=params['n_neighbors'], include_self=False)


    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    # انتصاب الگوریتم شیفت میانگین که از یک کرنل فلت استفاده میکند به متغیر ms -
    #  جهت عملکرد بهتر connectivity  قبلا محاسبه شده است که به جهت استفاده از کرنل RBF مهم است
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

    # انتساب الگوریتمMiniBatchKMeans با متغیر two_means
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])

    # یک نوع الگوریتم سلسله مراتبی هستش که اشیا خوشه بندی سلسله مراتبی را با استفاده از رویکرد پایین به بالا انجام می ده
    # که متغیر linkage مجموع اختلاف مربع در تمام خوشه ها را کاهش می دهد
    ward = cluster.AgglomerativeClustering( n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)

    # استفاده از الگوریتم SpectralClustering
    spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack',affinity="nearest_neighbors")

    # استفاده از الگوریتم DBSCAN
    dbscan = cluster.DBSCAN(eps=params['eps'])

    # الگوریتم AffinityPropagation خوشه ها را با ارسال پیام ها بین جفت نمونه تا همگرایی ایجاد می کند
    affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])

    # یک نوع الگوریتم سلسله مراتبی هستش که اشیا خوشه بندی سلسله مراتبی را با استفاده از رویکرد پایین به بالا انجام می ده
    # که متغیر linkage مجموع اختلاف مربع در تمام خوشه ها را کاهش می دهد
    # در اینجا برای فاصله از فاصله منهتن استفاده کرده است
    average_linkage = cluster.AgglomerativeClustering( linkage="average", affinity="cityblock",n_clusters=params['n_clusters'], connectivity=connectivity)

    # الگوریتم Birch یک درخت به نام خصوصیت ویژگی درخت (CFT) برای داده های داده شده ایجاد می کند
    birch = cluster.Birch(n_clusters=params['n_clusters'])

    # استفاده از الگوریتم ماتریس گوسی
    gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')

    # ایجاد متغیر آرایه با استفاده از متغیرهای الگوریتم های ایجاد شده
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


#در اینجا یک حلقه ایجاد شده  که الگوریتم های ایجاد شده در بلارا برو داده های گرفته شده ارز حلقه بالا اجرا میکند
    for name, algorithm in clustering_algorithms:

        #شروع زمان اجرای الگوریتم
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        #در اینجا به جهت جلوگیری از خطا هایی که ممکن است در الگوریتم kneighbors_graph رخ دهد خطاها چک میشود
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
            #اگر خطا رخ ندهد الگوریتم را بر روی دیتاست اعمال میکند
            algorithm.fit(X)
        #زمان تمام شدن الگوریتم
        t1 = time.time()
        #چک میکند که labels_  در خروجی الگوریتم وجود دارد یا خیر اگر لیبلی در خورد دیتا وجود داشته باشد در اینجا متوجه میشود
        #کلن خوشه بندی های به دونوع تقسیم میشود : یکی اینکه لیبل های خوشه بندی جزو یکی از ویژگی ها است
        # و دیگر اینکه با فیت کردن لیل های خروجی برای هر خوشه مشخص شود
        if hasattr(algorithm, 'labels_'):
            #اگر لیبل خوشه جزو یکی از ویژگی ها باشد در اینجا در y_pred قرار میگیرد
            y_pred = algorithm.labels_.astype(np.int)
        else:
            #پیشبینی خوشه که در هر نمونه در کدام دسته قرار میگیرد
            y_pred = algorithm.predict(X)
        #سابپلات های مختلف برای هر خوش در هر الگوریتم درست میکند
        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)

        if i_dataset == 0:
            plt.title(name, size=10)
        #رنگهای مختلف برای استفاده در پلات
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        #یک اسکاتر درست میکند برای قرارداد شکل های ایجاد شده در پلات
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        #تنظیمات مربوط به نمایش پلات
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        #مقدار زمان اجرای الگوریتم را در هر شکل قرار میدهد
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'), transform=plt.gca().transAxes, size=15,horizontalalignment='right')
        #یک واحد به شماره پلات برای استفاده در دور بعدی استفاده میکند
        plot_num += 1
#نمایش پلات و نمودارها
plt.show()

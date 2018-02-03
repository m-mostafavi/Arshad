#در ابتدا قصد داریم یک نمودار خطی ساده رسم کنیم. بدین منظور دو لیست با نام های x و y ایجاد می‌کنیم؛ به طوری مقدار عناصر y برابر با مربع مقدار عناصر x است. سپس با استفاده از متدهای plot و show، به رسم نمودار می‌پردازیم.

import  matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[1,4,9,16,25]
plt.plot(x,y)
plt.show()
#می‌توان با استفاده از متدهای xlabel و ylabel به محورهای x و y در نمودار نامی اختصاص داد و با استفاده از متد title برای نمودار عنوانی مشخص کرد.
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[1,4,9,16,25]
plt.xlabel('X')
plt.ylabel('Y=X^2')
plt.plot(x,y)
plt.show()
# دو نمودار ساده را در یک صفحه مختصات رسم می‌کنیم و با بکار گیری امکانات موجود در Matplotlib آن‌ها را متمایز از یکدیگر نشان داده و همچنین شرح نمودارها (legend)  را نیز در طرح ترسیم شده‌ی خود نشان می‌دهیم.
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y1=[1,4,9,16,25]
y2=[1,8,27,64,125]

plt.plot(x,y1,marker='*', linestyle='-',color='g',label='X^2')
plt.plot(x,y2,marker='o', linestyle='--',color='r',label='X^3')

plt.xlabel('X')
plt.ylabel('Ys')

plt.title('خطوص ساده')
plt.legend(loc='lower right')
plt.show()

#برای رسم نمودار پراکندگی مربوط به داده‌ها از متد scatter استفاده می‌شود. کار با این متد همانند کار با متد plot است و می‌توان تنظیمات یکسانی برای آن در حین آماده سازی نمودار در نظر گرفت
print(__doc__)
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[1,4,9,16,25]


plt.scatter(x,y1,marker='*', color='g')

plt.xlabel('X')
plt.ylabel('Y')
plt.axis([0,5,0,30])
plt.title('scatter')
#plt.legend(loc='lower right')
plt.show()
#================================================================================
#کشیدن خطوط در یک خط
import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()

#==================================================================================
#در یک نمودار میله‌ای، در محور x نام متغیرها یا مشخصه‌های موجود در مجموعه داده‌ی ما قرار خواهد گرفت و در محور y بازه‌ی مقادیر مشخص شده است. برای هر مشخصه، میله‌ای به اندازه‌ی مقدار آن رسم می‌شود.
import pandas as pa
import numpy as np
import random as ra
import matplotlib.pyplot as plt
value=ra.sample(range(30),10)
label1=ra.sample(range(20,30),10)
label2=['a','b','c','d','e','f','g','h','i','j','k',]
plt.bar(label1,value)
plt.xticks(label1,label2)
plt.show()
plt.close()
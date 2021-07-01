import pathlib
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import cv2
import binascii
import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster


plt.style.use("ggplot")

# img_path = r'D:\Python\git\Python_GUI\data\img\FLIR_09312.jpg'
img_path = r'C:\Users\ipx\Desktop\Personal\Python_GUI\data\img\FLIR_09312.jpg'
# img_path = r'C:\Users\ipx\Desktop\Personal\Python_GUI\data\img\test.png'


img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128,128))

ar = np.asarray(img)
shape = ar.shape
ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)


from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture(n_components=20, random_state=1, weight_concentration_prior_type="dirichlet_process", init_params="kmeans")

bgm.fit(ar)


print(
    "Number of clusters:\t",len(bgm.means_)
)

print(
    bgm.weights_
)








vecs, dist = scipy.cluster.vq.vq(ar, bgm.means_)
counts, bins = np.histogram(vecs, len(bgm.means_))



for idx, (k,v) in enumerate(zip(bgm.means_,counts)):
    
    # plt.hist([idx]*counts[idx], color=(1,0,0))
    plt.hist([idx]*round(100*v/(img.shape[0]*img.shape[1])), color=k/255, rwidth=19)



print(set(vecs))





# for i, (m, w) in enumerate(zip(bgm.means_, bgm.weights_)):
    # print(m, w)

plt.show()



# exit()
import pandas as pd



distortions = []

for i in range(1,20):
    codes, dist = scipy.cluster.vq.kmeans(ar, i)
    distortions.append(dist)
    

x = np.arange(1,20)
y = np.array(distortions)

z = np.polyfit(x,y,3)
print(z)

d2 = np.roots([6*z[0],2*z[1]])
d1 = np.roots([3*z[0], 2*z[1], z[2]])


print(d1)
print(d2)

elbow_plot = pd.DataFrame({'num_clusters' : np.arange(1,20),
                           'distortions' : distortions})

import seaborn as sns
sns.lineplot(x = 'num_clusters', y = 'distortions', data = elbow_plot)
# plt.show()
# plt.clf()

print('reading image')

# NUM_CLUSTERS = 20

# print('finding clusters')
# codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
# print('cluster centres:\n', codes)

# vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
# counts, bins = np.histogram(vecs, len(codes))    # count occurrences


for idx, (k,v) in enumerate(zip(codes,counts)):
    
    # plt.hist([idx]*counts[idx], color=(1,0,0))
    plt.hist([idx]*round(100*v/(img.shape[0]*img.shape[1])), color=k/255, rwidth=19)

# N, bins, patches = plt.hist(vecs, bins =bins)

# plt.show()


index_max = np.argmax(counts)                    # find most frequent
peak = codes[index_max]
colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
print('most frequent is %s (#%s)' % (peak, colour))

# plt.style.use("dark_background")


codes = bgm.means_[np.where(bgm.weights_ > 0.001)]


row = 3
col = len(codes)//3 + 1

fig, ax = plt.subplots(row,col)


ax = ax.flatten()



import imageio
c = ar.copy()


for i, code in enumerate(codes):
    
    print(code)
    
    d = np.ones_like(c)
    
    c[scipy.r_[np.where(vecs==i)],:] = code
    
    d[scipy.r_[np.where(vecs==i)],:] = code
    
    

    tmp = np.array(d.reshape(*shape), dtype=int)
    
    ax[i].set_title(np.array(code, dtype=int))
    ax[i].imshow(tmp)
    
fig.tight_layout()
plt.show()




print("EOF")
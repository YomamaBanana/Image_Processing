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
img_path = r'D:\Python\git\Python_GUI\data\img\test.png'


img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128,128))

ar = np.asarray(img)
shape = ar.shape
ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)


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
# plt.xticks(np.arange(1,6))
plt.show()

plt.clf()

# exit()
print('reading image')
# im = Image.open('image.jpg')
# im = im.resize((150, 150))      # optional, to reduce time

# im = cv2.resize(img, (150,150))

NUM_CLUSTERS = 15

print('finding clusters')
codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
print('cluster centres:\n', codes)

vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
counts, bins = np.histogram(vecs, len(codes))    # count occurrences


for idx, (k,v) in enumerate(zip(codes,counts)):
    
    # plt.hist([idx]*counts[idx], color=(1,0,0))
    plt.hist([idx]*round(100*v/(img.shape[0]*img.shape[1])), color=k/255, rwidth=19)

# N, bins, patches = plt.hist(vecs, bins =bins)

# print(codes[0])

# for idx, patch in enumerate(patches):
    # patches[idx].set_facecolor()

plt.show()


index_max = np.argmax(counts)                    # find most frequent
peak = codes[index_max]
colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
print('most frequent is %s (#%s)' % (peak, colour))

# plt.imshow([colour])
# plt.show()

plt.style.use("dark_background")


row = 3
col = len(codes)//3 + 1

fig, ax = plt.subplots(row,col)

# fig.axis("off")

ax = ax.flatten()



import imageio
c = ar.copy()

for i, code in enumerate(codes):
    d = np.ones_like(c)
    
    c[scipy.r_[np.where(vecs==i)],:] = code
    
    d[scipy.r_[np.where(vecs==i)],:] = 255
    
    

    tmp = np.array(d.reshape(*shape), dtype=int)
    
    ax[i].set_title(np.array(code, dtype=int))
    ax[i].imshow(tmp)
    
        
    # plt.imshow(tmp)

# plt.imshow(c.reshape(*shape).astype(np.uint8))
    
    
# imageio.imwrite('clusters.png', c.reshape(*shape).astype(np.uint8))
# print('saved clustered image')
plt.show()




print("EOF")